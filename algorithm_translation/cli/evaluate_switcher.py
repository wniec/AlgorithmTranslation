"""Benchmark a translating switcher on held-out BBOB problems.

    uv run eval-switch PSO CMAES --model models/PSO_CMAES_ppo.pt \
        --n-switches 6 --fe-multiplier 2000 --cdb 1.0 --switch-mode random

An *agent* runs each BBOB problem for the full FE budget while **randomly
switching** between the two named algorithms across ``--n-switches + 1``
segments (see ``algorithm_translation.evaluation.switcher``). At every boundary
where the algorithm actually changes, the source optimizer's internal state is
handed to the target. We compare five ways of running that same schedule:

* ``<A>_only`` / ``<B>_only`` — no switching at all, one algorithm for the whole
  budget (the "would switching have helped?" baselines);
* ``cold_switch``           — same schedule, but every hand-off **cold-starts**
  the next optimizer (state discarded — the naive switcher);
* ``lossy_switch``          — same schedule, hand-offs carry only the shared
  population (positions/values/best), discarding algorithm-specific structure
  (the current *default* hand-off);
* ``translated_switch``     — same schedule, hand-offs run through the trained
  translator (velocities <-> covariance, etc.).

All five see the same problem, the same segment seeds, and (the switchers) the
same random algorithm schedule and switch points, so differences are attributable
to the hand-off alone.

The primary metric is **AOCC** (Area Over the Convergence Curve, in [0, 1],
higher = better anytime performance), computed over each run's best-so-far
trajectory exactly as DynamicAlgorithmSelection's ``agents/agent_utils`` — the
precision ``best_y - optimum`` is clipped to [1e-8, 1e8], log10-normalized, and
``1 - that`` is averaged over the FE budget. Methods are ranked per run by AOCC
(mean rank reported, lower = better) with head-to-head AOCC win rates. **ERT**
(Expected Running Time) is also reported per method, pooled over all runs, across
a fixed precision-target ladder (1e-3, 1e-4, ..., 1e-7).

Switch points come from one of two modes (``--switch-mode``), both honoring the
concentration-of-decision-boundaries knob ``--cdb``:

* ``random`` — sampled log-uniformly per problem via ``sample_switch_points``
  (``--cdb`` warps *where in log-FE space* they land: >1 earlier, <1 later);
* ``cdb``    — deterministic exponentially-spaced checkpoints via
  ``get_checkpoints`` (``--cdb`` is the growth ratio: 1.0 = uniform spacing,
  >1 = shorter early segments).
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from algorithm_translation.cli import ALGO_CHOICES, population_sizes
from algorithm_translation.cli.scenario import exclude_train_ids
from algorithm_translation.evaluation.anytime import ERT_TARGETS, average_ranks, ert
from algorithm_translation.evaluation.switcher import evaluate_problem
from algorithm_translation.suite import IOHSuite, problem_dim
from algorithm_translation.suite.bbob_splits import (
    ALL_DIMS,
    ALL_FUNCTIONS,
    INSTANCE_IDS,
    build_problem_ids,
)
from algorithm_translation.train_loop import (
    load_checkpoint_split,
    load_translator,
    resolve_model_path,
)


def summarize(rows, args):
    if not rows:
        print("no evaluable problems")
        return
    a, b = args.algo_a, args.algo_b
    methods = [f"{a}_only", f"{b}_only", "cold_switch", "lossy_switch", "translated_switch"]
    n = len(rows)

    aocc = {m: np.array([r[m]["aocc"] for r in rows], dtype=float) for m in methods}
    best = {m: np.array([r[m]["best"] for r in rows], dtype=float) for m in methods}
    # fett[m] is (n_runs, n_targets): FE-to-target per run for each ERT_TARGETS entry.
    fett = {m: np.array([r[m]["fett"] for r in rows], dtype=float) for m in methods}
    opt = np.array([r["optimum"] for r in rows], dtype=float)
    budgets = np.array([r["budget"] for r in rows], dtype=float)

    # Primary metric: mean rank by AOCC (higher AOCC = better -> rank on -aocc).
    rank_sums = dict.fromkeys(methods, 0.0)
    for idx in range(n):
        ranks = average_ranks([-rows[idx][m]["aocc"] for m in methods])
        for m, rk in zip(methods, ranks, strict=True):
            rank_sums[m] += rk
    mean_rank = {m: rank_sums[m] / n for m in methods}

    print(
        f"\n=== {a} <-> {b}  |  switch-mode={args.switch_mode}  cdb={args.cdb}  "
        f"n_switches={args.n_switches}  fe={args.fe_multiplier}/dim  "
        f"repeats={args.repeats}  |  {n} runs ==="
    )
    print(f"  {'method':20s} {'AOCC_rank':>9s} {'mean_AOCC':>10s}  median precision")
    for m in methods:
        star = " *" if mean_rank[m] == min(mean_rank.values()) else "  "
        prec = np.median(best[m] - opt)
        print(f"{star}{m:20s} {mean_rank[m]:>9.3f} {aocc[m].mean():>10.4f}  {prec:.4g}")

    # ERT across the fixed precision-target ladder, pooled over all runs. NOTE:
    # pooling across heterogeneous functions/dims is non-standard COCO usage (ERT
    # is normally per function+dim); read it as a coarse cross-suite summary.
    # Cell = ERT in FEs ('inf' = no run reached that target); (k) = # successes.
    print(f"\n  ERT (FEs) to precision target, pooled over {n} runs  [(k)=#runs reaching it]:")
    header = "  " + f"{'method':20s}" + "".join(f"{t:>13.0e}" for t in ERT_TARGETS)
    print(header)
    for m in methods:
        cells = []
        for j in range(len(ERT_TARGETS)):
            value, n_succ = ert(fett[m][:, j], budgets)
            cells.append(("inf" if not np.isfinite(value) else f"{value:.0f}") + f"({n_succ})")
        print("  " + f"{m:20s}" + "".join(f"{c:>13s}" for c in cells))

    # Head-to-head on AOCC: is the learned hand-off worth it? (higher = win)
    t = aocc["translated_switch"]
    print("\n  translated_switch AOCC vs:")
    best_standalone = np.maximum(aocc[f"{a}_only"], aocc[f"{b}_only"])
    for label, base_arr in (
        ("cold_switch", aocc["cold_switch"]),
        ("lossy_switch", aocc["lossy_switch"]),
        ("best standalone", best_standalone),
    ):
        ties_mask = np.isclose(t, base_arr)
        wins = int(((t > base_arr) & ~ties_mask).sum())
        ties = int(ties_mask.sum())
        losses = n - wins - ties
        print(f"    {label:16s}: {wins} win / {ties} tie / {losses} loss")

    fb = np.array([r["translated_fallbacks"] for r in rows])
    if fb.sum():
        print(
            f"\n  note: {int((fb > 0).sum())}/{n} runs had a degenerate source state "
            f"served lossily ({int(fb.sum())} hand-offs total)"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("algo_a", choices=ALGO_CHOICES)
    p.add_argument("algo_b", choices=ALGO_CHOICES)
    p.add_argument(
        "--model",
        default=None,
        help="translator checkpoint (default: models/<A>_<B>_ppo.pt or "
        "models/<A>_<B>_grpo.pt, whichever exists)",
    )
    p.add_argument("-d", "--dims", type=int, nargs="+", default=None,
                   help=f"BBOB dimensions (available: {ALL_DIMS}; default: the "
                        "checkpoint's, else 2 3 5)")
    p.add_argument("--instances", type=int, nargs="+", default=None,
                   help=f"BBOB instance ids (default: all {len(INSTANCE_IDS)} of them)")
    p.add_argument("--functions", type=int, nargs="+", default=None,
                   help="BBOB function ids 1-24 (default: the whole suite)")
    # Problems the checkpoint was trained on (its stored lopo/loio split) are
    # always excluded; with neither --functions nor --instances given, the
    # checkpoint's held-out test set is used as-is.
    p.add_argument("--fe-multiplier", type=int, default=2000, help="FE budget = this * dim")
    p.add_argument("--n-switches", type=int, default=6, help="number of hand-offs per run")
    p.add_argument("--cdb", type=float, default=1.0,
                   help="concentration of switch points (see module docstring)")
    p.add_argument("--switch-mode", choices=["random", "cdb"], default="random",
                   help="'random' samples switch points log-uniformly per run; "
                        "'cdb' places them deterministically via get_checkpoints")
    p.add_argument("--n-individuals", type=int, default=None, help="shared population size")
    p.add_argument("--n-individuals-a", type=int, default=None, help="override for algo_a")
    p.add_argument("--n-individuals-b", type=int, default=None, help="override for algo_b")
    p.add_argument("--repeats", type=int, default=1,
                   help="random schedules evaluated per problem (averaged into the stats)")
    p.add_argument("--max-problems", type=int, default=None, help="cap the (shuffled) problem set")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    model = resolve_model_path(args.model, args.algo_a, args.algo_b)
    pair = load_translator(model, device=args.device)
    print(f"loaded translator {pair.algo_a}<->{pair.algo_b} from {model}")

    n_for = population_sizes(args).get

    ckpt_split = load_checkpoint_split(model)
    if ckpt_split and not args.functions and not args.instances:
        ids = list(ckpt_split["test_ids"])
        if args.dims:
            dims = set(args.dims)
            ids = [i for i in ids if problem_dim(i) in dims]
        print(f"[split] using checkpoint's {ckpt_split['scenario']} test set")
    else:
        funcs = set(args.functions) if args.functions else ALL_FUNCTIONS
        instances = args.instances if args.instances else INSTANCE_IDS
        ids = build_problem_ids(funcs, args.dims or [2, 3, 5], instances)
    ids = exclude_train_ids(ids, ckpt_split)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    if args.max_problems is not None:
        ids = ids[: args.max_problems]
    suite = IOHSuite()

    rows = []
    for idx, pid in enumerate(tqdm(ids, desc="benchmark", unit="prob")):
        rows.extend(
            evaluate_problem(
                pair,
                pid,
                suite,
                algo_a=args.algo_a,
                algo_b=args.algo_b,
                n_for=n_for,
                fe_multiplier=args.fe_multiplier,
                n_switches=args.n_switches,
                switch_mode=args.switch_mode,
                cdb=args.cdb,
                repeats=args.repeats,
                seed=args.seed,
                problem_idx=idx,
            )
        )
    summarize(rows, args)


if __name__ == "__main__":
    main()
