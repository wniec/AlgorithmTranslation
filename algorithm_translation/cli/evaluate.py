"""Evaluate a trained translator: does the learned hand-off beat the default?

    uv run eval PSO CMAES --model models/PSO_CMAES_ppo.pt [options]

For each BBOB **test** problem we warm up the *source* optimizer to build a
genuine internal state, then switch to the *target* optimizer for the remaining
budget under three hand-offs:

* **translated** — the source state is run through the trained translator to
  produce the target's algorithm-specific fields (velocities <-> covariance);
* **lossy**      — the current default: carry only the shared population
  (positions / values / best), discarding all algorithm-specific structure;
* **cold**       — a fresh target optimizer that ignores the source entirely.

We report the best objective value each hand-off reaches. Success = the
translated hand-off is at least as good as lossy on average.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from algorithm_translation.cli import ALGO_CHOICES
from algorithm_translation.cli.scenario import add_scenario_args, resolve_test_ids
from algorithm_translation.models.norm import NormContext
from algorithm_translation.optimizers.base import sample_switch_points
from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.rl.reward import run_term
from algorithm_translation.state.canonical import (
    CanonicalState,
    augment_native,
    collate,
    from_canonical,
    has_full_state,
    to_canonical,
    warm_start_optimizer,
)
from algorithm_translation.suite import IOHSuite
from algorithm_translation.train_loop import load_checkpoint_split, load_translator


def _problem_cfg(prob):
    return {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }


def _make(algo, cfg, max_fe, target_fe, best_y, n_individuals, seed):
    options = {
        "max_function_evaluations": max_fe,
        "target_fe": target_fe,
        "best_so_far_y": best_y if best_y < float("inf") else float("inf"),
        "seed_rng": seed,
        "verbose": False,
    }
    if n_individuals is not None:
        options["n_individuals"] = n_individuals
    return PORTFOLIO[algo](cfg, options)


def translate_single(pair, state: CanonicalState, target_algo: str) -> CanonicalState:
    batch = collate([state])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    with torch.no_grad():
        out = pair.translate(batch, target_algo, ctx)
    return out.index(0)


def _run_target_result(algo, cfg, max_fe, n_fe_start, n_individuals, seed, native=None):
    """Run target optimizer from ``n_fe_start`` to ``max_fe``; return its result dict."""
    opt = _make(algo, cfg, max_fe, max_fe, float("inf"), n_individuals, seed)
    opt.n_function_evaluations = n_fe_start
    if native is None:
        opt.set_data()  # cold
    else:
        warm_start_optimizer(opt, native)
    return opt.optimize()


def run_target(algo, cfg, max_fe, n_fe_start, n_individuals, seed, native=None):
    """Run target optimizer for the remaining budget; return final best_y."""
    result = _run_target_result(algo, cfg, max_fe, n_fe_start, n_individuals, seed, native)
    return float(result.get("best_so_far_y", float("inf")))


def evaluate_problem(
    pair,
    source,
    target,
    problem_id,
    suite,
    *,
    fe_multiplier,
    n_individuals,
    seed,
    problem_idx,
    n_individuals_target=None,
    reward_measure="aocc",
    aocc_decades=4.0,
):
    """``n_individuals`` sizes the source (and, unless ``n_individuals_target``
    is given, the target) population.

    Also reports ``reward_coldtarget``: the training env's 'coldtarget' reward
    for this switch — score(translated run) - score(target run cold from FE 0
    over the full budget), each scored by ``reward_measure`` from the incumbent
    at the switch (see ``algorithm_translation.rl.reward.run_term``)."""
    prob = suite.get_problem(problem_id)
    dim = prob.dimension
    cfg = _problem_cfg(prob)
    max_fe = fe_multiplier * dim
    pop = n_individuals if n_individuals is not None else 20
    n_tgt = n_individuals if n_individuals_target is None else n_individuals_target

    base_seed = (seed * 1_000_000 + problem_idx * 1_000) % (2**31)

    # One switch per episode (source -> target), positioned log-uniformly over
    # the FE budget and sampled independently per problem — so the hand-off is
    # tested at randomized, early-biased points rather than a fixed mid-run one.
    sw_rng = np.random.default_rng((seed * 7_919 + problem_idx) % (2**31))
    switch = int(sample_switch_points(2, max_fe, pop, sw_rng)[0])

    # --- warm up the source optimizer to `switch` FEs -------------------- #
    src = _make(source, cfg, max_fe, switch, float("inf"), n_individuals, base_seed)
    src.optimize()
    native_src = augment_native(src, dim)
    if not has_full_state(source, native_src):
        return None
    best_after_warmup = float(src.best_so_far_y)
    n_fe = src.n_function_evaluations

    state = to_canonical(source, native_src)

    # --- three hand-offs over the remaining budget ----------------------- #
    # `translated_native` already bundles the shared population (from the source
    # state's positions) plus the decoded target-specific fields.
    translated_native = from_canonical(translate_single(pair, state, target))
    # The lossy hand-off carries only the shared population — the current default.
    shared = {
        "x": native_src["x"],
        "y": native_src["y"],
        "best_x": native_src["best_x"],
        "best_y": native_src["best_y"],
    }

    translated_res = _run_target_result(
        target, cfg, max_fe, n_fe, n_tgt, base_seed + 1, translated_native
    )
    results = {
        "translated": float(translated_res.get("best_so_far_y", float("inf"))),
        "lossy": run_target(
            target, cfg, max_fe, n_fe, n_tgt, base_seed + 1, shared
        ),
        "cold": run_target(
            target, cfg, max_fe, n_fe, n_tgt, base_seed + 1, None
        ),
    }
    # 'coldtarget' reference: the target run alone from scratch (FE 0) over
    # the whole budget, same seed — as in TranslationEnv._coldtarget_term.
    coldtarget_res = _run_target_result(
        target, cfg, max_fe, 0, n_tgt, base_seed + 1, None
    )
    optimum = getattr(prob, "optimum", None)
    opt_val = float(optimum) if optimum is not None else None
    rng_range = max(best_after_warmup - opt_val, 1e-12) if opt_val is not None else 1.0

    def _score(res):
        return run_term(
            res,
            best_after_warmup,
            rng_range,
            measure=reward_measure,
            optimum=opt_val,
            decades=aocc_decades,
        )

    return {
        "problem_id": problem_id,
        "optimum": opt_val if opt_val is not None else float("nan"),
        "best_after_warmup": best_after_warmup,
        **{f"best_{k}": v for k, v in results.items()},
        "reward_coldtarget": _score(translated_res) - _score(coldtarget_res),
    }


def summarize(rows, source, target):
    if not rows:
        print("no evaluable problems")
        return
    arr = {
        k: np.array([r[f"best_{k}"] for r in rows])
        for k in ("translated", "lossy", "cold")
    }
    print(f"\n=== {source} -> {target}  ({len(rows)} problems) ===")
    for k in ("translated", "lossy", "cold"):
        print(
            f"  {k:11s} best_y: mean={arr[k].mean():.4g}  median={np.median(arr[k]):.4g}"
        )
    # Lower best_y is better. Win = translated strictly better than lossy.
    wins = int((arr["translated"] < arr["lossy"]).sum())
    ties = int(np.isclose(arr["translated"], arr["lossy"]).sum())
    print(f"  translated beats lossy on {wins}/{len(rows)} problems ({ties} ties)")


HANDOFFS = ("translated", "lossy", "cold")


def heldout_metrics(rows) -> dict:
    """Aggregate per-problem rows of one direction into scalar metrics.

    ``log10_prec/*`` is the mean log10 of ``best_y - optimum`` (clipped at 1e-8;
    lower = better) so problems of very different scale are comparable;
    ``win_rate/*`` is the fraction of problems where translated is strictly
    better than the named baseline.
    """
    if not rows:
        return {"n_problems": 0}
    best = {k: np.array([r[f"best_{k}"] for r in rows]) for k in HANDOFFS}
    opt = np.array([r.get("optimum", np.nan) for r in rows])
    m = {"n_problems": len(rows)}
    if np.isfinite(opt).all():
        for k in HANDOFFS:
            prec = np.maximum(best[k] - opt, 1e-8)
            m[f"log10_prec/{k}"] = float(np.log10(prec).mean())
    for k in ("lossy", "cold"):
        m[f"win_rate/translated_vs_{k}"] = float((best["translated"] < best[k]).mean())
    rc = [r["reward_coldtarget"] for r in rows if "reward_coldtarget" in r]
    if rc:
        m["reward/coldtarget"] = float(np.mean(rc))
    return m


def evaluate_heldout(
    pair,
    test_ids,
    directions,
    *,
    fe_multiplier,
    n_for,
    seed=123,
    max_problems=None,
    suite=None,
    progress=False,
    reward_measure="aocc",
    aocc_decades=4.0,
):
    """Evaluate ``pair`` on held-out ``test_ids`` for each (source, target) in
    ``directions``. ``n_for(algo)`` gives each algorithm's population size.

    The problem subset (shuffled with ``seed``, capped at ``max_problems``) and
    all switch points/optimizer seeds are a pure function of ``seed``, so
    repeated calls — e.g. at successive training checkpoints — score the same
    episodes and are directly comparable. Runs on a CPU copy of ``pair``.

    Returns ``(metrics, rows_by_direction)`` where ``metrics`` is flat:
    ``{"<SRC>_to_<TGT>/<metric>": value, ...}`` plus direction-averaged
    ``mean/<metric>`` entries.
    """
    pair = copy.deepcopy(pair).cpu().eval()
    suite = suite or IOHSuite()
    ids = list(test_ids)
    np.random.default_rng(seed).shuffle(ids)
    if max_problems is not None:
        ids = ids[:max_problems]

    metrics, all_rows, per_dir = {}, {}, []
    for source, target in directions:
        it = ids
        if progress:
            from tqdm import tqdm

            it = tqdm(ids, desc=f"{source}->{target}", unit="prob", leave=False)
        rows = []
        for idx, pid in enumerate(it):
            row = evaluate_problem(
                pair,
                source,
                target,
                pid,
                suite,
                fe_multiplier=fe_multiplier,
                n_individuals=n_for(source),
                n_individuals_target=n_for(target),
                seed=seed,
                problem_idx=idx,
                reward_measure=reward_measure,
                aocc_decades=aocc_decades,
            )
            if row is not None:
                rows.append(row)
        all_rows[(source, target)] = rows
        dm = heldout_metrics(rows)
        per_dir.append(dm)
        metrics.update({f"{source}_to_{target}/{k}": v for k, v in dm.items()})
    if len(per_dir) > 1:
        keys = set.intersection(*(set(d) for d in per_dir))
        for k in sorted(keys - {"n_problems"}):
            metrics[f"mean/{k}"] = float(np.mean([d[k] for d in per_dir]))
    return metrics, all_rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algo_a", choices=ALGO_CHOICES)
    p.add_argument("algo_b", choices=ALGO_CHOICES)
    p.add_argument(
        "--model",
        default=None,
        help="translator checkpoint (default models/<A>_<B>_ppo.pt)",
    )
    add_scenario_args(p, for_eval=True)
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument("--n-individuals", type=int, default=None)
    p.add_argument("--n-individuals-a", type=int, default=None)
    p.add_argument("--n-individuals-b", type=int, default=None)
    p.add_argument("--max-problems", type=int, default=60)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model or f"models/{args.algo_a}_{args.algo_b}_ppo.pt"
    pair = load_translator(model, device=args.device)
    print(f"loaded translator {pair.algo_a}<->{pair.algo_b} from {model}")

    test_ids = resolve_test_ids(args, load_checkpoint_split(model))
    n_a = args.n_individuals_a if args.n_individuals_a is not None else args.n_individuals
    n_b = args.n_individuals_b if args.n_individuals_b is not None else args.n_individuals

    def n_for(algo):
        return n_a if algo == args.algo_a else n_b

    directions = ((args.algo_a, args.algo_b), (args.algo_b, args.algo_a))
    _, rows = evaluate_heldout(
        pair,
        test_ids,
        directions,
        fe_multiplier=args.fe_multiplier,
        n_for=n_for,
        seed=args.seed,
        max_problems=args.max_problems,
        progress=True,
    )
    for (source, target), r in rows.items():
        summarize(r, source, target)


if __name__ == "__main__":
    main()
