"""Evaluate a trained translator: does the learned hand-off beat the default?

    uv run eval PSO CMAES --model models/PSO_CMAES_ppo.pt [options]

On each held-out problem the source optimizer warms up to a switch point drawn
as in training, then the target runs the rest of the budget under three
hand-offs (see ``algorithm_translation.evaluation.heldout``):

* **translated** — the source state run through the trained translator;
* **lossy**      — only the shared population (positions / values / best);
* **cold**       — a fresh target that ignores the source.

Reports the best objective value each hand-off reaches. Success = translated is
at least as good as lossy on average.
"""

from __future__ import annotations

import argparse

import numpy as np

from algorithm_translation.cli import ALGO_CHOICES, population_sizes
from algorithm_translation.cli.scenario import add_scenario_args, resolve_test_ids
from algorithm_translation.evaluation.heldout import HANDOFFS, evaluate_heldout
from algorithm_translation.train_loop import (
    load_checkpoint_split,
    load_translator,
    resolve_model_path,
)


def summarize(rows, source, target):
    if not rows:
        print("no evaluable problems")
        return
    best = {k: np.array([r[f"best_{k}"] for r in rows]) for k in HANDOFFS}
    print(f"\n=== {source} -> {target}  ({len(rows)} problems) ===")
    for k in HANDOFFS:
        print(f"  {k:11s} best_y: mean={best[k].mean():.4g}  median={np.median(best[k]):.4g}")
    # Lower best_y is better. Win = translated strictly better than lossy.
    wins = int((best["translated"] < best["lossy"]).sum())
    ties = int(np.isclose(best["translated"], best["lossy"]).sum())
    print(f"  translated beats lossy on {wins}/{len(rows)} problems ({ties} ties)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algo_a", choices=ALGO_CHOICES)
    p.add_argument("algo_b", choices=ALGO_CHOICES)
    p.add_argument(
        "--model",
        default=None,
        help="translator checkpoint (default: models/<A>_<B>_ppo.pt or "
        "models/<A>_<B>_grpo.pt, whichever exists)",
    )
    add_scenario_args(p, for_eval=True)
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument("--n-individuals", type=int, default=None)
    p.add_argument("--n-individuals-a", type=int, default=None)
    p.add_argument("--n-individuals-b", type=int, default=None)
    p.add_argument("--max-problems", type=int, default=60)
    p.add_argument(
        "--switch-cdb",
        type=float,
        default=None,
        help="switch-point warp, as the trainer's --switch-cdb. Default: the "
        "value stored in the checkpoint (1.0 if it has none)",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    model = resolve_model_path(args.model, args.algo_a, args.algo_b)
    pair = load_translator(model, device=args.device)
    print(f"loaded translator {pair.algo_a}<->{pair.algo_b} from {model}")

    ckpt_split = load_checkpoint_split(model)
    test_ids = resolve_test_ids(args, ckpt_split)
    switch_cdb = args.switch_cdb
    if switch_cdb is None:
        switch_cdb = float((ckpt_split or {}).get("switch_cdb") or 1.0)

    directions = ((args.algo_a, args.algo_b), (args.algo_b, args.algo_a))
    _, rows = evaluate_heldout(
        pair,
        test_ids,
        directions,
        fe_multiplier=args.fe_multiplier,
        n_for=population_sizes(args).get,
        seed=args.seed,
        max_problems=args.max_problems,
        progress=True,
        switch_cdb=switch_cdb,
    )
    for (source, target), r in rows.items():
        summarize(r, source, target)


if __name__ == "__main__":
    main()
