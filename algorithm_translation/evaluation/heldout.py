"""Held-out evaluation with one switch per problem.

For each held-out problem the source optimizer warms up to a switch point drawn
exactly as in training (``rl.env.sample_episode_start``), then the target runs
to the end of the budget under three hand-offs:

* **translated** — the source state run through the translator;
* **lossy**      — only the shared population (the default without a network);
* **cold**       — a fresh target that ignores the source.

Each row also carries the ``coldtarget`` reward of the translated run and the
size of the deterministic action.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from algorithm_translation.models.norm import NormContext
from algorithm_translation.rl.env import sample_episode_start
from algorithm_translation.rl.scoring import SwitchPoint, SwitchScorer
from algorithm_translation.runs import make_optimizer, problem_cfg, shared_handoff
from algorithm_translation.seeding import Stream, derive_seed
from algorithm_translation.state.canonical import (
    CanonicalState,
    augment_native,
    collate,
    from_canonical,
    has_full_state,
    to_canonical,
)
from algorithm_translation.suite import IOHSuite

HANDOFFS = ("translated", "lossy", "cold")
# An action counts as non-zero when any coordinate exceeds this in magnitude.
ACTION_ZERO_TOL = 1e-6


def translate_with_action(
    pair, state: CanonicalState, target_algo: str
) -> tuple[CanonicalState, np.ndarray]:
    """Translate ``state`` deterministically and also return the action: the
    flat normalized coordinates the decoder head emits before they are rebuilt
    into the target's fields. For CMA-ES these are the residual around the
    computed prior, so an all-zero action means "prior only"."""
    batch = collate([state])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    head = pair.decoders[target_algo]
    with torch.no_grad():
        z = pair.encode(batch, ctx)
        coords = head.action_mean(z, batch.positions, ctx)
        specific = head.build_state(coords, ctx, batch)
    out = CanonicalState(
        algo=target_algo,
        positions=batch.positions,
        values=batch.values,
        best_x=batch.best_x,
        best_y=batch.best_y,
        specific=specific,
    )
    return out.index(0), coords[0].cpu().numpy()


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
    seed_rep=0,
    n_individuals_target=None,
    reward_measure="aocc",
    aocc_decades=4.0,
    switch_cdb=1.0,
):
    """One held-out episode. Returns a row dict, or None when the source has
    no full state at the switch.

    ``n_individuals`` sizes the source (and, unless ``n_individuals_target`` is
    given, the target) population. ``switch_cdb`` should be the trainer's
    ``--switch-cdb``. ``seed_rep`` re-runs the same problem and switch point
    with different optimizer seeds. ``reward_coldtarget`` is the training
    env's ``coldtarget`` reward for the translated hand-off."""
    prob = suite.get_problem(problem_id)
    dim = prob.dimension
    cfg = problem_cfg(prob)
    max_fe = fe_multiplier * dim
    n_tgt = n_individuals if n_individuals_target is None else n_individuals_target
    pop = max(n_individuals or 20, n_tgt or 20)  # as TranslationEnv.reset

    rng = np.random.default_rng(derive_seed(seed, problem_idx, Stream.EPISODE))
    switch, x_probe, y_probe = sample_episode_start(prob, max_fe, pop, rng, switch_cdb)
    i_best = int(np.argmin(y_probe))

    # --- warm up the source optimizer to the switch ---------------------- #
    src = make_optimizer(
        source,
        cfg,
        max_fe=max_fe,
        target_fe=switch,
        seed=derive_seed(seed, problem_idx, Stream.WARMUP, seed_rep),
        n_individuals=n_individuals,
        best_y=float(y_probe[i_best]),
        n_fe_start=len(y_probe),
    )
    src.set_data(best_x=x_probe[i_best], best_y=float(y_probe[i_best]))
    src.optimize()
    native_src = augment_native(src, dim)
    if not has_full_state(source, native_src):
        return None
    best_after_warmup = float(src.best_so_far_y)
    optimum = getattr(prob, "optimum", None)
    opt_val = float(optimum) if optimum is not None else None

    seg_seed = derive_seed(seed, problem_idx, Stream.SEGMENT, seed_rep)
    scorer = SwitchScorer(
        SwitchPoint(
            cfg=cfg,
            max_fe=max_fe,
            n_fe=src.n_function_evaluations,
            best_y=best_after_warmup,
            scale=max(best_after_warmup - opt_val, 1e-12) if opt_val is not None else 1.0,
            optimum=opt_val,
            source_algo=source,
            target_algo=target,
            source_native=native_src,
            n_source=n_individuals,
            n_target=n_tgt,
        ),
        modes=["coldtarget"],
        seeds=[seg_seed],
        measure=reward_measure,
        decades=aocc_decades,
    )

    # --- three hand-offs over the remaining budget, same seed ------------ #
    translated_state, action = translate_with_action(
        pair, to_canonical(source, native_src), target
    )
    _, translated_res = scorer.run_target(from_canonical(translated_state), seg_seed)
    _, lossy_res = scorer.run_target(shared_handoff(native_src), seg_seed)
    _, cold_res = scorer.run_target(None, seg_seed)
    results = {"translated": translated_res, "lossy": lossy_res, "cold": cold_res}

    return {
        "problem_id": problem_id,
        "optimum": opt_val if opt_val is not None else float("nan"),
        "best_after_warmup": best_after_warmup,
        **{
            f"best_{k}": float(r.get("best_so_far_y", float("inf")))
            for k, r in results.items()
        },
        "reward_coldtarget": scorer.term(translated_res)
        - scorer.baseline("coldtarget", seg_seed),
        # A policy collapsed to (or stuck at) the all-zero action shows here.
        "action_rms": float(np.sqrt(np.mean(action**2))) if action.size else 0.0,
        "action_max_abs": float(np.abs(action).max()) if action.size else 0.0,
    }


def heldout_metrics(rows) -> dict:
    """Aggregate the rows of one direction into scalar metrics.

    ``log10_prec/*`` is the mean log10 of ``best_y - optimum`` (clipped at 1e-8;
    lower = better), comparable across problems of very different scale;
    ``win_rate/*`` is the fraction of rows where translated is strictly better
    than the named baseline.
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
    if all("action_rms" in r for r in rows):
        rms = np.array([r["action_rms"] for r in rows])
        mx = np.array([r["action_max_abs"] for r in rows])
        m["action/rms"] = float(rms.mean())
        m["action/max_abs"] = float(mx.max())
        m["action/nonzero_frac"] = float((mx > ACTION_ZERO_TOL).mean())
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
    seed_reps=1,
    switch_cdb=1.0,
):
    """Evaluate ``pair`` on ``test_ids`` for each (source, target) in
    ``directions``; ``n_for(algo)`` gives each algorithm's population size.

    The problem subset (shuffled with ``seed``, capped at ``max_problems``),
    switch points and optimizer seeds are a pure function of ``seed``, so
    repeated calls (e.g. successive checkpoints) score the same episodes. Each
    episode runs ``seed_reps`` times with different optimizer seeds; all runs
    are pooled into the metrics (``n_problems`` counts distinct problems,
    ``n_runs`` problem x seed runs). Runs on a CPU copy of ``pair``.

    Returns ``(metrics, rows_by_direction)``: ``metrics`` is flat,
    ``{"<SRC>_to_<TGT>/<metric>": value}`` plus direction-averaged
    ``mean/<metric>``.
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
            for rep in range(seed_reps):
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
                    seed_rep=rep,
                    reward_measure=reward_measure,
                    aocc_decades=aocc_decades,
                    switch_cdb=switch_cdb,
                )
                if row is not None:
                    rows.append(row)
        all_rows[(source, target)] = rows
        dm = heldout_metrics(rows)
        if rows:
            dm["n_runs"] = len(rows)
            dm["n_problems"] = len({r["problem_id"] for r in rows})
        per_dir.append(dm)
        metrics.update({f"{source}_to_{target}/{k}": v for k, v in dm.items()})
    if len(per_dir) > 1:
        keys = set.intersection(*(set(d) for d in per_dir))
        for k in sorted(keys - {"n_problems", "n_runs"}):
            metrics[f"mean/{k}"] = float(np.mean([d[k] for d in per_dir]))
    return metrics, all_rows
