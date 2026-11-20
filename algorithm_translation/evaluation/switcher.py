"""Multi-switch evaluation: a random algorithm schedule over the whole budget.

Each run splits the budget into ``n_switches + 1`` segments, each assigned one
of the two algorithms at random. The same problem, schedule, switch points and
segment seeds are run five ways:

* ``<A>_only`` / ``<B>_only`` — one algorithm for the whole budget;
* ``cold_switch``            — every real switch cold-starts the next optimizer;
* ``lossy_switch``           — real switches carry only the shared population;
* ``translated_switch``      — real switches go through the translator.

The three switchers differ only at real algorithm changes, so differences are
attributable to the hand-off.
"""

from __future__ import annotations

import numpy as np
import torch

from algorithm_translation.evaluation.anytime import Recorder, budget_aocc, fe_to_targets
from algorithm_translation.models.norm import NormContext
from algorithm_translation.optimizers.base import get_checkpoints, sample_switch_points
from algorithm_translation.runs import problem_cfg, run_segment, shared_handoff
from algorithm_translation.seeding import Stream, derive_seed
from algorithm_translation.state.canonical import (
    CanonicalState,
    augment_native,
    collate,
    from_canonical,
    has_full_state,
    to_canonical,
)

# (hand-off mode, row key) for the three switchers.
SWITCH_MODES = (
    ("cold", "cold_switch"),
    ("lossy", "lossy_switch"),
    ("translated", "translated_switch"),
)


def translate_native(pair, native: dict, source_algo: str, target_algo: str) -> dict:
    """The target's full warm-start dict for a source native state."""
    batch = collate([to_canonical(source_algo, native)])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    with torch.no_grad():
        out: CanonicalState = pair.translate(batch, target_algo, ctx)
    return from_canonical(out.index(0))


def _metrics(rec: Recorder, budget: int, optimum: float) -> dict:
    return {
        "best": rec.best,
        "aocc": budget_aocc(rec.history, budget, optimum),
        "fett": fe_to_targets(rec.history, optimum),  # one per ERT target
    }


def run_standalone(algo, problem, max_fe, optimum, n_individuals, seed) -> dict:
    """One algorithm for the whole budget."""
    rec = Recorder(problem)
    cfg = {**problem_cfg(problem), "fitness_function": rec}
    run_segment(algo, cfg, max_fe=max_fe, target_fe=max_fe, seed=seed, n_individuals=n_individuals)
    return _metrics(rec, max_fe, optimum)


def _handoff(mode, pair, native, cur_algo, next_algo):
    """Payload for the segment after ``cur_algo``, and whether a translated
    switch had to fall back because the source state was degenerate."""
    full = has_full_state(cur_algo, native)
    has_shared = native.get("x") is not None and native.get("y") is not None
    shared = shared_handoff(native) if has_shared else None
    if next_algo == cur_algo:
        # Staying: warm-continue on the algorithm's own state in every mode.
        return (native if full else shared), False
    if mode == "cold":
        return None, False
    if mode == "translated":
        if full:
            return translate_native(pair, native, cur_algo, next_algo), False
        return shared, True
    return shared, False  # lossy


def run_switcher(mode, algos, checkpoints, problem, optimum, n_for, segment_seeds, pair) -> dict:
    """Run the algorithm schedule ``algos`` (segment ``i`` ends at
    ``checkpoints[i]`` and uses ``segment_seeds[i]``) with hand-off ``mode``:
    ``translated``, ``lossy`` or ``cold``. Returns the run's metrics plus
    ``fallbacks``: real switches the translated mode served without a full
    source state."""
    rec = Recorder(problem)
    cfg = {**problem_cfg(problem), "fitness_function": rec}
    dim = cfg["ndim_problem"]
    max_fe = int(checkpoints[-1])
    n_fe, native, cur_algo, fallbacks = 0, None, None, 0

    for i, algo in enumerate(algos):
        if i == 0:
            payload = None
        else:
            payload, fell_back = _handoff(mode, pair, native, cur_algo, algo)
            fallbacks += int(fell_back)
        # A cold segment knows no incumbent; a warm one keeps the champion.
        opt, _ = run_segment(
            algo,
            cfg,
            max_fe=max_fe,
            target_fe=int(checkpoints[i]),
            seed=segment_seeds[i],
            n_individuals=n_for(algo),
            n_fe_start=n_fe,
            best_y=rec.best if payload is not None else float("inf"),
            payload=payload,
        )
        n_fe = opt.n_function_evaluations
        native = augment_native(opt, dim)
        cur_algo = algo

    return {**_metrics(rec, max_fe, optimum), "fallbacks": fallbacks}


def random_schedule(algo_a, algo_b, n_switches, rng) -> list[str]:
    """A random sequence of ``n_switches + 1`` algorithms with at least one real
    switch (when ``n_switches >= 1``), so the hand-off is actually exercised."""
    algos = [algo_a if rng.random() < 0.5 else algo_b for _ in range(n_switches + 1)]
    if n_switches >= 1 and len(set(algos)) == 1:
        flip = int(rng.integers(1, n_switches + 1))
        algos[flip] = algo_b if algos[flip] == algo_a else algo_a
    return algos


def segment_ends(mode, n_switches, max_fe, pop, cdb, rng) -> np.ndarray:
    """The ``n_switches + 1`` cumulative segment-end FEs (the last is max_fe):
    log-uniformly sampled (``random``) or deterministic (``cdb``)."""
    n = n_switches + 1
    if mode == "cdb":
        return get_checkpoints(n, max_fe, pop, cdb)
    return sample_switch_points(n, max_fe, pop, rng, cdb=cdb)


def evaluate_problem(
    pair,
    problem_id,
    suite,
    *,
    algo_a,
    algo_b,
    n_for,
    fe_multiplier,
    n_switches,
    switch_mode,
    cdb,
    repeats,
    seed,
    problem_idx,
) -> list[dict]:
    """``repeats`` random schedules on one problem; one row per schedule."""
    problem = suite.get_problem(problem_id)
    dim = problem.dimension
    max_fe = fe_multiplier * dim
    pop = max(n_for(algo_a) or 20, n_for(algo_b) or 20)
    optimum = getattr(problem, "optimum", None)
    if optimum is None:
        raise RuntimeError(f"{problem_id}: problem exposes no optimum; cannot score AOCC")
    optimum = float(optimum)

    # The standalone baselines don't depend on the schedule: run them once.
    standalone = {
        f"{algo}_only": run_standalone(
            algo, problem, max_fe, optimum, n_for(algo),
            derive_seed(seed, problem_idx, Stream.STANDALONE, k),
        )
        for k, algo in enumerate((algo_a, algo_b))
    }

    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(derive_seed(seed, problem_idx, Stream.SCHEDULE, rep))
        algos = random_schedule(algo_a, algo_b, n_switches, rng)
        checkpoints = segment_ends(switch_mode, n_switches, max_fe, pop, cdb, rng)
        segment_seeds = [
            derive_seed(seed, problem_idx, Stream.SEGMENT, rep, i) for i in range(len(algos))
        ]
        row = {
            "problem_id": problem_id,
            "dimension": dim,
            "optimum": optimum,
            "budget": max_fe,
            **standalone,
        }
        for mode, key in SWITCH_MODES:
            row[key] = run_switcher(
                mode, algos, checkpoints, problem, optimum, n_for, segment_seeds, pair
            )
        row["translated_fallbacks"] = row["translated_switch"]["fallbacks"]
        rows.append(row)
    return rows
