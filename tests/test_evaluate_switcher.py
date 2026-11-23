"""The multi-switch benchmark (``uv run eval-switch``): its anytime metrics and
one schedule run end to end."""

import math

import numpy as np

from algorithm_translation.evaluation.anytime import (
    ERT_TARGETS,
    average_ranks,
    budget_aocc,
    ert,
    fe_to_targets,
)
from algorithm_translation.evaluation.switcher import evaluate_problem, random_schedule
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.suite import IOHSuite, build_problem_ids


def test_budget_aocc_bounds_and_plateau():
    assert budget_aocc([], 100, 0.0) == 0.0
    # Precision 1e8 (the clip ceiling) the whole time -> 0; at the optimum -> 1.
    assert budget_aocc([(1, 1e9)], 100, 0.0) == 0.0
    assert math.isclose(budget_aocc([(1, 0.0)], 100, 0.0), 1.0)
    # Better precision scores higher, and an intermediate plateau lowers it.
    assert budget_aocc([(1, 1e-3)], 100, 0.0) > budget_aocc([(1, 1.0)], 100, 0.0)
    direct = budget_aocc([(1, 1.0), (10, 0.0)], 100, 0.0)
    stalled = budget_aocc([(1, 1.0), (50, 0.5), (90, 0.0)], 100, 0.0)
    assert 0.0 < stalled < direct < 1.0


def test_fe_to_targets_and_ert():
    fett = fe_to_targets([(5, 1e-2), (20, 1e-4), (50, 1e-8)], 0.0)
    assert len(fett) == len(ERT_TARGETS)
    assert fett[:2] == [20.0, 20.0] and fett[-1] == 50.0
    assert fe_to_targets([(5, 1.0)], 0.0) == [math.inf] * len(ERT_TARGETS)
    # Two runs, one success at 20 FEs and one failure with a 100-FE budget.
    value, n_succ = ert(np.array([20.0, math.inf]), np.array([100.0, 100.0]))
    assert (value, n_succ) == (120.0, 1)
    assert ert(np.array([math.inf]), np.array([100.0])) == (math.inf, 0)


def test_average_ranks_share_ties():
    assert average_ranks([3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0]
    assert average_ranks([1.0, 1.0, 2.0]) == [1.5, 1.5, 3.0]


def test_random_schedule_always_switches():
    rng = np.random.default_rng(0)
    for _ in range(50):
        algos = random_schedule("PSO", "CMAES", 3, rng)
        assert len(algos) == 4 and len(set(algos)) == 2


def test_evaluate_problem_runs_all_five_methods():
    pair = ActorCritic("PSO", "CMAES", hidden=16, n_layers=1).translator
    pid = build_problem_ids({1}, dims=[2], instances=[1])[0]
    rows = evaluate_problem(
        pair,
        pid,
        IOHSuite(),
        algo_a="PSO",
        algo_b="CMAES",
        n_for={"PSO": 12, "CMAES": 12}.get,
        fe_multiplier=200,
        n_switches=2,
        switch_mode="random",
        cdb=1.0,
        repeats=2,
        seed=0,
        problem_idx=0,
    )
    assert len(rows) == 2
    for row in rows:
        for m in ("PSO_only", "CMAES_only", "cold_switch", "lossy_switch", "translated_switch"):
            assert 0.0 <= row[m]["aocc"] <= 1.0, m
            assert math.isfinite(row[m]["best"]), m
        assert row["translated_fallbacks"] >= 0
