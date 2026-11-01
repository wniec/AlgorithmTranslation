"""Unit tests for the pure per-segment reward scoring in ``cat.rl.reward``.

These exercise the AOCC / endpoint math directly (bounds, monotonicity, the
floor clamp, the empty-history and no-optimum fallbacks) rather than through the
env, so the scoring contract is pinned independently of the optimizer runs.
"""

import math

import numpy as np

from cat.rl.reward import (
    _REWARD_EPS,
    _log_scale,
    aocc,
    log_scaled_improvement,
    run_term,
    scaled_improvement,
)


def test_scaled_improvement_bounds():
    assert scaled_improvement(10.0, 5.0, 5.0) == 1.0  # full gap closed -> 1
    assert scaled_improvement(5.0, 5.0, 5.0) == 0.0  # no change
    assert scaled_improvement(5.0, 6.0, 5.0) == 0.0  # worse -> clamped to 0
    assert 0.0 < scaled_improvement(10.0, 9.0, 5.0) < 1.0


def test_log_scale_zero_and_monotonic():
    assert _log_scale(0.0) == 0.0
    # Strictly increasing in the scaled improvement.
    assert _log_scale(0.1) < _log_scale(0.5) < _log_scale(1.0)
    # A full [0,1] improvement maps to log(1 + 1/eps).
    assert math.isclose(_log_scale(1.0), math.log(1 + 1 / _REWARD_EPS), rel_tol=1e-6)


def test_aocc_none_on_empty_history():
    assert aocc(None, prev_best=10.0, optimum=0.0, decades=4.0) is None
    assert aocc([], prev_best=10.0, optimum=0.0, decades=4.0) is None
    assert aocc(np.array([]), prev_best=10.0, optimum=0.0, decades=4.0) is None


def test_aocc_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.uniform(0.0, 10.0, size=rng.integers(1, 40))
        v = aocc(y, prev_best=10.0, optimum=0.0, decades=4.0)
        assert 0.0 - 1e-12 <= v <= 1.0 + 1e-12


def test_aocc_rewards_earlier_convergence():
    # Same endpoint (1.0), but 'fast' reaches it immediately -> higher area.
    fast = aocc(np.array([1.0] * 50), prev_best=10.0, optimum=0.0, decades=4.0)
    slow = aocc(np.array([9.0] * 49 + [1.0]), prev_best=10.0, optimum=0.0, decades=4.0)
    assert fast > slow
    # fast sits one decade below the gap-at-switch the whole time: v = 1/4.
    assert math.isclose(fast, 0.25, rel_tol=1e-9)


def test_aocc_caps_at_one_when_optimum_reached():
    # Reaching the optimum clamps the gap to the floor g_lo -> full credit 1.0.
    v = aocc(np.array([0.0] * 10), prev_best=10.0, optimum=0.0, decades=4.0)
    assert math.isclose(v, 1.0, rel_tol=1e-9)


def test_aocc_zero_when_never_below_incumbent():
    # A run that never improves on prev_best earns zero area.
    assert aocc(np.array([10.0] * 10), prev_best=10.0, optimum=0.0, decades=4.0) == 0.0
    assert aocc(np.array([20.0] * 10), prev_best=10.0, optimum=0.0, decades=4.0) == 0.0


def test_run_term_aocc_path_uses_history():
    v = run_term(
        {"y_history": np.array([1.0] * 50)},
        prev_best=10.0,
        rng_range=999.0,  # ignored on the aocc path
        measure="aocc",
        optimum=0.0,
        decades=4.0,
    )
    assert math.isclose(v, 0.25, rel_tol=1e-9)


def test_run_term_falls_back_to_endpoint_without_optimum():
    # optimum is None (suite doesn't expose it) -> endpoint measure.
    got = run_term(
        {"best_so_far_y": 5.0},
        prev_best=10.0,
        rng_range=10.0,
        measure="aocc",
        optimum=None,
        decades=4.0,
    )
    assert math.isclose(got, log_scaled_improvement(10.0, 5.0, 10.0), rel_tol=1e-9)


def test_run_term_falls_back_to_endpoint_without_history():
    # optimum known but no history -> endpoint measure.
    got = run_term(
        {"best_so_far_y": 5.0},
        prev_best=10.0,
        rng_range=10.0,
        measure="aocc",
        optimum=0.0,
        decades=4.0,
    )
    assert math.isclose(got, log_scaled_improvement(10.0, 5.0, 10.0), rel_tol=1e-9)


def test_run_term_endpoint_measure_ignores_history():
    # Even with a history present, measure='endpoint' scores only the boundary.
    got = run_term(
        {"best_so_far_y": 5.0, "y_history": np.array([1.0] * 50)},
        prev_best=10.0,
        rng_range=10.0,
        measure="endpoint",
        optimum=0.0,
        decades=4.0,
    )
    assert math.isclose(got, log_scaled_improvement(10.0, 5.0, 10.0), rel_tol=1e-9)
