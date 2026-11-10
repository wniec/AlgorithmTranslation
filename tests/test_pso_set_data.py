"""PSO.set_data must reindex per-particle warm-start fields (v, p_x, p_y, n_x)
consistently with the fitness-sort permutation applied to x/y — otherwise a
particle's velocity/personal-best gets silently attached to the wrong
individual after every warm start."""

import numpy as np

from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.suite import IOHSuite


def _cfg(problem_id="bbob_f001_i01_d05"):
    prob = IOHSuite().get_problem(problem_id)
    return {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }


def _pso(n_individuals=5, dim=5):
    cfg = _cfg()
    return PORTFOLIO["PSO"](
        cfg,
        {
            "max_function_evaluations": 100,
            "target_fe": 100,
            "n_individuals": n_individuals,
            "seed_rng": 0,
            "verbose": False,
        },
    )


def test_set_data_reindexes_per_particle_fields_with_x_y():
    n, d = 5, 5
    opt = _pso(n_individuals=n, dim=d)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, d))
    # Deliberately non-sorted fitness so argsort(y) != identity.
    y = np.array([3.0, 1.0, 4.0, 0.0, 2.0])
    v = rng.normal(size=(n, d))
    p_x = rng.normal(size=(n, d))
    p_y = rng.normal(size=n)
    n_x = rng.normal(size=(n, d))

    opt.set_data(x=x, y=y, v=v, p_x=p_x, p_y=p_y, n_x=n_x)
    ws = opt.get_data()

    expected_idx = np.argsort(y)
    assert np.allclose(ws["x"], x[expected_idx])
    assert np.allclose(ws["y"], y[expected_idx])
    # v/p_x/p_y/n_x must follow the SAME permutation as x/y, not be used as-is.
    assert np.allclose(ws["v"], v[expected_idx])
    assert np.allclose(ws["p_x"], p_x[expected_idx])
    assert np.allclose(ws["p_y"], p_y[expected_idx])
    assert np.allclose(ws["n_x"], n_x[expected_idx])


def test_set_data_without_velocity_defaults_to_zero_not_random():
    """A population hand-off that omits velocities (the lossy default / a non-PSO
    source) must start the swarm at rest. Random dispersal velocity would fling
    an inherited converged swarm off its best point and make the ``* -> PSO``
    relative-reward baseline pathologically weak (never negative). A genuine cold
    start (no population at all) still randomizes velocity."""
    n, d = 5, 5
    opt = _pso(n_individuals=n, dim=d)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    opt.set_data(x=x, y=y)  # no velocity supplied
    assert np.count_nonzero(opt.get_data()["v"]) == 0  # started at rest


def test_set_data_truncates_larger_population_consistently():
    n_target, n_source, d = 4, 7, 5
    opt = _pso(n_individuals=n_target, dim=d)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n_source, d))
    y = rng.normal(size=n_source)
    v = rng.normal(size=(n_source, d))
    p_x = rng.normal(size=(n_source, d))
    n_x = rng.normal(size=(n_source, d))

    opt.set_data(x=x, y=y, v=v, p_x=p_x, n_x=n_x)
    ws = opt.get_data()

    expected_idx = np.argsort(y)[:n_target]
    assert ws["x"].shape == (n_target, d)
    assert np.allclose(ws["x"], x[expected_idx])
    assert np.allclose(ws["v"], v[expected_idx])
    assert np.allclose(ws["p_x"], p_x[expected_idx])
    assert np.allclose(ws["n_x"], n_x[expected_idx])
