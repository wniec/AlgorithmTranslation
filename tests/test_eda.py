"""EDA optimizers (smoothed UMDA_c, AMaLGaM) and their translation.

Mirrors ``test_bobyqa.py``: the optimizers solve easy problems, their native
state round-trips through the canonical adapter and warm-starts the next run,
the computed prior equals the estimate each EDA makes on its own from a
population-only hand-off, and the translator maps into and out of them.
"""

import numpy as np
import pytest
import torch

from algorithm_translation.models import NormContext, TranslatorPair
from algorithm_translation.models.prior import computed_specific, prior_native
from algorithm_translation.optimizers.EDA.amalgam import estimate_amalgam
from algorithm_translation.optimizers.EDA.umda import estimate_umda
from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.state import collate, from_canonical, to_canonical
from algorithm_translation.state.canonical import (
    CanonicalState,
    augment_native,
    has_full_state,
    warm_start_optimizer,
)
from algorithm_translation.suite import IOHSuite
from algorithm_translation.suite.bbob_splits import build_problem_ids

EDAS = ("UMDA", "AMALGAM")


def _make(algo, problem_id="bbob_f001_i01_d05", **opts):
    prob = IOHSuite().get_problem(problem_id)
    cfg = {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }
    base = {"max_function_evaluations": 5000, "seed_rng": 1, "verbose": False}
    return prob, PORTFOLIO[algo](cfg, {**base, **opts})


# -- optimizers ------------------------------------------------------------ #


@pytest.mark.parametrize("algo", EDAS)
def test_eda_solves_sphere(algo):
    prob, opt = _make(algo)
    res = opt.optimize()
    assert res["best_so_far_y"] <= prob.optimum + 1e-4


def test_amalgam_solves_ellipsoid():
    """The full covariance learns an ill-conditioned (rotated) quadratic."""
    prob, opt = _make("AMALGAM", "bbob_f010_i01_d05", max_function_evaluations=20000)
    res = opt.optimize()
    assert res["best_so_far_y"] <= prob.optimum + 1e-2


@pytest.mark.parametrize("algo", EDAS)
def test_state_roundtrips_and_has_full_state(algo):
    _, opt = _make(algo, target_fe=600)
    opt.optimize()
    native = augment_native(opt, 5)
    assert has_full_state(algo, native)
    back = from_canonical(to_canonical(algo, native))
    for key in to_canonical(algo, native).specific:
        np.testing.assert_allclose(np.asarray(back[key]), np.asarray(native[key]), rtol=1e-5)


@pytest.mark.parametrize("algo", EDAS)
def test_warm_start_samples_from_the_given_distribution(algo):
    """A handed-over distribution is used as-is by the next generation."""
    _, opt = _make(algo, max_function_evaluations=40)
    centre = np.full(5, 2.0)
    if algo == "UMDA":
        state = {"mean": centre, "std": np.full(5, 1e-3)}
    else:
        state = {"mean": centre, "cov": 1e-6 * np.eye(5), "ams_shift": np.zeros(5), "c_mult": 1.0}
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, (30, 5))
    warm_start_optimizer(opt, {"x": x, "y": np.arange(30.0), **state})
    opt.optimize()
    sampled = np.array(opt.x_history)
    assert np.abs(sampled - centre).max() < 0.05


@pytest.mark.parametrize("algo", EDAS)
def test_continuation_keeps_the_state(algo):
    """Stopping and restarting from get_data() continues the same run."""
    prob, first = _make(algo, target_fe=1500)
    first.optimize()
    native = augment_native(first, 5)
    _, second = _make(algo, seed_rng=2)
    second.n_function_evaluations = first.n_function_evaluations
    warm_start_optimizer(second, native)
    res = second.optimize()
    assert res["best_so_far_y"] <= first.best_so_far_y
    assert res["best_so_far_y"] <= prob.optimum + 1e-4


# -- computed prior --------------------------------------------------------- #


def _points_state(algo, n, d=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = (x**2).sum(axis=1) + rng.normal(scale=0.1, size=n)
    native = {"x": x, "y": y, "best_x": x[np.argmin(y)], "best_y": y.min()}
    return x, y, native


@pytest.mark.parametrize("n", [6, 50])  # 6: fewer selected points than dims
def test_prior_equals_the_edas_own_estimate(n):
    x, y, native = _points_state("CMAES", n)
    batch = collate(
        [
            CanonicalState(
                "UMDA",
                torch.as_tensor(x, dtype=torch.float64),
                torch.as_tensor(y, dtype=torch.float64),
                torch.as_tensor(native["best_x"], dtype=torch.float64),
                torch.as_tensor(native["best_y"], dtype=torch.float64),
                {},
            )
        ]
    )
    ctx = NormContext.from_shared(batch.positions, batch.values)

    mean, std = estimate_umda(x, y)
    got = computed_specific("UMDA", batch, ctx)
    np.testing.assert_allclose(got["mean"][0].numpy(), mean, rtol=1e-9)
    np.testing.assert_allclose(got["std"][0].numpy(), std, rtol=1e-9)

    mean, cov = estimate_amalgam(x, y)
    got = computed_specific("AMALGAM", batch, ctx)
    np.testing.assert_allclose(got["mean"][0].numpy(), mean, rtol=1e-9)
    np.testing.assert_allclose(got["cov"][0].numpy(), cov, rtol=1e-9, atol=1e-15)
    assert float(got["c_mult"][0]) == 1.0
    assert torch.count_nonzero(got["ams_shift"]) == 0


def test_prior_native_warm_starts_amalgam():
    _, pso = _make("PSO", target_fe=300, n_individuals=20)
    pso.optimize()
    native = prior_native("PSO", augment_native(pso, 5), "AMALGAM")
    assert {"mean", "cov", "ams_shift", "c_mult"} <= set(native)
    _, opt = _make("AMALGAM")
    opt.n_function_evaluations = pso.n_function_evaluations
    warm_start_optimizer(opt, native)
    assert opt.optimize()["best_so_far_y"] <= pso.best_so_far_y


# -- translator ------------------------------------------------------------- #


def _source_state(algo, B=2, N=12, D=4):
    g = torch.Generator().manual_seed(7)
    pos, vals = torch.randn(B, N, D, generator=g), torch.randn(B, N, generator=g)
    bx, by = torch.randn(B, D, generator=g), torch.randn(B, generator=g)
    A = torch.randn(B, D, D, generator=g)
    specific = {
        "PSO": {"velocity": torch.randn(B, N, D, generator=g), "pbest_x": pos.clone()},
        "UMDA": {
            "mean": torch.randn(B, D, generator=g),
            "std": torch.rand(B, D, generator=g) + 0.1,
        },
        "AMALGAM": {
            "mean": torch.randn(B, D, generator=g),
            "cov": A @ A.transpose(-2, -1) + torch.eye(D),
            "ams_shift": torch.randn(B, D, generator=g),
            "c_mult": torch.rand(B, generator=g) + 0.5,
        },
    }[algo]
    return CanonicalState(algo, pos, vals, bx, by, specific)


@pytest.mark.parametrize(
    "source,target", [("PSO", "UMDA"), ("UMDA", "AMALGAM"), ("AMALGAM", "PSO")]
)
def test_translation_produces_valid_state(source, target):
    pair = TranslatorPair(source, target).eval()
    s = _source_state(source)
    ctx = NormContext.from_shared(s.positions, s.values)
    with torch.no_grad():
        out = pair.translate(s, target, ctx)
    if target == "UMDA":
        assert (out.specific["std"] > 0).all()
    if target == "AMALGAM":
        assert (torch.linalg.eigvalsh(out.specific["cov"]) > 0).all()
        assert (out.specific["c_mult"] > 0).all()
    for v in out.specific.values():
        assert torch.isfinite(v).all()


@pytest.mark.parametrize("target", EDAS)
def test_zero_action_reproduces_prior(target):
    pair = TranslatorPair("PSO", target).eval()
    s = _source_state("PSO")
    ctx = NormContext.from_shared(s.positions, s.values)
    dec = pair.decoders[target]
    zero = torch.zeros(s.positions.shape[0], dec.action_dim(s.d, s.n))
    built = dec.build_state(zero, ctx, s)
    prior = computed_specific(target, s, ctx)
    for k in prior:
        assert torch.allclose(built[k], prior[k])


def test_umda_log_std_residual_can_narrow_one_coordinate():
    """Unlike an additive F F^T term, UMDA's per-coordinate residual is signed."""
    pair = TranslatorPair("PSO", "UMDA").eval()
    s = _source_state("PSO")
    ctx = NormContext.from_shared(s.positions, s.values)
    dec = pair.decoders["UMDA"]
    B, D = s.positions.shape[0], s.d
    coords = torch.zeros(B, dec.action_dim(D, s.n))
    coords[:, 1] = -1.0  # shift block is (D, 2) row-major: coordinate 0, log-std column
    built = dec.build_state(coords, ctx, s)
    prior = computed_specific("UMDA", s, ctx)
    assert (built["std"][:, 0] < prior["std"][:, 0]).all()
    assert torch.allclose(built["std"][:, 1:], prior["std"][:, 1:])


@pytest.mark.parametrize("pair", [("PSO", "AMALGAM"), ("UMDA", "CMAES")])
def test_env_episode_with_an_eda(pair):
    env = TranslationEnv(
        *pair,
        build_problem_ids({1, 2}, dims=[2], instances=[1]),
        IOHSuite(),
        fe_multiplier=200,
        n_individuals=12,
        seed=0,
    )
    obs, _ = env.reset()
    n = obs["native"]
    lossy = {"x": n["x"], "y": n["y"], "best_x": n["best_x"], "best_y": n["best_y"]}
    _, reward, *_ = env.step(lossy)
    assert np.isfinite(reward)
