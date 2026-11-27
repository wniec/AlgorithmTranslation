"""Computed BOBYQA warm start: BOBYQA's own model fit of the handed-over points,
independent of the source optimizer, and the residual decoder built on it."""

import math

import numpy as np
import torch

from algorithm_translation.models.heads import StateDecoder
from algorithm_translation.models.norm import NormContext
from algorithm_translation.models.prior import bobyqa_specific, prior_native
from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.runs import make_optimizer, problem_cfg, shared_handoff
from algorithm_translation.state.canonical import CanonicalState, warm_start_optimizer
from algorithm_translation.suite import IOHSuite, build_problem_ids


def _points(B=2, N=12, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, D, generator=g, dtype=torch.float64), g


def _state(algo, X, F, specific=None):
    ib = F.argmin(dim=1)
    b = torch.arange(X.shape[0])
    return CanonicalState(algo, X, F, X[b, ib], F[b, ib], specific or {})


def _ctx(s):
    return NormContext.from_shared(s.positions, s.values)


def test_recovers_an_exact_quadratic():
    """Enough points for a full quadratic -> the fit is the function itself."""
    D = 3
    X, g = _points(B=1, N=20, D=D)
    A = torch.randn(D, D, generator=g, dtype=torch.float64)
    H = A @ A.T + torch.eye(D, dtype=torch.float64)
    b = torch.randn(D, generator=g, dtype=torch.float64)
    F = 0.5 * torch.einsum("bnd,de,bne->bn", X, H, X) + X @ b
    s = _state("PSO", X, F)
    out = bobyqa_specific(s, _ctx(s))
    centre = s.best_x[0]
    assert torch.allclose(out["hessian"][0], H, atol=1e-8)
    assert torch.allclose(out["grad"][0], H @ centre + b, atol=1e-8)


def test_same_points_give_the_same_model_from_any_source():
    X, g = _points()
    F = (X**2).sum(-1)
    B, N, D = X.shape
    sources = {
        "PSO": {
            "velocity": torch.randn(B, N, D, generator=g, dtype=X.dtype),
            "pbest_x": torch.randn(B, N, D, generator=g, dtype=X.dtype),
        },
        "CMAES": {
            "mean": torch.randn(B, D, generator=g, dtype=X.dtype),
            "sigma_cov": torch.eye(D, dtype=X.dtype).expand(B, D, D) * 3.0,
            "p_c": torch.zeros(B, D, dtype=X.dtype),
            "p_s": torch.zeros(B, D, dtype=X.dtype),
        },
        "BOBYQA": {
            "grad": torch.randn(B, D, generator=g, dtype=X.dtype),
            "hessian": torch.eye(D, dtype=X.dtype).expand(B, D, D) * 7.0,
            "radius": torch.ones(B, dtype=X.dtype),
        },
    }
    outs = [bobyqa_specific(st, _ctx(st)) for st in
            (_state(a, X, F, spec) for a, spec in sources.items())]
    for other in outs[1:]:
        for k in outs[0]:
            assert torch.equal(outs[0][k], other[k]), k


def test_prior_is_what_bobyqa_fits_on_its_own():
    """Warm-starting with the prior leaves BOBYQA's start-up model as it is
    with the population alone."""
    prob = IOHSuite().get_problem("bbob_f008_i01_d05")
    cfg = problem_cfg(prob)
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(12, 5))
    native = {"x": x, "y": np.array([prob(xi) for xi in x])}
    native |= {"best_x": x[native["y"].argmin()], "best_y": native["y"].min()}

    def startup_model(payload):
        opt = make_optimizer("BOBYQA", cfg, max_fe=500, target_fe=500, seed=0)
        warm_start_optimizer(opt, payload)
        opt._setup()
        return opt._g, opt._H, opt.delta

    g0, H0, r0 = startup_model(shared_handoff(native))
    swarm = {**native, "v": np.zeros_like(x), "p_x": x, "p_y": native["y"]}
    g1, H1, r1 = startup_model(prior_native("PSO", swarm, "BOBYQA"))
    assert np.allclose(g0, g1, rtol=1e-4, atol=1e-4 * np.abs(g0).max())
    assert np.allclose(H0, H1, rtol=1e-4, atol=1e-4 * np.abs(H0).max())
    assert math.isclose(r0, r1, rel_tol=1e-5)


def test_zero_action_reproduces_prior_and_residuals_act():
    X, _ = _points()
    s = _state("PSO", X, (X**2).sum(-1))
    ctx = _ctx(s)
    dec = StateDecoder("BOBYQA", cov_rank=2)
    D = s.d
    assert dec.action_dim(D, s.n) == 3 * D + 2
    prior = bobyqa_specific(s, ctx)
    zero = dec.build_state(torch.zeros(2, dec.action_dim(D, s.n), dtype=X.dtype), ctx, s)
    for k in prior:
        assert torch.equal(zero[k], prior[k]), k

    wider = bobyqa_specific(s, ctx, factor=torch.ones(2, D, 1, dtype=X.dtype))
    added = torch.linalg.eigvalsh(wider["hessian"] - prior["hessian"])
    assert (added > -1e-9).all()  # the first factor column adds curvature
    assert added.max() > 1e-6
    f = torch.zeros(2, D, 2, dtype=X.dtype)
    f[..., 1] = 1.0  # the second column removes it, so the Hessian can turn indefinite
    flatter = bobyqa_specific(s, ctx, factor=f)
    removed = torch.linalg.eigvalsh(flatter["hessian"] - prior["hessian"])
    assert (removed < 1e-9).all() and removed.min() < -1e-6
    doubled = bobyqa_specific(s, ctx, log_radius=torch.full((2,), math.log(2.0)))
    assert torch.allclose(doubled["radius"], 2 * prior["radius"])
    scaled = bobyqa_specific(s, ctx, log_curvature=torch.full((2,), math.log(3.0)))
    assert torch.allclose(scaled["hessian"], 3 * prior["hessian"])


def test_unevaluated_points_are_ignored():
    X, _ = _points(B=1, N=12, D=2)
    F = (X**2).sum(-1)
    F_inf = F.clone()
    F_inf[0, 0 if F[0, 0] != F.min() else 1] = math.inf
    out = bobyqa_specific(_state("PSO", X, F_inf), _ctx(_state("PSO", X, F)))
    assert all(torch.isfinite(v).all() for v in out.values())


def test_relative_baseline_into_bobyqa_is_the_computed_model():
    env = TranslationEnv(
        "PSO",
        "BOBYQA",
        build_problem_ids({1, 8}, dims=[5], instances=[1]),
        IOHSuite(),
        fe_multiplier=200,
        n_individuals=12,
        random_start=False,
        reward_mode="relative",
        seed=3,
    )
    obs, _ = env.reset()
    assert obs["target_algo"] == "BOBYQA"
    native = env.relative_baseline_native()
    assert {"grad", "hessian", "radius"} <= set(native)
    assert native["hessian"].shape == (5, 5) and native["radius"] > 0
