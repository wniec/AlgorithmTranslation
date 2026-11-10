"""Computed CMA-ES warm start: which points it uses, its formulas, the residual
decoder built on it, and its role as the 'relative' reward baseline."""

import math

import numpy as np
import torch

from algorithm_translation.models.heads import StateDecoder
from algorithm_translation.models.norm import NormContext
from algorithm_translation.models.prior import PRIOR_BLEND, cmaes_specific, prior_distribution
from algorithm_translation.models.translator import TranslatorPair
from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.state.canonical import (
    CanonicalState,
    collate,
    from_canonical,
    resample_population,
    to_canonical,
)
from algorithm_translation.suite import IOHSuite, build_problem_ids


def _swarm(B=2, N=20, D=5, seed=0, with_pbest_y=True):
    g = torch.Generator().manual_seed(seed)
    aux = {"pbest_y": torch.randn(B, N, generator=g)} if with_pbest_y else {}
    return CanonicalState(
        "PSO",
        torch.randn(B, N, D, generator=g),
        torch.randn(B, N, generator=g),
        torch.randn(B, D, generator=g),
        torch.randn(B, generator=g),
        {
            "velocity": torch.randn(B, N, D, generator=g),
            "pbest_x": torch.randn(B, N, D, generator=g),
        },
        aux,
    )


def _ctx(s):
    return NormContext.from_shared(s.positions, s.values)


def _expected(points, values, mu):
    """Reference formulas, written out for one un-batched set of points."""
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    elites = points[np.argsort(values)[:mu]]
    diff = elites - points.mean(0)
    return w @ elites, (diff * w[:, None]).T @ diff


def test_prior_uses_personal_bests_when_ranked():
    s = _swarm()
    mean, cov = prior_distribution(s)
    # D=5: CMA-ES's default lambda is 8, so 4 parents.
    m, c = _expected(s.specific["pbest_x"][0].numpy(), s.aux["pbest_y"][0].numpy(), 4)
    assert np.allclose(mean[0].numpy(), m, atol=1e-5)
    assert np.allclose(cov[0].numpy(), c, atol=1e-5)


def test_prior_falls_back_to_population():
    s = _swarm(with_pbest_y=False)
    mean, cov = prior_distribution(s)
    m, c = _expected(s.positions[1].numpy(), s.values[1].numpy(), 4)
    assert np.allclose(mean[1].numpy(), m, atol=1e-5)
    assert np.allclose(cov[1].numpy(), c, atol=1e-5)


def test_prior_is_blended_and_full_rank_with_paths_at_zero():
    s = _swarm()
    out = cmaes_specific(s, _ctx(s))
    _, cov = prior_distribution(s)
    tau2 = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
    # 4 parents give a rank-deficient estimate; the blend must lift every eigenvalue.
    assert torch.linalg.eigvalsh(cov).min() < 1e-4
    lowest = torch.linalg.eigvalsh(out["sigma_cov"]).min(-1).values
    assert (lowest >= PRIOR_BLEND * tau2 * 0.99).all()
    assert (out["p_c"] == 0).all() and (out["p_s"] == 0).all()


def test_coincident_points_still_give_a_usable_covariance():
    s = _swarm()
    s.specific["pbest_x"] = torch.ones_like(s.specific["pbest_x"])
    native = from_canonical(
        CanonicalState(
            "CMAES", s.positions, s.values, s.best_x, s.best_y, cmaes_specific(s, _ctx(s))
        ).index(0)
    )
    assert np.isfinite(native["cm"]).all() and native["sigma"] > 0


def test_pbest_y_follows_every_state_operation():
    D, N = 3, 6
    rng = np.random.default_rng(0)
    native = {
        "x": rng.normal(size=(N, D)),
        "y": rng.normal(size=N),
        "best_x": rng.normal(size=D),
        "best_y": 0.0,
        "v": rng.normal(size=(N, D)),
        "p_x": rng.normal(size=(N, D)),
        "p_y": rng.normal(size=N),
    }
    s = to_canonical("PSO", native)
    assert np.allclose(s.aux["pbest_y"].numpy(), native["p_y"])
    assert collate([s, s]).index(1).aux["pbest_y"].shape == (N,)
    assert s.to("cpu").detach_clone().aux["pbest_y"].shape == (N,)

    small = resample_population(s, 3, rng)
    kept = torch.argsort(s.values)[:3]
    assert torch.equal(small.aux["pbest_y"], s.aux["pbest_y"][kept])
    assert resample_population(s, 9, rng).aux["pbest_y"].shape == (9,)
    # Not part of the optimizer-facing payload beyond its native key.
    assert "pbest_y" not in from_canonical(s)


def test_zero_action_reproduces_prior_exactly():
    s = _swarm()
    dec = StateDecoder("CMAES")
    coords = torch.zeros(s.positions.shape[0], dec.action_dim(s.d, s.n))
    decoded = dec.build_state(coords, _ctx(s), s)
    prior = cmaes_specific(s, _ctx(s))
    for k in prior:
        assert torch.equal(decoded[k], prior[k]), k


def test_untrained_translation_starts_near_prior():
    s = _swarm()
    pair = TranslatorPair("PSO", "CMAES").eval()
    with torch.no_grad():
        out = pair.translate(s, "CMAES", _ctx(s)).specific
    prior = cmaes_specific(s, _ctx(s))
    tau = torch.diagonal(prior["sigma_cov"], dim1=-2, dim2=-1).mean(-1).sqrt()
    assert ((out["mean"] - prior["mean"]).norm(dim=-1) < tau).all()
    ratio = torch.diagonal(out["sigma_cov"], dim1=-2, dim2=-1) / torch.diagonal(
        prior["sigma_cov"], dim1=-2, dim2=-1
    )
    assert ((ratio > 0.5) & (ratio < 2.0)).all()


def test_action_size_and_rank_zero():
    for rank, expected in ((4, 5 * 5 + 2), (1, 2 * 5 + 2), (0, 5 + 2)):
        dec = StateDecoder("CMAES", cov_rank=rank)
        assert dec.action_dim(5, 12) == expected
        assert dec.action_dim(5, 40) == expected  # independent of N
        s = _swarm()
        coords = torch.randn(s.positions.shape[0], expected)
        cov = dec.build_state(coords, _ctx(s), s)["sigma_cov"]
        assert torch.linalg.eigvalsh(cov).min() > 0


def test_residuals_act_as_documented():
    s = _swarm()
    ctx = _ctx(s)
    base = cmaes_specific(s, ctx)
    _, raw = prior_distribution(s)
    tau = torch.diagonal(raw, dim1=-2, dim2=-1).mean(-1).sqrt()

    shifted = cmaes_specific(s, ctx, mean_shift=torch.ones(2, 5))
    assert torch.allclose(shifted["mean"] - base["mean"], tau[:, None].expand(2, 5), atol=1e-5)

    doubled = cmaes_specific(s, ctx, log_sigma=torch.full((2,), math.log(2.0)))
    assert torch.allclose(doubled["sigma_cov"], 4 * base["sigma_cov"], rtol=1e-3)

    f = torch.zeros(2, 5, 1)
    f[:, 0, 0] = 1.0
    wider = cmaes_specific(s, ctx, factor=f)
    added = wider["sigma_cov"][:, 0, 0] - base["sigma_cov"][:, 0, 0]
    assert torch.allclose(added, tau**2, atol=1e-5)

    pure = cmaes_specific(s, ctx, blend_logit=torch.full((2,), 100.0))
    off = pure["sigma_cov"][0] - torch.diag(torch.diagonal(pure["sigma_cov"][0]))
    assert off.abs().max() < 1e-5


def _env(**kw):
    return TranslationEnv(
        "PSO",
        "CMAES",
        build_problem_ids({1, 8}, dims=[5], instances=[1]),
        IOHSuite(),
        fe_multiplier=200,
        n_individuals=12,
        random_start=False,
        reward_mode="relative",
        **kw,
    )


def test_baseline_into_cmaes_is_computed_not_a_restart():
    env = _env(seed=3)
    obs, _ = env.reset()
    native = env.relative_baseline_native()
    n = obs["native"]
    m, _c = _expected(n["p_x"], n["p_y"], 4)
    assert np.allclose(native["mean"], m, atol=1e-5)
    assert native["sigma"] > 0 and native["cm"].shape == (5, 5)


def test_baseline_into_pso_stays_the_lossy_default():
    env = _env(seed=3)
    obs, _ = env.reset()
    obs, *_ = env.step(env.relative_baseline_native())
    assert obs["target_algo"] == "PSO"
    assert set(env.relative_baseline_native()) == {"x", "y", "best_x", "best_y"}


def test_deviating_from_the_baseline_changes_the_reward():
    """A much worse hand-off (step size blown up) must score below the baseline."""
    env = _env(seed=3)
    env.reset()
    bad = dict(env.relative_baseline_native())
    bad["sigma"] = bad["sigma"] * 1e3
    _obs, r, *_ = env.step(bad)
    assert r < 0
