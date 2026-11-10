"""BOBYQA model-based trust-region optimizer + its quadratic-model translation.

Mirrors ``test_madde.py``: the optimizer solves smooth problems as a strong
local method, its native state round-trips through the canonical adapter, and
the neural translator maps its gradient/Hessian/radius both directions.
"""

import numpy as np
import torch

from cat.models import NormContext, TranslatorPair
from cat.optimizers.portfolio import PORTFOLIO
from cat.state import from_canonical, get_spec, to_canonical
from cat.state.canonical import CanonicalState, warm_start_optimizer
from cat.suite import IOHSuite


def _cfg(problem_id="bbob_f001_i01_d05"):
    prob = IOHSuite().get_problem(problem_id)
    return prob, {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }


def _augment(opt, dim):
    native = {k: v for k, v in opt.get_data().items() if v is not None}
    native["sigma"] = getattr(opt, "sigma", None)
    native["best_x"] = (
        opt.best_so_far_x if opt.best_so_far_x is not None else np.zeros(dim)
    )
    native["best_y"] = opt.best_so_far_y
    return native


def _make(problem_id, **opts):
    prob, cfg = _cfg(problem_id)
    base = {"max_function_evaluations": 3000, "seed_rng": 1, "verbose": False}
    return prob, PORTFOLIO["BOBYQA"](cfg, {**base, **opts})


# -- optimizer ----------------------------------------------------------- #


def test_bobyqa_solves_sphere():
    prob, opt = _make("bbob_f001_i01_d05", target_fe=3000)
    res = opt.optimize()
    assert res["best_so_far_y"] <= prob.optimum + 1e-3  # exact on a quadratic


def test_bobyqa_strong_local_on_rosenbrock():
    """A strong local method should drive the (unimodal) Rosenbrock valley to
    near the optimum given enough budget."""
    prob, opt = _make("bbob_f008_i01_d05", target_fe=5000, max_function_evaluations=5000)
    res = opt.optimize()
    assert res["best_so_far_y"] <= prob.optimum + 1e-2


def test_bobyqa_fixed_pop_interpolation_set():
    _, opt = _make("bbob_f001_i01_d05", target_fe=2000, n_individuals=12)
    opt.optimize()
    data = opt.get_data()
    assert len(data["x"]) == 12  # interpolation-set size held at the configured N
    assert data["hessian"].shape == (5, 5)
    assert data["grad"].shape == (5,)
    assert data["radius"] > 0.0
    assert {"x", "y", "grad", "hessian", "radius"}.issubset(data)


def test_bobyqa_warmstart_from_shared_only():
    """A PSO/CMA-ES hand-off supplies only x/y (no model); BOBYQA must refit a
    valid model and still run to a finite result."""
    _, a = _make("bbob_f001_i01_d05", target_fe=800, n_individuals=12)
    a.optimize()
    native = _augment(a, 5)
    _, b = _make("bbob_f001_i01_d05", target_fe=1600, n_individuals=12, seed_rng=2)
    warm_start_optimizer(
        b,
        {
            "x": native["x"],
            "y": native["y"],
            "best_x": native["best_x"],
            "best_y": native["best_y"],
        },
    )
    res = b.optimize()
    assert np.isfinite(res["best_so_far_y"])


def test_bobyqa_warmstart_uses_translated_model():
    """A full hand-off (with model gradient/Hessian/radius) is accepted and the
    optimizer continues to a finite, improved result."""
    _, a = _make("bbob_f001_i01_d05", target_fe=800, n_individuals=12)
    a.optimize()
    native = _augment(a, 5)
    st = to_canonical("BOBYQA", native)
    back = from_canonical(st)
    _, b = _make("bbob_f001_i01_d05", target_fe=1600, n_individuals=12, seed_rng=3)
    warm_start_optimizer(b, {**back, "best_x": native["best_x"], "best_y": native["best_y"]})
    res = b.optimize()
    assert np.isfinite(res["best_so_far_y"])
    assert res["best_so_far_y"] <= native["best_y"] + 1e-9  # never worse than handed


# -- canonical / translation --------------------------------------------- #


def test_bobyqa_has_full_state():
    from cat.state.canonical import has_full_state

    _, opt = _make("bbob_f001_i01_d05", target_fe=800, n_individuals=12)
    opt.optimize()
    assert has_full_state("BOBYQA", _augment(opt, 5))


def test_bobyqa_canonical_shapes_and_roundtrip():
    _, opt = _make("bbob_f001_i01_d05", target_fe=800, n_individuals=12)
    opt.optimize()
    native = _augment(opt, 5)
    st = to_canonical("BOBYQA", native)
    assert st.positions.shape == (12, 5)
    assert st.specific["grad"].shape == (5,)
    assert st.specific["hessian"].shape == (5, 5)
    assert st.specific["radius"].shape == ()
    back = from_canonical(st)
    st2 = to_canonical("BOBYQA", {**native, **back})
    for name in get_spec("BOBYQA").field_names:
        assert np.allclose(
            st.specific[name].numpy(), st2.specific[name].numpy(), atol=1e-5
        ), f"{name} not preserved"


def _rand_state(algo, B=2, N=10, D=4):
    g = torch.Generator().manual_seed(abs(hash(algo)) % 2**31)
    pos, vals = torch.randn(B, N, D, generator=g), torch.randn(B, N, generator=g)
    bx, by = torch.randn(B, D, generator=g), torch.randn(B, generator=g)
    A = torch.randn(B, D, D, generator=g)
    if algo == "BOBYQA":
        spec = {
            "grad": torch.randn(B, D, generator=g),
            "hessian": A @ A.transpose(-2, -1) + torch.eye(D),
            "radius": torch.rand(B, generator=g) + 0.5,
        }
    else:  # PSO source
        spec = {
            "velocity": torch.randn(B, N, D, generator=g),
            "pbest_x": torch.randn(B, N, D, generator=g),
        }
    return CanonicalState(algo, pos, vals, bx, by, spec)


def test_translate_to_bobyqa_produces_psd_model():
    pair = TranslatorPair("PSO", "BOBYQA").eval()
    s = _rand_state("PSO", N=10, D=4)
    ctx = NormContext.from_shared(s.positions, s.values)
    with torch.no_grad():
        out = pair.translate(s, "BOBYQA", ctx)
    assert sorted(out.specific) == ["grad", "hessian", "radius"]
    assert out.specific["hessian"].shape == (2, 4, 4)
    assert out.specific["radius"].shape == (2,)
    # The Hessian decodes through the PSD covariance head (convex local model).
    assert (torch.linalg.eigvalsh(out.specific["hessian"]) > -1e-6).all()
    # Radius is decoded in log-space -> strictly positive.
    assert (out.specific["radius"] > 0).all()
    assert torch.isfinite(out.specific["hessian"]).all()


def test_bobyqa_translates_both_directions():
    for a, b in (("PSO", "BOBYQA"), ("BOBYQA", "PSO")):
        pair = TranslatorPair(a, b).eval()
        s = _rand_state(a)
        ctx = NormContext.from_shared(s.positions, s.values)
        with torch.no_grad():
            out = pair.translate(s, b, ctx)
        assert sorted(out.specific) == sorted(f.name for f in get_spec(b).specific)
        assert all(torch.isfinite(v).all() for v in out.specific.values())


def test_bobyqa_specific_fields_survive_resample():
    """grad/hessian/radius are per-distribution, not per-individual, so resizing
    the interpolation set must leave them untouched (like CMA-ES's mean/cov)."""
    from cat.state.canonical import resample_population

    _, opt = _make("bbob_f001_i01_d05", target_fe=800, n_individuals=12)
    opt.optimize()
    st = to_canonical("BOBYQA", _augment(opt, 5))
    down = resample_population(st, st.n - 3, np.random.default_rng(0))
    up = resample_population(st, st.n + 4, np.random.default_rng(0))
    for out in (down, up):
        for name in ("grad", "hessian", "radius"):
            assert torch.allclose(out.specific[name], st.specific[name])
