"""Population-conditioned, coordinate-equivariant exploration std.

The policy's per-coordinate std is a learnable per-segment base plus a bounded
residual read from the encoder latent: per dimension-token for the D-indexed
segments, one global value for scalar segments. These tests pin the three
properties that matter — backward-compatible init, genuine per-dimension
adaptivity, and coordinate-permutation equivariance."""

import torch

from algorithm_translation.models.norm import NormContext
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.state.canonical import CanonicalState


def _state(algo, B=1, N=8, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos, vals = torch.randn(B, N, D, generator=g), torch.randn(B, N, generator=g)
    bx, by = torch.randn(B, D, generator=g), torch.randn(B, generator=g)
    A = torch.randn(B, D, D, generator=g)
    if algo == "BOBYQA":
        spec = {
            "grad": torch.randn(B, D, generator=g),
            "hessian": A @ A.transpose(-2, -1) + torch.eye(D),
            "radius": torch.rand(B, generator=g) + 0.5,
        }
    elif algo == "CMAES":
        spec = {
            "mean": torch.randn(B, D, generator=g),
            "sigma_cov": A @ A.transpose(-2, -1) + torch.eye(D),
            "p_c": torch.randn(B, D, generator=g),
            "p_s": torch.randn(B, D, generator=g),
        }
    else:  # PSO
        spec = {
            "velocity": torch.randn(B, N, D, generator=g),
            "pbest_x": torch.randn(B, N, D, generator=g),
        }
    return CanonicalState(algo, pos, vals, bx, by, spec)


def _ctx(s):
    return NormContext.from_shared(s.positions, s.values)


def test_std_shape_and_finite_all_targets():
    """Runs for every segment kind: ppd (PSO), cov+scalar (BOBYQA), cov no-scalar
    (CMAES), set (MADDE)."""
    for a, b in (("PSO", "BOBYQA"), ("CMAES", "BOBYQA"), ("PSO", "CMAES"), ("PSO", "MADDE")):
        ac = ActorCritic(a, b, hidden=32, n_layers=1).eval()
        for target in (a, b):
            s = _state(a)
            z = ac.translator.encode(s, _ctx(s))
            std = ac.action_std(target, z, s.n)
            dec = ac.translator.decoders[target]
            assert std.shape == (1, dec.action_dim(s.d, s.n))
            assert torch.isfinite(std).all()
            assert (std > 0).all()


def test_std_matches_base_at_init():
    """Zero-init std heads => std is state-independent and equals exp(base) per
    segment, i.e. identical to the pre-adaptive per-segment behaviour."""
    ac = ActorCritic("PSO", "BOBYQA", hidden=32, n_layers=1).eval()
    s = _state("PSO")
    z = ac.translator.encode(s, _ctx(s))
    dec = ac.translator.decoders["BOBYQA"]
    std = ac.action_std("BOBYQA", z, s.n)
    sizes = torch.tensor(dec.segment_sizes(s.d, s.n))
    expected = ac.log_std["BOBYQA"].repeat_interleave(sizes).exp().unsqueeze(0)
    assert torch.allclose(std, expected, atol=1e-6)


def test_std_becomes_per_dimension_after_head_learns():
    """Once the token head is non-trivial, a D-indexed segment's std varies
    across dimensions (it is no longer one flat value per segment)."""
    ac = ActorCritic("PSO", "BOBYQA", hidden=32, n_layers=1).eval()
    with torch.no_grad():
        ac.std_token_head["BOBYQA"].weight.normal_(0, 0.5)
        ac.std_token_head["BOBYQA"].bias.normal_(0, 0.5)
    s = _state("PSO")
    z = ac.translator.encode(s, _ctx(s))
    std = ac.action_std("BOBYQA", z, s.n)
    grad_seg = std[0, : s.d]  # first segment is grad, one coord per dimension
    assert grad_seg.std() > 1e-4  # genuinely varies across dimensions


def test_std_is_coordinate_permutation_equivariant():
    """Permuting the coordinate axes of the source state permutes the per-
    dimension std identically — the policy stays coordinate-equivariant."""
    ac = ActorCritic("PSO", "BOBYQA", hidden=32, n_layers=1).eval()
    with torch.no_grad():  # give the residual head real, non-zero structure
        ac.std_token_head["BOBYQA"].weight.normal_(0, 0.7)
        ac.std_token_head["BOBYQA"].bias.normal_(0, 0.3)

    s = _state("PSO", D=5, seed=1)
    perm = torch.randperm(s.d)
    s_perm = CanonicalState(
        "PSO",
        s.positions[..., perm],
        s.values,
        s.best_x[..., perm],
        s.best_y,
        {k: v[..., perm] for k, v in s.specific.items()},
    )
    with torch.no_grad():
        std = ac.action_std("BOBYQA", ac.translator.encode(s, _ctx(s)), s.n)
        std_perm = ac.action_std("BOBYQA", ac.translator.encode(s_perm, _ctx(s_perm)), s.n)

    D = s.d
    # grad segment: one std per dimension -> should permute with the coords.
    assert torch.allclose(std[0, :D][perm], std_perm[0, :D], atol=1e-4)
    # cov_d segment (the covariance diagonal) is also one-per-dimension.
    dec = ac.translator.decoders["BOBYQA"]
    sizes = dec.segment_sizes(D, s.n)
    names = dec.segment_names()
    off = sum(sizes[: names.index("cov_d")])
    cov_d = std[0, off : off + D]
    cov_d_perm = std_perm[0, off : off + D]
    assert torch.allclose(cov_d[perm], cov_d_perm, atol=1e-4)
