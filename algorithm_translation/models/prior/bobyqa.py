"""Computed BOBYQA warm start: the quadratic model BOBYQA itself would fit.

BOBYQA keeps a quadratic surrogate ``m(c + s) = f(c) + g.s + 1/2 s.H s`` centred
on its best interpolation point ``c``. Every source hands over a population of
evaluated points, and that is all this prior reads: positions and values, never
a source-specific field. The same points therefore give the same model whichever
optimizer produced them.

The prior is exactly the model BOBYQA builds from those points on its own
(``BOBYQA._setup`` -> ``_refit`` from a zero model): the minimum-norm least-
squares quadratic through them. A zero action therefore reproduces the plain
hand-off, which is also the ``relative`` reward's baseline for switches into
BOBYQA. (Filling the model's under-determined part with a curvature guessed
from the population's covariance was tried and did no better than this on BBOB.)
The radius is BOBYQA's own default for a warm start, the population's mean
per-coordinate std.

BOBYQA re-fits ``(g, H)`` against the same points on start-up, changing them as
little as possible. A residual on top of the prior therefore survives only in
the part of the model the points leave open, which is the part a network can
add information to.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from algorithm_translation.models.norm import NormContext
from algorithm_translation.state.canonical import CanonicalState

# Smallest radius and residual length scale, relative to the population's spread.
_MIN_STD = 1e-4
# Size of a unit action coordinate (see the CMA-ES prior).
_GRAD_STEP = 0.5
_FACTOR_STEP = 0.3
_LOG_SCALE_STEP = 0.5
_MAX_LOG_SCALE = 6.0


def _design(d: Tensor) -> Tensor:
    """BOBYQA's quadratic basis for displacements ``d`` (B, N, D):
    ``[1, d_i, 1/2 d_i^2, d_i d_j (i < j)]`` (see ``BOBYQA._design``)."""
    B, N, D = d.shape
    iu, ju = torch.triu_indices(D, D, offset=1, device=d.device)
    return torch.cat(
        [
            torch.ones(B, N, 1, dtype=d.dtype, device=d.device),
            d,
            0.5 * d**2,
            d[..., iu] * d[..., ju],
        ],
        dim=-1,
    )


def fit_model(state: CanonicalState, ctx: NormContext) -> dict[str, Tensor]:
    """The prior model of a batched state, from its positions and values only.

    Returns ``grad`` (B, D), ``hessian`` (B, D, D), ``radius`` (B,) and the
    reference scales the decoder's residuals are measured in: ``h_ref`` (B,),
    a curvature, and ``g_ref`` (B,), a gradient component.
    """
    dtype = state.positions.dtype
    X = state.positions.double()  # (B, N, D)
    F = state.values.double()  # (B, N)
    B, _, D = X.shape
    batch = torch.arange(B, device=X.device)

    finite = torch.isfinite(F)
    ib = torch.where(finite, F, torch.full_like(F, math.inf)).argmin(dim=1)  # (B,)
    d = X - X[batch, ib][:, None, :]  # (B, N, D)
    w = finite.to(X.dtype).unsqueeze(-1)  # unevaluated points carry no information
    r = torch.where(finite, F - F[batch, ib][:, None], torch.zeros_like(F))

    coef = (torch.linalg.pinv(_design(d) * w) @ r.unsqueeze(-1)).squeeze(-1)
    grad = coef[:, 1 : 1 + D]
    hessian = torch.diag_embed(coef[:, 1 + D : 1 + 2 * D])
    iu, ju = torch.triu_indices(D, D, offset=1, device=X.device)
    hessian[:, iu, ju] = coef[:, 1 + 2 * D :]
    hessian[:, ju, iu] = coef[:, 1 + 2 * D :]

    spread = X.std(dim=1, unbiased=False)  # (B, D)
    floor = _MIN_STD * ctx.scale.double()
    radius = torch.maximum(spread.mean(dim=-1), floor)

    # Residual units: the curvature the value spread implies over the
    # population's spread (exact in expectation for an isotropic quadratic),
    # and the gradient that curvature gives one population-spread away.
    tau = spread.pow(2).mean(dim=-1).sqrt().clamp_min(floor)
    h_ref = ctx.val_std.double() / (tau.pow(2) * math.sqrt(D))
    g_ref = h_ref * tau
    out = {"grad": grad, "hessian": hessian, "radius": radius, "h_ref": h_ref, "g_ref": g_ref}
    return {k: v.to(dtype) for k, v in out.items()}


def curvature_signs(rank: int, dtype=None, device=None) -> Tensor:
    """Fixed signs of the factor columns, alternating ``+1, -1, +1, ...``.

    A Hessian, unlike a covariance, need not be positive semidefinite (saddles,
    concave directions), so the residual ``F diag(s) F^T`` adds curvature along
    the even columns and removes it along the odd ones. Any symmetric matrix of
    rank <= R with at most ``ceil(R/2)`` positive and ``floor(R/2)`` negative
    eigenvalues is reachable, at no extra action coordinates. Rank 1 can only add.
    """
    s = torch.ones(rank, dtype=dtype, device=device)
    s[1::2] = -1.0
    return s


def bobyqa_specific(
    state: CanonicalState,
    ctx: NormContext,
    grad_shift: Tensor | None = None,
    factor: Tensor | None = None,
    log_curvature: Tensor | None = None,
    log_radius: Tensor | None = None,
) -> dict[str, Tensor]:
    """BOBYQA's specific fields from the computed prior, plus optional residuals:

    * ``grad_shift`` (B, D): added to the gradient, in units of ``g_ref``;
    * ``factor`` (B, D, R): adds the signed ``h_ref * F diag(s) F^T`` to the
      Hessian, ``s = curvature_signs(R)``;
    * ``log_curvature`` (B,): multiplies the Hessian by ``exp(log_curvature)``;
    * ``log_radius`` (B,): multiplies the radius by ``exp(log_radius)``.

    Leaving them all out gives the prior itself.
    """
    m = fit_model(state, ctx)
    B = m["grad"].shape[0]
    grad, hessian, radius = m["grad"], m["hessian"], m["radius"]
    if grad_shift is not None:
        grad = grad + m["g_ref"].unsqueeze(-1) * grad_shift
    if factor is not None:
        s = curvature_signs(factor.shape[-1], factor.dtype, factor.device)
        signed = (factor * s) @ factor.transpose(-2, -1)
        hessian = hessian + m["h_ref"].reshape(B, 1, 1) * signed
    if log_curvature is not None:
        scale = log_curvature.clamp(-_MAX_LOG_SCALE, _MAX_LOG_SCALE).exp()
        hessian = hessian * scale.reshape(B, 1, 1)
    if log_radius is not None:
        radius = radius * log_radius.clamp(-_MAX_LOG_SCALE, _MAX_LOG_SCALE).exp()
    return {"grad": grad, "hessian": hessian, "radius": radius}


def bobyqa_from_action(
    state: CanonicalState,
    ctx: NormContext,
    shift: Tensor | None,
    factor: Tensor | None,
    scalars: Tensor | None,
) -> dict[str, Tensor]:
    """``bobyqa_specific`` from the decoder's action blocks: ``shift`` (B, D)
    moves the gradient, ``factor`` (B, D, R) adds or removes curvature along R
    directions, ``scalars`` (B, 2)
    are the log curvature-scale and log radius shifts."""
    return bobyqa_specific(
        state,
        ctx,
        grad_shift=None if shift is None else _GRAD_STEP * shift,
        factor=None if factor is None else _FACTOR_STEP * factor,
        log_curvature=None if scalars is None else _LOG_SCALE_STEP * scalars[:, 0],
        log_radius=None if scalars is None else _LOG_SCALE_STEP * scalars[:, 1],
    )
