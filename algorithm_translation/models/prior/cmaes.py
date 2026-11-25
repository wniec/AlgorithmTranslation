"""Computed CMA-ES warm start: a search distribution from the handed-over points.

A CMA-ES search distribution can be estimated directly from the points the
source optimizer already evaluated: the mean is CMA-ES's own weighted
recombination of the best points, and the covariance has the form of its rank-mu
update, ``sum w_i (x_i - m_all)(x_i - m_all)^T``, so it stretches along the
direction of improvement. A swarm's personal bests are the preferred points
(its memory of good positions); any other source uses its current population.
Evolution paths start at zero, as on any fresh CMA-ES run.

This state is both the ``relative`` reward's baseline for switches into CMA-ES
and the point the CMA-ES decoder head predicts residuals around, so a zero action
reproduces the baseline exactly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from algorithm_translation.models.norm import NormContext
from algorithm_translation.state.canonical import CanonicalState

# Share of the covariance replaced by its isotropic part: a small set of parents
# gives a rank-deficient estimate, which CMA-ES could never search out of.
PRIOR_BLEND = 0.1
# Smallest search std, relative to the population's spread (for coincident points).
_MIN_STD = 1e-4
_MAX_LOG_SIGMA = 6.0
# Size of a unit action coordinate. Exploration noise lands directly on these,
# so they keep a std-1 action within a plausible range of the prior.
_MEAN_STEP = 0.5
_FACTOR_STEP = 0.3
_LOG_SIGMA_STEP = 0.5


def _n_parents(n: int, d: int) -> int:
    """CMA-ES's default parent count for dimension ``d``, capped at half the points."""
    return max(1, min(n // 2, (4 + int(3 * math.log(d))) // 2))


def _search_points(state: CanonicalState) -> tuple[Tensor, Tensor]:
    """The ranked points to estimate from: personal bests when known."""
    if "pbest_y" in state.aux:
        return state.specific["pbest_x"], state.aux["pbest_y"]
    return state.positions, state.values


def prior_distribution(state: CanonicalState) -> tuple[Tensor, Tensor]:
    """Mean (B, D) and raw rank-mu covariance (B, D, D) of a batched state."""
    points, values = _search_points(state)
    B, N, D = points.shape
    mu = _n_parents(N, D)
    weights = math.log(mu + 0.5) - torch.arange(1, mu + 1, device=points.device).log()
    weights = (weights / weights.sum()).to(points.dtype)

    idx = values.argsort(dim=1)[:, :mu]
    elites = points.gather(1, idx.unsqueeze(-1).expand(B, mu, D))  # (B, mu, D)
    mean = (weights[None, :, None] * elites).sum(dim=1)
    diff = elites - points.mean(dim=1, keepdim=True)
    cov = (diff.transpose(1, 2) * weights[None, None, :]) @ diff
    return mean, cov


def cmaes_specific(
    state: CanonicalState,
    ctx: NormContext,
    mean_shift: Tensor | None = None,
    factor: Tensor | None = None,
    log_sigma: Tensor | None = None,
    blend_logit: Tensor | None = None,
) -> dict[str, Tensor]:
    """CMA-ES's specific fields from the computed prior, plus optional residuals.

    Residuals are relative to the prior's own scale ``tau`` (its RMS std per
    dimension), so they mean the same whether the swarm is wide or converged:

    * ``mean_shift`` (B, D): added to the mean, in units of ``tau``;
    * ``factor`` (B, D, R): adds ``tau**2 * factor factor^T`` to the covariance;
    * ``log_sigma`` (B,): multiplies the covariance by ``exp(2 * log_sigma)``;
    * ``blend_logit`` (B,): shifts ``PRIOR_BLEND`` in logit space.

    Leaving them all out gives the prior itself.
    """
    mean, cov = prior_distribution(state)
    B, D = mean.shape
    tau2 = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1)  # (B,)
    eye = torch.eye(D, device=cov.device, dtype=cov.dtype)

    logit = math.log(PRIOR_BLEND / (1.0 - PRIOR_BLEND))
    if blend_logit is not None:
        logit = logit + blend_logit
    blend = torch.sigmoid(torch.as_tensor(logit, device=cov.device, dtype=cov.dtype))
    blend = blend.reshape(-1, 1, 1)
    sigma_cov = (1.0 - blend) * cov + blend * tau2.reshape(B, 1, 1) * eye

    if factor is not None:
        sigma_cov = sigma_cov + tau2.reshape(B, 1, 1) * factor @ factor.transpose(-2, -1)
    if log_sigma is not None:
        scale2 = torch.exp(2.0 * log_sigma.clamp(-_MAX_LOG_SIGMA, _MAX_LOG_SIGMA))
        sigma_cov = sigma_cov * scale2.reshape(B, 1, 1)
    sigma_cov = sigma_cov + (_MIN_STD * ctx.scale).pow(2).reshape(B, 1, 1) * eye

    if mean_shift is not None:
        mean = mean + tau2.sqrt().unsqueeze(-1) * mean_shift
    return {
        "mean": mean,
        "sigma_cov": sigma_cov,
        "p_c": torch.zeros_like(mean),
        "p_s": torch.zeros_like(mean),
    }


def cmaes_from_action(
    state: CanonicalState,
    ctx: NormContext,
    shift: Tensor | None,
    factor: Tensor | None,
    scalars: Tensor | None,
) -> dict[str, Tensor]:
    """``cmaes_specific`` from the decoder's action blocks: ``shift`` (B, D) moves
    the mean, ``factor`` (B, D, R) widens the covariance, ``scalars`` (B, 2) are
    the log step-size and blend-logit shifts. ``None`` blocks leave the prior."""
    return cmaes_specific(
        state,
        ctx,
        mean_shift=None if shift is None else _MEAN_STEP * shift,
        factor=None if factor is None else _FACTOR_STEP * factor,
        log_sigma=None if scalars is None else _LOG_SIGMA_STEP * scalars[:, 0],
        blend_logit=None if scalars is None else scalars[:, 1],
    )
