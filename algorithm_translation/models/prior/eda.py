"""Computed EDA warm starts: the search distribution an EDA estimates by itself.

Both EDAs re-estimate their Gaussian from truncation-selected points, so the
natural hand-off is that estimate from the points the source already evaluated
(a swarm's personal bests when known, as for CMA-ES; otherwise its population).
With a population-only hand-off, the EDA builds exactly this estimate on its own
(``optimizers/EDA``), so the prior is also the ``relative`` reward's baseline.

The decoder predicts residuals around it, and a zero action reproduces it:

* **UMDA** (diagonal Gaussian): shift column 0 moves the mean, in units of each
  coordinate's std; shift column 1 is a per-coordinate log-std change (signed,
  so a single coordinate can be narrowed as well as widened); one global
  log-std scalar.
* **AMaLGaM** (full Gaussian): shift column 0 moves the mean and column 1 sets
  the anticipated mean shift (zero in the estimate), both in units of the
  estimate's RMS std ``tau``; the factor adds ``tau**2 F F^T`` to the
  covariance, as for CMA-ES; scalars are a global log-scale of the covariance
  and ``log c_mult`` (1 in the estimate).
"""

from __future__ import annotations

import torch
from torch import Tensor

from algorithm_translation.models.norm import NormContext
from algorithm_translation.models.prior.cmaes import _search_points
from algorithm_translation.optimizers.EDA import amalgam as _am
from algorithm_translation.optimizers.EDA import umda as _um
from algorithm_translation.state.canonical import CanonicalState

# Size of a unit action coordinate (see the CMA-ES prior).
_MEAN_STEP = 0.5
_SHIFT_STEP = 0.5
_FACTOR_STEP = 0.3
_LOG_STEP = 0.5
_MAX_LOG = 6.0


def _selected(state: CanonicalState, k_of) -> Tensor:
    """The best ``k_of(n_points)`` search points, (B, k, D)."""
    points, values = _search_points(state)
    B, N, D = points.shape
    k = k_of(N)
    idx = values.argsort(dim=1)[:, :k]
    return points.gather(1, idx.unsqueeze(-1).expand(B, k, D))


def umda_estimate(state: CanonicalState) -> tuple[Tensor, Tensor]:
    """``estimate_umda`` on a batched state: mean (B, D), std (B, D)."""
    pts = _selected(state, _um.n_selected)
    mean = pts.mean(dim=1)
    std = (pts - mean.unsqueeze(1)).pow(2).mean(dim=1).sqrt().clamp_min(_um.MIN_STD)
    return mean, std


def amalgam_estimate(state: CanonicalState) -> tuple[Tensor, Tensor]:
    """``estimate_amalgam`` on a batched state: mean (B, D), cov (B, D, D)."""
    pts = _selected(state, _am.n_selected)
    B, k, D = pts.shape
    mean = pts.mean(dim=1)
    diff = pts - mean.unsqueeze(1)
    cov = diff.transpose(1, 2) @ diff / k
    eye = torch.eye(D, device=pts.device, dtype=pts.dtype)
    if k <= D:
        tau2 = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1).reshape(B, 1, 1)
        b = _am.SMALL_SAMPLE_BLEND
        cov = (1.0 - b) * cov + b * tau2 * eye
    return mean, cov + _am.MIN_VAR * eye


def _shift_cols(shift: Tensor | None, k: int) -> list[Tensor | None]:
    """Split a (B, D, k) shift block into its columns (None -> no residual)."""
    if shift is None:
        return [None] * k
    if shift.dim() == 2:  # a single column
        return [shift] + [None] * (k - 1)
    return [shift[..., i] for i in range(k)]


def umda_from_action(
    state: CanonicalState,
    ctx: NormContext,
    shift: Tensor | None,
    factor: Tensor | None,
    scalars: Tensor | None,
) -> dict[str, Tensor]:
    mean, std = umda_estimate(state)
    d_mean, d_log_std = _shift_cols(shift, 2)
    if d_mean is not None:
        mean = mean + _MEAN_STEP * std * d_mean
    log_scale = torch.zeros_like(std)
    if d_log_std is not None:
        log_scale = log_scale + d_log_std
    if scalars is not None:
        log_scale = log_scale + scalars[:, :1]
    std = std * torch.exp((_LOG_STEP * log_scale).clamp(-_MAX_LOG, _MAX_LOG))
    return {"mean": mean, "std": std.clamp_min(_um.MIN_STD)}


def amalgam_from_action(
    state: CanonicalState,
    ctx: NormContext,
    shift: Tensor | None,
    factor: Tensor | None,
    scalars: Tensor | None,
) -> dict[str, Tensor]:
    mean, cov = amalgam_estimate(state)
    B = mean.shape[0]
    tau = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1).sqrt()  # (B,)
    d_mean, d_ams = _shift_cols(shift, 2)
    if d_mean is not None:
        mean = mean + _MEAN_STEP * tau.unsqueeze(-1) * d_mean
    ams = torch.zeros_like(mean)
    if d_ams is not None:
        ams = _SHIFT_STEP * tau.unsqueeze(-1) * d_ams
    if factor is not None:
        f = _FACTOR_STEP * factor
        cov = cov + tau.pow(2).reshape(B, 1, 1) * f @ f.transpose(-2, -1)
    c_mult = torch.ones_like(tau)
    if scalars is not None:
        scale2 = torch.exp((2.0 * _LOG_STEP * scalars[:, 0]).clamp(-_MAX_LOG, _MAX_LOG))
        cov = cov * scale2.reshape(B, 1, 1)
        c_mult = torch.exp((_LOG_STEP * scalars[:, 1]).clamp(-_MAX_LOG, _MAX_LOG))
    return {"mean": mean, "cov": cov, "ams_shift": ams, "c_mult": c_mult}


__all__ = [
    "amalgam_estimate",
    "amalgam_from_action",
    "umda_estimate",
    "umda_from_action",
]
