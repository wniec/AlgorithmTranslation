"""Computed warm starts: the target state a switch can derive without a network.

Some targets can build a sensible state from the handed-over population alone:
a CMA-ES search distribution (``cmaes``), BOBYQA's own quadratic model fit
(``bobyqa``). For those, the decoder head predicts only a residual around the
computed prior, and a zero action reproduces it exactly. That makes the prior both the starting
point of training and the ``relative`` reward's baseline.

Every prior has the same action layout, so the decoder handles all of them the
same way:

* ``shift`` (D,): a per-coordinate shift (CMA-ES mean / BOBYQA gradient);
* ``factor`` (D, R): a low-rank PSD term ``F F^T`` added to the matrix (CMA-ES
  covariance / BOBYQA Hessian), omitted when the rank is 0;
* ``scalars`` (``n_scalar``,): global log-scale shifts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from algorithm_translation.models.norm import NormContext
from algorithm_translation.models.prior.bobyqa import bobyqa_from_action, bobyqa_specific
from algorithm_translation.models.prior.cmaes import (
    PRIOR_BLEND,
    cmaes_from_action,
    cmaes_specific,
    prior_distribution,
)
from algorithm_translation.state.canonical import (
    CanonicalState,
    collate,
    from_canonical,
    resample_population,
    to_canonical,
)

__all__ = [
    "PRIORS",
    "PRIOR_BLEND",
    "Prior",
    "bobyqa_specific",
    "cmaes_specific",
    "computed_specific",
    "has_prior",
    "prior_distribution",
    "prior_native",
]

# build(state, ctx, shift, factor, scalars) -> raw specific fields; any block
# may be None (no residual).
Build = Callable[
    [CanonicalState, NormContext, Tensor | None, Tensor | None, Tensor | None],
    dict[str, Tensor],
]


@dataclass(frozen=True)
class Prior:
    n_scalar: int
    build: Build


PRIORS: dict[str, Prior] = {
    "CMAES": Prior(n_scalar=2, build=cmaes_from_action),
    "BOBYQA": Prior(n_scalar=2, build=bobyqa_from_action),
}


def has_prior(algo: str) -> bool:
    return algo in PRIORS


def computed_specific(
    target_algo: str, state: CanonicalState, ctx: NormContext
) -> dict[str, Tensor]:
    """The prior's specific fields for ``target_algo`` from a batched state."""
    return PRIORS[target_algo].build(state, ctx, None, None, None)


def prior_native(
    source_algo: str,
    native: dict,
    target_algo: str,
    n_target: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Native warm-start dict for ``target_algo`` computed from a source
    optimizer's native state, resized to the target's population like a
    translated action."""
    batch = collate([to_canonical(source_algo, native)])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    state = CanonicalState(
        target_algo,
        batch.positions,
        batch.values,
        batch.best_x,
        batch.best_y,
        computed_specific(target_algo, batch, ctx),
    ).index(0)
    if n_target is not None and n_target != state.n:
        state = resample_population(state, n_target, rng or np.random.default_rng(0))
    return from_canonical(state)
