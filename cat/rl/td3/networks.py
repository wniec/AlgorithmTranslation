"""TD3 networks: the deterministic actor (the shared ``TranslatorPair``) and the
twin Q-critics, plus the action-as-state decode trick that lets a fixed-size
critic score a variable-dimension action.

The Q-network's structural challenge: the action is not a fixed-size vector
across the board — its dimension depends on ``(target_algo, D, N)`` (see
``StateDecoder.action_dim`` in ``cat/models/heads.py``), so a plain
``concat(state, action) -> MLP`` critic can't have a fixed input size. The fix
reuses the trick that makes the *state* encoder shape-agnostic: decode the
action back into its structured per-algorithm fields (``build_state``, entirely
parameter-free — pure reshape + denormalize + the static
``CovFactorHead.assemble`` or the computed CMA-ES prior of the stored source
state — so replaying an old, off-policy action through the
*current* decoder always reconstructs bit-identical fields, no staleness
concern) and encode *that* with the target algorithm's own ``StateEncoder``,
exactly like encoding a real state. The critic then combines the source state's
embedding with this "action-as-state" embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from cat.models.encoder import StateEncoder
from cat.models.layers import MLP
from cat.models.norm import NormContext
from cat.models.translator import TranslatorPair
from cat.rl.td3.config import CONTEXT_DIM
from cat.state.canonical import (
    CanonicalState,
    collate,
    from_canonical,
    resample_population,
    to_canonical,
)


def _ctx(state: CanonicalState) -> NormContext:
    return NormContext.from_shared(state.positions, state.values)


def _action_state(
    actor: TD3Actor, target_algo: str, action: Tensor, batch: CanonicalState, ctx
) -> CanonicalState:
    """Decode a flat action vector into a full (batched) ``CanonicalState`` on
    ``target_algo``, sharing ``batch``'s population (see module docstring: this
    reconstruction is parameter-free, so it's safe regardless of which actor
    instance/weights are passed — an ``actor`` argument is only needed because
    ``build_state`` lives on an ``nn.Module`` decoder head)."""
    specific = actor.translator.decoders[target_algo].build_state(action, ctx, batch)
    return CanonicalState(
        algo=target_algo,
        positions=batch.positions,
        values=batch.values,
        best_x=batch.best_x,
        best_y=batch.best_y,
        specific=specific,
    )


@dataclass
class TD3Step:
    """Everything produced for one acted transition (used by the replay buffer)."""

    src_state: CanonicalState  # un-batched, on cpu
    target_algo: str
    action: Tensor  # (action_dim,) raw mean(+noise) action, on the SOURCE's own N
    native: dict  # target warm-start dict (resized to target's own N) for env.step


class TD3Actor(nn.Module):
    def __init__(
        self,
        algo_a: str,
        algo_b: str,
        hidden: int = 64,
        n_layers: int = 2,
        cov_rank: int = 4,
        resample_seed: int | None = None,
    ):
        super().__init__()
        # Drives the population-resizing jitter in _assemble_native, exactly
        # like PPO's ActorCritic (cat/rl/policy.py) — reproducible given a seed.
        self._resample_rng = np.random.default_rng(resample_seed)
        self.translator = TranslatorPair(
            algo_a, algo_b, hidden=hidden, n_layers=n_layers, cov_rank=cov_rank
        )
        self.algo_a = algo_a
        self.algo_b = algo_b

    def forward(self, target_algo: str, batch: CanonicalState, ctx) -> Tensor:
        z = self.translator.encode(batch, ctx)
        return self.translator.decoders[target_algo].action_mean(z, batch.positions, ctx)

    def obs_to_state(self, obs: dict, device="cpu") -> CanonicalState:
        return to_canonical(obs["source_algo"], obs["native"], device=device)

    @torch.no_grad()
    def act(
        self,
        obs: dict,
        expl_noise: float,
        rng: np.random.Generator,
        device: str = "cpu",
        context=None,
    ) -> TD3Step:
        src = self.obs_to_state(obs, device)
        target = obs["target_algo"]
        batch = collate([src])
        ctx = _ctx(batch)

        mean = self.forward(target, batch, ctx)
        if expl_noise > 0:
            noise = torch.as_tensor(
                rng.normal(scale=expl_noise, size=tuple(mean.shape)),
                dtype=mean.dtype,
                device=device,
            )
            action = mean + noise
        else:
            action = mean

        native = self._assemble_native(action, target, batch, ctx, obs.get("target_n"))
        return TD3Step(
            src_state=src.to("cpu"),
            target_algo=target,
            action=action.squeeze(0).cpu(),
            native=native,
        )

    def _assemble_native(
        self, action: Tensor, target: str, batch: CanonicalState, ctx, n_target=None
    ) -> dict:
        """Mirrors ``ActorCritic._assemble_native`` (cat/rl/policy.py): the
        action is always decoded onto the source's own population (keeping the
        replay buffer's action_dim a pure function of (target_algo, D, N) for
        homogeneous batching); only the assembled warm-start payload handed to
        env.step is resized to the target's own N."""
        tgt_state = _action_state(self, target, action, batch, ctx).index(0)
        if n_target is not None and n_target != tgt_state.n:
            tgt_state = resample_population(tgt_state, n_target, self._resample_rng)
        return from_canonical(tgt_state)

    def cycle_drift(self, batch: CanonicalState, ctx, cycle_mode: str = "field"):
        """Mean A->B->A drift for a batch; see ``TranslatorPair.cycle_drift``
        (shared with PPO's ``ActorCritic``)."""
        return self.translator.cycle_drift(batch, ctx, cycle_mode)


class QCritic(nn.Module):
    """One of the two twins. Encodes the source state AND the proposed
    action-as-state (see module docstring), each with their own dedicated
    per-algorithm ``StateEncoder`` — kept separate from the actor's own
    encoder for the same reason PPO's critic does (cat/rl/policy.py): sharing
    would make the critic chase a representation the actor/cycle losses
    constantly reshape."""

    def __init__(self, algo_a: str, algo_b: str, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.src_encoders = nn.ModuleDict(
            {a: StateEncoder(a, hidden, n_layers) for a in (algo_a, algo_b)}
        )
        self.act_encoders = nn.ModuleDict(
            {a: StateEncoder(a, hidden, n_layers) for a in (algo_a, algo_b)}
        )
        self.head = MLP([2 * hidden + CONTEXT_DIM, hidden, 1])

    def forward(
        self,
        source_algo: str,
        target_algo: str,
        src_batch: CanonicalState,
        act_batch: CanonicalState,
        ctx,
        context: Tensor,
    ) -> Tensor:
        zs = self.src_encoders[source_algo](src_batch, ctx)
        za = self.act_encoders[target_algo](act_batch, ctx)
        feats = torch.cat([zs.global_vec, za.global_vec, context], dim=-1)
        return self.head(feats).squeeze(-1)


class TwinQCritic(nn.Module):
    def __init__(self, algo_a: str, algo_b: str, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.q1 = QCritic(algo_a, algo_b, hidden, n_layers)
        self.q2 = QCritic(algo_a, algo_b, hidden, n_layers)

    def forward(
        self, source_algo, target_algo, src_batch, act_batch, ctx, context
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1(source_algo, target_algo, src_batch, act_batch, ctx, context),
            self.q2(source_algo, target_algo, src_batch, act_batch, ctx, context),
        )


def _polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(p, alpha=tau)
