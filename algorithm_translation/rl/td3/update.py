"""The TD3 update: ``gradient_steps`` critic updates on shape-homogeneous
minibatches, with the delayed (every ``policy_freq``) actor + Polyak target
updates."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from algorithm_translation.rl.td3.buffer import TD3Transition
from algorithm_translation.rl.td3.config import TD3Config
from algorithm_translation.rl.td3.networks import (
    TD3Actor,
    TwinQCritic,
    _action_state,
    _ctx,
    _polyak_update,
)
from algorithm_translation.state.canonical import collate


def _group_key(tr: TD3Transition):
    return (tr.source_algo, tr.src_state.d, tr.src_state.n)


def update(
    actor: TD3Actor,
    target_actor: TD3Actor,
    critic: TwinQCritic,
    target_critic: TwinQCritic,
    actor_opt,
    critic_opt,
    transitions: list[TD3Transition],
    cfg: TD3Config,
    rng: np.random.Generator,
    grad_step: int,
) -> tuple[dict, int]:
    """``gradient_steps`` critic updates (every one), each on a shape-
    homogeneous minibatch sampled (with replacement) from one random
    ``(source_algo, D, N)`` group of the replay buffer — same grouping trick
    PPO uses, since the encoders need uniform shapes within a batch, and
    since the target algorithm is a deterministic function of the source
    algorithm here too (env alternates strictly between ``algo_a``/``algo_b``,
    each with its own fixed configured population size), every transition's
    ``next_state`` in a group shares the same algo/D/N as well. The actor and
    both target networks are updated only every ``policy_freq`` critic steps
    (TD3's delayed policy update)."""
    device = torch.device(cfg.device)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, tr in enumerate(transitions):
        groups[_group_key(tr)].append(i)
    group_keys = list(groups.keys())

    metrics = {"critic_loss": 0.0, "q_value": 0.0, "actor_loss": 0.0}
    n_steps = 0
    actor_updates = 0
    for _ in range(cfg.gradient_steps):
        gi = int(rng.integers(len(group_keys)))
        idxs = groups[group_keys[gi]]
        size = min(cfg.minibatch_size, len(idxs))
        pick = rng.choice(len(idxs), size=size, replace=len(idxs) < cfg.minibatch_size)
        mb = [idxs[i] for i in pick]
        source_algo = transitions[mb[0]].source_algo
        target_algo = transitions[mb[0]].next_state.algo
        # Deterministic pairing: s' is encoded as target_algo, and a' decodes
        # back onto source_algo.
        next_source_algo, next_target_algo = target_algo, source_algo

        batch = collate([transitions[i].src_state.to(device) for i in mb])
        ctx = _ctx(batch)
        contexts = torch.tensor(
            np.stack([transitions[i].context for i in mb]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.stack([transitions[i].action for i in mb]).to(device)
        rewards = torch.tensor(
            [transitions[i].reward for i in mb], dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [transitions[i].done for i in mb], dtype=torch.float32, device=device
        )

        next_batch = collate([transitions[i].next_state.to(device) for i in mb])
        next_ctx = _ctx(next_batch)
        next_contexts = torch.tensor(
            np.stack([transitions[i].next_context for i in mb]),
            dtype=torch.float32,
            device=device,
        )

        # ---- critic update ----
        with torch.no_grad():
            a2_mean = target_actor(next_target_algo, next_batch, next_ctx)
            noise = (torch.randn_like(a2_mean) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            a2 = a2_mean + noise
            act_state2 = _action_state(actor, next_target_algo, a2, next_batch, next_ctx)
            q1_t, q2_t = target_critic(
                next_source_algo, next_target_algo, next_batch, act_state2, next_ctx,
                next_contexts,
            )
            q_next = torch.min(q1_t, q2_t)
            y = rewards + cfg.gamma * (1.0 - dones) * q_next

        act_state = _action_state(actor, target_algo, actions, batch, ctx)
        q1, q2 = critic(source_algo, target_algo, batch, act_state, ctx, contexts)
        critic_loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        critic_opt.step()

        metrics["critic_loss"] += float(critic_loss.detach())
        metrics["q_value"] += float(q1.mean().detach())
        n_steps += 1
        grad_step += 1

        # ---- delayed actor + target updates ----
        if grad_step % cfg.policy_freq == 0:
            a_new = actor(target_algo, batch, ctx)
            act_state_new = _action_state(actor, target_algo, a_new, batch, ctx)
            q1_new = critic.q1(source_algo, target_algo, batch, act_state_new, ctx, contexts)
            actor_loss = -q1_new.mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            _polyak_update(target_actor, actor, cfg.tau)
            _polyak_update(target_critic, critic, cfg.tau)

            metrics["actor_loss"] += float(actor_loss.detach())
            actor_updates += 1

    n_steps = max(n_steps, 1)
    actor_updates = max(actor_updates, 1)
    out = {
        "critic_loss": metrics["critic_loss"] / n_steps,
        "q_value": metrics["q_value"] / n_steps,
        "actor_loss": metrics["actor_loss"] / actor_updates,
    }
    return out, grad_step
