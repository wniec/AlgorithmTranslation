"""The TD3 training loop: build actor/critic + their Polyak targets, optimizers,
normalizers and replay buffer, then collect → update once per update step."""

from __future__ import annotations

import copy

import numpy as np
import torch

from cat.rl.normalize import RunningMeanStd
from cat.rl.td3.buffer import ReplayBuffer
from cat.rl.td3.config import CONTEXT_DIM, TD3Config, TD3Log
from cat.rl.td3.networks import TD3Actor, TwinQCritic
from cat.rl.td3.rollout import collect_rollout
from cat.rl.td3.update import update


def train_td3(
    actor: TD3Actor,
    critic: TwinQCritic,
    envs,
    cfg: TD3Config,
    log_fn=None,
    on_update=None,
) -> TD3Log:
    """Run TD3. ``log_fn(metrics)`` is called once per update (for W&B;
    optional). Mirrors ``cat.rl.ppo.train_ppo``'s outer loop/logging shape so
    the two algorithms' runs are directly comparable. ``on_update(u, log)`` runs after
    update ``u`` is applied and logged (checkpointing)."""
    actor.to(cfg.device)
    critic.to(cfg.device)
    target_actor = copy.deepcopy(actor).to(cfg.device)
    target_critic = copy.deepcopy(critic).to(cfg.device)
    for net in (target_actor, target_critic):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    ep_rng = np.random.default_rng(cfg.seed)
    train_rng = np.random.default_rng(cfg.seed + 1)
    log = TD3Log()
    ret_rms = RunningMeanStd(()) if cfg.norm_reward else None
    obs_rms = RunningMeanStd((CONTEXT_DIM,)) if cfg.norm_obs else None
    log.ret_rms, log.obs_rms = ret_rms, obs_rms

    buffer = ReplayBuffer(cfg.buffer_capacity)
    grad_step = 0

    for u in range(cfg.updates):
        fresh, ep_returns, switch_rewards = collect_rollout(
            actor, envs, cfg, ep_rng, ret_rms, obs_rms
        )
        buffer.extend(fresh)
        m, grad_step = update(
            actor,
            target_actor,
            critic,
            target_critic,
            actor_opt,
            critic_opt,
            buffer.as_list(),
            cfg,
            train_rng,
            grad_step,
        )
        m["update"] = u
        m["mean_return"] = float(np.mean(ep_returns)) if ep_returns else 0.0
        # Per-switch-direction mean reward (comparable baselines; empty for a
        # pure 'absolute' reward), plus each summed component's mean by mode name.
        for switch, rs in switch_rewards.items():
            m[f"reward/{switch}"] = float(np.mean(rs))
        m["n_transitions"] = len(buffer)
        m["n_fresh"] = len(fresh)
        log.history.append(m)
        if log_fn is not None:
            log_fn(m)
        if cfg.log_every and u % cfg.log_every == 0:
            print(
                f"update {u:3d} | return {m['mean_return']:.4f} "
                f"| critic {m['critic_loss']:.4f} | actor {m['actor_loss']:.4f} "
                f"| q {m['q_value']:.4f}"
            )
        if on_update is not None:
            on_update(u, log)
    return log
