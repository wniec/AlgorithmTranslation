"""The PPO training loop: build the optimizer/normalizers/buffer (and optional
worker pool), then collect → update once per update step, logging as it goes."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from cat.rl.normalize import RunningMeanStd
from cat.rl.policy import ActorCritic
from cat.rl.ppo.config import PPOConfig, PPOLog, RolloutBuffer
from cat.rl.ppo.rollout import (
    collect_rollout,
    collect_rollout_parallel,
    make_executor,
)
from cat.rl.ppo.update import update


def train_ppo(ac: ActorCritic, envs, cfg: PPOConfig, log_fn=None) -> PPOLog:
    """Run PPO. ``log_fn(metrics)`` is called once per update (for W&B; optional).

    The reward/observation normalizers (running statistics) are attached to the
    returned log as ``log.ret_rms`` / ``log.obs_rms`` so they can be checkpointed.
    """
    ac.to(cfg.device)
    other_params = [
        p for n, p in ac.named_parameters() if not n.startswith("log_std.")
    ]
    opt = torch.optim.Adam(
        [
            {"params": other_params},
            {"params": list(ac.log_std.values()), "lr": cfg.lr * cfg.std_lr_mult},
        ],
        lr=cfg.lr,
    )
    ep_rng = np.random.default_rng(cfg.seed)
    log = PPOLog()
    ret_rms = RunningMeanStd(()) if cfg.norm_reward else None
    obs_rms = RunningMeanStd((ac.context_dim,)) if cfg.norm_obs else None
    log.ret_rms, log.obs_rms = ret_rms, obs_rms

    # Persistent sliding-window buffer: each record lives for several updates.
    # Clamp capacity up so at least one full fresh rollout always fits.
    capacity = max(cfg.buffer_capacity or cfg.rollout_steps, cfg.rollout_steps)
    buffer = RolloutBuffer(capacity)

    # Optional parallel rollout: a persistent pool built once and reused across
    # updates, so each worker's envs (and their problem walk) stay alive.
    executor = make_executor(ac, envs, cfg) if cfg.n_workers > 1 else None
    try:
        _train_loop(ac, envs, cfg, opt, ep_rng, ret_rms, obs_rms, buffer, log,
                    log_fn, executor)
    finally:
        if executor is not None:
            executor.shutdown()
    return log


def _train_loop(ac, envs, cfg, opt, ep_rng, ret_rms, obs_rms, buffer, log,
                log_fn, executor):
    pbar = tqdm(range(cfg.updates), desc="training", unit="upd", dynamic_ncols=True)
    for u in pbar:
        if executor is not None:
            fresh, ep_returns, switch_rewards = collect_rollout_parallel(
                executor, ac, cfg, ret_rms, obs_rms
            )
        else:
            fresh, ep_returns, switch_rewards = collect_rollout(
                ac, envs, cfg, ep_rng, ret_rms, obs_rms
            )
        buffer.extend(fresh)
        m = update(ac, opt, buffer.as_list(), cfg)
        m["update"] = u
        # Report the raw return of the freshly collected episodes only, so the
        # learning curve reflects the current policy (not stale window records).
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
        pbar.set_postfix(
            ret=f"{m['mean_return']:.4f}",
            ev=f"{m['explained_variance']:.3f}",
            std=f"{m['policy_std']:.4f}",
            refresh=False,
        )
        if cfg.log_every and u % cfg.log_every == 0:
            # tqdm.write keeps the per-update log line from corrupting the bar.
            tqdm.write(
                f"update {u:3d} | return {m['mean_return']:.4f} "
                f"| pi {m['policy_loss']:.4f} | vf {m['value_loss']:.4f} "
                f"| ev {m['explained_variance']:.3f} "
                f"| std {m['policy_std']:.4f} | cycle {m['cycle']:.4f}"
            )
    pbar.close()
    return log
