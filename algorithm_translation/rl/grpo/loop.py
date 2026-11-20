"""``train_grpo``: collect groups, compute group-relative advantages, update."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from algorithm_translation.rl.grpo.config import (
    Group,
    GRPOConfig,
    GRPOLog,
    Sample,
    group_advantages,
)
from algorithm_translation.rl.grpo.rollout import (
    collect_groups,
    collect_groups_parallel,
    make_executor,
)
from algorithm_translation.rl.grpo.update import update
from algorithm_translation.rl.policy import ActorCritic


def groups_to_samples(groups: list[Group], scale_by_std: bool) -> list[Sample]:
    samples: list[Sample] = []
    for g in groups:
        adv = group_advantages(g.rewards, scale_by_std)
        samples.extend(
            Sample(step=st, reward=r, adv=float(a))
            for st, r, a in zip(g.steps, g.rewards, adv, strict=True)
        )
    return samples


def rollout_metrics(groups: list[Group]) -> dict:
    """Reward statistics of the freshly collected groups."""
    rewards = [r for g in groups for r in g.rewards]
    stds = np.array([np.std(g.rewards) for g in groups])
    m = {
        "mean_return": float(np.mean(rewards)) if rewards else 0.0,
        # How much the action matters within a state: the signal GRPO learns from.
        "group_reward_std": float(stds.mean()) if len(stds) else 0.0,
        "zero_adv_group_frac": float((stds <= 1e-8).mean()) if len(stds) else 0.0,
        "n_groups": len(groups),
        "n_samples": len(rewards),
    }
    by_key: dict[str, list[float]] = defaultdict(list)
    for g in groups:
        if g.log_direction:
            by_key[g.switch].extend(g.rewards)
        for mode, vals in g.parts.items():
            by_key[mode].extend(vals)
    for k, v in by_key.items():
        m[f"reward/{k}"] = float(np.mean(v))
    return m


def train_grpo(
    ac: ActorCritic, envs, cfg: GRPOConfig, log_fn=None, on_update=None
) -> GRPOLog:
    """Train the actor with GRPO. The critic inside ``ActorCritic`` is unused
    and left untrained. ``on_update(u, log)`` runs after each update (e.g. the
    autosave/eval ``Checkpointer``)."""
    ac.to(cfg.device)
    std_params = list(ac.log_std.values())
    std_ids = {id(p) for p in std_params}
    other = [p for p in ac.actor_parameters() if id(p) not in std_ids]
    opt = torch.optim.Adam(
        [
            {"params": other},
            {"params": std_params, "lr": cfg.lr * cfg.std_lr_mult},
        ],
        lr=cfg.lr,
    )
    ep_rng = np.random.default_rng(cfg.seed)
    log = GRPOLog()
    executor = make_executor(ac, envs, cfg) if cfg.n_workers > 1 else None
    try:
        pbar = tqdm(range(cfg.updates), desc="training", unit="upd", dynamic_ncols=True)
        for u in pbar:
            if executor is not None:
                groups = collect_groups_parallel(executor, ac, cfg)
            else:
                groups = collect_groups(
                    ac, envs, cfg, ep_rng, cfg.groups_per_update, cfg.device
                )
            samples = groups_to_samples(groups, cfg.scale_adv_by_std)
            m = update(ac, opt, samples, cfg, update_idx=u)
            m.update(rollout_metrics(groups))
            m["update"] = u
            log.history.append(m)
            if log_fn is not None:
                log_fn(m)
            pbar.set_postfix(
                ret=f"{m['mean_return']:.4f}",
                gstd=f"{m['group_reward_std']:.4f}",
                std=f"{m['policy_std']:.4f}",
                refresh=False,
            )
            if cfg.log_every and u % cfg.log_every == 0:
                tqdm.write(
                    f"update {u:3d} | return {m['mean_return']:.4f} "
                    f"| group std {m['group_reward_std']:.4f} "
                    f"| pi {m['policy_loss']:.4f} | kl {m['approx_kl']:.4f} "
                    f"| clip {m['clip_frac']:.3f} | std {m['policy_std']:.4f}"
                )
            if on_update is not None:
                on_update(u, log)
        pbar.close()
    finally:
        if executor is not None:
            executor.shutdown()
    return log
