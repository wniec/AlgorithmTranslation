"""GRPO configuration, the data structures the loop carries around, and the
group-relative advantage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algorithm_translation.rl.policy import Step


@dataclass
class GRPOConfig:
    updates: int = 100
    # Each update samples this many switch states (episodes), and for each one
    # a group of `group_size` candidate translations scored from that same
    # state under the same post-switch seeds.
    groups_per_update: int = 128
    group_size: int = 8
    epochs: int = 4
    minibatch_size: int = 256
    clip: float = 0.2
    ent_coef: float = 0.0
    lr: float = 1e-4
    # log_std (ActorCritic.log_std) is a handful of per-segment base scalars;
    # like PPO they get their own Adam param group at lr * std_lr_mult.
    std_lr_mult: float = 30.0
    max_grad_norm: float = 1.0
    # Divide each group's centered rewards by the group's std (standard GRPO).
    # False keeps them unscaled (mean-centered only, as in Dr. GRPO), so groups
    # where the action barely matters also contribute barely any gradient.
    scale_adv_by_std: bool = True
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1
    # Rollout workers (as in PPO): each collects whole groups with a frozen
    # copy of the current policy. 1 keeps the single-process path.
    n_workers: int = 1


@dataclass
class Group:
    """One switch state and the candidate actions scored from it."""

    steps: list[Step]
    rewards: list[float]
    switch: str  # "<source>-><target>"
    log_direction: bool  # False for a pure 'absolute' reward (not comparable)
    parts: dict[str, list[float]] = field(default_factory=dict)  # per-mode rewards


@dataclass
class Sample:
    step: Step
    reward: float
    adv: float


@dataclass
class GRPOLog:
    history: list[dict] = field(default_factory=list)


def group_advantages(
    rewards, scale_by_std: bool = True, eps: float = 1e-8
) -> np.ndarray:
    """Group-relative advantage: each reward minus the group mean, optionally
    divided by the group std. A group whose rewards are all equal (the action
    made no difference) gets all-zero advantages."""
    r = np.asarray(rewards, dtype=np.float64)
    adv = r - r.mean()
    if scale_by_std:
        sd = r.std()
        adv = adv / (sd + eps) if sd > eps else np.zeros_like(adv)
    return adv
