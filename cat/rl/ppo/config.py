"""PPO configuration and the data structures the loop carries around."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cat.rl.normalize import RunningMeanStd
from cat.rl.policy import Step


@dataclass
class PPOConfig:
    updates: int = 50
    rollout_steps: int = 1024
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.25
    # Value-function clip window, decoupled from the policy ratio `clip`. None
    # (the default) disables value clipping — the standard choice, and it frees
    # the critic to track returns that span many units (the old behavior reused
    # `clip`=0.2, which throttled it, worst under --no-norm-reward). Set a float
    # to re-enable (e.g. 0.2 reproduces the old throttle; a larger value is a
    # looser, decoupled window).
    vf_clip: float | None = None
    lambda_cycle: float = 0.05
    cycle_mode: str = "field"  # "field" (decoded fields) or "latent" (re-encoded latents)
    lr: float = 3e-4
    # log_std is a single global scalar (see ActorCritic.log_std): at the shared
    # network lr it can only drift by ~lr per opt.step(), which is too slow to
    # track a moving optimum over the handful of steps in one update. Give it
    # its own Adam param group at lr * std_lr_mult instead.
    std_lr_mult: float = 20.0
    max_grad_norm: float = 1.0
    # Sliding-window rollout buffer: each update collects ~rollout_steps fresh
    # transitions, appends them to a deque of this capacity, and trains on the
    # whole window — so each record is reused across roughly
    # ``buffer_capacity / rollout_steps`` updates. ``None`` keeps only the latest
    # rollout (textbook on-policy PPO). Clamped up to rollout_steps so a full
    # fresh rollout always fits.
    buffer_capacity: int | None = None
    norm_reward: bool = True
    norm_obs: bool = True
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1
    # Rollout collection is the wall-clock bottleneck (the counterfactual
    # optimizer runs in env.step dominate). With n_workers > 1 the rollout is
    # split across that many worker processes, each rebuilding its own envs and
    # running whole episodes with a frozen snapshot of the current policy +
    # normalizers; the main process merges the results and runs the PPO update.
    # 1 keeps the exact single-process path. CPU rollout is assumed.
    n_workers: int = 1


@dataclass
class Transition:
    step: Step
    reward: float
    context: np.ndarray
    adv: float = 0.0
    ret: float = 0.0


@dataclass
class PPOLog:
    history: list[dict] = field(default_factory=list)
    ret_rms: RunningMeanStd | None = None
    obs_rms: RunningMeanStd | None = None


class RolloutBuffer:
    """Sliding-window store of transitions, retained across PPO updates.

    Mirrors ``DynamicAlgorithmSelection``'s ``RolloutBuffer``: rather than a
    fresh buffer per update (textbook on-policy PPO), it keeps the most recent
    ``capacity`` transitions so each record is reused over several updates. The
    clipped PPO ratio (``exp(logp - old_logp)``) tolerates the mild
    off-policyness of older records still inside the window.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def clear(self) -> None:
        self._buf.clear()

    def extend(self, transitions: list[Transition]) -> None:
        # deque(maxlen) evicts the oldest transitions once capacity is exceeded.
        self._buf.extend(transitions)

    def as_list(self) -> list[Transition]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
