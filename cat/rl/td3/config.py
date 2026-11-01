"""TD3 configuration, log container, and the context-dimension constants."""

from __future__ import annotations

from dataclasses import dataclass, field

from cat.rl.normalize import RunningMeanStd

PROGRESS_DIM = 2  # kept in sync with cat.rl.policy.PROGRESS_DIM (no ELA side-input)
CONTEXT_DIM = PROGRESS_DIM


@dataclass
class TD3Config:
    updates: int = 500
    rollout_steps: int = 1024  # env steps collected (noisy actor) per update
    gradient_steps: int = 64  # critic updates performed per update
    minibatch_size: int = 256
    buffer_capacity: int = 20_000
    gamma: float = 0.8
    actor_lr: float = 8e-5
    critic_lr: float = 3e-4
    expl_noise: float = 0.1  # std of Gaussian exploration noise (action space)
    policy_noise: float = 0.2  # std of target policy smoothing noise
    noise_clip: float = 0.5  # clip range for target policy smoothing noise
    policy_freq: int = 2  # delayed actor/target update frequency, in gradient steps
    tau: float = 0.005  # Polyak averaging coefficient for target networks
    max_grad_norm: float = 1.0
    lambda_cycle: float = 0.0  # optional cycle-consistency penalty on the actor loss
    cycle_mode: str = "field"
    norm_reward: bool = True
    norm_obs: bool = True
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1


@dataclass
class TD3Log:
    history: list[dict] = field(default_factory=list)
    ret_rms: RunningMeanStd | None = None
    obs_rms: RunningMeanStd | None = None
