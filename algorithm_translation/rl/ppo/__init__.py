"""Custom PPO loop for the state translator.

Standard clipped-surrogate PPO with GAE, specialized for **structured,
variable-shape** observations and actions (the optimizer states): the update
groups transitions by ``(target_algo, D, N)`` and each minibatch is drawn from
one shape-homogeneous group.

Split across ``config`` (``PPOConfig`` + data structures), ``rollout`` (episode
collection + the parallel worker pool), ``update`` (the clipped-surrogate step),
and ``loop`` (``train_ppo``).
"""

from __future__ import annotations

from algorithm_translation.rl.ppo.config import PPOConfig, PPOLog, RolloutBuffer, Transition
from algorithm_translation.rl.ppo.loop import train_ppo
from algorithm_translation.rl.ppo.rollout import (
    collect_rollout,
    collect_rollout_parallel,
    make_executor,
)
from algorithm_translation.rl.ppo.update import explained_variance, update

__all__ = [
    "PPOConfig",
    "PPOLog",
    "RolloutBuffer",
    "Transition",
    "collect_rollout",
    "collect_rollout_parallel",
    "explained_variance",
    "make_executor",
    "train_ppo",
    "update",
]
