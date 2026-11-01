"""Custom PPO loop for the state translator.

Standard clipped-surrogate PPO with GAE, specialized in two ways:

* transitions carry **structured, variable-shape** observations and actions
  (the optimizer states), so the update groups them by ``(target_algo, D, N)``
  and each minibatch is drawn from one shape-homogeneous group;
* the actor objective adds a differentiable **cycle-consistency penalty**
  (``lambda_cycle · A->B->A drift``), realizing the "improvement + cycle penalty"
  objective more sample-efficiently than routing a differentiable quantity
  through the scalar reward. The drift is scored via ``cycle_mode`` — "field"
  (decoded fields, ``cat.losses.field_distance``, default) or "latent"
  (re-encoded latents, ``cat.losses.latent_cycle_loss``) — and skipped entirely
  when ``lambda_cycle == 0``.

Split across ``config`` (``PPOConfig`` + data structures), ``rollout`` (episode
collection + the parallel worker pool), ``update`` (the clipped-surrogate step),
and ``loop`` (``train_ppo``).
"""

from __future__ import annotations

from cat.rl.ppo.config import PPOConfig, PPOLog, RolloutBuffer, Transition
from cat.rl.ppo.loop import train_ppo
from cat.rl.ppo.rollout import (
    collect_rollout,
    collect_rollout_parallel,
    make_executor,
)
from cat.rl.ppo.update import explained_variance, update

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
