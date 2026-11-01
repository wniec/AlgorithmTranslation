"""Twin Delayed DDPG (TD3), an off-policy alternative to PPO for the same
``TranslationEnv`` (``cat/rl/env.py``) and the same *continuous* action: the
target optimizer's full decoded warm-start, produced by the set-equivariant
``TranslatorPair`` (``cat/models/translator.py``).

Unlike PPO's stochastic Gaussian policy, TD3's actor is **deterministic** — it
outputs the translator's mean action, with exploration coming from Gaussian
noise added in action space during rollout, and training from a replay buffer
(off-policy). Two independent critics (``TwinQCritic``) estimate Q(s, a); the
*minimum* of the two is used for the TD-target and (via target policy smoothing)
to curb value-overestimation; the actor is updated only every ``policy_freq``
critic updates and both target networks are Polyak-averaged.

Split across ``config``, ``networks`` (actor + twin critics + the
action-as-state trick), ``buffer`` (replay memory), ``rollout``, ``update``, and
``loop`` (``train_td3``).
"""

from __future__ import annotations

from cat.rl.td3.buffer import ReplayBuffer, TD3Transition
from cat.rl.td3.config import CONTEXT_DIM, PROGRESS_DIM, TD3Config, TD3Log
from cat.rl.td3.loop import train_td3
from cat.rl.td3.networks import (
    QCritic,
    TD3Actor,
    TD3Step,
    TwinQCritic,
    _polyak_update,
)
from cat.rl.td3.rollout import collect_rollout
from cat.rl.td3.update import update

__all__ = [
    "CONTEXT_DIM",
    "PROGRESS_DIM",
    "QCritic",
    "ReplayBuffer",
    "TD3Actor",
    "TD3Config",
    "TD3Log",
    "TD3Step",
    "TD3Transition",
    "TwinQCritic",
    "_polyak_update",
    "collect_rollout",
    "train_td3",
    "update",
]
