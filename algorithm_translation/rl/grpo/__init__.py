"""GRPO (Group Relative Policy Optimization) for the state translator.

Each episode has a single switch, so the task is a contextual bandit. Instead of
a learned critic baseline, GRPO samples a *group* of candidate translations for
every switch state, scores each from that same state under the same post-switch
seeds (``TranslationEnv.evaluate_action``), and uses each candidate's reward
relative to its group as the advantage. Between-problem and seed variance cancel
within a group, and the action-independent reward baselines (``noswitch`` /
``relative`` / ``coldtarget``) cancel exactly, so they only matter for logging.
The update is PPO's clipped surrogate over those advantages, with no critic.

Split across ``config`` (``GRPOConfig``, data structures, the advantage),
``rollout`` (group collection + worker pool), ``update`` and ``loop``
(``train_grpo``).
"""

from __future__ import annotations

from algorithm_translation.rl.grpo.config import (
    Group,
    GRPOConfig,
    GRPOLog,
    Sample,
    group_advantages,
)
from algorithm_translation.rl.grpo.loop import groups_to_samples, rollout_metrics, train_grpo
from algorithm_translation.rl.grpo.rollout import (
    collect_groups,
    collect_groups_parallel,
    make_executor,
)
from algorithm_translation.rl.grpo.update import update

__all__ = [
    "GRPOConfig",
    "GRPOLog",
    "Group",
    "Sample",
    "collect_groups",
    "collect_groups_parallel",
    "group_advantages",
    "groups_to_samples",
    "make_executor",
    "rollout_metrics",
    "train_grpo",
    "update",
]
