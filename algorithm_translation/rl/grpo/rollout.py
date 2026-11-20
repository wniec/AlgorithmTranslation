"""Group collection for GRPO: for each switch state, sample a group of
candidate translations and score each from that same state
(``env.evaluate_action``), single-process or over the worker pool
(``rl.workers``)."""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch

from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.rl.grpo.config import Group, GRPOConfig
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.rl.workers import (
    WORKER,
    load_worker_policy,
    make_executor,
    policy_weights,
)

__all__ = ["collect_groups", "collect_groups_parallel", "make_executor"]


def collect_groups(
    ac: ActorCritic, envs, cfg: GRPOConfig, ep_rng, n_groups: int, device="cpu"
) -> list[Group]:
    """Collect ``n_groups`` groups. Episodes whose source has no full state at
    the switch are skipped, as in PPO."""
    groups: list[Group] = []
    with torch.no_grad():
        while len(groups) < n_groups:
            env: TranslationEnv = envs[ep_rng.integers(len(envs))]
            obs, _ = env.reset()
            if not env.source_has_full_state():
                continue
            steps = ac.act_group(obs, cfg.group_size, device=device)
            rewards: list[float] = []
            parts: dict[str, list[float]] = defaultdict(list)
            for st in steps:
                reward, info = env.evaluate_action(st.native)
                rewards.append(float(reward))
                for mode, value in info.get("reward_parts", {}).items():
                    parts[mode].append(float(value))
            groups.append(
                Group(
                    steps=steps,
                    rewards=rewards,
                    switch=f"{obs['source_algo']}->{obs['target_algo']}",
                    log_direction=env.logs_per_direction,
                    parts=dict(parts),
                )
            )
    return groups


def _worker_collect(payload):
    """Collect ``n_groups`` groups with the given weights (in a worker)."""
    state_dict, n_groups = payload
    ac = load_worker_policy(state_dict)
    groups = collect_groups(ac, WORKER["envs"], WORKER["cfg"], WORKER["ep_rng"], n_groups)
    # Plain pickle copies tensor data inline (see ppo.rollout._worker_collect).
    return pickle.dumps(groups, protocol=pickle.HIGHEST_PROTOCOL)


def collect_groups_parallel(
    executor: ProcessPoolExecutor, ac: ActorCritic, cfg: GRPOConfig
) -> list[Group]:
    """Split ``cfg.groups_per_update`` groups over the worker pool."""
    per = math.ceil(cfg.groups_per_update / cfg.n_workers)
    payload = (policy_weights(ac), per)
    groups: list[Group] = []
    for b in executor.map(_worker_collect, [payload] * cfg.n_workers):
        groups.extend(pickle.loads(b))
    return groups
