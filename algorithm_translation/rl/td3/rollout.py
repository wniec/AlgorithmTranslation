"""TD3 rollout: run whole episodes with the noisy deterministic actor, storing
each transition (with a well-formed ``next_state``) for the replay buffer."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.rl.normalize import RunningMeanStd, normalize_rewards
from algorithm_translation.rl.td3.buffer import TD3Transition
from algorithm_translation.rl.td3.config import TD3Config
from algorithm_translation.rl.td3.networks import TD3Actor
from algorithm_translation.state.canonical import to_canonical


def collect_rollout(
    actor: TD3Actor,
    envs,
    cfg: TD3Config,
    ep_rng: np.random.Generator,
    ret_rms: RunningMeanStd | None = None,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[list[TD3Transition], list[float], dict[str, list[float]]]:
    """Run whole episodes (actor + Gaussian exploration noise) until at least
    rollout_steps transitions are gathered. Mirrors ``algorithm_translation.rl.ppo.collect_rollout``
    (same env, same reward/obs normalization). Unlike PPO's GAE (which only needs the
    reward/value sequence), TD3 needs a well-formed ``next_state`` right away to
    bootstrap from, so a transition whose resulting state is degenerate is
    dropped and the rest of the episode abandoned.

    The third return value maps each switch type (``"<source>-><target>"``) to
    its raw per-step rewards, for every comparable-baseline reward mode, plus
    each summed component by mode name (see ``algorithm_translation.rl.ppo.collect_rollout``)."""
    transitions: list[TD3Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    while len(transitions) < cfg.rollout_steps:
        env: TranslationEnv = envs[ep_rng.integers(len(envs))]
        obs, _ = env.reset()
        ep: list[TD3Transition] = []
        rewards: list[float] = []
        done = False
        while not done:
            if not env.source_has_full_state():
                break  # degenerate source; abandon this episode
            raw_ctx = np.asarray(obs["context"], dtype=np.float32)
            if obs_rms is not None:
                obs_rms.update(raw_ctx[None])
                ctx = obs_rms.normalize(raw_ctx)
            else:
                ctx = raw_ctx

            switch = f"{obs['source_algo']}->{obs['target_algo']}"
            step = actor.act(
                obs, expl_noise=cfg.expl_noise, rng=ep_rng, device=cfg.device, context=ctx
            )
            next_obs, reward, term, trunc, info = env.step(step.native)
            done = term or trunc

            if not env.source_has_full_state():
                break  # resulting state degenerate; drop this transition too

            next_raw_ctx = np.asarray(next_obs["context"], dtype=np.float32)
            if obs_rms is not None:
                obs_rms.update(next_raw_ctx[None])
                next_ctx = obs_rms.normalize(next_raw_ctx)
            else:
                next_ctx = next_raw_ctx
            next_state = to_canonical(next_obs["source_algo"], next_obs["native"]).to(
                "cpu"
            )

            ep.append(
                TD3Transition(
                    src_state=step.src_state,
                    source_algo=obs["source_algo"],
                    action=step.action,
                    reward=reward,
                    context=ctx,
                    next_state=next_state,
                    next_context=next_ctx,
                    done=done,
                )
            )
            rewards.append(reward)
            if env.logs_per_direction:
                switch_rewards[switch].append(reward)  # raw, per switch direction
            # For a summed (multi-mode) reward, also break out each additive
            # component so its mean is logged (reward/<mode>, e.g. reward/noswitch).
            for part, value in info.get("reward_parts", {}).items():
                switch_rewards[part].append(value)
            obs = next_obs
        if not ep:
            continue
        ep_returns.append(float(sum(rewards)))  # report the RAW return
        train_rewards = (
            normalize_rewards(rewards, cfg.gamma, ret_rms)
            if ret_rms is not None
            else rewards
        )
        for tr, r in zip(ep, train_rewards, strict=True):
            tr.reward = r
        transitions.extend(ep)
    return transitions, ep_returns, switch_rewards
