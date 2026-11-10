"""Rollout collection for PPO: whole-episode gathering with GAE, plus the
parallel worker pool that fans collection out over processes.

Rollout collection is the wall-clock bottleneck (the counterfactual optimizer
runs inside ``env.step`` dominate), so ``collect_rollout_parallel`` runs a
persistent pool of worker processes, each owning its own envs and a frozen
snapshot of the policy + normalizers, and the main process merges the results.
"""

from __future__ import annotations

import contextlib
import copy
import math
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as torch_mp

from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.rl.normalize import (
    RunningMeanStd,
    discounted_returns,
    normalize_rewards,
    scale_by_ret_std,
)
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.rl.ppo.config import PPOConfig, Transition


def _gae(rewards, values, gamma, lam):
    T = len(rewards)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0  # episodes terminate -> 0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns


def _collect_episodes(
    ac: ActorCritic,
    envs,
    cfg: PPOConfig,
    ep_rng,
    ret_rms: RunningMeanStd | None,
    obs_rms: RunningMeanStd | None,
    min_steps: int,
    device: str,
    *,
    frozen: bool,
):
    """Run whole episodes until at least ``min_steps`` transitions are gathered.

    ``frozen`` selects how the normalizers are used:

    * ``False`` (single-process path) — ``obs_rms``/``ret_rms`` are the live
      estimators and are updated inline as collection proceeds, exactly as the
      original loop did.
    * ``True`` (parallel worker path) — they are read-only snapshots: the
      context is normalized and the rewards scaled against the frozen stats, and
      the raw contexts + discounted returns are accumulated and returned so the
      main process can merge them into the global estimators once per update.

    Returns ``(transitions, ep_returns, switch_rewards, raw_ctxs, discounted)``.
    The last two are empty unless ``frozen``.
    """
    transitions: list[Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    raw_ctxs: list[np.ndarray] = []
    discounted: list[float] = []
    while len(transitions) < min_steps:
        env: TranslationEnv = envs[ep_rng.integers(len(envs))]
        obs, _ = env.reset()
        ep: list[Transition] = []
        rewards, values = [], []
        done = False
        while not done:
            if not env.source_has_full_state():
                break  # degenerate source; abandon this episode
            # Critic side-input: the env context (progress scalars, plus the
            # scale/ELA features when enabled).
            raw_ctx = np.asarray(obs["context"], dtype=np.float32)
            if obs_rms is not None:
                if frozen:
                    raw_ctxs.append(raw_ctx)  # merged centrally after the rollout
                else:
                    obs_rms.update(raw_ctx[None])
                ctx = obs_rms.normalize(raw_ctx)
            else:
                ctx = raw_ctx
            switch = f"{obs['source_algo']}->{obs['target_algo']}"
            step = ac.act(obs, device=device, context=ctx)
            next_obs, reward, term, trunc, info = env.step(step.native)
            ep.append(Transition(step=step, reward=reward, context=ctx))
            rewards.append(reward)
            values.append(step.value)
            if env.logs_per_direction:
                switch_rewards[switch].append(reward)  # raw, per switch direction
            # For a summed (multi-mode) reward, also break out each additive
            # component so its mean is logged (reward/<mode>, e.g. reward/noswitch).
            for part, value in info.get("reward_parts", {}).items():
                switch_rewards[part].append(value)
            done = term or trunc
            obs = next_obs
        if not ep:
            continue
        ep_returns.append(float(sum(rewards)))  # report the RAW return
        if ret_rms is None:
            train_rewards = rewards
        elif frozen:
            discounted.extend(discounted_returns(rewards, cfg.gamma))
            train_rewards = scale_by_ret_std(rewards, ret_rms)
        else:
            train_rewards = normalize_rewards(rewards, cfg.gamma, ret_rms)
        adv, ret = _gae(train_rewards, values, cfg.gamma, cfg.gae_lambda)
        for tr, a, r in zip(ep, adv, ret, strict=True):
            tr.adv, tr.ret = a, r
        transitions.extend(ep)
    return transitions, ep_returns, switch_rewards, raw_ctxs, discounted


def collect_rollout(
    ac: ActorCritic,
    envs,
    cfg: PPOConfig,
    ep_rng,
    ret_rms: RunningMeanStd | None = None,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[list[Transition], list[float], dict[str, list[float]]]:
    """Single-process rollout: run whole episodes until at least rollout_steps
    transitions are gathered, updating the normalizers inline.

    Reward normalization (``ret_rms``) and context/observation normalization
    (``obs_rms``) use running statistics that persist across updates; the
    normalized context is stored in each transition so the PPO update is
    consistent with what the policy saw at collection time.

    The third return value maps each switch type (``"<source>-><target>"``) to
    its raw per-step rewards, for every reward mode whose baseline is comparable
    across the hand-off directions (each a log-scaled improvement over that same
    direction's own baseline: the network-free hand-off for ``relative``, the source
    continuing for ``noswitch``, the cold target run for ``coldtarget``). A pure
    ``absolute`` reward is a raw, scale-varying improvement, so it is not broken
    out per direction (``env.logs_per_direction``). When the reward sums several
    modes (``mixed`` or an explicit list), the same dict additionally carries
    each additive component under its mode name (aggregated over all directions),
    so each part's mean is logged alongside the per-direction totals.
    """
    transitions, ep_returns, switch_rewards, _, _ = _collect_episodes(
        ac, envs, cfg, ep_rng, ret_rms, obs_rms, cfg.rollout_steps, cfg.device,
        frozen=False,
    )
    return transitions, ep_returns, switch_rewards


# --------------------------------------------------------------------------- #
# Parallel rollout: a persistent pool of worker processes, each owning its own
# envs and a copy of the policy, collecting a share of the rollout per update.
# --------------------------------------------------------------------------- #

_WORKER: dict = {}


def _use_file_system_sharing() -> None:
    """Have torch share tensors between processes via a shm *file* (mmap'd, with
    the FD closed right after mapping) instead of the default 'file_descriptor'
    strategy, which passes one open FD per storage over the pool socket.

    The parallel rollout returns transitions holding tensors, and the
    sliding-window ``RolloutBuffer`` retains them across updates. Under
    'file_descriptor' those FDs accumulate past the process's open-file limit,
    so the worker socket eventually delivers none of them -- the
    ``received 0 items of ancdata`` -> ``BrokenProcessPool`` failure. Under
    'file_system' a retained tensor costs shm pages, not a permanent FD.
    """
    # Suppress: strategy unavailable on this platform -> keep torch's default.
    with contextlib.suppress(RuntimeError, ValueError):
        torch_mp.set_sharing_strategy("file_system")


def _worker_init(ac: ActorCritic, specs, base_seed: int, cfg: PPOConfig, counter):
    """Runs once per worker process. Builds this worker's envs (with a distinct
    seed offset so workers don't duplicate each other's episode stream) and
    stores the policy + a persistent env-selection RNG in module globals."""
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    # One optimizer run per worker at a time: let each worker use a single
    # thread so N worker processes don't oversubscribe the cores with N * (BLAS
    # threads) — that makes more workers slower, not faster.
    torch.set_num_threads(1)
    _use_file_system_sharing()  # the worker is the sender of the buffered tensors
    ac.to("cpu")
    envs = [TranslationEnv(**spec) for spec in _seeded_specs(specs, wid)]
    _WORKER["ac"] = ac
    _WORKER["cfg"] = cfg
    _WORKER["envs"] = envs
    _WORKER["ep_rng"] = np.random.default_rng(base_seed + 10_000 * (wid + 1))


def _seeded_specs(specs, wid: int):
    """Offset each env spec's seed by the worker id so every worker walks a
    different problem/instance/boundary stream."""
    offset = 1_000_003 * (wid + 1)
    return [{**s, "seed": int(s["seed"]) + offset} for s in specs]


def _worker_collect(payload):
    """Collect ~``n_steps`` transitions with the given policy weights, using
    frozen normalizer snapshots. Runs in a worker process."""
    state_dict, obs_rms, ret_rms, n_steps = payload
    ac: ActorCritic = _WORKER["ac"]
    ac.load_state_dict(state_dict)
    cfg: PPOConfig = _WORKER["cfg"]
    with torch.no_grad():
        trs, ep_returns, switch_rewards, raw_ctxs, disc = _collect_episodes(
            ac, _WORKER["envs"], cfg, _WORKER["ep_rng"], ret_rms, obs_rms,
            n_steps, "cpu", frozen=True,
        )
    # Plain pickle copies tensor data inline. Returning the objects directly
    # would let torch's ForkingPickler hand each tensor over as its own
    # shared-memory mapping; the buffer keeps thousands of transitions alive,
    # so those mappings pile up until vm.max_map_count (65530) is hit and the
    # pool dies with "unable to mmap ... Cannot allocate memory".
    return pickle.dumps(
        (trs, ep_returns, dict(switch_rewards), raw_ctxs, disc),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def collect_rollout_parallel(
    executor: ProcessPoolExecutor,
    ac: ActorCritic,
    cfg: PPOConfig,
    ret_rms: RunningMeanStd | None,
    obs_rms: RunningMeanStd | None,
) -> tuple[list[Transition], list[float], dict[str, list[float]]]:
    """Fan the rollout out over ``cfg.n_workers`` worker processes, each using a
    frozen snapshot of the current policy + normalizers, then merge the results
    and fold the new context/return statistics back into the live estimators."""
    n = cfg.n_workers
    per = math.ceil(cfg.rollout_steps / n)
    state_dict = {k: v.detach().cpu() for k, v in ac.state_dict().items()}
    obs_snap = copy.deepcopy(obs_rms)
    ret_snap = copy.deepcopy(ret_rms)
    payload = (state_dict, obs_snap, ret_snap, per)
    results = [pickle.loads(b) for b in executor.map(_worker_collect, [payload] * n)]

    transitions: list[Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    all_ctxs: list[np.ndarray] = []
    all_disc: list[float] = []
    for trs, eprs, sw, ctxs, disc in results:
        transitions.extend(trs)
        ep_returns.extend(eprs)
        for k, v in sw.items():
            switch_rewards[k].extend(v)
        all_ctxs.extend(ctxs)
        all_disc.extend(disc)
    # Merge the rollout's new statistics once (workers left the snapshots frozen).
    if obs_rms is not None and all_ctxs:
        obs_rms.update(np.asarray(all_ctxs, dtype=np.float64))
    if ret_rms is not None and all_disc:
        ret_rms.update(np.asarray(all_disc, dtype=np.float64)[:, None])
    return transitions, ep_returns, switch_rewards


def make_executor(ac: ActorCritic, envs, cfg: PPOConfig) -> ProcessPoolExecutor:
    """Build the persistent worker pool. ``spawn`` is used so workers start from
    a clean interpreter (the live ioh problems in ``envs`` are not fork-safe to
    reuse); each worker rebuilds its own envs from the picklable specs."""
    # Children inherit these at spawn and import numpy fresh, so their BLAS
    # (the CMA-ES eigendecomposition) stays single-threaded — see _worker_init.
    # setdefault leaves any explicit user choice untouched; the already-imported
    # parent BLAS is unaffected.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _use_file_system_sharing()  # parent side: it also receives (and re-sends) tensors
    mp_ctx = mp.get_context("spawn")
    counter = mp_ctx.Value("i", 0)
    specs = [e.spec() for e in envs]
    template = copy.deepcopy(ac).cpu()  # pickled to each worker once, at startup
    return ProcessPoolExecutor(
        max_workers=cfg.n_workers,
        mp_context=mp_ctx,
        initializer=_worker_init,
        initargs=(template, specs, cfg.seed, cfg, counter),
    )
