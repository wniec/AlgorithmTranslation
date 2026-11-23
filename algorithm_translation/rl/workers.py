"""Persistent process pool for parallel rollout collection (PPO and GRPO).

Rollout collection is the wall-clock bottleneck (the optimizer runs inside the
env dominate), so each trainer can fan it out over worker processes. Every
worker rebuilds its own envs from picklable specs (with its own seeds) and
keeps a copy of the policy; per update the trainer sends fresh weights and gets
pickled results back. A trainer's collect function, run in the worker, reads
the worker's state from ``WORKER``.
"""

from __future__ import annotations

import contextlib
import copy
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as torch_mp

from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.seeding import Stream, derive_seed

# Per-process worker state: "ac" (policy), "cfg", "envs", "ep_rng".
WORKER: dict = {}


def use_file_system_sharing() -> None:
    """Share tensors between processes through shm files instead of one open
    file descriptor per tensor. Retained rollout tensors otherwise exhaust the
    open-file limit and the pool fails with ``received 0 items of ancdata``."""
    with contextlib.suppress(RuntimeError, ValueError):  # unavailable: keep default
        torch_mp.set_sharing_strategy("file_system")


def worker_env_specs(specs: list[dict], worker_id: int) -> list[dict]:
    """Env specs re-seeded for one worker, so workers walk different problem
    orders, switch points and seeds."""
    return [
        {**s, "seed": derive_seed(int(s["seed"]), worker_id, Stream.WORKER)} for s in specs
    ]


def _init_worker(ac, specs, base_seed: int, cfg, counter) -> None:
    with counter.get_lock():
        worker_id = counter.value
        counter.value += 1
    # One optimizer run per worker at a time: N workers x many BLAS threads
    # would oversubscribe the cores.
    torch.set_num_threads(1)
    use_file_system_sharing()
    WORKER["ac"] = ac.to("cpu")
    WORKER["cfg"] = cfg
    WORKER["envs"] = [TranslationEnv(**spec) for spec in worker_env_specs(specs, worker_id)]
    WORKER["ep_rng"] = np.random.default_rng(
        derive_seed(base_seed, worker_id, Stream.WORKER_EPISODES)
    )


def load_worker_policy(state_dict: dict):
    """The worker's policy with the trainer's current weights loaded."""
    ac = WORKER["ac"]
    ac.load_state_dict(state_dict)
    return ac


def policy_weights(ac) -> dict:
    """The policy's weights as CPU tensors, ready to send to the workers."""
    return {k: v.detach().cpu() for k, v in ac.state_dict().items()}


def make_executor(ac, envs, cfg) -> ProcessPoolExecutor:
    """A persistent ``spawn`` pool of ``cfg.n_workers`` workers.

    ``spawn`` gives each worker a clean interpreter (the live ioh problems in
    ``envs`` are not fork-safe); the workers rebuild their envs from specs."""
    # Inherited at spawn, before numpy is imported, so each worker's BLAS stays
    # single-threaded. An explicit user setting is left alone.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    use_file_system_sharing()  # the parent also receives and re-sends tensors
    mp_ctx = mp.get_context("spawn")
    counter = mp_ctx.Value("i", 0)
    return ProcessPoolExecutor(
        max_workers=cfg.n_workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(copy.deepcopy(ac).cpu(), [e.spec() for e in envs], cfg.seed, cfg, counter),
    )
