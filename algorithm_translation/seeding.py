"""Seed derivation.

Every random stream in training and evaluation is seeded from a user seed plus
a few integers saying *which* stream it is (episode index, worker id, repeat,
...). ``derive_seed`` mixes them through ``numpy.random.SeedSequence`` so
different tuples give unrelated seeds, instead of hand-rolled arithmetic like
``seed * 1_000_003 + idx`` whose streams can collide.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Stream(IntEnum):
    """Tags that keep independent random streams apart."""

    EPISODE = 1  # switch direction, switch point and probe of one episode
    WARMUP = 2  # source optimizer run up to the switch
    SEGMENT = 3  # post-switch runs (translated + counterfactual baselines)
    PROBLEM_ORDER = 4  # an env's permutation of its problems for one pass
    ENV = 5  # one training env's base seed
    WORKER = 6  # a rollout worker's env seeds
    WORKER_EPISODES = 7  # a rollout worker's env-selection RNG
    MINIBATCH = 8  # minibatch shuffling in one update
    STANDALONE = 9  # eval-switch: single-algorithm baseline runs
    SCHEDULE = 10  # eval-switch: random algorithm schedule + switch points


def derive_seed(*parts: int) -> int:
    """A seed in ``[0, 2**31)`` determined by ``parts`` (any integers)."""
    entropy = [int(p) % 2**64 for p in parts]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0] >> 1)
