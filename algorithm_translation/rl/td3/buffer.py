"""TD3 replay memory: the stored transition record and the sliding-window buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from algorithm_translation.state.canonical import CanonicalState


@dataclass
class TD3Transition:
    src_state: CanonicalState  # s, un-batched, cpu
    source_algo: str
    action: Tensor  # (action_dim,), cpu — raw mean(+noise) action taken
    reward: float
    context: np.ndarray  # normalized context at s
    # Always a well-formed CanonicalState (see collect_rollout's guard) with
    # algo == the deterministic partner algorithm; `done` masks its
    # contribution to the TD-target rather than needing a None special case.
    next_state: CanonicalState
    next_context: np.ndarray
    done: bool


class ReplayBuffer:
    """Sliding-window replay memory (FIFO once ``capacity`` is exceeded)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[TD3Transition] = deque(maxlen=capacity)

    def extend(self, transitions: list[TD3Transition]) -> None:
        self._buf.extend(transitions)

    def as_list(self) -> list[TD3Transition]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
