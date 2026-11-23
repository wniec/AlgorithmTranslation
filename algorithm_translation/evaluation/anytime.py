"""Anytime-performance metrics for whole-budget runs (``uv run eval-switch``).

``budget_aocc`` follows DynamicAlgorithmSelection's ``get_runtime_stats``: the
precision ``best_y - optimum`` is clipped to [1e-8, 1e8], log10-normalized onto
[0, 1], and ``1 - that`` is averaged over the whole FE budget. It is a different
quantity from the training reward's ``rl.reward.segment_aocc``, which is
relative to the gap at a switch and covers only the post-switch segment.
"""

from __future__ import annotations

import numpy as np

_PREC_LB, _PREC_UB = 1e-8, 1e8
_LOG_LB, _LOG_UB = -8.0, 8.0

# Precision targets ERT is reported against.
ERT_TARGETS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


class Recorder:
    """Wraps a problem and records every best-so-far improvement as ``(fe,
    y)`` over one whole run, across all its segments. ``fe`` counts actual calls,
    i.e. the run's true FE spend (cold restarts re-evaluate populations, etc.)."""

    def __init__(self, problem):
        self._problem = problem
        self.fe = 0
        self.best = float("inf")
        self.history: list[tuple[int, float]] = []

    def __call__(self, x) -> float:
        y = float(self._problem(x))
        self.fe += 1
        if y < self.best:
            self.best = y
            self.history.append((self.fe, y))
        return y


def _normalized_log_precision(y: float, optimum: float) -> float:
    clipped = float(np.clip(y - optimum, _PREC_LB, _PREC_UB))
    return (np.log10(clipped) - _LOG_LB) / (_LOG_UB - _LOG_LB)


def budget_aocc(history, budget: int, optimum: float) -> float:
    """Area over the convergence curve in [0, 1] (higher = better), averaged
    over ``budget`` FEs. Each improvement's value covers the interval that ends
    at its FE, and the final value extends to the end of the budget."""
    if not history or budget <= 0:
        return 0.0
    area = 0.0
    last_fe = 0
    for fe, y in history:
        area += (1.0 - _normalized_log_precision(y, optimum)) * (fe - last_fe)
        last_fe = fe
    final = 1.0 - _normalized_log_precision(history[-1][1], optimum)
    area += final * max(0, budget - history[-1][0])
    return area / budget


def fe_to_targets(history, optimum: float) -> list[float]:
    """First FE at which the precision reaches each of ``ERT_TARGETS`` (``inf``
    when never reached), in ``ERT_TARGETS`` order."""
    result = [float("inf")] * len(ERT_TARGETS)
    for fe, y in history:
        prec = y - optimum
        for j, target in enumerate(ERT_TARGETS):
            if result[j] == float("inf") and prec <= target:
                result[j] = float(fe)
        if all(r != float("inf") for r in result):
            break
    return result


def ert(fe_to_target: np.ndarray, budgets: np.ndarray) -> tuple[float, int]:
    """Expected Running Time (COCO definition) for one target: FEs spent by all
    runs (successes up to the target, failures their whole budget) divided by
    the number of successes. Returns ``(ert, n_successes)``; ``inf`` when no run
    reached the target."""
    success = np.isfinite(fe_to_target)
    n_succ = int(success.sum())
    if n_succ == 0:
        return float("inf"), 0
    spent = fe_to_target[success].sum() + budgets[~success].sum()
    return float(spent / n_succ), n_succ


def average_ranks(values: list[float]) -> list[float]:
    """1-based ranks (smallest value -> rank 1); ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks
