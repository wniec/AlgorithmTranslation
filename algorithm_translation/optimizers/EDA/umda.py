"""UMDA: continuous Univariate Marginal Distribution Algorithm (UMDA_c), smoothed.

A compact implementation of UMDA_c (Larrañaga & Lozano, 2002): every generation
samples from a Gaussian with *diagonal* covariance, truncation-selects the best
fraction of the samples and re-estimates the per-coordinate mean and standard
deviation from them.

The classic algorithm has no memory: its distribution is a function of the last
selected population, which the shared fields already carry, so there would be
nothing algorithm-specific to translate. Here the estimates are *smoothed*
across generations with a learning rate, as in PBIL_c (Sebag & Ducoulombier,
1998) and the cross-entropy method::

    mean <- (1 - alpha) * mean + alpha * mean(selected)
    std  <- (1 - alpha) * std  + alpha * std(selected)

so ``(mean, std)`` are genuine state. ``learning_rate = 1`` recovers plain
UMDA_c.

Warm-start state
----------------
Shared:   ``x`` / ``y`` — the last population.
Specific: ``mean`` ``(D,)``, ``std`` ``(D,)`` — the distribution the *next*
generation samples from.

Without ``mean`` / ``std`` (a population-only hand-off) the distribution is
estimated from the handed-over points (``estimate_umda``), which is also the
computed prior the translator predicts residuals around
(``algorithm_translation/models/prior/eda.py``).
"""

from __future__ import annotations

import numpy as np

from algorithm_translation.optimizers.base import SubOptimizer

SELECTION_RATIO = 0.3
LEARNING_RATE = 0.7
# Absolute std floor, so coincident points never give a degenerate distribution.
MIN_STD = 1e-12


def n_selected(n_points: int, ratio: float = SELECTION_RATIO) -> int:
    """Truncation-selection size for ``n_points`` points (at least 2)."""
    return int(min(n_points, max(2, round(ratio * n_points))))


def estimate_umda(
    x: np.ndarray, y: np.ndarray, ratio: float = SELECTION_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood ``(mean, std)`` of the best ``ratio`` of the points."""
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    sel = np.argsort(y)[: n_selected(len(y), ratio)]
    pts = x[sel]
    return pts.mean(axis=0), np.maximum(pts.std(axis=0), MIN_STD)


class UMDA(SubOptimizer):
    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        n = self.ndim_problem
        if self.n_individuals is None:
            self.n_individuals = max(20, 10 * n)
        self.n_individuals = max(int(self.n_individuals), 4)
        self.selection_ratio: float = options.get("selection_ratio", SELECTION_RATIO)
        self.n_parents = n_selected(self.n_individuals, self.selection_ratio)
        self.learning_rate: float = options.get("learning_rate", LEARNING_RATE)
        self._n_generations = 0

    # ------------------------------------------------------------------ #

    def initialize(self, mean=None, std=None, x=None, y=None):
        n, lam = self.ndim_problem, self.n_individuals
        if mean is not None and std is not None:
            if x is None or y is None:
                x, y = np.empty((0, n)), np.empty(0)
            return mean, np.maximum(std, MIN_STD), x, y
        if x is None or y is None or np.isfinite(y).sum() < 2:
            # Cold start: a uniform population, then a full (alpha = 1) estimate.
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary, size=(lam, n)
            )
            y = np.full(lam, np.inf)
            for i in range(lam):
                if self._check_terminations():
                    break
                y[i] = self._evaluate_fitness(x[i])
            if np.isfinite(y).sum() < 2:
                self._store(None, None, x, y)
                return None, None, x, y
        mean, std = estimate_umda(x, y, self.selection_ratio)
        return mean, std, x, y

    def iterate(self, mean, std, x, y):
        n, lam = self.ndim_problem, self.n_individuals
        new_x = np.empty((lam, n))
        new_y = np.full(lam, np.inf)
        z = self.rng_optimization.standard_normal((lam, n))
        done = 0
        for i in range(lam):
            if self._check_terminations():
                break
            new_x[i] = np.clip(mean + std * z[i], self.lower_boundary, self.upper_boundary)
            new_y[i] = self._evaluate_fitness(new_x[i])
            done += 1
        if done < lam:  # budget ran out mid-generation: keep the distribution
            return mean, std, new_x[:done], new_y[:done]

        sel = np.argsort(new_y)[: self.n_parents]
        a = self.learning_rate
        mean = (1.0 - a) * mean + a * new_x[sel].mean(axis=0)
        std = np.maximum((1.0 - a) * std + a * new_x[sel].std(axis=0), MIN_STD)
        self._n_generations += 1
        return mean, std, new_x, new_y

    def _store(self, mean, std, x, y):
        self._warm_start = {"x": x, "y": y, "mean": mean, "std": std}

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        ws = self._warm_start
        mean, std, x, y = self.initialize(ws.get("mean"), ws.get("std"), ws.get("x"), ws.get("y"))
        if mean is not None:
            self._store(mean, std, x, y)
            while not self._check_terminations():
                mean, std, gx, gy = self.iterate(mean, std, x, y)
                if len(gy):
                    x, y = gx, gy
                self._store(mean, std, x, y)
        return self._collect(fitness)

    # ------------------------------------------------------------------ #

    def set_data(self, x=None, y=None, best_x=None, best_y=None, mean=None, std=None, **kwargs):
        n = self.ndim_problem

        def _vec(v):
            return None if v is None or np.asarray(v).shape != (n,) else np.asarray(v, float)

        self._warm_start = {"mean": _vec(mean), "std": _vec(std), "x": None, "y": None}
        # Every handed-over point is kept (not truncated to n_individuals), so a
        # population-only hand-off estimates from exactly the points the
        # computed prior sees. The first generation then resizes the population.
        if x is not None and y is not None and len(x) > 0:
            self._warm_start["x"] = np.asarray(x, float)
            self._warm_start["y"] = np.asarray(y, float).reshape(-1)
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        return dict(self._warm_start)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result
