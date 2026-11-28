"""AMaLGaM: Adapted Maximum-Likelihood Gaussian Model Iterated Density-Estimation
Evolutionary Algorithm (full-covariance variant, single population).

A compact implementation of the method of Bosman, Grahl & Thierens (2008,
"Enhancing the performance of maximum-likelihood Gaussian EDAs using
anticipated mean shift", PPSN X) with the parameter-free settings of Bosman,
Grahl & Thierens (2013, "Benchmarking parameter-free AMaLGaM on functions with
and without noise", Evolutionary Computation 21(3)). It is not a bit-exact port
of the reference C code, and the restart scheme with growing populations is
left out (as for the CMA-ES here).

One generation, given the distribution ``N(mean, c_mult * cov)`` and the
anticipated mean shift ``ams_shift``:

1. sample ``N - 1`` offspring; the first ``alpha_AMS * (N - 1)`` of them are moved
   by ``delta_AMS * c_mult * ams_shift`` (anticipated mean shift);
2. the next population is the best old individual (elitism) plus the offspring;
3. adapt the distribution multiplier ``c_mult`` (SDR + no-improvement stretch):
   offspring better than the previous best reset the stretch and, when their
   mean lies far from ``mean`` in standard-deviation units (SDR > 1), widen the
   distribution; without improvement, ``c_mult`` shrinks after ``NIS_MAX``
   generations;
4. truncation-select the best ``tau * N``, re-estimate ``mean`` by maximum
   likelihood, blend the ML covariance into ``cov`` with memory ``eta_Sigma``,
   and update ``ams_shift`` with memory ``eta_shift``.

Warm-start state
----------------
Shared:   ``x`` / ``y`` — the population.
Specific: ``mean`` ``(D,)``, ``cov`` ``(D, D)`` (memory-smoothed, so it is *not*
recoverable from the population), ``ams_shift`` ``(D,)`` and the scalar
``c_mult``: the state the *next* generation samples from.

Without them (a population-only hand-off) ``mean`` / ``cov`` are estimated from
the handed-over points (``estimate_amalgam``), ``ams_shift = 0`` and
``c_mult = 1``. That is also the computed prior the translator predicts
residuals around (``algorithm_translation/models/prior/eda.py``).
"""

from __future__ import annotations

import math

import numpy as np

from algorithm_translation.optimizers.base import SubOptimizer

TAU = 0.35
DELTA_AMS = 2.0
ETA_DEC = 0.9
THETA_SDR = 1.0
# Share of the ML covariance replaced by its isotropic part when no more points
# are selected than there are dimensions: such an estimate is rank-deficient
# and the algorithm could never sample out of its subspace.
SMALL_SAMPLE_BLEND = 0.1
MIN_VAR = 1e-24
_C_MULT_RANGE = (1e-10, 1e10)


def n_selected(n_points: int, tau: float = TAU) -> int:
    return int(min(n_points, max(2, math.floor(tau * n_points))))


def estimate_amalgam(
    x: np.ndarray, y: np.ndarray, tau: float = TAU
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood ``(mean, cov)`` of the best ``tau`` of the points."""
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    k = n_selected(len(y), tau)
    pts = x[np.argsort(y)[:k]]
    mean = pts.mean(axis=0)
    diff = pts - mean
    cov = diff.T @ diff / k
    d = x.shape[1]
    if k <= d:
        tau2 = np.trace(cov) / d
        cov = (1.0 - SMALL_SAMPLE_BLEND) * cov + SMALL_SAMPLE_BLEND * tau2 * np.eye(d)
    return mean, cov + MIN_VAR * np.eye(d)


class AMALGAM(SubOptimizer):
    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        n = self.ndim_problem
        if self.n_individuals is None:
            self.n_individuals = int(17 + 3 * n**1.5)
        self.n_individuals = max(int(self.n_individuals), 4)
        N = self.n_individuals
        self.tau: float = options.get("tau", TAU)
        self.n_parents = n_selected(N, self.tau)
        s = self.n_parents
        self.eta_sigma = 1.0 - math.exp(-1.1 * s**1.2 / n**1.6)
        self.eta_shift = 1.0 - math.exp(-1.2 * s**0.31 / n**0.5)
        self.alpha_ams = 0.5 * self.tau * N / (N - 1)
        self.n_ams = int(math.floor(self.alpha_ams * (N - 1)))
        self.nis_max = 25 + n
        self._n_generations = 0

    # ------------------------------------------------------------------ #

    def _chol(self, cov: np.ndarray) -> np.ndarray:
        cov = 0.5 * (cov + cov.T)
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(cov)
            return v * np.sqrt(np.maximum(w, MIN_VAR))

    def initialize(self, mean=None, cov=None, ams_shift=None, c_mult=None, nis=0, x=None, y=None):
        n, lam = self.ndim_problem, self.n_individuals
        if x is None or y is None or np.isfinite(y).sum() < 2:
            if mean is None or cov is None:
                # Cold start: a uniform population, then a full ML estimate.
                x = self.rng_initialization.uniform(
                    self.initial_lower_boundary, self.initial_upper_boundary, size=(lam, n)
                )
                y = np.full(lam, np.inf)
                for i in range(lam):
                    if self._check_terminations():
                        break
                    y[i] = self._evaluate_fitness(x[i])
                if np.isfinite(y).sum() < 2:
                    return None, x, y
            else:
                x, y = mean.reshape(1, n), np.full(1, np.inf)
        if mean is None or cov is None:
            mean, cov = estimate_amalgam(x, y, self.tau)
            ams_shift, c_mult, nis = None, None, 0
        state = {
            "mean": np.asarray(mean, float),
            "cov": np.asarray(cov, float),
            "ams_shift": np.zeros(n) if ams_shift is None else np.asarray(ams_shift, float),
            "c_mult": 1.0 if c_mult is None else float(np.clip(c_mult, *_C_MULT_RANGE)),
            "nis": int(nis),
        }
        return state, x, y

    def iterate(self, st: dict, x: np.ndarray, y: np.ndarray):
        n, lam = self.ndim_problem, self.n_individuals
        mean, cov, shift, c_mult = st["mean"], st["cov"], st["ams_shift"], st["c_mult"]
        finite = np.isfinite(y)
        best_prev = float(np.min(y[finite])) if finite.any() else np.inf
        elite = int(np.argmin(np.where(finite, y, np.inf)))

        L = self._chol(cov)
        z = self.rng_optimization.standard_normal((lam - 1, n))
        off = mean + math.sqrt(c_mult) * z @ L.T
        off[: self.n_ams] += DELTA_AMS * c_mult * shift
        off = np.clip(off, self.lower_boundary, self.upper_boundary)

        new_x = np.empty((lam, n))
        new_y = np.full(lam, np.inf)
        new_x[0], new_y[0] = x[elite], y[elite]
        done = 0
        for i in range(lam - 1):
            if self._check_terminations():
                break
            new_x[i + 1] = off[i]
            new_y[i + 1] = self._evaluate_fitness(off[i])
            done += 1
        if done < lam - 1:  # budget ran out mid-generation: keep the distribution
            return st, new_x[: done + 1], new_y[: done + 1]

        # -- distribution multiplier: SDR + no-improvement stretch ---------- #
        improved = new_y[1:] < best_prev
        nis = st["nis"]
        if improved.any():
            nis = 0
            c_mult = max(c_mult, 1.0)
            x_imp = new_x[1:][improved].mean(axis=0)
            sdr = float(np.max(np.abs(np.linalg.solve(L, x_imp - mean))))
            if sdr > THETA_SDR:
                c_mult /= ETA_DEC
        else:
            if c_mult <= 1.0:
                nis += 1
            if c_mult > 1.0 or nis >= self.nis_max:
                c_mult *= ETA_DEC
            if c_mult < 1.0 and nis < self.nis_max:
                c_mult = 1.0
        c_mult = float(np.clip(c_mult, *_C_MULT_RANGE))

        # -- selection and ML estimation with memory ------------------------ #
        sel = new_x[np.argsort(new_y)[: self.n_parents]]
        old_mean = mean
        mean = sel.mean(axis=0)
        diff = sel - mean
        cov_ml = diff.T @ diff / len(sel)
        cov = (1.0 - self.eta_sigma) * cov + self.eta_sigma * cov_ml
        cov = 0.5 * (cov + cov.T) + MIN_VAR * np.eye(n)
        shift = (1.0 - self.eta_shift) * shift + self.eta_shift * (mean - old_mean)

        self._n_generations += 1
        st = {"mean": mean, "cov": cov, "ams_shift": shift, "c_mult": c_mult, "nis": nis}
        return st, new_x, new_y

    def _store(self, st, x, y):
        self._warm_start = {"x": x, "y": y, **(st or {})}

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        ws = self._warm_start
        st, x, y = self.initialize(
            ws.get("mean"),
            ws.get("cov"),
            ws.get("ams_shift"),
            ws.get("c_mult"),
            ws.get("nis", 0),
            ws.get("x"),
            ws.get("y"),
        )
        self._store(st, x, y)
        if st is not None:
            while not self._check_terminations():
                st, x, y = self.iterate(st, x, y)
                self._store(st, x, y)
        return self._collect(fitness)

    # ------------------------------------------------------------------ #

    def set_data(
        self,
        x=None,
        y=None,
        best_x=None,
        best_y=None,
        mean=None,
        cov=None,
        ams_shift=None,
        c_mult=None,
        nis=0,
        **kwargs,
    ):
        n = self.ndim_problem

        def _arr(v, shape):
            return None if v is None or np.asarray(v).shape != shape else np.asarray(v, float)

        self._warm_start = {
            "mean": _arr(mean, (n,)),
            "cov": _arr(cov, (n, n)),
            "ams_shift": _arr(ams_shift, (n,)),
            "c_mult": None if c_mult is None else float(c_mult),
            "nis": int(nis),
            "x": None,
            "y": None,
        }
        # Every handed-over point is kept, so a population-only hand-off
        # estimates from exactly the points the computed prior sees.
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
