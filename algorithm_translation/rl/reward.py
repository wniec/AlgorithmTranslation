"""Scoring one optimizer segment for the training reward.

Pure functions of plain values (no optimizer or env state). ``rl.scoring`` runs
the translated and counterfactual segments and scores each run here; the
baseline subtraction and reward-mode summing happen there.

Two measures:

* ``aocc`` (default, ``segment_aocc``) — Area Over the Convergence Curve: the mean, over the
  segment's evaluations, of the normalized log-precision closed toward the
  optimum. Rewards converging FASTER/LOWER, so the hand-off's transient value
  survives even one long segment (where the endpoint washes out). Bounded
  ``[0, 1]``; needs the optimum, and falls back to ``endpoint`` when it's absent.
* ``endpoint`` — log-scaled best-so-far improvement at the segment
  boundary only.
"""

from __future__ import annotations

import numpy as np

# Log-scaling floor. reward = log(scaled + eps) - log(eps) = log(1 + scaled/eps)
# maps a [0, 1] scaled improvement to [0, log(1 + 1/eps)] (~9.2 for eps=1e-4),
# with 0 improvement -> 0. Log scaling emphasizes the small improvements that
# dominate late-stage refinement near the optimum.
_REWARD_EPS = 1e-4


def scaled_improvement(prev_best: float, new_best: float, rng_range: float) -> float:
    """Best-so-far improvement over a segment, scaled to [0, 1]."""
    improvement = max(0.0, prev_best - new_best)
    return float(np.clip(improvement / max(rng_range, 1e-12), 0.0, 1.0))


def _log_scale(scaled: float) -> float:
    """Map a [0, 1] scaled improvement through a log transform (0 -> 0)."""
    return float(np.log(scaled + _REWARD_EPS) - np.log(_REWARD_EPS))


def log_scaled_improvement(
    prev_best: float, new_best: float, rng_range: float
) -> float:
    """Log-scaled best-so-far improvement over a segment."""
    return _log_scale(scaled_improvement(prev_best, new_best, rng_range))


def segment_aocc(
    y_history, prev_best: float, optimum: float, decades: float, skip: int = 0
) -> float | None:
    """Area Over the Convergence Curve for one run, in [0, 1].

    The best-so-far curve is seeded at ``prev_best`` (the incumbent at the
    switch) and run over the segment's evaluations. Each point's precision is the
    fraction of the log-gap closed from the gap-at-switch (``g_hi``, so the curve
    starts at 0) down to the floor ``g_lo = g_hi * 10**(-decades)`` (full
    credit). AOCC is the mean over evaluations — reaching high precision EARLIER
    raises it.

    ``skip`` drops the first ``skip`` evaluations from the average while still
    letting them count toward the best-so-far. It is for a run that started
    *before* the switch (the cold-target reference, run from FE 0): scoring it
    over only the post-switch window ``[switch, max_fe]`` keeps it on the same
    FE window as the translated run, instead of diluting its average with the
    pre-switch evaluations that a post-switch run never has. Returns None when
    there is no history (or nothing is left after ``skip``)."""
    if y_history is None or len(y_history) == 0:
        return None
    y = np.asarray(y_history, dtype=np.float64).ravel()
    g_hi = max(prev_best - optimum, 1e-12)
    g_lo = g_hi * (10.0 ** (-decades))
    run_best = np.minimum.accumulate(y)[max(int(skip), 0):]
    if run_best.size == 0:
        return None
    best = np.minimum(prev_best, run_best)  # best-so-far curve over the window
    gap = np.clip(best - optimum, g_lo, g_hi)
    v = np.log10(g_hi / gap) / decades  # per-eval precision in [0, 1]
    return float(v.mean())


def run_term(
    result: dict,
    prev_best: float,
    rng_range: float,
    *,
    measure: str,
    optimum: float | None,
    decades: float,
    skip: int = 0,
) -> float:
    """Score one optimizer run's segment — the building block every reward
    mode's translated/baseline term shares, so they stay comparable within a
    step.

    ``aocc`` needs the optimum; when it is ``None`` (or the run has no history)
    this falls back to the ``endpoint`` measure (log-scaled boundary
    improvement). ``skip`` is passed to ``segment_aocc`` (see there); the endpoint
    measure only looks at the final best, so it needs no window."""
    if measure == "aocc" and optimum is not None:
        area = segment_aocc(result.get("y_history"), prev_best, optimum, decades, skip=skip)
        if area is not None:
            return area
    best = result.get("best_so_far_y", float("inf"))
    improvement = max(0.0, prev_best - best)
    return _log_scale(float(np.clip(improvement / rng_range, 0.0, 1.0)))
