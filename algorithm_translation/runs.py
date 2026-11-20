"""Building and running portfolio optimizers on a problem.

Shared by the training env (``rl``), the reward scorer and both evaluators, so
an optimizer segment is constructed the same way everywhere.
"""

from __future__ import annotations

from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.state.canonical import warm_start_optimizer


def problem_cfg(problem) -> dict:
    """The problem dict every portfolio optimizer takes."""
    return {
        "fitness_function": problem,
        "ndim_problem": problem.dimension,
        "lower_boundary": problem.lower_bounds,
        "upper_boundary": problem.upper_bounds,
    }


def make_optimizer(
    algo: str,
    cfg: dict,
    *,
    max_fe: int,
    target_fe: int,
    seed: int,
    n_individuals: int | None = None,
    best_y: float = float("inf"),
    n_fe_start: int = 0,
):
    """An optimizer that runs from ``n_fe_start`` until ``target_fe`` FEs
    (within a total budget of ``max_fe``), knowing ``best_y`` as the incumbent.
    ``n_individuals=None`` keeps the algorithm's own default population."""
    options = {
        "max_function_evaluations": int(max_fe),
        "target_fe": int(target_fe),
        "best_so_far_y": float(best_y),
        "seed_rng": int(seed),
        "verbose": False,
    }
    if n_individuals is not None:
        options["n_individuals"] = n_individuals
    opt = PORTFOLIO[algo](cfg, options)
    opt.n_function_evaluations = int(n_fe_start)
    return opt


def run_segment(
    algo: str,
    cfg: dict,
    *,
    max_fe: int,
    target_fe: int,
    seed: int,
    n_individuals: int | None = None,
    n_fe_start: int = 0,
    best_y: float = float("inf"),
    payload: dict | None = None,
):
    """Run one segment warm-started from ``payload`` (a native state dict), or
    cold-started when it is None. Returns ``(optimizer, result)``."""
    opt = make_optimizer(
        algo,
        cfg,
        max_fe=max_fe,
        target_fe=target_fe,
        seed=seed,
        n_individuals=n_individuals,
        best_y=best_y,
        n_fe_start=n_fe_start,
    )
    if payload is None:
        opt.set_data()  # cold start
    else:
        warm_start_optimizer(opt, payload)
    return opt, opt.optimize()


def shared_handoff(native: dict) -> dict:
    """The lossy default hand-off: only the shared population (positions,
    values, best), dropping every algorithm-specific field."""
    return {
        "x": native["x"],
        "y": native["y"],
        "best_x": native["best_x"],
        "best_y": native["best_y"],
    }
