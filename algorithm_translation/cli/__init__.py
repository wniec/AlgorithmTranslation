"""Command-line entry points, exposed as console scripts (see
``[project.scripts]`` in pyproject.toml): ``train`` / ``train-grpo`` for the PPO
and GRPO trainers, ``eval`` / ``eval-switch`` for the two evaluators. Each module
defines a ``main()`` the script wraps; shared CLI/env plumbing lives in
``algorithm_translation.cli.common``.
"""

from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.state.schema import has_spec

# Optimizers the CLIs accept: those in the portfolio that have a StateSpec.
ALGO_CHOICES = sorted(a for a in PORTFOLIO if has_spec(a))


def population_sizes(args) -> dict[str, int | None]:
    """``{algo: population size}`` from ``--n-individuals`` and its per-algo
    overrides ``--n-individuals-a/-b`` (None = the algorithm's own default)."""

    def pick(override):
        return override if override is not None else args.n_individuals

    return {args.algo_a: pick(args.n_individuals_a), args.algo_b: pick(args.n_individuals_b)}
