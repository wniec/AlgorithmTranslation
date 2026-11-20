"""Command-line entry points, exposed as console scripts (see
``[project.scripts]`` in pyproject.toml): ``train`` / ``train-grpo`` for the PPO
and GRPO trainers, ``eval`` / ``eval-switch`` for the two evaluators. Each module
defines a ``main()`` the script wraps; shared CLI/env plumbing lives in
``algorithm_translation.cli.common``.
"""

import argparse
import re

from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.state.schema import has_spec

# Optimizers the CLIs accept: those in the portfolio that have a StateSpec.
ALGO_CHOICES = sorted(a for a in PORTFOLIO if has_spec(a))


# argparse only recognizes "-1" / "-0.5" as negative numbers, so a value such as
# "-1e-3" is taken for an option and the flag before it reports "expected one
# argument". Also accept scientific notation.
_NEGATIVE_NUMBER = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def make_parser(**kwargs) -> argparse.ArgumentParser:
    """An ``ArgumentParser`` whose options accept negative values in any
    float syntax, e.g. ``--ent-coef -1e-3``."""
    parser = argparse.ArgumentParser(**kwargs)
    parser._negative_number_matcher = _NEGATIVE_NUMBER
    return parser


def population_sizes(args) -> dict[str, int | None]:
    """``{algo: population size}`` from ``--n-individuals`` and its per-algo
    overrides ``--n-individuals-a/-b`` (None = the algorithm's own default)."""

    def pick(override):
        return override if override is not None else args.n_individuals

    return {args.algo_a: pick(args.n_individuals_a), args.algo_b: pick(args.n_individuals_b)}
