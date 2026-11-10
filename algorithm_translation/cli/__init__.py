"""Command-line entry points, exposed as console scripts (see
``[project.scripts]`` in pyproject.toml): ``train`` / ``train-td3`` for the PPO
and TD3 trainers, ``eval`` / ``eval-switch`` for the two evaluators. Each module
defines a ``main()`` the script wraps; shared CLI/env plumbing lives in
``algorithm_translation.cli.common``.
"""

from algorithm_translation.optimizers.portfolio import PORTFOLIO
from algorithm_translation.state.schema import has_spec

# Optimizers the CLIs accept: those in the portfolio that have a StateSpec.
ALGO_CHOICES = sorted(a for a in PORTFOLIO if has_spec(a))
