"""Command-line entry points, exposed as console scripts (see
``[project.scripts]`` in pyproject.toml): ``train`` / ``train-td3`` for the PPO
and TD3 trainers, ``eval`` / ``eval-switch`` for the two evaluators. Each module
defines a ``main()`` the script wraps; shared CLI/env plumbing lives in
``cat.cli.common``.
"""
