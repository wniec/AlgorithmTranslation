"""Public-surface contract: the console-script entry points import and expose a
callable ``main``, and the package ``__init__`` re-exports stay intact. These
catch rename/move breakage (missing modules, dropped re-exports) cheaply — the
class of failure the algorithm_translation/cli + algorithm_translation/rl package
split could reintroduce.
"""

import importlib


def test_cli_modules_expose_main():
    for name in ("train_ppo", "train_td3", "evaluate", "evaluate_switcher"):
        mod = importlib.import_module(f"algorithm_translation.cli.{name}")
        assert callable(mod.main), f"algorithm_translation.cli.{name}.main missing/not callable"


def test_package_reexports_intact():
    import algorithm_translation.rl.env as env
    import algorithm_translation.rl.ppo as ppo
    import algorithm_translation.rl.reward as reward
    import algorithm_translation.rl.td3 as td3
    import algorithm_translation.state as state

    for n in (
        "PPOConfig",
        "PPOLog",
        "RolloutBuffer",
        "Transition",
        "train_ppo",
        "collect_rollout",
        "collect_rollout_parallel",
        "make_executor",
        "update",
        "explained_variance",
    ):
        assert hasattr(ppo, n), f"algorithm_translation.rl.ppo.{n} missing"
    for n in (
        "TD3Actor",
        "TD3Config",
        "TwinQCritic",
        "QCritic",
        "TD3Step",
        "TD3Transition",
        "ReplayBuffer",
        "train_td3",
        "_polyak_update",
    ):
        assert hasattr(td3, n), f"algorithm_translation.rl.td3.{n} missing"
    for n in (
        "TranslationEnv",
        "scaled_improvement",
        "log_scaled_improvement",
        "_log_scale",
        "_REWARD_EPS",
    ):
        assert hasattr(env, n), f"algorithm_translation.rl.env.{n} missing"
    for n in ("aocc", "run_term", "scaled_improvement", "log_scaled_improvement"):
        assert hasattr(reward, n), f"algorithm_translation.rl.reward.{n} missing"
    for n in (
        "CanonicalState",
        "augment_native",
        "has_full_state",
        "collate",
        "to_canonical",
        "from_canonical",
        "warm_start_optimizer",
    ):
        assert hasattr(state, n), f"algorithm_translation.state.{n} missing"
