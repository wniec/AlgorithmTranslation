"""Public-surface contract: the console-script entry points import and expose a
callable ``main``, and the package ``__init__`` re-exports stay intact. These
catch rename/move breakage (missing modules, dropped re-exports) cheaply — the
class of failure the algorithm_translation/cli + algorithm_translation/rl package
split could reintroduce.
"""

import importlib


def test_cli_modules_expose_main():
    for name in ("train_ppo", "train_grpo", "evaluate", "evaluate_switcher"):
        mod = importlib.import_module(f"algorithm_translation.cli.{name}")
        assert callable(mod.main), f"algorithm_translation.cli.{name}.main missing/not callable"


def test_package_reexports_intact():
    import algorithm_translation.rl.env as env
    import algorithm_translation.rl.grpo as grpo
    import algorithm_translation.rl.ppo as ppo
    import algorithm_translation.rl.reward as reward
    import algorithm_translation.rl.scoring as scoring
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
        "GRPOConfig",
        "GRPOLog",
        "Group",
        "Sample",
        "train_grpo",
        "collect_groups",
        "collect_groups_parallel",
        "make_executor",
        "group_advantages",
        "update",
    ):
        assert hasattr(grpo, n), f"algorithm_translation.rl.grpo.{n} missing"
    for n in (
        "TranslationEnv",
        "sample_episode_start",
    ):
        assert hasattr(env, n), f"algorithm_translation.rl.env.{n} missing"
    for n in (
        "SwitchPoint",
        "SwitchScorer",
        "ScoredAction",
        "normalize_reward_modes",
    ):
        assert hasattr(scoring, n), f"algorithm_translation.rl.scoring.{n} missing"
    for n in ("segment_aocc", "run_term", "scaled_improvement", "log_scaled_improvement"):
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
