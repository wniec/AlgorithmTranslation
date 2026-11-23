"""GRPO: group-relative advantages, scoring several candidate actions from one
switch state without advancing the env, and an end-to-end training run."""

import math

import numpy as np
import torch

from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.rl.grpo import GRPOConfig, group_advantages, train_grpo
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.suite import IOHSuite, build_problem_ids


def _env(reward_mode="noswitch", **kw):
    pids = build_problem_ids({1, 2}, dims=[2], instances=[1])
    return TranslationEnv(
        "PSO",
        "CMAES",
        pids,
        IOHSuite(),
        fe_multiplier=200,
        n_individuals=12,
        reward_mode=reward_mode,
        random_start=False,
        seed=0,
        **kw,
    )


def _reset_with_state(env):
    for _ in range(10):
        obs, _ = env.reset()
        if env.source_has_full_state():
            return obs
    raise AssertionError("no episode with a full source state")


def test_group_advantages():
    adv = group_advantages([1.0, 2.0, 3.0, 6.0])
    assert abs(adv.mean()) < 1e-12
    assert abs(adv.std() - 1.0) < 1e-6
    raw = group_advantages([1.0, 2.0, 3.0, 6.0], scale_by_std=False)
    np.testing.assert_allclose(raw, [-2.0, -1.0, 0.0, 3.0])
    # The action made no difference -> no signal.
    np.testing.assert_array_equal(group_advantages([0.5, 0.5, 0.5]), np.zeros(3))


def test_evaluate_action_scores_without_advancing_and_caches_baselines():
    env = _env(reward_mode="noswitch,coldtarget", reward_seeds=2)
    obs = _reset_with_state(env)
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    steps = ac.act_group(obs, 3)

    calls = {"noswitch": 0, "coldtarget": 0}
    scorer = env._scorer
    run_baseline = scorer._run_baseline

    def counting(mode, seed):
        calls[mode] += 1
        return run_baseline(mode, seed)

    scorer._run_baseline = counting
    n_fe, best = env._n_fe, env._best_y
    rewards = [env.evaluate_action(s.native)[0] for s in steps]
    # The episode stays at the switch ...
    assert (env._n_fe, env._best_y, env._done) == (n_fe, best, False)
    # ... and each baseline ran once per seed, not once per candidate.
    assert calls == {"noswitch": 2, "coldtarget": 2}
    # Same action, same seeds -> same reward; step() agrees with evaluate_action.
    again, info = env.evaluate_action(steps[0].native)
    assert again == rewards[0]
    assert set(info["reward_parts"]) == {"noswitch", "coldtarget"}
    _, r_step, term, _, _ = env.step(steps[0].native)
    assert term and r_step == rewards[0]


def test_act_group_log_probs_match_policy():
    env = _env()
    obs = _reset_with_state(env)
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    steps = ac.act_group(obs, 4)
    assert len(steps) == 4
    actions = torch.stack([s.action for s in steps])
    assert not torch.allclose(actions[0], actions[1])  # distinct samples
    logp, ent, _ = ac.log_prob_entropy(
        [s.src_state for s in steps], "CMAES", actions
    )
    np.testing.assert_allclose(
        logp.detach().numpy(), [s.log_prob for s in steps], rtol=1e-4, atol=1e-4
    )
    assert torch.isfinite(ent).all()


def test_grpo_runs_and_metrics_finite():
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    critic_before = {k: v.clone() for k, v in ac.critic.state_dict().items()}
    cfg = GRPOConfig(
        updates=2,
        groups_per_update=3,
        group_size=4,
        epochs=2,
        minibatch_size=8,
        seed=1,
        log_every=0,
    )
    log = train_grpo(ac, [_env()], cfg)
    assert len(log.history) == 2
    for row in log.history:
        assert row["n_samples"] == 12
        for k in ("policy_loss", "approx_kl", "clip_frac", "mean_return",
                  "group_reward_std", "policy_std"):
            assert math.isfinite(row[k]), k
    # The critic is not part of GRPO's optimizer.
    for k, v in ac.critic.state_dict().items():
        assert torch.equal(v, critic_before[k]), k
