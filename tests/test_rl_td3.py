"""TD3 loop runs end-to-end on a tiny problem set and produces finite metrics —
mirrors tests/test_rl_ppo.py so the two trainers are covered symmetrically."""

import math

from cat.rl.env import TranslationEnv
from cat.rl.td3 import (
    ReplayBuffer,
    TD3Actor,
    TD3Config,
    TD3Transition,
    TwinQCritic,
    train_td3,
)
from cat.suite import IOHSuite, build_problem_ids


def _envs():
    pids = build_problem_ids({1, 2}, dims=[2], instances=[1])
    return [
        TranslationEnv(
            "PSO",
            "CMAES",
            pids,
            IOHSuite(),
            fe_multiplier=200,
            n_individuals=12,
            seed=0,
        )
    ]


def _envs_independent_sizes():
    pids = build_problem_ids({1, 2}, dims=[2], instances=[1])
    return [
        TranslationEnv(
            "PSO",
            "CMAES",
            pids,
            IOHSuite(),
            fe_multiplier=200,
            n_individuals_a=8,
            n_individuals_b=20,
            seed=0,
        )
    ]


def _actor_critic(**actor_kwargs):
    actor = TD3Actor("PSO", "CMAES", hidden=32, n_layers=1, **actor_kwargs)
    critic = TwinQCritic("PSO", "CMAES", hidden=32, n_layers=1)
    return actor, critic


def test_td3_runs_and_metrics_finite():
    actor, critic = _actor_critic()
    cfg = TD3Config(
        updates=2,
        rollout_steps=48,
        gradient_steps=8,
        minibatch_size=16,
        policy_freq=2,
        seed=1,
        log_every=0,
    )
    log = train_td3(actor, critic, _envs(), cfg)
    assert len(log.history) == 2
    for row in log.history:
        for k in ("critic_loss", "q_value", "actor_loss", "cycle", "mean_return"):
            assert math.isfinite(row[k]), f"{k} not finite"
        assert row["n_transitions"] >= 1


def test_td3_runs_with_independent_population_sizes():
    """Each algo running its own population size (PSO=8, CMAES=20) must not
    disturb training: the action is always decoded onto the source's own N
    (see TD3Actor.act / _assemble_native) — only the assembled warm-start is
    resized. Metrics must stay finite end-to-end through rollout + update."""
    actor, critic = _actor_critic(resample_seed=0)
    cfg = TD3Config(
        updates=2,
        rollout_steps=48,
        gradient_steps=8,
        minibatch_size=16,
        policy_freq=2,
        seed=1,
        log_every=0,
    )
    log = train_td3(actor, critic, _envs_independent_sizes(), cfg)
    for row in log.history:
        for k in ("critic_loss", "q_value", "actor_loss", "mean_return"):
            assert math.isfinite(row[k]), f"{k} not finite"


def test_td3_cycle_penalty_field_and_latent_modes_finite():
    for mode in ("field", "latent"):
        actor, critic = _actor_critic()
        cfg = TD3Config(
            updates=1,
            rollout_steps=32,
            gradient_steps=6,
            minibatch_size=8,
            policy_freq=2,
            lambda_cycle=0.1,
            cycle_mode=mode,
            seed=1,
            log_every=0,
        )
        log = train_td3(actor, critic, _envs(), cfg)
        for row in log.history:
            assert math.isfinite(row["cycle"]), mode
            assert math.isfinite(row["actor_loss"]), mode


def test_td3_lambda_cycle_zero_skips_cycle_computation():
    actor, critic = _actor_critic()
    cfg = TD3Config(
        updates=1,
        rollout_steps=32,
        gradient_steps=6,
        minibatch_size=8,
        policy_freq=2,
        lambda_cycle=0.0,
        seed=1,
        log_every=0,
    )
    log = train_td3(actor, critic, _envs(), cfg)
    for row in log.history:
        assert row["cycle"] == 0.0


def test_replay_buffer_window_evicts_oldest():
    """The buffer keeps only the most recent ``capacity`` transitions (FIFO)."""
    buf = ReplayBuffer(capacity=3)
    trs = [
        TD3Transition(
            src_state=None,
            source_algo="PSO",
            action=None,
            reward=float(i),
            context=None,
            next_state=None,
            next_context=None,
            done=False,
        )
        for i in range(5)
    ]
    buf.extend(trs[:2])
    assert len(buf) == 2
    buf.extend(trs[2:])  # total 5 added, capacity 3 -> oldest two evicted
    assert len(buf) == 3
    assert [t.reward for t in buf.as_list()] == [2.0, 3.0, 4.0]


def test_parameters_change_after_update():
    actor, critic = _actor_critic()
    before_a = [p.detach().clone() for p in actor.parameters()]
    before_c = [p.detach().clone() for p in critic.parameters()]
    train_td3(
        actor,
        critic,
        _envs(),
        TD3Config(
            updates=1,
            rollout_steps=48,
            gradient_steps=8,
            minibatch_size=16,
            policy_freq=2,
            seed=2,
            log_every=0,
        ),
    )
    after_a = list(actor.parameters())
    after_c = list(critic.parameters())
    assert any((a - b).abs().sum().item() > 0 for a, b in zip(after_a, before_a, strict=True))
    assert any((a - b).abs().sum().item() > 0 for a, b in zip(after_c, before_c, strict=True))


def test_polyak_update_moves_target_toward_online():
    """Direct, deterministic check of the Polyak-averaging math TD3's delayed
    target-network updates rely on: target <- (1-tau)*target + tau*online."""
    import torch
    from torch import nn

    from cat.rl.td3 import _polyak_update

    target = nn.Linear(3, 3, bias=False)
    online = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        target.weight.fill_(0.0)
        online.weight.fill_(1.0)

    _polyak_update(target, online, tau=0.1)
    assert torch.allclose(target.weight, torch.full((3, 3), 0.1))

    _polyak_update(target, online, tau=0.1)
    assert torch.allclose(target.weight, torch.full((3, 3), 0.19), atol=1e-6)
