"""Train the state translator with PPO (reinforcement learning).

    python train_ppo.py PSO CMAES [options]

Each episode is a BBOB run that switches between the two named algorithms at
log-uniformly sampled points; at every switch the policy (the set-equivariant
translator) maps the source optimizer's internal state into the target's. The
reward is the true downstream best-so-far improvement over the next segment, and
the A->B->A cycle drift enters as a differentiable penalty — so the network
learns hand-offs that are both *useful* and *reversible*, optimizing the real
(non-differentiable) objective directly.

PPO's action is a stochastic Gaussian over the translator's decoded fields,
trained on-policy. See train_td3.py for an off-policy alternative that trains
the same continuous action deterministically with twin critics + a replay
buffer; shared CLI/env-building logic lives in cat/cli/common.py.

Saves a checkpoint compatible with evaluate.py (`models/<A>_<B>_ppo.pt`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cat.cli.common import (
    add_algo_args,
    add_env_args,
    add_logging_args,
    add_runtime_args,
    build_envs,
    build_logger,
)
from cat.rl.policy import ActorCritic
from cat.rl.ppo import PPOConfig, train_ppo
from cat.train_loop import resolve_device, translator_meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_algo_args(p)
    add_env_args(p)
    # network
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--cov-rank", type=int, default=4)
    p.add_argument("--log-std-init", type=float, default=0.0)
    p.add_argument(
        "--std-lr-mult",
        type=float,
        default=30.0,
        help="log_std gets its own Adam param group at lr * std-lr-mult, since "
        "it's a single global scalar that needs a much bigger step than the "
        "rest of the network to track its optimum in a reasonable number of updates",
    )
    p.add_argument(
        "--resample-seed",
        type=int,
        default=None,
        help="seed for the population-resizing jitter (when n-individuals-a/b differ)",
    )
    # PPO
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="parallel rollout worker processes (default 1 = single process). "
        ">1 splits each rollout across that many processes, each running its "
        "own envs/optimizers; the counterfactual optimizer runs in env.step are "
        "the bottleneck, so this scales collection roughly linearly on CPU.",
    )
    p.add_argument(
        "--buffer-capacity",
        type=int,
        default=8192,
        help="sliding-window rollout-buffer size; transitions are reused across "
        "~buffer_capacity/rollout_steps updates (default 8192 ~ 8 updates). Set "
        "equal to --rollout-steps for textbook single-use on-policy PPO.",
    )
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--gae-lambda", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=0.25)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument(
        "--vf-clip",
        type=float,
        default=None,
        help="value-function clip window, decoupled from --clip. Default: none "
        "(no value clipping — frees the critic to track wide returns). Pass 0.2 "
        "to reproduce the old throttle, or a larger value for a looser window.",
    )
    p.add_argument(
        "--scale-feature",
        dest="scale_feature",
        action="store_true",
        help="add a problem-scale channel (log10 of the initial optimum gap) to "
        "the critic context, so it can tell different-magnitude problems apart "
        "(raises the achievable explained variance).",
    )
    p.add_argument(
        "--use-ela",
        dest="use_ela",
        action="store_true",
        help="append 22 Exploratory Landscape Analysis features (pflacco), "
        "computed from the points evaluated up to each switch (capped at 500), "
        "to the critic context. Landscape descriptors correlate with the "
        "fractional improvement a problem allows — the between-problem signal "
        "that caps explained variance. Critic-side only; adds a pflacco call "
        "(~0.4s) per acted step, so prefer running with --n-workers.",
    )
    p.add_argument("--lambda-cycle", type=float, default=0.0)
    p.add_argument(
        "--cycle-mode",
        choices=["field", "latent"],
        default="field",
        help="score the A->B->A cycle penalty on decoded fields or re-encoded latents",
    )
    p.add_argument(
        "--no-norm-reward",
        dest="norm_reward",
        action="store_false",
        help="disable reward normalization (running std of the discounted return)",
    )
    p.add_argument(
        "--no-norm-obs",
        dest="norm_obs",
        action="store_false",
        help="disable observation (context) normalization",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    add_runtime_args(p)
    add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = resolve_device(args.device)

    envs = build_envs(args)
    print(
        f"PPO training {args.algo_a}<->{args.algo_b} | dims={args.dims} "
        f"| {len(envs)} env(s) | device={args.device}"
    )

    ac = ActorCritic(
        args.algo_a,
        args.algo_b,
        hidden=args.hidden,
        n_layers=args.n_layers,
        cov_rank=args.cov_rank,
        log_std_init=args.log_std_init,
        resample_seed=args.resample_seed,
        context_dim=envs[0].context_dim,
    )
    cfg = PPOConfig(
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        buffer_capacity=args.buffer_capacity,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        vf_clip=args.vf_clip,
        lambda_cycle=args.lambda_cycle,
        cycle_mode=args.cycle_mode,
        lr=args.lr,
        std_lr_mult=args.std_lr_mult,
        norm_reward=args.norm_reward,
        norm_obs=args.norm_obs,
        device=args.device,
        seed=args.seed,
        n_workers=args.n_workers,
    )
    logger = build_logger(args, "ppo")
    ppo_log = train_ppo(ac, envs, cfg, log_fn=logger.log)
    logger.finish()

    out = args.out or os.path.join("models", f"{args.algo_a}_{args.algo_b}_ppo.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Save the actor's TranslatorPair under "state_dict" so evaluate.py /
    # load_translator can consume it directly; keep the full actor-critic and the
    # running normalizer statistics too (for resuming / reproducibility).
    ckpt = {
        "state_dict": ac.translator.state_dict(),
        "actor_critic_state_dict": ac.state_dict(),
        **translator_meta(ac.translator),
    }
    if ppo_log.ret_rms is not None:
        ckpt["ret_rms"] = ppo_log.ret_rms.state_dict()
    if ppo_log.obs_rms is not None:
        ckpt["obs_rms"] = ppo_log.obs_rms.state_dict()
    torch.save(ckpt, out)
    print(f"saved PPO translator to {out}")


if __name__ == "__main__":
    main()
