"""Train the state translator with GRPO (Group Relative Policy Optimization).

    uv run train-grpo PSO CMAES [options]

Same environment, reward and checkpoint format as PPO (``uv run train``). Each
episode has a single switch, so instead of a learned critic baseline every
switch state gets a *group* of ``--group-size`` sampled translations, scored from
that same state under the same post-switch seeds; each candidate's advantage is
its reward relative to the group (mean-centered, divided by the group std unless
``--no-adv-std-scale``). The policy update is PPO's clipped surrogate, without a
critic. Shared CLI/env-building logic lives in algorithm_translation/cli/common.py.

Saves a checkpoint compatible with `uv run eval` (`models/<A>_<B>_grpo.pt`).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from algorithm_translation.cli import make_parser
from algorithm_translation.cli.common import (
    Checkpointer,
    add_algo_args,
    add_checkpoint_args,
    add_env_args,
    add_logging_args,
    add_runtime_args,
    build_envs,
    build_logger,
    build_split,
)
from algorithm_translation.cli.scenario import split_record
from algorithm_translation.rl.grpo import GRPOConfig, train_grpo
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.train_loop import resolve_device, translator_meta


def parse_args():
    p = make_parser(description=__doc__)
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
        help="log_std gets its own Adam param group at lr * std-lr-mult",
    )
    p.add_argument(
        "--resample-seed",
        type=int,
        default=None,
        help="seed for the population-resizing jitter (when n-individuals-a/b differ)",
    )
    # GRPO
    p.add_argument("--updates", type=int, default=100)
    p.add_argument(
        "--groups-per-update",
        type=int,
        default=128,
        help="switch states (episodes) sampled per update",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="candidate translations sampled and scored per switch state",
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument(
        "--no-adv-std-scale",
        dest="scale_adv_by_std",
        action="store_false",
        help="only mean-center rewards within a group (Dr. GRPO); by default "
        "they are also divided by the group's std (standard GRPO)",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=">1 collects groups in that many worker processes",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    add_runtime_args(p)
    add_checkpoint_args(p)
    add_logging_args(p)
    args = p.parse_args()
    if args.group_size < 2:
        p.error("--group-size must be >= 2 (advantages are relative to the group)")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = resolve_device(args.device)

    train_ids, test_ids = build_split(args)
    envs = build_envs(args, train_ids)
    print(
        f"GRPO training {args.algo_a}<->{args.algo_b} | dims={args.dims} "
        f"| {len(envs)} env(s) | group {args.group_size} x {args.groups_per_update} "
        f"| device={args.device}"
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
    cfg = GRPOConfig(
        updates=args.updates,
        groups_per_update=args.groups_per_update,
        group_size=args.group_size,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        clip=args.clip,
        ent_coef=args.ent_coef,
        lr=args.lr,
        std_lr_mult=args.std_lr_mult,
        scale_adv_by_std=args.scale_adv_by_std,
        device=args.device,
        seed=args.seed,
        n_workers=args.n_workers,
    )
    out = args.out or os.path.join("models", f"{args.algo_a}_{args.algo_b}_grpo.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    split = split_record(args, train_ids, test_ids)

    def make_ckpt(_log=None) -> dict:
        # Same layout as the PPO checkpoint, so `uv run eval` loads it directly.
        return {
            "state_dict": ac.translator.state_dict(),
            "actor_critic_state_dict": ac.state_dict(),
            **translator_meta(ac.translator),
            "split": split,
            "trainer": "grpo",
        }

    logger = build_logger(args, "grpo")
    checkpointer = Checkpointer(args, out, make_ckpt, ac.translator, test_ids, logger)
    train_grpo(ac, envs, cfg, log_fn=logger.log, on_update=checkpointer)
    logger.finish()
    torch.save(make_ckpt(), out)
    print(f"saved GRPO translator to {out}")


if __name__ == "__main__":
    main()
