"""Shared CLI argument groups and env/logger construction for train_ppo.py and
train_td3.py — the two trainers train the same ``TranslationEnv`` (see
``cat/rl/env.py``), so problem-set/episode configuration, W&B logging, and
device/seed/output plumbing are identical between them; only the network and
algorithm hyperparameters differ (kept in each script)."""

from __future__ import annotations

import argparse

from cat.optimizers.portfolio import PORTFOLIO
from cat.rl.env import TranslationEnv
from cat.suite import IOHSuite
from cat.suite.bbob_splits import ALL_FUNCTIONS, EASY_TRAIN_FUNCTIONS, build_problem_ids
from cat.tracking import WandbLogger


def add_algo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("algo_a", choices=sorted(PORTFOLIO))
    p.add_argument("algo_b", choices=sorted(PORTFOLIO))


def add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-d", "--dims", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--split", choices=["easy", "all"], default="easy")
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument("--n-switches", type=int, default=6)
    p.add_argument(
        "--n-individuals",
        type=int,
        default=None,
        help="shared population size for both algos; omit (with -a/-b also "
        "unset) to let each algorithm use its own built-in default",
    )
    p.add_argument(
        "--n-individuals-a",
        type=int,
        default=None,
        help="override --n-individuals for algo_a only",
    )
    p.add_argument(
        "--n-individuals-b",
        type=int,
        default=None,
        help="override --n-individuals for algo_b only",
    )
    p.add_argument(
        "--switch-cdb",
        type=float,
        default=1.0,
        help="base of the switch-point log-FE warp: 1.0 (default) = plain "
        "log-uniform; >1 concentrates switches earlier; 0<cdb<1 later.",
    )
    p.add_argument(
        "--fixed-order",
        action="store_true",
        help="always start episodes with algo_a (A, B, A, ...); by default the "
        "starting algorithm is drawn at random per episode",
    )
    p.add_argument(
        "--endpoint-reward",
        dest="reward_measure",
        action="store_const",
        const="endpoint",
        default="aocc",
        help="score each segment by the best-so-far improvement at its boundary "
        "only (the original measure). Default is 'aocc' (Area Over the "
        "Convergence Curve), which rewards converging faster/lower over the "
        "whole segment — so the hand-off's transient value survives a long "
        "single segment instead of being washed out by the target's adaptation.",
    )
    p.add_argument(
        "--aocc-decades",
        type=float,
        default=4.0,
        help="AOCC log-precision span: how many decades of the gap-at-switch map "
        "onto [0, 1] (floor g_lo = g_hi * 10**-decades). Only used by the "
        "default 'aocc' measure.",
    )
    p.add_argument(
        "--reward-seeds",
        type=int,
        default=1,
        help="average the per-step reward over this many optimizer seeds "
        "(paired per seed for common random numbers). 1 (default) = original "
        "single-sample reward. The single-segment reward is a high-variance "
        "difference of stochastic runs, so K>1 cuts reward noise ~1/K — "
        "denoising both the critic target and the policy gradient — at ~Kx the "
        "counterfactual optimizer cost (offset by --n-workers). Seed 0 is the "
        "run that advances the episode.",
    )
    p.add_argument(
        "--reward-mode",
        default="noswitch",
        help="every mode is log_scale(translated) - log_scale(baseline). "
        "'noswitch' (default) baseline = the source optimizer continuing "
        "(signed switch-worthiness: + if the switch beats staying, - if worse); "
        "'absolute' baseline = 0 (raw improvement); 'relative' baseline = the "
        "lossy default hand-off (translation quality); 'coldtarget' baseline = "
        "the target (second) optimizer run cold from the start (beat running "
        "the second algorithm alone). Pass a comma-separated list "
        "(e.g. 'noswitch,relative,coldtarget') to sum several modes' rewards; "
        "'mixed' is the alias for 'noswitch,relative'. Each non-absolute mode "
        "runs one extra counterfactual optimizer per step (~2x cost each).",
    )


def add_logging_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wandb", action="store_true", help="log metrics to Weights & Biases"
    )
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' uses CUDA when available else CPU; or pass cuda / cuda:0 / cpu / mps",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)


def build_envs(args) -> list[TranslationEnv]:
    funcs = EASY_TRAIN_FUNCTIONS if args.split == "easy" else ALL_FUNCTIONS
    suite = IOHSuite()
    envs = []
    for dim in args.dims:
        ids = build_problem_ids(funcs, [dim], args.instances)
        envs.append(
            TranslationEnv(
                args.algo_a,
                args.algo_b,
                ids,
                suite,
                fe_multiplier=args.fe_multiplier,
                n_switches=args.n_switches,
                n_individuals=args.n_individuals,
                n_individuals_a=args.n_individuals_a,
                n_individuals_b=args.n_individuals_b,
                reward_mode=args.reward_mode,
                switch_cdb=args.switch_cdb,
                random_start=not args.fixed_order,
                scale_feature=getattr(args, "scale_feature", False),
                use_ela=getattr(args, "use_ela", False),
                reward_seeds=getattr(args, "reward_seeds", 1),
                reward_measure=getattr(args, "reward_measure", "aocc"),
                aocc_decades=getattr(args, "aocc_decades", 4.0),
                seed=args.seed * 100 + dim,
            )
        )
    return envs


def build_logger(args, trainer: str) -> WandbLogger:
    return WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"{trainer}_{args.algo_a}_{args.algo_b}",
        group=f"{args.algo_a}_{args.algo_b}",
        config={**vars(args), "trainer": trainer},
        x_axis="update",
    )
