"""Shared CLI argument groups and env/logger construction for train_ppo.py and
train_td3.py — the two trainers train the same ``TranslationEnv`` (see
``algorithm_translation/rl/env.py``), so problem-set/episode configuration, W&B logging, and
device/seed/output plumbing are identical between them; only the network and
algorithm hyperparameters differ (kept in each script)."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from algorithm_translation.cli import ALGO_CHOICES
from algorithm_translation.cli.evaluate import evaluate_heldout
from algorithm_translation.cli.scenario import add_scenario_args, split_from_args
from algorithm_translation.rl.env import TranslationEnv
from algorithm_translation.suite import IOHSuite
from algorithm_translation.tracking import WandbLogger


def add_algo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("algo_a", choices=ALGO_CHOICES)
    p.add_argument("algo_b", choices=ALGO_CHOICES)


def add_env_args(p: argparse.ArgumentParser) -> None:
    add_scenario_args(p)
    p.add_argument("--fe-multiplier", type=int, default=2000)
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
        "log-uniform; >1 moves the switch earlier; 0<cdb<1 later.",
    )
    p.add_argument(
        "--fixed-order",
        action="store_true",
        help="always switch algo_a -> algo_b; by default each episode's "
        "direction is drawn at random",
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
        help="every mode is score(translated) - score(baseline), each run's "
        "segment scored by AOCC (or --endpoint-reward). "
        "'noswitch' (default) baseline = the source optimizer continuing "
        "(signed switch-worthiness: + if the switch beats staying, - if worse); "
        "'absolute' baseline = 0 (the translated run's own score); "
        "'relative' baseline = the "
        "hand-off available without a network (translation quality): the "
        "computed warm start into CMA-ES, the lossy default otherwise; "
        "'coldtarget' baseline = "
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


def add_checkpoint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="autosave the weights every N updates (epochs) and evaluate that "
        "checkpoint on the held-out test split; 0 disables",
    )
    p.add_argument(
        "--eval-max-problems",
        type=int,
        default=20,
        help="held-out problems scored per direction at each autosave",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=123,
        help="fixes the held-out subset, switch points and optimizer seeds, so "
        "every autosave is scored on the same episodes",
    )


def build_envs(args, train_ids: list[str]) -> list[TranslationEnv]:
    """One env per dimension, each walking only the *training* problems."""
    suite = IOHSuite()
    by_dim: dict[int, list[str]] = defaultdict(list)
    for pid in train_ids:
        by_dim[int(pid.rsplit("_d", 1)[1])].append(pid)
    envs = []
    for dim in args.dims:
        ids = by_dim[dim]
        if not ids:
            continue
        envs.append(
            TranslationEnv(
                args.algo_a,
                args.algo_b,
                ids,
                suite,
                fe_multiplier=args.fe_multiplier,
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


def build_split(args) -> tuple[list[str], list[str]]:
    """Resolve the lopo/loio split and report it."""
    train_ids, test_ids = split_from_args(args)
    print(
        f"scenario={args.scenario}"
        + (f" ({args.split})" if args.scenario == "lopo" else "")
        + f" | {len(train_ids)} train / {len(test_ids)} held-out problems"
    )
    return train_ids, test_ids


class Checkpointer:
    """Saves the weights every ``--save-every`` updates and evaluates that
    checkpoint on the held-out split, logging ``eval/*`` metrics to W&B.

    ``make_ckpt(train_log)`` returns the checkpoint dict to save (built by the
    trainer; ``train_log`` carries the running normalizers); ``translator`` is
    the live TranslatorPair being trained. Pass the instance as the training
    loop's ``on_update`` hook.
    """

    def __init__(self, args, out: str, make_ckpt, translator, test_ids, logger):
        self.args = args
        self.every = args.save_every
        self.stem, self.ext = os.path.splitext(out)
        self.make_ckpt = make_ckpt
        self.translator = translator
        self.test_ids = test_ids
        self.logger = logger
        n_a = args.n_individuals_a if args.n_individuals_a is not None else args.n_individuals
        n_b = args.n_individuals_b if args.n_individuals_b is not None else args.n_individuals
        self._n = {args.algo_a: n_a, args.algo_b: n_b}
        self.directions = [(args.algo_a, args.algo_b)]
        if not args.fixed_order:  # training sees both directions -> score both
            self.directions.append((args.algo_b, args.algo_a))

    def __call__(self, u: int, train_log) -> None:
        done = u + 1  # updates completed so far
        if not self.every or done % self.every:
            return
        path = f"{self.stem}_upd{done:05d}{self.ext or '.pt'}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = self.make_ckpt(train_log)
        torch.save(ckpt, path)

        t0 = time.time()
        metrics, _ = evaluate_heldout(
            self.translator,
            self.test_ids,
            self.directions,
            fe_multiplier=self.args.fe_multiplier,
            n_for=self._n.get,
            seed=self.args.eval_seed,
            max_problems=self.args.eval_max_problems,
            suite=IOHSuite(),
            reward_measure=getattr(self.args, "reward_measure", "aocc"),
            aocc_decades=getattr(self.args, "aocc_decades", 4.0),
        )
        metrics = {f"eval/{k}": v for k, v in metrics.items()}
        # Store the scores alongside the weights they belong to.
        ckpt["eval"] = {"update": done, **metrics}
        torch.save(ckpt, path)

        self.logger.log({**metrics, "eval/seconds": time.time() - t0}, step=u)
        head = ", ".join(
            f"{k.removeprefix('eval/')}={v:.3g}"
            for k, v in metrics.items()
            if "win_rate" in k or "log10_prec/translated" in k or "reward/" in k
        )
        tqdm.write(f"[checkpoint] update {done}: saved {path} | {head}")
