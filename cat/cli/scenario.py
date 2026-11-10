"""Train/test generalization scenarios shared by the trainers and evaluators.

Every run is either

* **lopo** — leave-some-problems-out: test BBOB *functions* never trained on;
* **loio** — leave-some-instances-out: same functions, held-out *instances*.

The trainers build their envs from the train IDs only, evaluate checkpoints on
the test IDs, and store both lists in the checkpoint (``ckpt["split"]``) so the
standalone evaluators default to exactly that held-out set and never score a
problem the model was trained on.
"""

from __future__ import annotations

import argparse

from cat.suite.bbob_splits import SCENARIOS, scenario_split


def add_scenario_args(p: argparse.ArgumentParser, *, for_eval: bool = False) -> None:
    """Add the split flags. With ``for_eval`` the scenario/dims default to None
    (meaning: reuse the split stored in the checkpoint)."""
    p.add_argument(
        "-d", "--dims", type=int, nargs="+", default=None if for_eval else [2, 3, 5]
    )
    p.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None if for_eval else "lopo",
        help="lopo = leave-some-problems-out (held-out BBOB functions); "
        "loio = leave-some-instances-out (held-out instances of the same "
        "functions)."
        + (" Default: the split stored in the checkpoint." if for_eval else ""),
    )
    p.add_argument(
        "--split",
        choices=["easy", "hard", "random"],
        default="easy",
        help="lopo only: how functions are partitioned (easy = train on the "
        "easy set, test on the rest; hard = inverse; random = seeded 2/3-1/3)",
    )
    p.add_argument(
        "--functions",
        type=int,
        nargs="+",
        default=None,
        help="loio: the function pool (default all 24); lopo: restrict both sides to these",
    )
    p.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=None,
        help="training instances (default 1 2 3)",
    )
    p.add_argument(
        "--test-instances",
        type=int,
        nargs="+",
        default=None,
        help="held-out instances (lopo default: = --instances; loio default: "
        "every BBOB instance not in --instances)",
    )
    p.add_argument(
        "--split-seed", type=int, default=0, help="seed for --split random"
    )


def split_from_args(args) -> tuple[list[str], list[str]]:
    return scenario_split(
        args.scenario,
        args.dims or [2, 3, 5],
        split=args.split,
        functions=args.functions,
        train_instances=args.instances,
        test_instances=args.test_instances,
        seed=args.split_seed,
    )


def split_record(args, train_ids, test_ids) -> dict:
    """What gets stored in a checkpoint under ``"split"``."""
    return {
        "scenario": args.scenario,
        "split": args.split if args.scenario == "lopo" else None,
        "dims": list(args.dims),
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
    }


def _dim_of(pid: str) -> int:
    return int(pid.rsplit("_d", 1)[1])


def exclude_train_ids(ids, ckpt_split: dict | None) -> list[str]:
    """Drop any problem the checkpoint was trained on (warns if any dropped)."""
    if not ckpt_split:
        return list(ids)
    train = set(ckpt_split.get("train_ids", ()))
    kept = [i for i in ids if i not in train]
    if len(kept) < len(ids):
        print(
            f"[split] dropped {len(ids) - len(kept)} problem(s) the model was "
            f"trained on ({ckpt_split.get('scenario')})"
        )
    if not kept:
        raise SystemExit("no held-out problems left to evaluate")
    return kept


def resolve_test_ids(args, ckpt_split: dict | None) -> list[str]:
    """Held-out problems for a standalone evaluator.

    No ``--scenario`` and the checkpoint stores a split -> its test IDs (filtered
    to ``--dims`` if given). Otherwise the split is rebuilt from the flags. In
    both cases anything in the checkpoint's train set is removed.
    """
    if args.scenario is None and ckpt_split:
        ids = list(ckpt_split["test_ids"])
        if args.dims:
            ids = [i for i in ids if _dim_of(i) in set(args.dims)]
        print(f"[split] using checkpoint's {ckpt_split['scenario']} test set")
    else:
        if args.scenario is None:
            args.scenario = "lopo"
            print("[split] checkpoint has no stored split; defaulting to lopo/easy")
        _, ids = split_from_args(args)
    return exclude_train_ids(ids, ckpt_split)
