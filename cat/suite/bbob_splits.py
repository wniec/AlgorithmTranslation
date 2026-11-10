"""BBOB problem-set constants and train/test split helpers.

Vendored (and trimmed) from DynamicAlgorithmSelection2 (das/env/bbob_splits.py).
"""

from itertools import product

import numpy as np

ALL_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def build_problem_ids(
    functions: set[int],
    dims: list[int],
    instances: list[int] | None = None,
) -> list[str]:
    insts = instances if instances is not None else INSTANCE_IDS
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in product(insts, sorted(functions), dims)
    ]


def get_train_test_split(mode: str, dims: list[int]) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) for the given split mode and dimensions.

    Modes:
      easy   - train on easy BBOB functions, test on hard ones
      hard   - inverse of easy
      random - random 2/3 / 1/3 split on all problem IDs
    """
    if mode == "easy":
        return (
            build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
            build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
        )
    if mode == "hard":
        return (
            build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
            build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
        )
    # random 2/3 - 1/3 split
    all_ids = build_problem_ids(ALL_FUNCTIONS, dims)
    rng = np.random.default_rng()
    rng.shuffle(all_ids)
    split = 2 * len(all_ids) // 3
    return all_ids[:split], all_ids[split:]


# --------------------------------------------------------------------------- #
# Generalization scenarios                                                    #
# --------------------------------------------------------------------------- #

SCENARIOS = ("lopo", "loio")
DEFAULT_TRAIN_INSTANCES = [1, 2, 3]


def split_functions(split: str, seed: int = 0) -> tuple[set[int], set[int]]:
    """Partition the 24 BBOB functions into disjoint (train, test) sets.

    ``easy``: train on EASY_TRAIN_FUNCTIONS, test on the rest; ``hard``: the
    inverse; ``random``: a seeded 2/3 - 1/3 random partition.
    """
    if split == "easy":
        return set(EASY_TRAIN_FUNCTIONS), ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS
    if split == "hard":
        return ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, set(EASY_TRAIN_FUNCTIONS)
    if split == "random":
        funcs = sorted(ALL_FUNCTIONS)
        np.random.default_rng(seed).shuffle(funcs)
        cut = 2 * len(funcs) // 3
        return set(funcs[:cut]), set(funcs[cut:])
    raise ValueError(f"unknown function split {split!r}")


def scenario_split(
    scenario: str,
    dims: list[int],
    *,
    split: str = "easy",
    functions: list[int] | set[int] | None = None,
    train_instances: list[int] | None = None,
    test_instances: list[int] | None = None,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Return disjoint (train_ids, test_ids) for a generalization scenario.

    * ``lopo`` (leave-some-problems-out): the BBOB *functions* are partitioned by
      ``split`` (easy/hard/random); test problems are functions never seen in
      training. ``test_instances`` defaults to ``train_instances``.
    * ``loio`` (leave-some-instances-out): train and test use the same
      ``functions`` (default: all 24) but disjoint *instances*.
      ``test_instances`` defaults to every BBOB instance not trained on.

    Raises if the two sets would share a problem ID.
    """
    train_inst = list(train_instances or DEFAULT_TRAIN_INSTANCES)
    if scenario == "lopo":
        train_f, test_f = split_functions(split, seed)
        if functions is not None:  # optional restriction of the pool
            train_f &= set(functions)
            test_f &= set(functions)
        test_inst = list(test_instances or train_inst)
    elif scenario == "loio":
        train_f = test_f = set(functions) if functions else set(ALL_FUNCTIONS)
        test_inst = list(
            test_instances or [i for i in INSTANCE_IDS if i not in train_inst]
        )
        if set(train_inst) & set(test_inst):
            raise ValueError(
                f"loio: train instances {train_inst} and test instances "
                f"{test_inst} overlap"
            )
    else:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    train_ids = build_problem_ids(train_f, dims, train_inst)
    test_ids = build_problem_ids(test_f, dims, test_inst)
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"{scenario}: train/test overlap on {sorted(overlap)[:5]}...")
    if not train_ids or not test_ids:
        raise ValueError(f"{scenario}: empty train ({len(train_ids)}) or test "
                         f"({len(test_ids)}) problem set")
    return train_ids, test_ids
