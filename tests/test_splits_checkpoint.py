"""Generalization scenarios (lopo / loio) keep train and test disjoint, and the
trainers autosave + evaluate checkpoints on the held-out split only."""

import argparse
import sys

import pytest
import torch

from algorithm_translation.cli.scenario import add_scenario_args, resolve_test_ids
from algorithm_translation.suite.bbob_splits import (
    ALL_FUNCTIONS,
    EASY_TRAIN_FUNCTIONS,
    scenario_split,
)


def _parts(pid):
    f, i, d = pid.split("_")[1:]
    return int(f[1:]), int(i[1:]), int(d[1:])


def test_lopo_holds_out_functions():
    train, test = scenario_split("lopo", [2, 3], split="easy")
    train_f = {_parts(p)[0] for p in train}
    test_f = {_parts(p)[0] for p in test}
    assert train_f == EASY_TRAIN_FUNCTIONS
    assert test_f == ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS
    assert not train_f & test_f


def test_lopo_random_is_seeded_and_disjoint():
    a = scenario_split("lopo", [2], split="random", seed=3)
    b = scenario_split("lopo", [2], split="random", seed=3)
    assert a == b
    assert not {_parts(p)[0] for p in a[0]} & {_parts(p)[0] for p in a[1]}


def test_loio_holds_out_instances():
    train, test = scenario_split("loio", [2], train_instances=[1, 2], test_instances=[3])
    assert {_parts(p)[0] for p in train} == {_parts(p)[0] for p in test} == ALL_FUNCTIONS
    assert {_parts(p)[1] for p in train} == {1, 2}
    assert {_parts(p)[1] for p in test} == {3}


def test_loio_overlapping_instances_rejected():
    with pytest.raises(ValueError):
        scenario_split("loio", [2], train_instances=[1, 2], test_instances=[2, 3])


def test_eval_drops_training_problems():
    p = argparse.ArgumentParser()
    add_scenario_args(p, for_eval=True)
    # Ask for a loio set whose test instances overlap the checkpoint's train set.
    args = p.parse_args(["--scenario", "loio", "-d", "2", "--instances", "9",
                         "--test-instances", "1", "2", "--functions", "1"])
    ckpt_split = {"scenario": "loio", "train_ids": ["bbob_f001_i01_d02"],
                  "test_ids": ["bbob_f001_i02_d02"]}
    assert resolve_test_ids(args, ckpt_split) == ["bbob_f001_i02_d02"]


def test_ppo_autosaves_and_evaluates_heldout(tmp_path, monkeypatch):
    from algorithm_translation.cli import train_ppo

    out = tmp_path / "m.pt"
    argv = [
        "train", "PSO", "CMAES", "-d", "2", "--scenario", "lopo",
        "--functions", "1", "2", "4", "6", "--instances", "1",
        "--fe-multiplier", "200", "--n-individuals", "12", "--fixed-order",
        "--updates", "2", "--rollout-steps", "16", "--minibatch-size", "8",
        "--ppo-epochs", "1", "--hidden", "16", "--n-layers", "1",
        "--save-every", "1", "--eval-max-problems", "1", "--device", "cpu",
        "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_ppo.main()

    for name in ("m_upd00001.pt", "m_upd00002.pt", "m.pt"):
        assert (tmp_path / name).exists(), name
    ck = torch.load(tmp_path / "m_upd00002.pt", weights_only=False)
    split = ck["split"]
    assert split["scenario"] == "lopo"
    assert {_parts(p)[0] for p in split["train_ids"]} == {4, 6}
    assert {_parts(p)[0] for p in split["test_ids"]} == {1, 2}
    ev = ck["eval"]
    assert ev["update"] == 2
    assert ev["eval/PSO_to_CMAES/n_problems"] == 1
    assert "eval/PSO_to_CMAES/win_rate/translated_vs_lossy" in ev
    assert "eval/CMAES_to_PSO/n_problems" not in ev  # --fixed-order: one direction
