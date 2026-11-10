"""The evaluation path: an RL-shaped checkpoint round-trips through
``load_translator``, and ``evaluate_problem`` produces the headline
translated/lossy/cold comparison on a real BBOB problem.
"""

import math

import torch

from algorithm_translation.cli.evaluate import evaluate_problem, translate_single
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.state.canonical import to_canonical
from algorithm_translation.suite import IOHSuite, build_problem_ids
from algorithm_translation.train_loop import load_translator, translator_meta


def _pair():
    # The actor's TranslatorPair is exactly what train_ppo.py checkpoints.
    return ActorCritic("PSO", "CMAES", hidden=32, n_layers=1).translator


def test_rl_checkpoint_roundtrips_through_load_translator(tmp_path, pso_native):
    pair = _pair()
    # Mirror the checkpoint train_ppo.py writes: translator weights + arch meta.
    ckpt = {"state_dict": pair.state_dict(), **translator_meta(pair)}
    path = tmp_path / "PSO_CMAES_ppo.pt"
    torch.save(ckpt, path)

    loaded = load_translator(str(path))
    assert (loaded.algo_a, loaded.algo_b) == ("PSO", "CMAES")

    # Weights survive the round trip bit-for-bit (the meta rebuilt the arch).
    ref, got = pair.state_dict(), loaded.state_dict()
    assert ref.keys() == got.keys()
    for k in ref:
        assert torch.equal(ref[k].cpu(), got[k].cpu()), k

    # The reloaded translator produces a usable target state.
    out = translate_single(loaded, to_canonical("PSO", pso_native), "CMAES")
    assert out.algo == "CMAES"
    assert torch.isfinite(out.positions).all()


def test_evaluate_problem_reports_three_handoffs():
    pair = _pair()
    pid = build_problem_ids({1}, dims=[2], instances=[1])[0]
    row = evaluate_problem(
        pair,
        "PSO",
        "CMAES",
        pid,
        IOHSuite(),
        fe_multiplier=500,
        n_individuals=12,
        seed=0,
        problem_idx=0,
    )
    assert row is not None  # PSO builds a full state over this budget
    for k in ("best_translated", "best_lossy", "best_cold", "best_after_warmup"):
        assert k in row and math.isfinite(float(row[k])), k
