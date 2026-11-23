"""Shared helpers for building and (de)serializing the state translator.

The translator is trained by the RL trainers (``train_ppo.py`` / ``train_grpo.py``);
this module holds the device resolution and checkpoint round-tripping they and
the evaluators share.
"""

from __future__ import annotations

import os

import torch

from algorithm_translation.models.translator import TranslatorPair


def resolve_device(name: str = "auto") -> str:
    """Resolve a device string. ``"auto"`` picks CUDA when available, else CPU.

    An explicit ``"cuda"`` that isn't available falls back to CPU with a warning
    (so a CUDA command line still runs on a CPU-only box). Apple MPS is only used
    when requested explicitly — several linalg ops we rely on (cholesky/eigh)
    have spotty MPS support.
    """
    if name and name != "auto":
        if name.startswith("cuda") and not torch.cuda.is_available():
            print(f"[device] {name!r} requested but CUDA is unavailable; using cpu")
            return "cpu"
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_path(model: str | None, algo_a: str, algo_b: str) -> str:
    """``model`` if given; otherwise the one checkpoint the trainers save by
    default for this pair (``models/<A>_<B>_{ppo,grpo}.pt``). Exits with a
    message when there is none, or both exist."""
    if model:
        return model
    candidates = [os.path.join("models", f"{algo_a}_{algo_b}_{t}.pt") for t in ("ppo", "grpo")]
    found = [c for c in candidates if os.path.exists(c)]
    if len(found) == 1:
        return found[0]
    reason = "none of" if not found else "several of"
    raise SystemExit(f"--model not given and {reason} {candidates} exist; pass --model")


def translator_meta(pair: TranslatorPair) -> dict:
    """Architecture hyperparameters needed to rebuild a TranslatorPair."""
    enc = pair.encoders[pair.algo_a]
    dec = pair.decoders[pair.algo_a]
    return {
        "algo_a": pair.algo_a,
        "algo_b": pair.algo_b,
        "hidden": enc.hidden,
        "n_layers": len(enc.attn),
        "cov_rank": dec.cov_rank,
    }


def load_translator(path: str, device="cpu") -> TranslatorPair:
    """Rebuild a TranslatorPair from an RL checkpoint.

    The RL trainers (train_ppo.py, train_grpo.py) store the actor's TranslatorPair
    under the ``state_dict`` key, so this loads either transparently.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    pair = TranslatorPair(
        ckpt["algo_a"],
        ckpt["algo_b"],
        hidden=ckpt.get("hidden", 64),
        n_layers=ckpt.get("n_layers", 2),
        cov_rank=ckpt.get("cov_rank", 4),
    )
    pair.load_state_dict(ckpt["state_dict"])
    pair.to(device)
    pair.eval()
    return pair


def load_checkpoint_split(path: str) -> dict | None:
    """The train/test split a checkpoint was trained under (``None`` for
    checkpoints saved before splits were recorded)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("split")
