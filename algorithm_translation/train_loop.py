"""Shared helpers for building and (de)serializing the state translator.

The translator is trained by the RL trainers (``train_ppo.py`` / ``train_td3.py``);
this module holds the device resolution and checkpoint round-tripping they and
``evaluate.py`` share.
"""

from __future__ import annotations

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

    The RL trainers (train_ppo.py, train_td3.py) store the actor's TranslatorPair
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
