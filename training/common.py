"""Shared helpers for the language fine-tuning pipeline.

Everything here is intentionally dependency-light and reused across
``prepare_data``, ``extend_tokenizer``, ``precompute_features``, ``train``,
``merge_and_export`` and ``evaluate`` so the steps stay consistent.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

# Files that make up a complete multilingual checkpoint on the Hf Hub.
BASE_FILES = ["ve.pt", "s3gen.pt", "conds.pt", "Cangjie5_TC.json"]
DEFAULT_TOKENIZER_NAME = "grapheme_mtl_merged_expanded_v1.json"


# --------------------------------------------------------------------------- #
# Config / device / seeding
# --------------------------------------------------------------------------- #
def load_config(path: str):
    """Load a YAML config (via OmegaConf, already a chatterbox dependency)."""
    cfg = OmegaConf.load(path)
    return cfg


def iter_manifest(manifest_path: str) -> Iterator[dict]:
    """Yield rows of a JSONL manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str = "mps") -> str:
    """Resolve and validate the compute device, falling back gracefully."""
    requested = (requested or "").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("[common] MPS requested but unavailable - falling back to CPU.")
        return "cpu"
    if requested == "cuda":
        print("[common] CUDA requested but unavailable - falling back to CPU.")
        return "cpu"
    return "cpu" if requested in ("", "cpu") else requested


def map_location_for(device: str):
    # Mirror chatterbox: load CUDA-saved tensors to CPU first on cpu/mps.
    return torch.device("cpu") if device in ("cpu", "mps") else None


# --------------------------------------------------------------------------- #
# Base checkpoint download
# --------------------------------------------------------------------------- #
def download_base(cfg) -> Path:
    """Download the base multilingual checkpoint files and return their dir.

    Honours the ``HF_TOKEN`` env var. Only the files we need are fetched.
    """
    from huggingface_hub import snapshot_download

    allow = list(BASE_FILES) + [cfg.base_t3, DEFAULT_TOKENIZER_NAME]
    ckpt_dir = snapshot_download(
        repo_id=cfg.base_repo,
        repo_type="model",
        revision="main",
        allow_patterns=allow,
        token=os.getenv("HF_TOKEN"),
    )
    return Path(ckpt_dir)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(tokenizer_path: str) -> Tuple["object", int]:
    """Load an MTLTokenizer and return ``(tokenizer, vocab_size)``."""
    from chatterbox.models.tokenizers import MTLTokenizer

    tok = MTLTokenizer(str(tokenizer_path))
    return tok, tok.tokenizer.get_vocab_size()


# --------------------------------------------------------------------------- #
# T3 construction with warm start + vocab resize
# --------------------------------------------------------------------------- #
def build_warmstart_t3(base_dir: Path, base_t3: str, vocab_size: int, device: str = "cpu"):
    """Build a T3 sized for ``vocab_size`` and warm-start it from ``base_t3``.

    Overlapping parameters are copied verbatim. The text embedding and text head
    grow with the extended vocab; their first ``2454`` rows are copied from the
    pretrained checkpoint and the new rows keep their fresh initialization.

    Returns ``(t3, resized)`` where ``resized`` lists the keys whose shapes
    changed (for logging).
    """
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config

    t3 = T3(T3Config.multilingual(vocab_size))

    state = load_safetensors(str(Path(base_dir) / base_t3))
    if "model" in state.keys():  # some checkpoints wrap weights as {"model": [sd]}
        state = state["model"][0]

    model_sd = t3.state_dict()
    to_load = {}
    resized = []
    for k, v in state.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape == v.shape:
            to_load[k] = v
        else:
            # Copy the overlapping region (e.g. first 2454 vocab rows) into a
            # clone of the freshly-initialized parameter.
            new_t = model_sd[k].clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(new_t.shape, v.shape))
            new_t[slices] = v[slices]
            # The NEW (non-overlapping) text-embedding rows otherwise keep
            # nn.Embedding's default N(0,1) init — ~65x larger than the
            # pretrained rows (std ~0.015). Feeding ~66x-oversized vectors into
            # the RMSNorm backbone makes a short fine-tune spend its budget
            # renormalizing instead of learning the new tokens, so re-scale the
            # new rows to the pretrained embedding's std (mean already ~0).
            if k.endswith("text_emb.weight") and new_t.shape[0] > v.shape[0]:
                with torch.no_grad():
                    new_t[v.shape[0]:].normal_(0.0, float(v.std()))
            to_load[k] = new_t
            resized.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing, unexpected = t3.load_state_dict(to_load, strict=False)
    # `missing` are params not present in the base checkpoint (should be none
    # for a matching multilingual checkpoint); they keep their init.
    if missing:
        print(f"[common] warm-start: {len(missing)} params kept their init (not in base).")
    if unexpected:
        print(f"[common] warm-start: {len(unexpected)} unexpected base keys ignored.")
    t3.to(device)
    return t3, resized
