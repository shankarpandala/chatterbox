"""Step 4 (library) - dataset over pre-computed features + T3 conditioning.

Training uses ``batch_size=1`` because the custom ``T3.forward`` concatenates
[cond | text | speech] without an attention mask, so naive padding to a batch
max would let speech attend to text padding. Effective batch size is recovered
with gradient accumulation (see train.py). For large-GPU multi-sample training
you would add attention masking to T3.forward; that is out of scope here.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, index_path: str):
        with open(index_path, encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        if not self.items:
            raise RuntimeError(f"No features found in {index_path}. Run precompute_features first.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.items[i]["feature_path"])
        return {
            "text_tokens": torch.from_numpy(d["text_tokens"]).long(),
            "speech_tokens": torch.from_numpy(d["speech_tokens"]).long(),
            "cond_tokens": torch.from_numpy(d["cond_tokens"]).long(),
            "speaker_emb": torch.from_numpy(d["speaker_emb"]).float(),
        }


def collate_single(batch):
    """batch_size=1 collate: pass the single example straight through."""
    assert len(batch) == 1, "training/dataset only supports batch_size=1 (see module docstring)"
    return batch[0]


def build_t3_cond(sample: dict, device: str, exaggeration: float = 0.5):
    """Build a (batched, B=1) T3Cond from a feature sample."""
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    return T3Cond(
        speaker_emb=sample["speaker_emb"].unsqueeze(0),                 # (1, 256)
        cond_prompt_speech_tokens=sample["cond_tokens"].unsqueeze(0),    # (1, Lp)
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    ).to(device=device)
