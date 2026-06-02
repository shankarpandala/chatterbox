"""Step 3 - pre-compute acoustic features (the key MPS / 24 GB enabler).

Runs the language-agnostic stack ONCE over the corpus and caches to disk:
  - target speech tokens        (S3Tokenizer)         -> the training target
  - speaker embedding           (VoiceEncoder)        -> T3 conditioning
  - cond-prompt speech tokens   (S3Tokenizer on a reference clip)
  - text token ids              (extended MTLTokenizer)

After this, train.py never touches audio, S3Gen, the VoiceEncoder or the
S3Tokenizer, so only T3 (+ LoRA) needs to live in memory during training.

    python -m training.precompute_features --config training/configs/tamil.yaml
    # add --device cpu if torch.stft misbehaves on MPS
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.voice_encoder import VoiceEncoder
from training.common import (
    download_base,
    get_device,
    iter_manifest,
    load_config,
    load_tokenizer,
    map_location_for,
)

PLEN = T3Config.multilingual().speech_cond_prompt_len  # 150 cond-prompt tokens
REF_SECONDS = 6  # cond-prompt reference length (mirrors ChatterboxMTL.ENC_COND_LEN)


def _pick_reference(rows, mode):
    """Return ref_index[i]: which utterance supplies the cond-prompt for utt i."""
    by_spk = defaultdict(list)
    for i, r in enumerate(rows):
        by_spk[r["speaker"]].append(i)
    ref = list(range(len(rows)))
    if mode == "self":
        return ref
    for _spk, idxs in by_spk.items():
        if len(idxs) == 1:
            continue
        # round-robin: each utt references the NEXT same-speaker utt (avoids leakage)
        for pos, i in enumerate(idxs):
            ref[i] = idxs[(pos + 1) % len(idxs)]
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None, help="override cfg.device (cpu is safest here)")
    p.add_argument("--limit", type=int, default=None, help="debug: only process N utts")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device or cfg.device)
    max_samples_16 = int(cfg.train.max_audio_seconds * S3_SR)
    max_samples_24 = int(cfg.train.max_audio_seconds * S3GEN_SR)

    base_dir = download_base(cfg)
    map_loc = map_location_for(device)

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(base_dir / "ve.pt", map_location=map_loc, weights_only=True))
    ve.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(base_dir / "s3gen.pt", map_location=map_loc, weights_only=True))
    s3gen.to(device).eval()
    s3_tok = s3gen.tokenizer  # the language-agnostic S3Tokenizer

    tokenizer, vocab_size = load_tokenizer(cfg.paths.tokenizer)
    print(f"[precompute] device={device}, tokenizer vocab={vocab_size}")

    rows = list(iter_manifest(cfg.paths.manifest))
    if args.limit:
        rows = rows[: args.limit]
    ref_index = _pick_reference(rows, cfg.precompute.reference)

    feat_dir = Path(cfg.paths.features)
    feat_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(cfg.paths.feature_index)

    # Cache resampled 16k reference audio lazily to avoid re-loading.
    kept, skipped = 0, 0
    with torch.inference_mode(), open(index_path, "w", encoding="utf-8") as idx:
        for i, row in enumerate(rows):
            try:
                wav24, _ = librosa.load(row["audio_path"], sr=S3GEN_SR)
                wav24 = wav24[:max_samples_24]
                wav16 = librosa.resample(wav24, orig_sr=S3GEN_SR, target_sr=S3_SR)[:max_samples_16]
                if wav16.shape[0] < S3_SR // 2:  # < 0.5s
                    skipped += 1
                    continue

                # speaker embedding (256,)
                spk = ve.embeds_from_wavs([wav16], sample_rate=S3_SR)
                spk = np.asarray(spk).mean(axis=0).astype(np.float32)

                # target speech tokens
                tok, tlen = s3_tok.forward([wav16])
                speech_tokens = tok[0, : int(tlen[0])].cpu().numpy().astype(np.int64)
                if speech_tokens.shape[0] < 5:
                    skipped += 1
                    continue

                # cond-prompt tokens from the reference clip
                rj = ref_index[i]
                if rj == i:
                    ref16 = wav16
                else:
                    ref16, _ = librosa.load(rows[rj]["audio_path"], sr=S3_SR, duration=REF_SECONDS + 1.0)
                ref16 = ref16[: REF_SECONDS * S3_SR]
                ctok, clen = s3_tok.forward([ref16], max_len=PLEN)
                cond_tokens = ctok[0, : int(clen[0])].cpu().numpy().astype(np.int64)

                # text token ids (with language tag; SOT/EOT added at train time)
                text_tokens = tokenizer.text_to_tokens(
                    row["text"], language_id=cfg.language
                )[0].numpy().astype(np.int64)
                if text_tokens.shape[0] < 1:
                    skipped += 1
                    continue

                fpath = feat_dir / f"{i:08d}.npz"
                np.savez(
                    fpath,
                    speech_tokens=speech_tokens,
                    cond_tokens=cond_tokens,
                    text_tokens=text_tokens,
                    speaker_emb=spk,
                )
                idx.write(json.dumps({
                    "feature_path": str(fpath),
                    "speaker": row["speaker"],
                    "n_speech": int(speech_tokens.shape[0]),
                    "n_text": int(text_tokens.shape[0]),
                }) + "\n")
                kept += 1
            except Exception as e:  # keep going on a bad file
                skipped += 1
                print(f"[precompute] skip {row['audio_path']}: {e}")

            if (i + 1) % 200 == 0:
                print(f"[precompute] {i + 1}/{len(rows)} (kept={kept}, skipped={skipped})")

    print(f"[precompute] done: kept={kept}, skipped={skipped} -> {index_path}")


if __name__ == "__main__":
    main()
