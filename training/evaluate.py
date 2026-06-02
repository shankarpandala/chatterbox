"""Step 7 - synthesize eval sentences and score the fine-tuned model.

Loads the exported model dir, generates the (pure-script + code-switched) eval
sentences, writes WAVs, and reports speaker similarity (VoiceEncoder cosine vs a
reference voice). Optionally computes CER with an ASR model if you pass --asr.

    python -m training.evaluate --config training/configs/tamil.yaml \
        --ref reference_voice.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio as ta

from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from training.common import get_device, load_config


def _cer(ref: str, hyp: str) -> float:
    # Levenshtein distance over characters, normalized by reference length.
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0
    dp = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, dp[0] = dp[0], i
        for j, hc in enumerate(hyp, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (rc != hc))
    return dp[len(hyp)] / len(ref)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", default=None, help="defaults to cfg.paths.export")
    p.add_argument("--ref", default=None, help="reference voice wav (cloning + similarity)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--asr", default=None, help="optional HF Whisper id, e.g. openai/whisper-small")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device or cfg.device)
    model_dir = args.model_dir or cfg.paths.export
    t3_name = f"t3_mtl_{cfg.language}.safetensors"

    model = ChatterboxMultilingualTTS.from_local(model_dir, device, t3_model=t3_name)
    out_dir = Path(args.out_dir or (Path(model_dir) / "samples"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_emb = None
    if args.ref:
        ref16, _ = librosa.load(args.ref, sr=16000)
        ref_emb = model.ve.embeds_from_wavs([ref16], sample_rate=16000).mean(axis=0)

    asr = None
    if args.asr:
        try:
            from transformers import pipeline
            asr = pipeline("automatic-speech-recognition", model=args.asr, device=0 if device == "cuda" else -1)
        except Exception as e:
            print(f"[evaluate] ASR unavailable ({e}); skipping CER.")

    print(f"[evaluate] model_dir={model_dir} | device={device} | lang={cfg.language}")
    for i, text in enumerate(cfg.eval_sentences):
        wav = model.generate(text, language_id=cfg.language, audio_prompt_path=args.ref)
        wav_path = out_dir / f"sample_{i:02d}.wav"
        ta.save(str(wav_path), wav, model.sr)

        line = f"[{i:02d}] {wav_path.name}"
        if ref_emb is not None:
            gen16 = librosa.resample(wav.squeeze(0).numpy(), orig_sr=model.sr, target_sr=16000)
            gen_emb = model.ve.embeds_from_wavs([gen16], sample_rate=16000).mean(axis=0)
            line += f" | spk_sim={float(VoiceEncoder.voice_similarity(gen_emb, ref_emb)):.3f}"
        if asr is not None:
            hyp = asr(str(wav_path))["text"]
            line += f" | CER={_cer(text, hyp):.3f}"
        print(line + f'  "{text[:48]}"')

    print(f"[evaluate] wrote {len(cfg.eval_sentences)} samples -> {out_dir}")


if __name__ == "__main__":
    main()
