"""Step 1 - build a unified training manifest.

Reads one source at a time and appends rows to ``paths.manifest`` (JSONL), each:

    {"audio_path": "/abs/clip.wav", "text": "...", "lang": "ta", "speaker": "spkA"}

Supported ``--source-type`` values:

  ljspeech : a metadata file ("id<sep>text" or "id<sep>text<sep>speaker") + audio dir
             (covers LJSpeech, IIT-Madras IndicTTS, most studio corpora)
  csv      : a CSV with header; choose the audio/text/speaker columns
  hf       : a HuggingFace `datasets` dataset (audio decoded to wav under work_dir)

Run once per source (use --append for the 2nd, 3rd, ... source). Example mix for
a Tamil pilot: IIT-M IndicTTS (ljspeech) + OpenSLR SLR65 (csv) + IndicVoices-R
(hf) + IndicVoices conversational / MUCS code-switch (hf or ljspeech).

    python -m training.prepare_data --config training/configs/tamil.yaml \
        --source-type ljspeech --audio-dir /data/indictts/ta/wav \
        --metadata /data/indictts/ta/txt.done.data --speaker indictts_f
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from training.common import load_config


def _norm_text(text: str) -> str:
    # Light normalization; script-specific canonicalization happens in the
    # tokenizer at encode time. Keep code-switched English as-is.
    return " ".join(str(text).split()).strip()


def _append_rows(manifest_path: Path, rows, lang: str):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(manifest_path, "a", encoding="utf-8") as f:
        for audio_path, text, speaker in rows:
            text = _norm_text(text)
            if not text:
                continue
            if not os.path.isfile(audio_path):
                print(f"[prepare_data] skip missing audio: {audio_path}")
                continue
            f.write(json.dumps({
                "audio_path": os.path.abspath(audio_path),
                "text": text,
                "lang": lang,
                "speaker": str(speaker),
            }, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def _from_ljspeech(args):
    sep = args.sep
    audio_dir = Path(args.audio_dir)
    with open(args.metadata, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < 2:
                continue
            uid, text = parts[0].strip(), parts[1]
            speaker = parts[2].strip() if len(parts) > 2 else args.speaker
            # Allow uid with or without extension.
            cand = audio_dir / uid
            if not cand.suffix:
                cand = audio_dir / f"{uid}{args.ext}"
            yield str(cand), text, speaker


def _from_csv(args):
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=args.sep if args.sep != "|" else ",")
        for row in reader:
            audio = row[args.audio_col]
            if not os.path.isabs(audio) and args.audio_dir:
                audio = os.path.join(args.audio_dir, audio)
            speaker = row.get(args.speaker_col, args.speaker) if args.speaker_col else args.speaker
            yield audio, row[args.text_col], speaker


def _from_hf(args, cfg):
    import soundfile as sf
    from datasets import load_dataset

    cache_dir = Path(cfg.paths.work_dir) / "audio_cache" / (args.hf_config or "default")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
    for i, ex in enumerate(ds):
        audio = ex[args.audio_col]
        if isinstance(audio, dict) and "array" in audio:  # decoded audio
            wav_path = cache_dir / f"{i:08d}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            audio_path = str(wav_path)
        else:  # a path-like
            audio_path = audio["path"] if isinstance(audio, dict) else str(audio)
        speaker = str(ex.get(args.speaker_col, args.speaker)) if args.speaker_col else args.speaker
        yield audio_path, ex[args.text_col], speaker


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--source-type", required=True, choices=["ljspeech", "csv", "hf"])
    p.add_argument("--append", action="store_true", help="append to an existing manifest")
    p.add_argument("--speaker", default="spk0", help="default speaker id for the source")
    # ljspeech / csv
    p.add_argument("--audio-dir", default=None)
    p.add_argument("--metadata", default=None)
    p.add_argument("--ext", default=".wav")
    p.add_argument("--sep", default="|")
    # csv
    p.add_argument("--csv", default=None)
    p.add_argument("--audio-col", default="audio")
    p.add_argument("--text-col", default="text")
    p.add_argument("--speaker-col", default=None)
    # hf
    p.add_argument("--hf-dataset", default=None)
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default="train")
    args = p.parse_args()

    cfg = load_config(args.config)
    manifest = Path(cfg.paths.manifest)
    if manifest.exists() and not args.append:
        raise SystemExit(
            f"{manifest} already exists. Pass --append to add another source, "
            f"or delete it to start fresh."
        )

    if args.source_type == "ljspeech":
        assert args.audio_dir and args.metadata, "ljspeech needs --audio-dir and --metadata"
        rows = _from_ljspeech(args)
    elif args.source_type == "csv":
        assert args.csv, "csv needs --csv"
        rows = _from_csv(args)
    else:
        assert args.hf_dataset, "hf needs --hf-dataset"
        rows = _from_hf(args, cfg)

    kept = _append_rows(manifest, rows, cfg.language)
    total = sum(1 for _ in open(manifest, encoding="utf-8"))
    print(f"[prepare_data] added {kept} utterances from {args.source_type}. Manifest now: {total} rows -> {manifest}")


if __name__ == "__main__":
    main()
