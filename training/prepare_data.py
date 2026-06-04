"""Step 1 - build a unified training manifest.

Reads one source at a time and appends rows to ``paths.manifest`` (JSONL), each:

    {"audio_path": "/abs/clip.wav", "text": "...", "lang": "te", "speaker": "spkA"}

Supported ``--source-type`` values:

  hf       : a HuggingFace `datasets` dataset (audio decoded to wav under work_dir).
             Ungated + works out-of-the-box: google/fleurs (te_in) for a smoke test,
             ai4bharat/indicvoices_r (config Telugu) for scale.
  openslr  : auto-download + extract an OpenSLR set by id (e.g. 66 = Telugu) and
             parse its line_index.tsv. Ungated, CC-BY-SA-4.0, Mac-friendly size.
  ljspeech : a metadata file ("id<sep>text" or "id<sep>text<sep>speaker") + audio dir
             (covers LJSpeech, IIT-Madras IndicTTS, most local studio corpora)
  csv      : a CSV with header; choose the audio/text/speaker columns

Examples:

    # Pull EVERY source listed under `sources:` in the config into one manifest:
    python -m training.prepare_data --config training/configs/telugu.yaml --all

    # ...or one source at a time (use --append to add to an existing manifest):
    python -m training.prepare_data --config training/configs/telugu.yaml \
        --source-type hf --hf-dataset google/fleurs --hf-config te_in \
        --hf-split train --text-col transcription --audio-col audio --speaker fleurs
    python -m training.prepare_data --config training/configs/telugu.yaml \
        --append --source-type openslr --slr-id 66
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
import zipfile
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


def _iter_hf_dataset(ds, args, cache_dir):
    """Yield (audio_path, text, speaker) from a loaded `datasets` dataset.

    `datasets` >= 4 decodes audio through `torchcodec` (a heavy native dep that
    needs a matching FFmpeg/torch build, painful on Mac/MPS). We don't need its
    decoder: turn it off and decode the raw bytes ourselves with `soundfile`
    (already a dependency), which handles wav/flac/ogg out of the box. Decoded
    clips are cached as wav under `cache_dir` so later steps read plain files.
    """
    import io

    import soundfile as sf
    from datasets import Audio

    cache_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(ds.features.get(args.audio_col), Audio):
        ds = ds.cast_column(args.audio_col, Audio(decode=False))

    for i, ex in enumerate(ds):
        audio = ex[args.audio_col]
        wav_path = cache_dir / f"{i:08d}.wav"
        if isinstance(audio, dict) and audio.get("array") is not None:  # already decoded
            if not wav_path.exists():
                sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            audio_path = str(wav_path)
        elif isinstance(audio, dict) and audio.get("bytes") is not None:  # raw encoded bytes
            if not wav_path.exists():
                data, sr = sf.read(io.BytesIO(audio["bytes"]))
                sf.write(str(wav_path), data, sr)
            audio_path = str(wav_path)
        else:  # a path-like (decode disabled and only a path is available)
            audio_path = audio["path"] if isinstance(audio, dict) else str(audio)
        speaker = str(ex.get(args.speaker_col, args.speaker)) if args.speaker_col else args.speaker
        yield audio_path, ex[args.text_col], speaker


def _from_hf(args, cfg):
    from datasets import load_dataset

    cache_dir = Path(cfg.paths.work_dir) / "audio_cache" / (args.hf_config or "default")
    try:
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
    except Exception as e:
        # Some datasets (e.g. google/fleurs on older `datasets`) need this; newer
        # `datasets` removed the kwarg, so only retry with it when asked for.
        if "trust_remote_code" in str(e):
            ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, trust_remote_code=True)
        else:
            raise
    yield from _iter_hf_dataset(ds, args, cache_dir)


def _from_parquet(args, cfg):
    """Build rows straight from local parquet shard(s) via a glob.

    Lets you train on a PARTIALLY-downloaded HF parquet dataset: the normal
    loader refuses to resolve a split until every shard is present, but the
    shards already on disk are valid parquet, so we read them directly. Point
    ``--parquet-glob`` at the cached shards, e.g. (one line):

        ~/.cache/huggingface/hub/datasets--ai4bharat--indicvoices_r/snapshots/*/Telugu/train-*.parquet
    """
    import glob

    from datasets import load_dataset

    files = sorted(glob.glob(os.path.expanduser(args.parquet_glob)))
    if not files:
        raise SystemExit(f"parquet source: no files match {args.parquet_glob!r}")
    print(f"[prepare_data] parquet: {len(files)} shard(s) matched")
    cache_dir = Path(cfg.paths.work_dir) / "audio_cache" / (args.cache_name or "parquet")
    ds = load_dataset("parquet", data_files=files, split="train")
    yield from _iter_hf_dataset(ds, args, cache_dir)


# OpenSLR crowdsourced high-quality Indic sets (ungated, CC-BY-SA-4.0). Each entry:
#   slr_id: [(audio_zip, line_index_tsv, speaker_label), ...]
# The line-index TSVs are separate downloads from the zips and map: FileID<TAB>text.
OPENSLR_SETS = {
    66: [  # Telugu
        ("te_in_female.zip", "line_index_female.tsv", "slr66_f"),
        ("te_in_male.zip", "line_index_male.tsv", "slr66_m"),
    ],
    65: [  # Tamil
        ("ta_in_female.zip", "line_index_female.tsv", "slr65_f"),
        ("ta_in_male.zip", "line_index_male.tsv", "slr65_m"),
    ],
    64: [("mr_in_female.zip", "line_index.tsv", "slr64_f")],  # Marathi (female only)
    63: [  # Malayalam
        ("ml_in_female.zip", "line_index_female.tsv", "slr63_f"),
        ("ml_in_male.zip", "line_index_male.tsv", "slr63_m"),
    ],
}
OPENSLR_MIRRORS = ("https://www.openslr.org/resources", "https://us.openslr.org/resources")


def _download(url_paths, dest: Path):
    """Download the first reachable mirror URL to dest (skip if already present)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[prepare_data] cached {dest.name}")
        return
    last_err = None
    for url in url_paths:
        try:
            print(f"[prepare_data] downloading {url}")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # try the next mirror
            last_err = e
    raise SystemExit(f"Failed to download {dest.name}: {last_err}")


def _from_openslr(args, cfg):
    slr = int(args.slr_id)
    if slr not in OPENSLR_SETS:
        raise SystemExit(f"OpenSLR set {slr} not configured. Known: {sorted(OPENSLR_SETS)}")
    root = Path(cfg.paths.work_dir) / "openslr" / f"SLR{slr}"
    root.mkdir(parents=True, exist_ok=True)

    for zip_name, idx_name, speaker in OPENSLR_SETS[slr]:
        zip_path, idx_path = root / zip_name, root / f"{speaker}_{idx_name}"
        _download([f"{m}/{slr}/{zip_name}" for m in OPENSLR_MIRRORS], zip_path)
        _download([f"{m}/{slr}/{idx_name}" for m in OPENSLR_MIRRORS], idx_path)

        wav_dir = root / zip_name[:-4]
        if not wav_dir.exists():
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(wav_dir)
        # Map FileID -> wav path (files may sit in a nested folder inside the zip).
        wavs = {p.stem: p for p in wav_dir.rglob("*.wav")}

        with open(idx_path, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                fid, text = cols[0].strip(), cols[1].strip()
                wav = wavs.get(fid) or wavs.get(Path(fid).stem)
                if wav is not None:
                    # The coarse label (e.g. slr66_f) collapses ~24 real speakers
                    # into one id, which breaks `other_same_speaker` reference
                    # selection (it would prompt on a DIFFERENT physical voice).
                    # The OpenSLR FileID encodes the real speaker, e.g.
                    # tef_01033_<line> -> "tef_01033"; fall back to the label.
                    parts = Path(fid).stem.split("_")
                    real_spk = "_".join(parts[:2]) if len(parts) >= 2 else speaker
                    yield str(wav), text, real_spk


def _dispatch_rows(a, cfg):
    """Return a row generator for a single source described by namespace ``a``."""
    if a.source_type == "ljspeech":
        assert a.audio_dir and a.metadata, "ljspeech source needs audio_dir + metadata"
        return _from_ljspeech(a)
    if a.source_type == "csv":
        assert a.csv, "csv source needs `csv`"
        return _from_csv(a)
    if a.source_type == "openslr":
        assert a.slr_id, "openslr source needs slr_id (e.g. 66 for Telugu)"
        return _from_openslr(a, cfg)
    if a.source_type == "hf":
        assert a.hf_dataset, "hf source needs hf_dataset"
        return _from_hf(a, cfg)
    if a.source_type == "parquet":
        assert a.parquet_glob, "parquet source needs parquet_glob"
        return _from_parquet(a, cfg)
    raise SystemExit(f"Unknown source type: {a.source_type!r}")


def _source_namespace(src: dict):
    """Build an args-like namespace (matching the CLI attrs) from a config source dict."""
    import types

    d = dict(src)
    d.setdefault("source_type", d.pop("type", None))
    base = dict(
        source_type=None, speaker="spk0", audio_dir=None, metadata=None, ext=".wav",
        sep="|", csv=None, audio_col="audio", text_col="text", speaker_col=None,
        hf_dataset=None, hf_config=None, hf_split="train", slr_id=None,
        parquet_glob=None, cache_name=None,
    )
    base.update(d)
    return types.SimpleNamespace(**base)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--source-type", default=None, choices=["ljspeech", "csv", "hf", "openslr", "parquet"],
                   help="single source; omit and use --all to pull every source in the config")
    p.add_argument("--all", action="store_true",
                   help="download EVERY source listed under `sources:` in the config into one manifest")
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
    # openslr
    p.add_argument("--slr-id", type=int, default=None, help="OpenSLR set id, e.g. 66 (Telugu)")
    # parquet (local shards, e.g. a partially-downloaded HF parquet dataset)
    p.add_argument("--parquet-glob", default=None,
                   help="glob for local parquet shards, e.g. '~/.cache/huggingface/hub/"
                        "datasets--ai4bharat--indicvoices_r/snapshots/*/Telugu/train-*.parquet'")
    p.add_argument("--cache-name", default=None, help="audio_cache subdir name for parquet source")
    args = p.parse_args()

    cfg = load_config(args.config)
    manifest = Path(cfg.paths.manifest)

    if args.all:
        # Pull every source listed in the config into one combined manifest.
        from omegaconf import OmegaConf

        sources = cfg.get("sources")
        if not sources:
            raise SystemExit("--all requires a `sources:` list in the config.")
        # --all produces THE full manifest, so rebuild from scratch by default
        # (re-running shouldn't duplicate rows). Use --append to add instead.
        if manifest.exists() and not args.append:
            print(f"[prepare_data] --all: rebuilding {manifest} from scratch")
            manifest.unlink()
        print(f"[prepare_data] --all: pulling {len(sources)} source(s) for "
              f"'{cfg.language}' ({cfg.language_name}) — this language only")
        grand = 0
        failed = []  # (label, reason) for sources we skipped
        for i, src_cfg in enumerate(sources):
            a = _source_namespace(OmegaConf.to_container(src_cfg, resolve=True))
            label = a.hf_dataset or (f"SLR{a.slr_id}" if a.slr_id else a.source_type)
            print(f"[prepare_data] source {i + 1}/{len(sources)}: {a.source_type} ({label})")
            # `--all` aggregates independent public corpora; any one may be gated,
            # offline, or rate-limited at run time. Don't let a single unreachable
            # source throw away the rows already gathered — skip it and carry on.
            try:
                kept = _append_rows(manifest, _dispatch_rows(a, cfg), cfg.language)
            except Exception as e:  # noqa: BLE001 - report and continue
                failed.append((label, str(e).splitlines()[0]))
                print(f"[prepare_data]   !! skipped {label}: {failed[-1][1]}")
                continue
            grand += kept
            print(f"[prepare_data]   +{kept} utterances")
        total = sum(1 for _ in open(manifest, encoding="utf-8")) if manifest.exists() else 0
        print(f"[prepare_data] ALL sources done: added {grand}; manifest now {total} rows -> {manifest}")
        if failed:
            print(f"[prepare_data] {len(failed)}/{len(sources)} source(s) skipped:")
            for label, reason in failed:
                print(f"[prepare_data]   - {label}: {reason}")
            if grand == 0:
                raise SystemExit("[prepare_data] no sources succeeded — see errors above.")
        return

    if not args.source_type:
        raise SystemExit("Pass --source-type <ljspeech|csv|hf|openslr>, or --all to use the config's `sources:`.")
    if manifest.exists() and not args.append:
        raise SystemExit(
            f"{manifest} already exists. Pass --append to add to it, or delete it to start fresh."
        )

    kept = _append_rows(manifest, _dispatch_rows(args, cfg), cfg.language)
    total = sum(1 for _ in open(manifest, encoding="utf-8"))
    print(f"[prepare_data] added {kept} utterances from {args.source_type}. Manifest now: {total} rows -> {manifest}")


if __name__ == "__main__":
    main()
