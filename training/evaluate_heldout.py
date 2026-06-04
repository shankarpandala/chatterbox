"""Step 7b - TRUSTWORTHY held-out evaluation of the fine-tuned Telugu T3.

This supersedes ``training/evaluate.py`` (which scored only 3 in-sample,
no-ground-truth sentences with a circular speaker-similarity and a raw-codepoint
CER). Here we evaluate on the held-out validation split with real references:

  * CER/WER are computed against the *manifest* ground-truth text, after
    identical Unicode-NFC + casefold + punctuation/whitespace normalization on
    both sides, and over Telugu GRAPHEME CLUSTERS (aksharas) rather than raw
    codepoints (a Telugu akshara is 2-4 combining codepoints).
  * The ASR is MULTILINGUAL with NO forced language, so any incidental
    Latin/code-switch token is transcribed in its own alphabet instead of being
    force-mapped to Telugu (which would inflate CER as a pure artifact). The 210
    held-out rows are natural pure-Telugu, so this measures real generalization.
  * Speaker similarity is NON-CIRCULAR: synthesis clones a *different* same-speaker
    clip, and similarity is scored against a mean speaker centroid built from yet
    other same-speaker clips (excluding the prompt clip). FLEURS rows use a
    unique speaker-id per clip (no alternative clip exists), so they are skipped
    for spk_sim and reported as coverage; they still contribute to CER.

Run command (do NOT run until training + merge_and_export have finished):

    python -m training.evaluate_heldout \
        --config training/configs/telugu.yaml \
        --asr openai/whisper-small \
        --out-dir runs/te/eval_heldout
    # add --compare-base to also run the (Telugu-incapable) base as a CER floor,
    # --n 30 for a quick subset, --device cpu if MPS/Whisper misbehaves.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import traceback
import unicodedata
from collections import defaultdict
from pathlib import Path

# Repo root: this file lives at <root>/training/evaluate_heldout.py. The val
# index stores *repo-relative* feature paths, and the subagent cwd is not stable,
# so resolve everything against this constant rather than cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # `regex` provides \X extended-grapheme-cluster matching (akshara units).
    import regex as _regex  # type: ignore

    _HAS_REGEX = True
except ImportError:  # pragma: no cover - fallback keeps the script runnable.
    _HAS_REGEX = False


# --------------------------------------------------------------------------- #
# Text normalization + grapheme CER / WER
# --------------------------------------------------------------------------- #
def _normalize_for_cer(s: str) -> str:
    """Canonicalize text IDENTICALLY for ref and hyp before edit distance.

    Order matters:
      1. NFC      -> compose Telugu base+matra so visually identical strings have
                     identical codepoints (fixes NFC/NFD mismatch ASR-vs-GT).
      2. casefold -> aggressive lowercase for any Latin (code-switch) token;
                     a no-op for caseless Telugu.
      3. NFC again -> casefold can re-decompose a handful of codepoints.
      4. P/S categories (punctuation incl. danda '।', symbols) -> single space
         (NOT empty, so "a,b" -> "a b", never "ab").
      5. collapse whitespace runs, strip ends.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = unicodedata.normalize("NFC", s)
    s = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in s
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _segment(s: str) -> list[str]:
    """Grapheme-cluster (akshara) segmentation; codepoint fallback w/o `regex`."""
    if _HAS_REGEX:
        return _regex.findall(r"\X", s)
    return list(s)


def _levenshtein(ref_u: list, hyp_u: list) -> int:
    """Edit distance over two token sequences (two-row DP)."""
    prev = list(range(len(hyp_u) + 1))
    for i, rc in enumerate(ref_u, 1):
        cur = [i] + [0] * len(hyp_u)
        for j, hc in enumerate(hyp_u, 1):
            cur[j] = min(
                cur[j - 1] + 1,            # insertion
                prev[j] + 1,               # deletion
                prev[j - 1] + (rc != hc),  # substitution / match
            )
        prev = cur
    return prev[len(hyp_u)]


def cer(ref: str, hyp: str) -> float:
    """Grapheme-cluster character error rate, normalized identically on both sides."""
    ref_u = _segment(_normalize_for_cer(ref))
    hyp_u = _segment(_normalize_for_cer(hyp))
    if not ref_u:
        return 0.0 if not hyp_u else 1.0
    return _levenshtein(ref_u, hyp_u) / len(ref_u)


def wer(ref: str, hyp: str) -> float:
    """Word error rate on whitespace tokens of the normalized text.

    More meaningful than CER for code-switch lines (Latin word boundaries)."""
    ref_w = _normalize_for_cer(ref).split(" ")
    ref_w = [w for w in ref_w if w]
    hyp_w = [w for w in _normalize_for_cer(hyp).split(" ") if w]
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(ref_w, hyp_w) / len(ref_w)


# --------------------------------------------------------------------------- #
# Source bucketing (for per-source aggregation)
# --------------------------------------------------------------------------- #
def _source_of(speaker: str) -> str:
    """Map a manifest speaker id to its corpus source bucket."""
    if speaker.startswith("fleurs_"):
        return "fleurs"
    if speaker.startswith("tef_") or speaker.startswith("tem_"):
        return "slr66"
    return "indicvoices"


# --------------------------------------------------------------------------- #
# ASR (multilingual, NO forced language)
# --------------------------------------------------------------------------- #
def build_asr(device: str, model_id: str):
    """Return a `transcribe(wav16, sr=16000) -> str` closure, or None on failure.

    NO forced/target language: Whisper's own language head decides per utterance
    and may mix scripts. This is what removes the code-switch CER artifact. We
    feed an in-memory 16k mono float32 array (no disk round-trip).
    """
    import torch
    from transformers import pipeline

    use_cuda = device == "cuda"
    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if use_cuda else -1,  # HF ASR pipelines are flaky on MPS.
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            chunk_length_s=30,
            return_timestamps=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[evaluate_heldout] ASR unavailable ({e}); CER/WER will be skipped.")
        return None

    def transcribe(wav16, sr: int = 16000) -> str:
        # Critically: do NOT pass generate_kwargs={"language": ...}.
        out = asr({"array": wav16, "sampling_rate": sr})
        return out["text"]

    return transcribe


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def _load16(path: str):
    """Load any source (16k/48k, mono/stereo) as mono 16k float32.

    librosa downmixes to mono AND resamples; soundfile would not (it would hand
    a (N, 2) array to the VoiceEncoder). Returns None on failure.
    """
    import librosa

    try:
        wav16, _ = librosa.load(path, sr=16000, mono=True)
        return wav16
    except Exception as e:  # noqa: BLE001
        print(f"[evaluate_heldout] could not load {path}: {e}")
        return None


def _gen16(model, wav_tensor):
    """Resample a generated (1, N) tensor at model.sr to mono 16k float32."""
    import librosa

    wav24 = wav_tensor.squeeze(0).numpy()
    return librosa.resample(wav24, orig_sr=model.sr, target_sr=16000)


# --------------------------------------------------------------------------- #
# Non-circular speaker similarity
# --------------------------------------------------------------------------- #
def _target_spk_embed(ve, clip_paths, exclude_paths, max_clips=8):
    """Mean, L2-renormalized speaker centroid over clips NOT in exclude_paths.

    embeds_from_wavs(as_spk=False) returns L2-normed per-utterance embeds (B, E);
    utt_to_spk_embed means+renorms them. Returns None if no eligible clip.
    Caps at ``max_clips`` eligible clips — a speaker centroid is stable well
    before then, and SLR66 speakers have ~95 clips (one VE forward each would
    dominate runtime if all were embedded per row).
    """
    from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder

    wavs = []
    for p in clip_paths:
        if p in exclude_paths:
            continue
        w = _load16(p)
        if w is not None and w.size:
            wavs.append(w)
        if len(wavs) >= max_clips:
            break
    if not wavs:
        return None
    utt = ve.embeds_from_wavs(wavs, sample_rate=16000, as_spk=False)
    return VoiceEncoder.utt_to_spk_embed(utt)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _summary(values):
    """mean / median / std (population) / n for a list of floats."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def _fmt(stat, key="mean"):
    v = stat.get(key)
    return "n/a" if v is None else f"{v:.4f}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", default=None, help="defaults to cfg.paths.export")
    p.add_argument("--device", default=None, help="override cfg.device")
    p.add_argument(
        "--asr",
        default="openai/whisper-small",
        help="multilingual HF ASR id (fallback for large-v3 OOM/slowness)",
    )
    p.add_argument(
        "--n", type=int, default=None, help="limit held-out rows (default: all 210)"
    )
    p.add_argument("--out-dir", default=None, help="defaults to <model_dir>/eval_heldout")
    p.add_argument(
        "--compare-base",
        action="store_true",
        help="also run the base model as a Telugu-incapable CER floor/control",
    )
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    # Heavy imports inside main so --help and import-time are cheap.
    import librosa  # noqa: F401  (used transitively; ensures availability early)
    import numpy as np
    import torch
    import torchaudio as ta

    from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from training.common import get_device, iter_manifest, load_config, set_seed

    cfg = load_config(args.config)
    device = get_device(args.device or cfg.device)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve a relative export dir against the repo root (cwd is not stable when
    # launched as a module/subagent), mirroring the manifest/feature handling.
    _md = args.model_dir or cfg.paths.export
    model_dir = _md if Path(_md).is_absolute() else str(REPO_ROOT / _md)
    t3_name = f"t3_mtl_{cfg.language}.safetensors"
    out_dir = Path(args.out_dir or (Path(model_dir) / "eval_heldout"))
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate_heldout] model_dir={model_dir} | device={device} "
          f"| lang={cfg.language} | seed={args.seed}")
    print(f"[evaluate_heldout] grapheme segmentation: "
          f"{'regex \\X' if _HAS_REGEX else 'codepoint fallback (install `regex`)'}")

    # ----- load the fine-tuned export ------------------------------------- #
    model = ChatterboxMultilingualTTS.from_local(model_dir, device, t3_model=t3_name)

    # ----- optionally load the base (Telugu-incapable) control ------------ #
    base_model = None
    if args.compare_base:
        try:
            base_model = ChatterboxMultilingualTTS.from_pretrained(
                device, t3_model=cfg.base_t3
            )
            print("[evaluate_heldout] base model loaded as CER floor/control "
                  "(its tokenizer lacks Telugu graphemes -> ~1.0 CER expected).")
        except Exception as e:  # noqa: BLE001
            print(f"[evaluate_heldout] base model load FAILED ({e}); "
                  "continuing without the base control.")
            base_model = None

    # ----- ASR ------------------------------------------------------------ #
    transcribe = build_asr(device, args.asr)

    # ----- manifest + held-out val index ---------------------------------- #
    manifest = list(iter_manifest(str(REPO_ROOT / cfg.paths.manifest))
                    if not Path(cfg.paths.manifest).is_absolute()
                    else iter_manifest(cfg.paths.manifest))
    # spk -> sorted manifest indices (for deterministic same-speaker clip pick).
    spk2idx: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(manifest):
        spk2idx[row["speaker"]].append(i)

    _feat = Path(cfg.paths.features)
    feat_dir = _feat if _feat.is_absolute() else (REPO_ROOT / _feat)
    val_path = feat_dir / "index.val.jsonl"
    val_rows = list(iter_manifest(str(val_path)))
    if args.n:
        val_rows = val_rows[: args.n]
    print(f"[evaluate_heldout] held-out rows: {len(val_rows)} "
          f"(of {sum(1 for _ in open(val_path, encoding='utf-8'))} total)")

    # ----- per-row evaluation --------------------------------------------- #
    records = []
    skipped = 0
    for vi, vrow in enumerate(val_rows):
        try:
            mi = int(Path(vrow["feature_path"]).name.split(".")[0])
            mrow = manifest[mi]
            text = mrow["text"]
            gt_audio = mrow["audio_path"]
            speaker = vrow["speaker"]
            source = _source_of(speaker)

            # Same-speaker DIFFERENT clip for cloning (deterministic round-robin:
            # the next index after `mi` in the speaker's sorted list). Singleton
            # speakers (FLEURS) fall back to self-clip and are flagged.
            sib = spk2idx.get(speaker, [mi])
            prompt_is_self = len(sib) < 2
            if prompt_is_self:
                prompt_path = gt_audio
            else:
                pos = sib.index(mi)
                prompt_path = manifest[sib[(pos + 1) % len(sib)]]["audio_path"]

            # Synthesize (cloning from the prompt clip).
            with torch.inference_mode():
                wav = model.generate(
                    text, language_id=cfg.language, audio_prompt_path=prompt_path
                )
            wav_path = wav_dir / f"val_{vi:04d}_m{mi:08d}.wav"
            ta.save(str(wav_path), wav, model.sr)
            gen16 = _gen16(model, wav)

            rec = {
                "val_i": vi,
                "manifest_index": mi,
                "speaker": speaker,
                "source": source,
                "text": text,
                "gt_audio": gt_audio,
                "prompt_audio": prompt_path,
                "prompt_is_self": prompt_is_self,
                "wav": str(wav_path),
            }

            # CER / WER (fine-tuned).
            if transcribe is not None:
                hyp = transcribe(gen16, 16000)
                rec["hyp"] = hyp
                rec["cer"] = cer(text, hyp)
                rec["wer"] = wer(text, hyp)

            # Non-circular speaker similarity. Build target centroid from OTHER
            # same-speaker clips, excluding BOTH the prompt clip and the GT clip
            # being synthesized. Skip singleton (FLEURS) speakers entirely.
            rec["spk_sim"] = None
            if not prompt_is_self:
                clip_paths = [manifest[j]["audio_path"] for j in sib]
                tgt = _target_spk_embed(
                    model.ve, clip_paths, exclude_paths={prompt_path, gt_audio}
                )
                if tgt is not None:
                    gen_emb = model.ve.embeds_from_wavs(
                        [gen16], sample_rate=16000, as_spk=True
                    )
                    rec["spk_sim"] = float(VoiceEncoder.voice_similarity(gen_emb, tgt))

                    # Ceiling: GT clip vs its own speaker centroid (same-speaker
                    # upper bound). Anchors the otherwise-uncalibrated cosine.
                    gt16 = _load16(gt_audio)
                    if gt16 is not None and gt16.size:
                        gt_emb = model.ve.embeds_from_wavs(
                            [gt16], sample_rate=16000, as_spk=True
                        )
                        rec["spk_sim_ceiling"] = float(
                            VoiceEncoder.voice_similarity(gt_emb, tgt)
                        )

            # Base control on the SAME row (paired), if requested.
            if base_model is not None and transcribe is not None:
                try:
                    with torch.inference_mode():
                        bwav = base_model.generate(
                            text, language_id=cfg.language,
                            audio_prompt_path=prompt_path,
                        )
                    bgen16 = _gen16(base_model, bwav)
                    bhyp = transcribe(bgen16, 16000)
                    rec["base_hyp"] = bhyp
                    rec["base_cer"] = cer(text, bhyp)
                except Exception as e:  # noqa: BLE001
                    rec["base_cer"] = None
                    rec["base_error"] = str(e)

            records.append(rec)
        except Exception as e:  # noqa: BLE001 - never let one clip kill the run.
            skipped += 1
            print(f"[evaluate_heldout] SKIP val row {vi}: {e}")
            traceback.print_exc()

        # Release the MPS allocator's cached blocks each row: without this its
        # high-water mark climbs into swap over the run and each generation
        # slows from ~25 s to minutes (thrashing). Cheap vs a generation.
        if device == "mps":
            torch.mps.empty_cache()

        if (vi + 1) % 20 == 0:
            print(f"[evaluate_heldout] {vi + 1}/{len(val_rows)} "
                  f"(ok={len(records)}, skipped={skipped})")

    # ----- aggregate ------------------------------------------------------ #
    overall_cer = _summary([r.get("cer") for r in records])
    overall_wer = _summary([r.get("wer") for r in records])
    # spk_sim only over different-clip rows (headline); self-prompt excluded.
    spk_rows = [r for r in records if not r["prompt_is_self"]]
    overall_spk = _summary([r.get("spk_sim") for r in spk_rows])
    overall_spk_ceiling = _summary([r.get("spk_sim_ceiling") for r in spk_rows])

    per_source = {}
    for src in ("fleurs", "slr66", "indicvoices"):
        srecs = [r for r in records if r["source"] == src]
        per_source[src] = {
            "rows": len(srecs),
            "cer": _summary([r.get("cer") for r in srecs]),
            "wer": _summary([r.get("wer") for r in srecs]),
            "spk_sim": _summary(
                [r.get("spk_sim") for r in srecs if not r["prompt_is_self"]]
            ),
        }

    base_summary = None
    if base_model is not None:
        ft_paired = [r["cer"] for r in records
                     if r.get("cer") is not None and r.get("base_cer") is not None]
        base_paired = [r["base_cer"] for r in records
                       if r.get("cer") is not None and r.get("base_cer") is not None]
        deltas = [b - f for f, b in zip(ft_paired, base_paired)]  # base - finetuned
        base_summary = {
            "paired_n": len(deltas),
            "finetuned_cer": _summary(ft_paired),
            "base_cer": _summary(base_paired),
            "delta_base_minus_finetuned": _summary(deltas),
            "note": "Base lacks Telugu graphemes (all-[UNK]) -> expect base_cer ~1.0; "
                    "delta is how far below the floor fine-tuning gets.",
        }

    # ----- code-switch demo (3 cfg.eval_sentences, no GT audio/speaker) ---- #
    # Use a FIXED prompt clip for every sentence so the demo voice is controlled
    # and reproducible. Without an explicit audio_prompt_path, generate() reuses
    # model.conds, which the per-row loop above overwrote with the LAST val row's
    # speaker — making the demo voice order-dependent and non-deterministic.
    codeswitch = []
    cs_prompt = (records[0]["prompt_audio"] if records else
                 (manifest[0]["audio_path"] if manifest else None))
    if transcribe is not None:
        for si, sent in enumerate(list(cfg.eval_sentences)):
            try:
                with torch.inference_mode():
                    swav = model.generate(
                        sent, language_id=cfg.language, audio_prompt_path=cs_prompt
                    )
                spath = wav_dir / f"codeswitch_{si:02d}.wav"
                ta.save(str(spath), swav, model.sr)
                shyp = transcribe(_gen16(model, swav), 16000)
                codeswitch.append({
                    "i": si, "text": sent, "hyp": shyp, "wav": str(spath),
                    "prompt_audio": cs_prompt,
                    "cer": cer(sent, shyp), "wer": wer(sent, shyp),
                })
            except Exception as e:  # noqa: BLE001
                codeswitch.append({"i": si, "text": sent, "error": str(e)})

    results = {
        "config": str(args.config),
        "model_dir": str(model_dir),
        "device": device,
        "asr": args.asr,
        "seed": args.seed,
        "n_val_rows": len(val_rows),
        "n_evaluated": len(records),
        "n_skipped": skipped,
        "has_regex_graphemes": _HAS_REGEX,
        "overall": {
            "cer": overall_cer,
            "wer": overall_wer,
            "spk_sim": overall_spk,
            "spk_sim_ceiling": overall_spk_ceiling,
            "spk_sim_coverage": f"{overall_spk['n']}/{len(records)} rows "
                                f"(FLEURS singletons excluded)",
        },
        "per_source": per_source,
        "base_control": base_summary,
        "codeswitch_demo": codeswitch,
        "rows": records,
    }

    out_json = out_dir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ----- printed summary table ------------------------------------------ #
    print("\n" + "=" * 72)
    print("HELD-OUT EVALUATION SUMMARY (fine-tuned, pure-Telugu generalization)")
    print("=" * 72)
    print(f"evaluated {len(records)}/{len(val_rows)} rows  (skipped {skipped})")
    print(f"ASR: {args.asr}  |  device: {device}  |  seed: {args.seed}")
    print("-" * 72)
    print(f"{'metric':<16}{'mean':>10}{'median':>10}{'std':>10}{'n':>8}")
    for name, stat in (("CER", overall_cer), ("WER", overall_wer),
                       ("spk_sim", overall_spk),
                       ("spk_sim_ceiling", overall_spk_ceiling)):
        print(f"{name:<16}{_fmt(stat):>10}{_fmt(stat,'median'):>10}"
              f"{_fmt(stat,'std'):>10}{stat['n']:>8}")
    print(f"spk_sim coverage: {results['overall']['spk_sim_coverage']}")
    print("-" * 72)
    print(f"{'source':<14}{'rows':>6}{'CER':>10}{'WER':>10}{'spk_sim':>10}{'spk_n':>8}")
    for src, d in per_source.items():
        print(f"{src:<14}{d['rows']:>6}{_fmt(d['cer']):>10}{_fmt(d['wer']):>10}"
              f"{_fmt(d['spk_sim']):>10}{d['spk_sim']['n']:>8}")
    if base_summary is not None:
        print("-" * 72)
        print("BASE CONTROL (Telugu-incapable floor; paired with fine-tuned):")
        print(f"  paired rows           : {base_summary['paired_n']}")
        print(f"  base CER (floor)      : {_fmt(base_summary['base_cer'])}")
        print(f"  fine-tuned CER        : {_fmt(base_summary['finetuned_cer'])}")
        print(f"  delta (base - ft)     : "
              f"{_fmt(base_summary['delta_base_minus_finetuned'])}")
    if codeswitch:
        print("-" * 72)
        print("CODE-SWITCH DEMO (3 Tenglish sentences; CER vs written text, no GT):")
        for c in codeswitch:
            if "error" in c:
                print(f"  [{c['i']}] ERROR: {c['error']}")
            else:
                print(f"  [{c['i']}] CER={c['cer']:.3f} WER={c['wer']:.3f}  "
                      f"\"{c['text'][:40]}\"")
    print("=" * 72)
    print(f"[evaluate_heldout] wrote {len(records)} wavs -> {wav_dir}")
    print(f"[evaluate_heldout] results JSON -> {out_json}")


if __name__ == "__main__":
    main()
