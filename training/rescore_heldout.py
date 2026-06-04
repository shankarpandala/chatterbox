"""Re-score already-generated held-out wavs with a Telugu-specialist ASR.

``evaluate_heldout.py`` generated 60 wavs and scored them with a multilingual
ASR (openai/whisper-small), which transcribed correct Telugu speech into the
WRONG script (Devanagari/Kannada) — so the grapheme CER was a script-mismatch
artifact, not a TTS-quality signal. This re-transcribes the SAME wavs with a
Telugu ASR that emits Telugu script, and recomputes the normalized grapheme CER
(reusing evaluate_heldout.cer). No regeneration.

    python -m training.rescore_heldout --asr vasista22/whisper-telugu-small
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from training.evaluate_heldout import cer, wer

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="runs/te/export/eval_heldout/results.json")
    p.add_argument("--asr", default="vasista22/whisper-telugu-small")
    args = p.parse_args()

    import librosa
    from transformers import pipeline

    res = json.load(open(REPO_ROOT / args.results, encoding="utf-8"))
    rows = [r for r in res["rows"] if r.get("wav") and r.get("text")]
    print(f"[rescore] {len(rows)} wavs | ASR={args.asr}")

    asr = pipeline("automatic-speech-recognition", model=args.asr, device=-1)

    by_src = defaultdict(list)
    out = []
    for i, r in enumerate(rows):
        wav = r["wav"]
        if not Path(wav).is_absolute():
            wav = str(REPO_ROOT / wav)
        w16, _ = librosa.load(wav, sr=16000)
        hyp = asr({"array": w16, "sampling_rate": 16000})["text"]
        c, wr = cer(r["text"], hyp), wer(r["text"], hyp)
        by_src[r["source"]].append(c)
        out.append({"source": r["source"], "cer": c, "wer": wr,
                    "ref": r["text"], "hyp": hyp})
        if (i + 1) % 15 == 0:
            print(f"[rescore] {i+1}/{len(rows)}")

    allc = [o["cer"] for o in out]
    print("\n" + "=" * 60)
    print(f"RE-SCORED held-out CER with {args.asr}")
    print("=" * 60)
    print(f"OVERALL  CER mean={statistics.fmean(allc):.4f}  median={statistics.median(allc):.4f}  n={len(allc)}")
    for s, cs in sorted(by_src.items()):
        print(f"  {s:12s} n={len(cs):3d}  CER mean={statistics.fmean(cs):.4f}  median={statistics.median(cs):.4f}")
    print("-" * 60)
    print("sample REF/HYP (now same script):")
    for o in out[:4]:
        print(f"  CER={o['cer']:.2f} | REF {o['ref'][:55]}")
        print(f"           | HYP {o['hyp'][:55]}")

    out_path = REPO_ROOT / "runs/te/export/eval_heldout/rescore_telugu_asr.json"
    json.dump({"asr": args.asr, "overall_cer_mean": statistics.fmean(allc),
               "rows": out}, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[rescore] wrote -> {out_path}")


if __name__ == "__main__":
    main()
