"""Step 6 - merge the trained weights and export a self-contained model dir.

Rebuilds the warm-started T3, folds in the trained LoRA adapter + text
embedding/head (or loads the full fine-tune), and writes a directory that
``ChatterboxMultilingualTTS.from_local`` can load directly:

    export/
      t3_mtl_<lang>.safetensors             # the fine-tuned T3
      grapheme_mtl_merged_expanded_v1.json  # the extended tokenizer
      ve.pt  s3gen.pt  conds.pt             # reused, language-agnostic
      config.json  README.md               # metadata + model card

    python -m training.merge_and_export --config training/configs/telugu.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from training.common import DEFAULT_TOKENIZER_NAME, build_warmstart_t3, download_base, load_config, load_tokenizer


def resolve_checkpoint(cfg, checkpoint_arg):
    if checkpoint_arg:
        return Path(checkpoint_arg)
    latest = Path(cfg.paths.checkpoints) / "latest.json"
    if latest.exists():
        return Path(json.load(open(latest))["latest"])
    steps = sorted(Path(cfg.paths.checkpoints).glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not steps:
        raise SystemExit(f"No checkpoints found under {cfg.paths.checkpoints}")
    return steps[-1]


def model_card(cfg, t3_name, vocab_size) -> str:
    lang, name = cfg.language, cfg.language_name
    return f"""---
license: {cfg.upload.license}
language:
- {lang}
library_name: chatterbox
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- chatterbox
- {name.lower()}
- indian-languages
- code-switching
---

# Chatterbox-Multilingual fine-tuned for {name} ({lang})

A fine-tune of [`{cfg.base_repo}`](https://huggingface.co/{cfg.base_repo})
(`{cfg.base_t3}`) that adds **{name}** support, including **code-switched**
({name}+English) speech. The language-agnostic acoustic stack (S3Gen,
VoiceEncoder, S3Tokenizer) is reused unchanged; only the text tokenizer (vocab
extended to {vocab_size}) and the T3 text-to-token model were trained.

## Usage

```python
import torchaudio as ta
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

ckpt = snapshot_download("YOUR_USERNAME/REPO")   # this repo
model = ChatterboxMultilingualTTS.from_local(ckpt, device="mps", t3_model="{t3_name}")

wav = model.generate(
    "{cfg.eval_sentences[0]}",
    language_id="{lang}",
    audio_prompt_path="reference_voice.wav",   # 6-15s clip of the target voice
)
ta.save("out.wav", wav, model.sr)
```

All generated audio carries Resemble AI's **PerTh** neural watermark.

## ⚠️ Licensing of weights

These weights inherit obligations from the **training data**. Many open Indic
TTS corpora (e.g. IIT-Madras IndicTTS) are research-only. Verify every dataset's
license permits redistribution **before** making this model public.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="checkpoint dir (defaults to latest)")
    p.add_argument("--out", default=None, help="export dir (defaults to cfg.paths.export)")
    args = p.parse_args()

    cfg = load_config(args.config)
    base_dir = download_base(cfg)
    _, vocab_size = load_tokenizer(cfg.paths.tokenizer)
    t3, _ = build_warmstart_t3(base_dir, cfg.base_t3, vocab_size, device="cpu")

    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    meta = json.load(open(ckpt / "meta.json"))
    print(f"[export] merging checkpoint {ckpt} (lora={meta['lora']})")

    if meta["lora"]:
        from peft import PeftModel

        t3.tfmr = PeftModel.from_pretrained(t3.tfmr, str(ckpt / "adapter"))
        t3.tfmr = t3.tfmr.merge_and_unload()
        extras = load_file(str(ckpt / "extras.safetensors"))
        with torch.no_grad():
            t3.text_emb.weight.copy_(extras["text_emb.weight"])
            t3.text_head.weight.copy_(extras["text_head.weight"])
    else:
        t3.load_state_dict(load_file(str(ckpt / "t3_full.safetensors")))

    export = Path(args.out or cfg.paths.export)
    export.mkdir(parents=True, exist_ok=True)
    t3_name = f"t3_mtl_{cfg.language}.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in t3.state_dict().items()}, str(export / t3_name))

    # Tokenizer MUST keep the default filename so from_local discovers it.
    shutil.copyfile(cfg.paths.tokenizer, export / DEFAULT_TOKENIZER_NAME)
    for fn in ["ve.pt", "s3gen.pt", "conds.pt", "Cangjie5_TC.json"]:
        src = base_dir / fn
        if src.exists():
            shutil.copyfile(src, export / fn)

    json.dump(
        {"language": cfg.language, "language_name": cfg.language_name, "base_repo": cfg.base_repo,
         "base_t3": cfg.base_t3, "vocab_size": int(vocab_size), "t3_model": t3_name},
        open(export / "config.json", "w"), indent=2, ensure_ascii=False,
    )
    open(export / "README.md", "w", encoding="utf-8").write(model_card(cfg, t3_name, vocab_size))

    print(f"[export] wrote model dir -> {export}")
    print(f"[export] load with: ChatterboxMultilingualTTS.from_local('{export}', device='mps', t3_model='{t3_name}')")


if __name__ == "__main__":
    main()
