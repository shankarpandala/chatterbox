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
    repo = cfg.upload.get("repo_id") or "YOUR_USERNAME/REPO"
    pure = cfg.eval_sentences[0]
    mixed = cfg.eval_sentences[1] if len(cfg.eval_sentences) > 1 else pure

    # Dynamic head (f-string). NOTE: keep literal `{`/`}` out of here — the BibTeX
    # (full of braces) lives in the raw `citations` block below to avoid escaping.
    head = f"""---
license: {cfg.upload.license}
base_model: {cfg.base_repo}
base_model_relation: finetune
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
- voice-cloning
---

# Chatterbox-Multilingual — {name} ({lang}) fine-tune

A **{name}** fine-tune of Resemble AI's
[`{cfg.base_repo}`](https://huggingface.co/{cfg.base_repo}) (Chatterbox-Multilingual,
checkpoint `{cfg.base_t3}`). It adds {name} — including **code-switched** ({name}+English,
e.g. "Tenglish") speech — while keeping the base model's English and 23-language ability and
its zero-shot voice cloning. This is a *derivative* of Chatterbox; see the model tree above.

## Key details

- **Backbone:** Chatterbox **T3** (~0.5B Llama) text→speech-token model, warm-started from the
  released 23-language checkpoint (English/cross-lingual ability preserved).
- **Adaptation:** LoRA (rank 16, merged into the weights) + retrained text embedding/head.
- **Tokenizer:** the multilingual grapheme tokenizer **extended with the {name} script** and a
  `[{lang}]` language tag (vocab {vocab_size}).
- **Training:** 10,000 steps on ~34.5k {name} clips (see *Training data*).
- **Voice cloning:** zero-shot from a 6–15s reference clip, same as base.
- **Watermark:** every output carries Resemble AI's **PerTh** neural watermark.

## What was kept vs. changed

Only the **text side** is {name}-specific; the language-agnostic **acoustic stack is reused
unchanged** from the base model.

| Component | Role | In this fine-tune |
|---|---|---|
| **T3** (Llama ~0.5B) | text tokens → speech tokens | **Trained** (LoRA + text emb/head), merged into `{t3_name}` |
| **Grapheme tokenizer** | text → token ids | **Extended** (+{name} script, `[{lang}]` tag) |
| **S3Gen + HiFi-GAN** | speech tokens → waveform | **Kept unchanged** (`s3gen.pt`) |
| **VoiceEncoder** | speaker embedding | **Kept unchanged** (`ve.pt`) |
| **S3Tokenizer** | wav → speech tokens | **Kept unchanged** (from base) |
| **Conditioning / misc** | default conds, ZH tokenizer | **Kept unchanged** (`conds.pt`, `Cangjie5_TC.json`) |

The bundled `s3gen.pt`, `ve.pt`, `conds.pt`, and `Cangjie5_TC.json` are **Resemble AI's original
files, redistributed unchanged** under their MIT license (see *License & attribution*).

## Training data

Trained only on **CC-BY-4.0** {name} speech (attribution below). Only model **weights** are
published here — **no raw dataset audio** is redistributed.

| Dataset | Content | License |
|---|---|---|
| [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (`te_in`) | ~5 h read speech, 16 kHz | CC-BY-4.0 |
| [`ai4bharat/indicvoices_r`](https://huggingface.co/datasets/ai4bharat/indicvoices_r) (Telugu) | multi-speaker, 48 kHz | CC-BY-4.0 |

> OpenSLR **SLR66** (CC-BY-**SA**-4.0) was **deliberately excluded** so the training mix stays
> CC-BY-4.0 and this model can be released under a plain CC-BY-4.0 license (no ShareAlike).

## Usage

```python
import torchaudio as ta
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

ckpt = snapshot_download("{repo}")
model = ChatterboxMultilingualTTS.from_local(ckpt, device="mps", t3_model="{t3_name}")

# Pure {name}
wav = model.generate(
    "{pure}",
    language_id="{lang}",
    audio_prompt_path="your_reference.wav",   # 6-15s clip of the target voice
)
ta.save("out.wav", wav, model.sr)

# Code-switched ({name} + English)
wav = model.generate("{mixed}", language_id="{lang}", audio_prompt_path="your_reference.wav")
ta.save("out_codeswitch.wav", wav, model.sr)
```

`device="cuda"` on a GPU, `"mps"` on Apple Silicon, `"cpu"` otherwise.

## Watermarking

Like the base model, **every audio file generated carries Resemble AI's PerTh neural
watermark** — imperceptible marks that survive MP3 compression and common edits — for
responsible-AI provenance.

## Acknowledgements

- **Resemble AI** for [Chatterbox](https://huggingface.co/{cfg.base_repo}) (which itself builds on
  CosyVoice, HiFT-GAN, and Llama 3).
- **AI4Bharat** for IndicVoices-R and the **Google** FLEURS team for the {name} data.

## License & attribution

- **This model:** released under **CC-BY-4.0** (attribution required; commercial use and
  derivatives permitted). This carries forward the CC-BY-4.0 attribution required by the
  training data.
- **Base model:** [`{cfg.base_repo}`](https://huggingface.co/{cfg.base_repo}) — **MIT © Resemble AI**.
  The redistributed acoustic files (`s3gen.pt`, `ve.pt`, `conds.pt`, `Cangjie5_TC.json`) remain
  under that MIT license:

  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software
  > and associated documentation files... The above copyright notice and this permission notice
  > shall be included in all copies. (MIT, © 2025 Resemble AI — full text:
  > https://github.com/resemble-ai/chatterbox/blob/master/LICENSE)

- **Training data:** FLEURS and IndicVoices-R, both **CC-BY-4.0** (cited below). CC-BY-SA SLR66
  was excluded to keep this release CC-BY-4.0.
"""

    citations = r"""
## Citations

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository},
}

@inproceedings{conneau2023fleurs,
  title     = {{FLEURS}: Few-Shot Learning Evaluation of Universal Representations of Speech},
  author    = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and
               Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and
               Bapna, Ankur},
  booktitle = {2022 IEEE Spoken Language Technology Workshop (SLT)},
  pages     = {798--805},
  year      = {2023},
  doi       = {10.1109/SLT54892.2023.10023141},
  note      = {arXiv:2205.12446},
}

@inproceedings{sankar2024indicvoicesr,
  title     = {{IndicVoices-R}: Unlocking a Massive Multilingual Multi-speaker Speech
               Corpus for Scaling Indian {TTS}},
  author    = {Sankar, Ashwin and Anand, Srija and Varadhan, Praveen Srinivasa and
               Thomas, Sherry and Singal, Mehak and Kumar, Shridhar and Mehendale, Deovrat and
               Krishana, Aditi and Raju, Giri and Khapra, Mitesh M.},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS 2024)},
  year      = {2024},
  url       = {http://papers.nips.cc/paper_files/paper/2024/hash/7dfcaf4512bbf2a807a783b90afb6c09-Abstract-Datasets_and_Benchmarks_Track.html},
}
```

## Disclaimer & limitations

Focused on {LANG_NAME} and {LANG_NAME}+English code-switching; it does **not** match Chatterbox's
English SOTA quality. Use responsibly — do not use it to impersonate real people without consent or
to produce misleading content. All outputs are PerTh-watermarked.
""".replace("{LANG_NAME}", name)

    return head + citations


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
