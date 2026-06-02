# Adding Indian languages to Chatterbox (fine-tuning pipeline)

This package fine-tunes **Chatterbox-Multilingual** for a new language — one at a
time — on a single **Apple-Silicon Mac (MPS, 24 GB)**, with first-class support
for **code-switched** speech (Indian conversational speech mixes a lot of
English: "Hinglish", "Tanglish", …). The pilot config targets **Tamil (`ta`)**;
copy it to add the next language.

## How it works (and why it fits in 24 GB)

Chatterbox has four parts. Only the text side is language-specific:

| Component | Role | Here |
|---|---|---|
| **T3** (Llama-520M) | text tokens → speech tokens | **trained** (LoRA + text emb/head) |
| **MTLTokenizer** (grapheme BPE) | text → ids | **extended** with the new script |
| **S3Gen** + **HiFi-GAN** | speech tokens → waveform | **reused frozen** (acoustic) |
| **VoiceEncoder** + **S3Tokenizer** | speaker emb; wav → tokens | **reused frozen** (acoustic) |

Two design choices make it run on a Mac:
1. **Warm start** from the released 23-language checkpoint → English stays intact
   (so code-switching works) and you get cross-lingual transfer.
2. **Pre-compute acoustic features once** (`precompute_features.py`), then training
   loads *only* T3. With **LoRA**, trainable params are a few M and it fits in
   well under 12 GB.

> **Honest expectation.** A 24 GB Mac is great for LoRA fine-tunes on tens-to-low
> hundreds of hours and will give natural, code-switch-capable speech. Matching
> Chatterbox's *English* SOTA (trained on ~hundreds of thousands of hours across
> many GPUs) is **not** achievable on one Mac. The same configs run on a cloud
> GPU later — set `device: cuda`, raise `lora.rank`/data, and (optionally) switch
> to a full fine-tune — to push toward native quality.

## Install

```bash
pip install -e ".[training]"     # adds peft, datasets, soundfile, indic-nlp-library, huggingface_hub
```

## Datasets (open, for Indian-language TTS)

Blend studio-quality (voice quality) + multi-speaker (robustness) + conversational
(code-switching). Highlights:

- **Studio / TTS-grade:** IIT-Madras **IndicTTS** (13 langs, studio), **SYSPIN**
  (9 langs, CC-BY-4.0), AI4Bharat **Rasa** (expressive).
- **Large multi-speaker:** **IndicVoices-R** (`ai4bharat/indicvoices_r`, ~1700 h,
  22 langs), Google **OpenSLR** HQ sets (Tamil SLR65, Telugu SLR66, …).
- **Code-switching:** **OpenSLR MUCS-2021** (Hindi-English / Bengali-English
  subsets), **IndicVoices** conversational/extempore.
- **Crowd (robustness):** Mozilla **Common Voice** (Indic), AI4Bharat **Shrutilipi**.

Tamil pilot suggestion: IIT-M IndicTTS Tamil + OpenSLR SLR65 + IndicVoices-R Tamil
+ IndicVoices conversational / MUCS for "Tanglish". Start with ~20–50 h, expand.

> **Licensing:** several corpora are research-only. The fine-tuned weights inherit
> those terms — verify before publishing (see the upload step).

## Pipeline (run from the repo root)

```bash
CFG=training/configs/tamil.yaml

# 1) Build the manifest (run once per source, --append for the 2nd, 3rd, …)
python -m training.prepare_data --config $CFG --source-type ljspeech \
    --audio-dir /data/indictts/ta/wav --metadata /data/indictts/ta/metadata.txt --speaker indictts_f
python -m training.prepare_data --config $CFG --append --source-type hf \
    --hf-dataset ai4bharat/indicvoices_r --hf-config ta --hf-split train \
    --text-col text --audio-col audio --speaker-col speaker

# 2) Extend the tokenizer with the new script + [ta] tag
python -m training.extend_tokenizer --config $CFG

# 3) Pre-compute features (one-time; add --device cpu if MPS torch.stft errors)
python -m training.precompute_features --config $CFG

# 4) Fine-tune (resumes nothing; checkpoints land in runs/tamil/checkpoints)
python -m training.train --config $CFG

# 5) Merge LoRA + export a loadable model dir (runs/tamil/export)
python -m training.merge_and_export --config $CFG

# 6) Listen + score (pure-script and code-switched eval sentences)
python -m training.evaluate --config $CFG --ref reference_voice.wav
```

Use it from Python:

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio as ta

model = ChatterboxMultilingualTTS.from_local("runs/tamil/export", device="mps", t3_model="t3_mtl_ta.safetensors")
wav = model.generate("நான் office போறேன், meeting இருக்கு.", language_id="ta", audio_prompt_path="reference_voice.wav")
ta.save("out.wav", wav, model.sr)
```

## Upload to HuggingFace

```bash
huggingface-cli login                      # or: export HF_TOKEN=hf_xxx
# set upload.repo_id in the config, or pass --repo-id:
python -m training.push_to_hub --config $CFG --repo-id your-username/chatterbox-tamil
```

`merge_and_export` already wrote a model card (`export/README.md`) with usage and a
licensing notice. Repos default to **private** (`upload.private: true`) — flip it
only after confirming your data licenses allow redistribution.

## Add the NEXT language (1 by 1)

```bash
cp training/configs/tamil.yaml training/configs/telugu.yaml   # edit language: te, paths, eval_sentences
# then rerun steps 1–6 with --config training/configs/telugu.yaml
```

## Tips & troubleshooting

- **Quality first:** clean, single-speaker studio data drives voice quality; add
  multi-speaker + conversational data for robustness and code-switching.
- **Code-switching:** include code-mixed utterances in the manifest; the warm
  start already knows English and Latin tokens are preserved, so the model mainly
  learns the *transitions*. Test with mixed eval sentences (in the config).
- **MPS memory:** keep `batch_size: 1` and use `grad_accum` for effective batch.
  If you hit memory pressure, lower `max_audio_seconds` or `lora.rank`.
- **MPS precision:** `fp32` is the robust default. bf16 on MPS can be unstable.
- **`torch.stft` on MPS** (precompute): if it errors, run step 3 with `--device cpu`.
- **Sanity checks:** step 2 prints `[UNK]` count (should be ~0 on in-language text);
  step 4's `speech` loss should fall steadily; for a smoke test set `max_steps`
  small and `--limit` the features.
- **Scaling up (cloud GPU):** set `device: cuda`, increase data and `max_steps`,
  optionally `lora.enabled: false` for a full fine-tune.
