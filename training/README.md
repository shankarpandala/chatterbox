# Adding Indian languages to Chatterbox (fine-tuning pipeline)

Fine-tune **Chatterbox-Multilingual** for a new language — one at a time — with
first-class **code-switching** (Indian conversational speech mixes a lot of
English: "Tenglish", "Hinglish", …). The pilot targets **Telugu (`te`)**.

**One parametrized config, two platforms:**
- **Prove on your Mac first** (Apple-Silicon / MPS) — small smoke run.
- **Scale on Google Colab (T4 GPU)** — same commands, same config, switched with
  env vars.

## How it works (and why it fits a 24 GB Mac / a free T4)

Only the text side is language-specific:

| Component | Role | Here |
|---|---|---|
| **T3** (Llama-520M) | text tokens → speech tokens | **trained** (LoRA + text emb/head) |
| **MTLTokenizer** (grapheme BPE) | text → ids | **extended** with the new script |
| **S3Gen** + **HiFi-GAN** | speech tokens → waveform | **reused frozen** (acoustic) |
| **VoiceEncoder** + **S3Tokenizer** | speaker emb; wav → tokens | **reused frozen** (acoustic) |

1. **Warm-start** from the released 23-language checkpoint → English stays intact
   (code-switching works) + cross-lingual transfer.
2. **Pre-compute acoustic features once** → training loads *only* T3; with **LoRA**
   it fits in well under 12 GB (Mac) and easily on a T4.

> A 24 GB Mac / free T4 gives natural, code-switch-capable LoRA fine-tunes.
> Matching Chatterbox's *English* SOTA needs far more data/compute.

## One config, env-driven platforms

`configs/telugu.yaml` is fully parametrized. **All paths derive from `language`**,
and Mac/Colab differences are env vars — no file edits to switch platform:

| Env var | Default (Mac) | Colab T4 | Effect |
|---|---|---|---|
| `DEVICE` | `mps` | `cuda` | compute device |
| `PRECISION` | `fp32` | `fp16` | mixed precision (bf16 on A100/L4) |
| `MAX_STEPS` | `20000` | `40000` | training length (`200` for a Mac smoke test) |
| `WORK_DIR` | `./runs` | a Drive path | where all artifacts go (`$WORK_DIR/<language>`) |

## Add the NEXT language (1 by 1)

Copy the config and change **only the language fields** — every path follows
automatically:

```bash
cp training/configs/telugu.yaml training/configs/marathi.yaml
# edit marathi.yaml:  language: mr  |  language_name: Marathi  |  eval_sentences: (Marathi)
# then run the exact same steps with --config training/configs/marathi.yaml
```

## Install

**Mac:**
```bash
pip install -e ".[training]"     # peft, datasets, soundfile, indic-nlp-library, huggingface_hub
```

**Google Colab** (Runtime → Change runtime type → **T4 GPU**):
```python
!git clone https://github.com/shankarpandala/chatterbox.git
%cd chatterbox
!pip install -e ".[training]"
# Colab ships a CUDA torch; if the install re-pins it and CUDA breaks, reinstall:
# !pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Datasets (open, for Indian-language TTS)

Blend studio-quality (voice) + multi-speaker (robustness) + conversational
(code-switching):
- **Studio / TTS-grade:** IIT-Madras **IndicTTS** (incl. Telugu), **SYSPIN**
  (incl. Telugu, CC-BY-4.0), AI4Bharat **Rasa**.
- **Large multi-speaker:** **IndicVoices-R** (`ai4bharat/indicvoices_r`, te),
  Google **OpenSLR SLR66** (Telugu).
- **Code-switching:** **IndicVoices** conversational/extempore; MUCS-style mixed data.
- **Crowd (robustness):** Mozilla **Common Voice** (te), AI4Bharat **Shrutilipi**.

> Several corpora are research-only; the fine-tuned weights inherit those terms.

---

## Workflow A — Prove on your Mac (do this first)

A quick smoke test to confirm the loop runs and the loss falls, on a handful of clips.

```bash
CFG=training/configs/telugu.yaml

# 1) Manifest (run once per source; --append for more). Example: a studio corpus.
python -m training.prepare_data --config $CFG --source-type ljspeech \
    --audio-dir /data/indictts/te/wav --metadata /data/indictts/te/metadata.txt --speaker indictts_f

# 2) Extend the tokenizer with the Telugu script + [te] tag
python -m training.extend_tokenizer --config $CFG

# 3) Pre-compute features for just 50 clips (CPU avoids MPS torch.stft quirks)
python -m training.precompute_features --config $CFG --device cpu --limit 50

# 4) Short training run (override just the step count)
MAX_STEPS=200 python -m training.train --config $CFG

# 5) Export + 6) listen — confirm it speaks Telugu (+ code-switch)
python -m training.merge_and_export --config $CFG
python -m training.evaluate --config $CFG --ref reference_voice.wav
```

Watch the `speech` loss drop in step 4. Once that works, scale up.

## Workflow B — Scale on Google Colab (T4)

Same commands; set env vars once. Persist to Drive so a disconnect doesn't lose work.

```python
# In a Colab cell:
!nvidia-smi -L                                   # confirm a T4
from google.colab import drive; drive.mount('/content/drive')
from huggingface_hub import login; login()        # paste a token (or set HF_TOKEN)
```
```bash
# Point the whole run at the T4 + Drive, with a full schedule — no config edits:
export DEVICE=cuda PRECISION=fp16 MAX_STEPS=40000 \
       WORK_DIR=/content/drive/MyDrive/chatterbox_runs
CFG=training/configs/telugu.yaml

python -m training.prepare_data        --config $CFG --source-type hf \
    --hf-dataset ai4bharat/indicvoices_r --hf-config te --hf-split train \
    --text-col text --audio-col audio --speaker-col speaker
python -m training.extend_tokenizer    --config $CFG
python -m training.precompute_features --config $CFG          # CUDA handles torch.stft fine
python -m training.train               --config $CFG
python -m training.merge_and_export    --config $CFG
python -m training.evaluate            --config $CFG --ref reference_voice.wav
```

(On an A100/L4 instead of a T4, use `PRECISION=bf16` for better stability.)

## Use it from Python

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio as ta

# device="mps" on Mac, "cuda" on Colab
model = ChatterboxMultilingualTTS.from_local("runs/te/export", device="mps", t3_model="t3_mtl_te.safetensors")
wav = model.generate("నేను office కి వెళ్తున్నాను, meeting ఉంది.", language_id="te", audio_prompt_path="reference_voice.wav")
ta.save("out.wav", wav, model.sr)
```

## Upload to HuggingFace

```bash
huggingface-cli login        # or: export HF_TOKEN=hf_xxx
python -m training.push_to_hub --config $CFG --repo-id your-username/chatterbox-telugu
```

`merge_and_export` already wrote a model card (`export/README.md`). Repos default
to **private** — flip only after confirming your data licenses allow redistribution.

## Tips & troubleshooting

- **Quality first:** clean single-speaker studio data drives voice quality; add
  multi-speaker + conversational data for robustness and code-switching.
- **Code-switching:** include code-mixed utterances; the warm start already knows
  English, so the model mainly learns the *transitions*. Test with the mixed eval
  sentences in the config.
- **MPS memory:** keep `batch_size: 1`, use `grad_accum`; lower `max_audio_seconds`
  or `lora.rank` if memory is tight.
- **`torch.stft` on MPS** (precompute): if it errors, run step 3 with `--device cpu`.
- **Smoke test:** `--limit` the features + `MAX_STEPS=200`; just confirm `speech`
  loss falls before a full run.
- **Full fine-tune on a big GPU:** set `lora.enabled: false` (needs more memory).
