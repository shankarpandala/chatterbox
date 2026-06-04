# Citations, Licenses & Attribution

This project fine-tunes the **Chatterbox** multilingual TTS model on open speech
corpora. Every dataset and upstream repo below carries an attribution (and, in one
case, share-alike) obligation. If you train on these sources and **publish the
resulting weights, audio, or a paper, you must credit them** as described here.

> **Why this matters for the fine-tuned model.** A fine-tuned checkpoint is a
> *derivative* of its training data. Your weights inherit the **most restrictive**
> license among the corpora you actually trained on. Trim `sources:` in your config
> to match what you're allowed to redistribute, and keep the attributions below with
> any release.

---

## Data sources (`training/configs/telugu.yaml → sources:`)

### 1. FLEURS — `google/fleurs` (config `te_in`)

- **License:** CC-BY-4.0 (attribution required).
- **Page:** https://huggingface.co/datasets/google/fleurs
- **Used for:** ~5 h Telugu read speech, 16 kHz (smoke-test source).

```bibtex
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
```

### 2. OpenSLR SLR66 — Crowdsourced Telugu multi-speaker speech

- **License:** CC-BY-**SA**-4.0 — attribution **and ShareAlike**. Derivatives
  (including a model trained on this data, if you redistribute it) must be released
  under a compatible CC-BY-SA license. **Most restrictive source in this project.**
- **Page:** https://www.openslr.org/66/
- **Used for:** ~5 h Telugu, 16 kHz, crowdsourced male + female speakers.

```bibtex
@inproceedings{he-etal-2020-open,
  title     = {{Open-source Multi-speaker Speech Corpora for Building Gujarati, Kannada,
               Malayalam, Marathi, Tamil and Telugu Speech Synthesis Systems}},
  author    = {He, Fei and Chu, Shan-Hui Cathy and Kjartansson, Oddur and Rivera, Clara and
               Katanova, Anna and Gutkin, Alexander and Demirsahin, Isin and Johny, Cibu and
               Jansche, Martin and Sarin, Supheakmungkol and Pipatsrisawat, Knot},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference (LREC)},
  pages     = {6494--6503},
  month     = may,
  year      = {2020},
  address   = {Marseille, France},
  publisher = {European Language Resources Association (ELRA)},
  isbn      = {979-10-95546-34-4},
  url       = {https://aclanthology.org/2020.lrec-1.800},
}
```

### 3. IndicVoices-R — `ai4bharat/indicvoices_r` (config `Telugu`)

- **License:** CC-BY-4.0 (attribution required).
- **Access:** **Gated** — request access on the dataset page and authenticate with an
  `HF_TOKEN` before downloading.
- **Page:** https://huggingface.co/datasets/ai4bharat/indicvoices_r
- **Used for:** large multi-speaker Telugu, 48 kHz (scale source; sliced to
  `train[:5000]` in the default config).

```bibtex
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

---

## Upstream model & code

### Chatterbox — `ResembleAI/chatterbox` (Resemble AI)

- **License:** MIT (code and base model weights).
- **Page:** https://github.com/resemble-ai/chatterbox · https://huggingface.co/ResembleAI/chatterbox

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository},
}
```

---

## Compliance checklist before publishing weights or a paper

- [ ] Cite **every** corpus you actually trained on (the BibTeX above) plus Chatterbox.
- [ ] Reproduce the CC-BY attribution notices for FLEURS, SLR66, and IndicVoices-R.
- [ ] If **SLR66** is in your training mix and you redistribute the model, release the
      derivative under a **CC-BY-SA-4.0-compatible** license (ShareAlike). To avoid
      ShareAlike, drop SLR66 from `sources:` and retrain.
- [ ] Set `upload.private: true` (default) until you've confirmed redistribution rights.
- [ ] IndicVoices-R access is granted to your HF account only — do not redistribute the
      raw audio; share derived models per its CC-BY terms.
