"""Step 5 - fine-tune T3 for the new language (MPS / 24 GB friendly).

Warm-starts from the released multilingual checkpoint, resizes the text
embedding/head to the extended vocab, attaches LoRA to the Llama backbone, and
trains only the LoRA adapters + the (small) text embedding & head. The
language-agnostic acoustic stack is never loaded here - we train purely on the
pre-computed features.

Objective: autoregressive next-speech-token prediction. The speech input is
``[BOS] + tokens`` and the target is ``tokens + [EOS]`` so logits at position k
predict token k (teacher forcing), exactly matching inference in T3.inference.

    python -m training.train --config training/configs/tamil.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

from training.common import build_warmstart_t3, download_base, get_device, load_config, load_tokenizer, set_seed
from training.dataset import FeatureDataset, build_t3_cond, collate_single

IGNORE_ID = -100


def setup_model(cfg, base_dir, vocab_size, device):
    t3, resized = build_warmstart_t3(base_dir, cfg.base_t3, vocab_size, device="cpu")
    for k, src, dst in resized:
        print(f"[train] resized {k}: {src} -> {dst} (overlap copied, new rows fresh)")

    use_lora = bool(cfg.train.lora.enabled)
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise SystemExit("LoRA requires `peft`  ->  pip install peft") from e
        lcfg = LoraConfig(
            r=int(cfg.train.lora.rank),
            lora_alpha=int(cfg.train.lora.alpha),
            lora_dropout=float(cfg.train.lora.dropout),
            target_modules=list(cfg.train.lora.target_modules),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        t3.tfmr = get_peft_model(t3.tfmr, lcfg)

    # Freeze everything outside the backbone; (un)freeze the backbone per mode.
    for name, p in t3.named_parameters():
        if name.startswith("tfmr."):
            if not use_lora:
                p.requires_grad_(True)  # full fine-tune of the backbone
            # with LoRA: leave peft's setting (base frozen, adapters trainable)
            continue
        p.requires_grad_(False)
    if cfg.train.train_text_embeddings:
        t3.text_emb.weight.requires_grad_(True)
    if cfg.train.train_text_head:
        t3.text_head.weight.requires_grad_(True)

    t3.to(device)
    n_train = sum(p.numel() for p in t3.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in t3.parameters())
    print(f"[train] mode={'LoRA' if use_lora else 'full'} | trainable={n_train/1e6:.2f}M / {n_total/1e6:.1f}M")
    return t3, use_lora


def compute_loss(t3, sample, device, w_speech, w_text):
    hp = t3.hp
    sot, eot = hp.start_text_token, hp.stop_text_token
    bos, eos = hp.start_speech_token, hp.stop_speech_token

    text = sample["text_tokens"].to(device)
    text_in = torch.cat([
        torch.tensor([sot], device=device, dtype=torch.long),
        text,
        torch.tensor([eot], device=device, dtype=torch.long),
    ]).unsqueeze(0)
    text_lens = torch.tensor([text_in.size(1)], device=device, dtype=torch.long)

    raw = sample["speech_tokens"].to(device)
    speech_in = torch.cat([torch.tensor([bos], device=device, dtype=torch.long), raw]).unsqueeze(0)
    speech_tgt = torch.cat([raw, torch.tensor([eos], device=device, dtype=torch.long)]).unsqueeze(0)
    speech_lens = torch.tensor([speech_in.size(1)], device=device, dtype=torch.long)

    out = t3.forward(
        t3_cond=build_t3_cond(sample, device),
        text_tokens=text_in,
        text_token_lens=text_lens,
        speech_tokens=speech_in,
        speech_token_lens=speech_lens,
        training=True,
    )

    v = out.speech_logits.size(-1)
    loss_speech = F.cross_entropy(out.speech_logits.reshape(-1, v), speech_tgt.reshape(-1))
    loss = w_speech * loss_speech

    loss_text = torch.zeros((), device=device)
    if w_text > 0:
        vt = out.text_logits.size(-1)
        text_tgt = torch.full_like(text_in, IGNORE_ID)
        text_tgt[:, :-1] = text_in[:, 1:]  # next-text-token target
        loss_text = F.cross_entropy(out.text_logits.reshape(-1, vt), text_tgt.reshape(-1), ignore_index=IGNORE_ID)
        loss = loss + w_text * loss_text
    return loss, loss_speech.detach(), loss_text.detach()


def save_checkpoint(t3, cfg, use_lora, step, vocab_size):
    ckpt = Path(cfg.paths.checkpoints) / f"step_{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if use_lora:
        t3.tfmr.save_pretrained(str(ckpt / "adapter"))
        save_file(
            {
                "text_emb.weight": t3.text_emb.weight.detach().cpu().contiguous(),
                "text_head.weight": t3.text_head.weight.detach().cpu().contiguous(),
            },
            str(ckpt / "extras.safetensors"),
        )
    else:
        save_file({k: v.detach().cpu().contiguous() for k, v in t3.state_dict().items()},
                  str(ckpt / "t3_full.safetensors"))
    meta = {"lora": use_lora, "step": step, "vocab_size": int(vocab_size),
            "lang": cfg.language, "base_t3": cfg.base_t3}
    json.dump(meta, open(ckpt / "meta.json", "w"))
    json.dump({"latest": str(ckpt)}, open(Path(cfg.paths.checkpoints) / "latest.json", "w"))
    print(f"[train] saved checkpoint -> {ckpt}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.train.seed))
    device = get_device(cfg.device)
    if int(cfg.train.batch_size) != 1:
        raise SystemExit("train.batch_size must be 1 (see training/dataset.py). Use grad_accum for a larger effective batch.")

    base_dir = download_base(cfg)
    _, vocab_size = load_tokenizer(cfg.paths.tokenizer)
    t3, use_lora = setup_model(cfg, base_dir, vocab_size, device)
    t3.train()

    ds = FeatureDataset(cfg.paths.feature_index)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=int(cfg.train.num_workers), collate_fn=collate_single)
    print(f"[train] dataset: {len(ds)} utterances")

    trainable = [p for p in t3.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay), foreach=True)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.train.warmup_steps), int(cfg.train.max_steps))

    grad_accum = int(cfg.train.grad_accum)
    max_steps = int(cfg.train.max_steps)
    w_speech, w_text = float(cfg.train.speech_loss_weight), float(cfg.train.text_loss_weight)

    step, micro, done = 0, 0, False
    run_speech, run_text, t0 = 0.0, 0.0, time.time()
    opt.zero_grad()
    while not done:
        for sample in dl:
            loss, ls, lt = compute_loss(t3, sample, device, w_speech, w_text)
            (loss / grad_accum).backward()
            run_speech += float(ls); run_text += float(lt); micro += 1

            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.train.grad_clip))
                opt.step(); sched.step(); opt.zero_grad()
                step += 1

                if step % int(cfg.train.log_every) == 0:
                    n = int(cfg.train.log_every) * grad_accum
                    dt = time.time() - t0
                    print(f"[train] step {step}/{max_steps} | speech {run_speech/n:.4f} | "
                          f"text {run_text/n:.4f} | lr {sched.get_last_lr()[0]:.2e} | {n/dt:.1f} utt/s")
                    run_speech, run_text, t0 = 0.0, 0.0, time.time()

                if step % int(cfg.train.save_every) == 0:
                    save_checkpoint(t3, cfg, use_lora, step, vocab_size)
                if step >= max_steps:
                    done = True
                    break

    save_checkpoint(t3, cfg, use_lora, step, vocab_size)
    print("[train] done.")


if __name__ == "__main__":
    main()
