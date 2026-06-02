"""Step 2 - extend the grapheme tokenizer for the new language.

Downloads the base multilingual tokenizer, scans the training manifest for any
characters (after the same lowercase + NFKD + Indic normalization the model
applies at encode time) that are NOT already in the vocab, and adds them along
with the ``[<lang>]`` language tag.

New tokens are appended, so every original token id is preserved -> the warm
start in train.py can copy the pretrained embedding rows 1:1 and only the new
rows are learned.

    python -m training.extend_tokenizer --config training/configs/telugu.yaml
"""
from __future__ import annotations

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from training.common import DEFAULT_TOKENIZER_NAME, iter_manifest, load_config

# Same normalization the tokenizer applies (see MTLTokenizer.preprocess_text +
# the Indic dispatch in encode()).
from chatterbox.models.tokenizers.tokenizer import INDIC_LANGS, indic_normalize


def normalize_like_encoder(text: str, lang: str) -> str:
    txt = text.lower()
    txt = unicodedata.normalize("NFKD", txt)
    if lang in INDIC_LANGS:
        txt = indic_normalize(txt, lang)
    return txt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--min-count", type=int, default=1,
                   help="only add characters seen at least this many times")
    args = p.parse_args()

    cfg = load_config(args.config)
    lang = cfg.language

    base_path = hf_hub_download(repo_id=cfg.base_repo, filename=DEFAULT_TOKENIZER_NAME)
    tok = Tokenizer.from_file(base_path)
    base_size = tok.get_vocab_size()

    # Collect candidate characters from the corpus.
    counts: Counter = Counter()
    n_rows = 0
    for row in iter_manifest(cfg.paths.manifest):
        n_rows += 1
        for ch in normalize_like_encoder(row["text"], lang):
            if ch == " ":
                continue  # handled by the [SPACE] token
            counts[ch] += 1

    missing = [c for c, n in counts.items() if n >= args.min_count and tok.token_to_id(c) is None]
    missing = sorted(set(missing))

    to_add = list(missing)
    lang_tag = f"[{lang}]"
    if tok.token_to_id(lang_tag) is None:
        to_add.append(lang_tag)

    added = tok.add_tokens(to_add) if to_add else 0
    new_size = tok.get_vocab_size()

    out_path = Path(cfg.paths.tokenizer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    print(f"[extend_tokenizer] scanned {n_rows} utterances, language='{lang}'")
    print(f"[extend_tokenizer] base vocab={base_size} -> extended vocab={new_size} (+{added})")
    if missing:
        preview = "".join(missing[:60])
        print(f"[extend_tokenizer] added {len(missing)} new characters, e.g.: {preview}")
    if lang_tag in to_add:
        print(f"[extend_tokenizer] added language tag {lang_tag}")
    print(f"[extend_tokenizer] saved -> {out_path}")

    # Round-trip sanity check on the first row: no [UNK] should remain.
    sample = next(iter(iter_manifest(cfg.paths.manifest)))["text"]
    norm = normalize_like_encoder(sample, lang)
    ids = tok.encode(f"{lang_tag}{norm}".replace(' ', '[SPACE]')).ids
    unk_id = tok.token_to_id("[UNK]")
    n_unk = sum(1 for i in ids if i == unk_id)
    print(f"[extend_tokenizer] sanity: '{sample[:40]}...' -> {len(ids)} tokens, {n_unk} [UNK]")


if __name__ == "__main__":
    main()
