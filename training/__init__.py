"""Fine-tuning pipeline for adding new languages (e.g. Indian languages) to
Chatterbox-Multilingual.

The pipeline warm-starts from the released multilingual T3 checkpoint, extends
the grapheme tokenizer with the target script, and fine-tunes only the
text-side / backbone (via LoRA) while the language-agnostic vocoder stack
(S3Gen, VoiceEncoder, S3Tokenizer) is reused frozen.

See training/README.md for the end-to-end guide.
"""
