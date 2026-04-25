"""
Train a local BPE (ByteLevel) tokenizer on the corpus and encode/decode for the GPT.

Subword units get you much closer to real words than character tokens, with no
paid API. Requires: `pip install tokenizers` (see requirements.txt).
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

DEFAULT_BPE_VOCAB = 2048


def train_bpe_tokenizer(
    text: str,
    vocab_size: int = DEFAULT_BPE_VOCAB,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    ByteLevel BPE: can encode any UTF-8 string; good for English + punctuation.
    """
    unk = "<unk>"
    tokenizer = Tokenizer(BPE(unk_token=unk))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[unk],
    )

    # One string is fine for multi‑MB text; use an iterator of chunks if OOM.
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def bpe_tokenizer_path_for_checkpoint(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.stem + ".tokenizer.json")


def save_bpe_tokenizer(tokenizer: Tokenizer, ckpt: Path) -> Path:
    p = bpe_tokenizer_path_for_checkpoint(ckpt)
    p.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(p))
    return p


def load_bpe_tokenizer(ckpt: Path) -> Tokenizer:
    p = bpe_tokenizer_path_for_checkpoint(ckpt)
    if not p.is_file():
        raise FileNotFoundError(
            f"BPE tokenizer not found at {p} (expected next to the checkpoint {ckpt.name})"
        )
    return Tokenizer.from_file(str(p))


def encode_to_ids(tz: Tokenizer, text: str) -> list[int]:
    return tz.encode(text).ids


def decode_from_ids(tz: Tokenizer, ids: list[int]) -> str:
    return tz.decode(ids)
