"""
Data helpers for a character-level language model.

The network never sees raw letters — only integers (token ids). These helpers
bridge text ↔ ids so training and generation stay simple and explicit.

Naming:
  stoi — "string to int": map each character → id (0 .. vocab_size-1).
  itos — "int to string": inverse map for decoding predictions back to text.
"""

import os
import unicodedata
from pathlib import Path

import torch

from config import Config


def clean_text(s: str) -> str:
    """
    Normalize Unicode, line endings, and noisy whitespace.

    Does not remove real words — it makes the character stream more consistent
    so the model sees slightly cleaner "grammar of spacing" (paragraph breaks,
    no weird control bytes from PDF copy-paste).
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\n\t" or o >= 32:
            out.append(ch)
    s = "".join(out)
    while "\n\n\n\n" in s:
        s = s.replace("\n\n\n\n", "\n\n\n")
    return s.strip() + ("\n" if s.strip() else "")


def load_text(clean: bool | None = None):
    """
    Load the training corpus: one file (Config.dataset) or several in order (Config.corpus_files).

    When `corpus_files` is set, files are concatenated with a short header so the model sees
    which section is which. Put structured primers *first* for a stable "map" before long text.
    """
    if clean is None:
        clean = getattr(Config, "clean_corpus", True)
    paths = getattr(Config, "corpus_files", None)
    if not paths:
        paths = (Config.dataset,)

    chunks: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"Warning: corpus file missing, skipping: {path}")
            continue
        body = path.read_text(encoding="utf-8")
        chunks.append(f"[corpus file: {path.as_posix()}]\n{body}")

    text = "\n\n".join(chunks)
    if not text.strip():
        raise SystemExit("No corpus text loaded. Check Config.corpus_files / dataset paths.")
    if clean:
        text = clean_text(text)
    used = [Path(p).as_posix() for p in paths if Path(p).is_file()]
    print(
        f"Loaded {len(text):,} characters from {len(used)} file(s): {', '.join(used)}"
        + (" (cleaned)" if clean else "")
    )
    return text


def create_vocab(text):
    """
    Build the character vocabulary from the corpus.

    We take every *unique* character that appears, sort them (stable, reproducible
    ordering), and assign ids 0,1,2,... That fixed order becomes our "alphabet"
    for this run. We also return `vocab` as a single string (same order as ids)
    so checkpoints can save and reload the mapping without pickle quirks.
    """
    chars = sorted(set(text))
    vocab = "".join(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    n = len(chars)
    print(f"Vocabulary: {n} unique characters")
    return n, stoi, itos, vocab


def maps_from_vocab(vocab: str):
    """
    Rebuild stoi/itos from the vocabulary string stored in a checkpoint.

    `vocab[i]` must be the character with id `i` — same convention as create_vocab.
    """
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return len(vocab), stoi, itos


def encode(text, stoi):
    """Turn a string into a list of integer token ids (for feeding the model)."""
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    """Turn a list of token ids back into a string (for reading model output)."""
    return "".join(itos[i] for i in ids)


def text_to_tensor(text, stoi, device="cpu"):
    """Encode text and wrap as a 1D PyTorch tensor of int64 on the given device."""
    return torch.tensor(encode(text, stoi), dtype=torch.long, device=device)


def maybe_torch_compile(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optional `torch.compile` (PyTorch 2+). Set BABY_GPT_COMPILE=1 for training / inference.
    On CPU, speedup varies; safe to try on ThinkPad-class machines.
    """
    if os.environ.get("BABY_GPT_COMPILE", "").strip().lower() not in ("1", "true", "yes"):
        return model
    try:
        return torch.compile(model)  # type: ignore[return-value, assignment]
    except Exception as e:
        print(f"BABY_GPT_COMPILE: torch.compile skipped ({e})")
        return model


if __name__ == "__main__":
    t = load_text()
    n, stoi, itos, v = create_vocab(t)
    if n == 0:
        print("Add text to data/input.txt to test encode/decode.")
    else:
        s = t[: min(5, len(t))]
        x = encode(s, stoi)
        print(x, "->", decode(x, itos))
