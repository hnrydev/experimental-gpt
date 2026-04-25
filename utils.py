"""Load text, char-level vocabulary, encode/decode."""

import torch

from config import Config


def load_text():
    with open(Config.dataset, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from {Config.dataset}")
    return text


def create_vocab(text):
    """Return vocab size, stoi, itos (chars are sorted unique)."""
    chars = sorted(set(text))
    vocab = "".join(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    n = len(chars)
    print(f"Vocabulary: {n} unique characters")
    return n, stoi, itos, vocab


def maps_from_vocab(vocab: str):
    """Rebuild stoi/itos from a saved vocabulary string (index order = id)."""
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return len(vocab), stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def text_to_tensor(text, stoi, device="cpu"):
    return torch.tensor(encode(text, stoi), dtype=torch.long, device=device)


if __name__ == "__main__":
    t = load_text()
    n, stoi, itos, v = create_vocab(t)
    if n == 0:
        print("Add text to data/input.txt to test encode/decode.")
    else:
        s = t[: min(5, len(t))]
        x = encode(s, stoi)
        print(x, "->", decode(x, itos))
