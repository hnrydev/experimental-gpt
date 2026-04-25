"""
Load a trained checkpoint and sample new text (character by character).

Unlike training, we do not have "correct" next characters: we take the model's
predicted distribution at the last position, sample one id, append it, and repeat.
That is "autoregressive" generation. Temperature changes how random vs greedy
the choices are.
"""

import argparse
from pathlib import Path

import torch

from config import Config
from model import GPT
from utils import decode, maps_from_vocab, text_to_tensor


def load_trained_gpt(ck_path: str | Path, device: str | None = None):
    """
    Load weights + vocabulary from train.py output.

    Returns (model, stoi, itos) ready for generate() or an interactive loop.
    """
    device = device or Config.device
    ck_path = Path(ck_path)
    try:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ck_path, map_location=device)
    vocab = ck["vocab"]
    h = ck["hparams"]
    vocab_size, stoi, itos = maps_from_vocab(vocab)
    model = GPT(
        vocab_size=vocab_size,
        block_size=h["block_size"],
        n_embd=h["n_embd"],
        n_head=h["n_head"],
        n_layer=h["n_layer"],
        dropout=h["dropout"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, stoi, itos


def generate_continuation(
    model: GPT,
    stoi: dict,
    itos: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> str:
    """Run sampling and return the full string (prompt + new text)."""
    for ch in prompt:
        if ch not in stoi:
            raise ValueError(f"Character {ch!r} is not in the training vocabulary.")
    # Model was trained on sequences up to block_size; longer prompts use only the tail.
    bs = model.block_size
    if len(prompt) > bs:
        prompt = prompt[-bs:]
    idx = text_to_tensor(prompt, stoi, device=device).view(1, -1)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return decode(out[0].tolist(), itos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prompt",
        type=str,
        default="The ",
        help="Starting string; every character must appear in the training text (vocabulary).",
    )
    p.add_argument("--max-new", type=int, default=None, help="How many new characters to generate (default: config).")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature (default: config). Lower = more focused, higher = more random.",
    )
    p.add_argument("--checkpoint", type=str, default=Config.checkpoint, help="Path to the .pt file from train.py.")
    args = p.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_file():
        raise SystemExit(f"No checkpoint at {ck_path}. Train first: python train.py")

    max_new = args.max_new if args.max_new is not None else Config.max_new_tokens
    temp = args.temperature if args.temperature is not None else Config.temperature
    device = Config.device

    model, stoi, itos = load_trained_gpt(ck_path, device=device)
    try:
        text = generate_continuation(model, stoi, itos, args.prompt, max_new, temp, device)
    except ValueError as e:
        raise SystemExit(str(e))
    print(text)


if __name__ == "__main__":
    main()
