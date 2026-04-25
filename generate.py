"""Load checkpoint and sample text from the model."""

import argparse
from pathlib import Path

import torch

from config import Config
from model import GPT
from utils import decode, maps_from_vocab, text_to_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="The ", help="Starting string (chars must be in vocab)")
    p.add_argument("--max-new", type=int, default=None, help="Tokens to generate (default: config)")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: config)")
    p.add_argument("--checkpoint", type=str, default=Config.checkpoint)
    args = p.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_file():
        raise SystemExit(f"No checkpoint at {ck_path}. Train first: python train.py")

    max_new = args.max_new if args.max_new is not None else Config.max_new_tokens
    temp = args.temperature if args.temperature is not None else Config.temperature
    device = Config.device

    try:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ck_path, map_location=device)
    vocab = ck["vocab"]
    h = ck["hparams"]
    vocab_size, _stoi, itos = maps_from_vocab(vocab)

    for ch in args.prompt:
        if ch not in _stoi:
            raise SystemExit(f"Character {ch!r} is not in the training vocabulary.")

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

    idx = text_to_tensor(args.prompt, _stoi, device=device).view(1, -1)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new, temperature=temp)
    text = decode(out[0].tolist(), itos)
    print(text)


if __name__ == "__main__":
    main()
