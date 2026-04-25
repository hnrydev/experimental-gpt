"""
Interactive "chat" = type a prompt, see the model continue the text.

This is still a *character-level* language model: it was trained to predict
the next character in your `data/input.txt` style, not to hold a real dialogue.
You get open-ended *continuation*, like a fancy autocomplete.

Rules:
  - Use only characters that appear in the training data (same as generate.py).
  - Type a line and press Enter. Empty line or "quit" / "exit" stops.
  - Long lines are truncated to the last `block_size` characters (context limit).
"""

import argparse

from config import Config
from generate import generate_continuation, load_trained_gpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=Config.checkpoint)
    p.add_argument("--max-new", type=int, default=None, help="New characters per turn (default: config)")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: config)")
    args = p.parse_args()

    max_new = args.max_new if args.max_new is not None else Config.max_new_tokens
    temp = args.temperature if args.temperature is not None else Config.temperature
    device = Config.device

    model, stoi, itos = load_trained_gpt(args.checkpoint, device=device)
    print("Baby GPT — type a prompt (or quit to exit). Continuations are from your training text style.\n")

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        try:
            out = generate_continuation(model, stoi, itos, line, max_new, temp, device)
        except ValueError as e:
            print(f"  (skip) {e}")
            print("  Tip: only use letters/symbols that exist in data/input.txt.\n")
            continue
        if len(line) > model.block_size:
            print(f"  (used last {model.block_size} characters as context)\n")
        print(f"GPT> {out}\n")


if __name__ == "__main__":
    main()
