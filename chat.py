"""
Interactive session: you type a line, the model continues in *training* style
(classic LM continuation, not chat-tuned dialogue).

The model was trained on book-like text. Short chat prompts often produce weak
or blank-looking output — use several words of literate English for better results.
"""

import argparse
from pathlib import Path

from config import Config
from generate import (
    effective_prompt_string,
    generate_continuation,
    load_trained_gpt,
    missing_checkpoint_message,
)
from local_text_fix import trim_excessive_leading_newlines


def _suffix_after_model_prefix(full: str, effective_prompt: str) -> str:
    """Strip the same prefix the model saw (exact, or case-insensitive if lengths match)."""
    if full.startswith(effective_prompt):
        return full[len(effective_prompt) :]
    n = len(effective_prompt)
    if n and len(full) >= n and full[:n].lower() == effective_prompt.lower():
        return full[n:]
    return full


def _format_continuation(suffix: str) -> str:
    """If the model only generated whitespace, show a visible hint (common for small LMs)."""
    suffix = trim_excessive_leading_newlines(suffix, max_leading=2)
    if suffix.strip():
        return suffix.lstrip()
    if not suffix:
        return (
            "[no characters generated after your line — retrain, or try a 1–2 sentence book-style prompt]"
        )
    return f"(only blank characters: {repr(suffix[:120])})"


def _resolve_chat_sampling(args, coherent: bool):
    """Defaults are strict for small LMs; use --looser to widen sampling; --coherent is stricter still."""
    c = Config
    cap = int(getattr(c, "chat_safety_max_new", 200))
    if coherent:
        base_t = getattr(c, "chat_coherent_temperature", 0.45)
        base_p = getattr(c, "chat_coherent_top_p", 0.78)
        base_k = getattr(c, "chat_coherent_top_k", 12)
        base_r = getattr(c, "chat_coherent_repetition_penalty", 1.6)
        base_max = int(getattr(c, "chat_coherent_max_new", 120))
    else:
        base_t = getattr(c, "chat_temperature", c.temperature)
        base_p = getattr(c, "chat_top_p", c.top_p)
        # Finite top-k reduces junk tokens vs leaving None (full generate default top_k).
        base_k = int(getattr(c, "chat_top_k", 16))
        base_r = getattr(c, "chat_repetition_penalty", c.repetition_penalty)
        # Always cap by chat_safety_max_new to limit runaway gibberish
        base_max = min(int(c.max_new_tokens), cap)
    if args.looser:
        base_t = min(base_t + 0.2, 1.0)
        base_p = min(base_p + 0.1, 0.95)
        base_k = min(int(base_k * 1.4) + 4, 48)
    temp = base_t if args.temperature is None else args.temperature
    top_p = base_p if args.top_p is None else args.top_p
    if args.top_k is not None:
        top_k = args.top_k
    else:
        top_k = base_k
    rep = base_r if args.repetition_penalty is None else args.repetition_penalty
    if args.max_new is not None:
        max_new = args.max_new
    else:
        max_new = base_max
    return max_new, temp, top_p, top_k, rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=Config.checkpoint)
    p.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="New tokens to generate (char steps for char ckpt, BPE steps for BPE; default: config).",
    )
    p.add_argument(
        "--coherent",
        action="store_true",
        help="Tighter sampling (lower temp / top_p, smaller top_k): less gibberish, more boring. "
        "Does not fix an undertrained model; combine with more training steps.",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Argmax decoding (no randomness). Often less pseudo-word noise than sampling; "
        "can repeat; --coherent is ignored for decoding when this is set.",
    )
    p.add_argument(
        "--looser",
        action="store_true",
        help="Nudge sampling toward higher temp / top_p / top_k (more random, often messier on tiny LMs).",
    )
    p.add_argument("--temperature", type=float, default=None, help="Override chat_temperature")
    p.add_argument("--top-k", type=int, default=None, help="Top-k (default: config)")
    p.add_argument("--top-p", type=float, default=None, help="Nucleus p (default: chat_top_p in config)")
    p.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (default: chat_*)")
    args = p.parse_args()

    max_new, temp, top_p, top_k, rep = _resolve_chat_sampling(args, args.coherent)
    if args.greedy and args.max_new is None:
        max_new = getattr(Config, "chat_greedy_max_new", 220)
    device = Config.device
    suffix = getattr(Config, "chat_prompt_suffix", "\n")

    ck_path = Path(args.checkpoint)
    if not ck_path.is_file():
        raise SystemExit(missing_checkpoint_message(ck_path))
    try:
        model, stoi, itos, bpe = load_trained_gpt(args.checkpoint, device=device)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    except ImportError as e:
        raise SystemExit(str(e)) from e

    print("Baby GPT — type a line (or quit to exit).")
    if bpe is not None:
        print("(Loaded BPE / subword checkpoint.)\n")
    _tok = "token" if bpe is not None else "character"
    if args.greedy:
        print(f"Mode: --greedy (argmax each next {_tok}; no sampling — still not a chat model).\n")
    elif args.coherent:
        print("Mode: --coherent (strictest sampling; best match for a small continuation LM).\n")
    elif args.looser:
        print("Mode: --looser (wider sampling — you asked for it).\n")
    else:
        print("Default: strict chat decoding (see config: chat_*, top_k) — not a real chat model.\n")
    if not (args.greedy or args.looser) and not args.coherent:
        print(
            "Stricter: `python chat.py --coherent`  |  Even less noise: `python chat.py --greedy`  |  "
            "More variety (messier): `python chat.py --looser`  |  train longer: MAX_ITERS=5000+ .\n"
        )
    print(
        "This is a *text continuation* model (not chat-tuned), trained on text — not a yes/no assistant.\n"
    )

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        raw_prompt = line + suffix
        p_eff = effective_prompt_string(raw_prompt, model.block_size, bpe)
        rctx = getattr(Config, "chat_repetition_context_len", None)
        try:
            out = generate_continuation(
                model,
                stoi,
                itos,
                raw_prompt,
                max_new,
                temp,
                device,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep,
                repetition_context_len=rctx,
                greedy=args.greedy,
                bpe_tok=bpe,
            )
        except ValueError as e:
            print(f"  (skip) {e}")
            if bpe is None:
                print("  (char model: every character in your line must appear in the training text.)\n")
            else:
                print("  (BPE: use normal UTF-8; if this persists, file an issue.)\n")
            continue
        limit = "tokens" if bpe is not None else "characters"
        n_ctx = len(bpe.encode(raw_prompt).ids) if bpe is not None else len(raw_prompt)
        if n_ctx > model.block_size:
            print(f"  (note: only the last {model.block_size} {limit} of your line were used as context)\n")
        suffix_text = _suffix_after_model_prefix(out, p_eff)
        shown = _format_continuation(suffix_text)
        if line.strip() and suffix_text.strip() and line.strip().lower() == suffix_text.strip().lower():
            shown = (
                f"{shown}\n  "
                f"(The model’s continuation matched your line — a base LM often copies; "
                f"try a longer book-style line, `python chat.py --coherent`, or a BPE checkpoint.)\n"
            )
        print(f"GPT> {shown}\n")


if __name__ == "__main__":
    main()
