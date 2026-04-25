"""
Run the trained LM on a fixed set of prompts and write continuations to a file.

Use this to compare *grammar and sensibility* across training runs, not just val loss.
Default decoding favors stability (config chat_coherent_*), matching the "good text" goal.

  python sample_eval.py
  python sample_eval.py --checkpoint models/baby_gpt_fast_bpe.pt
  $env:BABY_GPT_BPE="1"; python sample_eval.py

Train then auto-eval (optional):

  $env:RUN_EVAL="1"; python train.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from config import Config
from generate import (
    effective_prompt_string,
    generate_continuation,
    load_trained_gpt,
    missing_checkpoint_message,
)


def default_prompts_path() -> Path:
    return Path(getattr(Config, "eval_prompts_file", "data/eval_prompts.txt"))


def load_eval_prompts(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Eval prompts not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    if not out:
        raise ValueError(f"No prompts in {path} (after removing blanks and # comments).")
    return out


def _suffix_after_effective_prefix(full: str, effective: str) -> str:
    if not effective:
        return full
    if full.startswith(effective):
        return full[len(effective) :]
    n = len(effective)
    if n and len(full) >= n and full[:n].lower() == effective.lower():
        return full[n:]
    return full


def run_fixed_prompt_eval(
    ck_path: Path,
    prompts_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    coherent: bool = True,
    greedy: bool = False,
    max_new: int | None = None,
    device: str | None = None,
) -> Path:
    """
    Load checkpoint, run each eval prompt, write a timestamped report under out_dir.
    Returns path to the written file.
    """
    ck_path = Path(ck_path)
    if not ck_path.is_file():
        raise FileNotFoundError(missing_checkpoint_message(ck_path))

    prompts_path = prompts_path or default_prompts_path()
    prompts = load_eval_prompts(Path(prompts_path))
    out_dir = out_dir or Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = device or Config.device
    model, stoi, itos, bpe = load_trained_gpt(ck_path, device=dev)
    block = model.block_size

    if greedy:
        # Argmax: temperature / top_p ignored inside generate; use shorter cap like chat --greedy.
        temp = 1.0
        top_p = 1.0
        top_k = 0
        rep = float(getattr(Config, "repetition_penalty", 1.22))
        rcl = int(getattr(Config, "repetition_context_len", 24))
        mx = max_new if max_new is not None else int(getattr(Config, "chat_greedy_max_new", 180))
    elif coherent:
        temp = float(getattr(Config, "chat_coherent_temperature", 0.55))
        top_p = float(getattr(Config, "chat_coherent_top_p", 0.85))
        top_k = int(getattr(Config, "chat_coherent_top_k", 20))
        rep = float(getattr(Config, "chat_coherent_repetition_penalty", 1.55))
        rcl = int(getattr(Config, "chat_repetition_context_len", 48))
        mx = max_new if max_new is not None else int(getattr(Config, "chat_coherent_max_new", 200))
    else:
        temp = float(getattr(Config, "temperature", 0.5))
        top_p = float(getattr(Config, "top_p", 0.88))
        top_k = int(getattr(Config, "top_k", 32))
        rep = float(getattr(Config, "repetition_penalty", 1.22))
        rcl = int(getattr(Config, "repetition_context_len", 24))
        mx = max_new if max_new is not None else int(getattr(Config, "max_new_tokens", 300))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode = "greedy" if greedy else ("coherent" if coherent else "sampling")
    out_path = out_dir / f"eval_samples_{ts}_{mode}.txt"

    if greedy:
        decode_line = f"decode: greedy argmax | rep={rep} rcl={rcl}"
    else:
        decode_line = f"decode: temp={temp} top_p={top_p} top_k={top_k} rep={rep} rcl={rcl}"
    n_l = len(model.blocks) if hasattr(model, "blocks") and model.blocks is not None else "n/a"
    lines: list[str] = [
        "sample_eval: fixed prompts for qualitative comparison (grammar, sense).",
        f"checkpoint: {ck_path.resolve()}",
        f"coherent: {coherent} | greedy: {greedy} | max_new: {mx}",
        decode_line,
        f"model: block_size={block} n_layer={n_l}",
        "",
    ]

    suffix_nl = str(getattr(Config, "chat_prompt_suffix", "\n") or "")

    for i, prompt in enumerate(prompts, 1):
        p_in = prompt
        if not p_in.endswith("\n") and suffix_nl:
            p_in = p_in + suffix_nl
        full = generate_continuation(
            model,
            stoi,
            itos,
            p_in,
            mx,
            temp,
            dev,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep,
            repetition_context_len=rcl,
            greedy=greedy,
            bpe_tok=bpe,
        )
        eff = effective_prompt_string(p_in, block, bpe)
        cont = _suffix_after_effective_prefix(full, eff)
        lines.append("-" * 72)
        lines.append(f"[{i}/{len(prompts)}] PROMPT (as typed, before suffix):")
        lines.append(prompt)
        lines.append("CONTINUATION (model, post local_text_fix on new part only):")
        lines.append(cont if cont.strip() else "(empty or whitespace)")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Run fixed eval prompts; write continuations to outputs/.")
    p.add_argument("--checkpoint", type=str, default=Config.checkpoint, help="Path to .pt from train.py")
    p.add_argument("--prompts", type=str, default=None, help="Path to eval prompt list (default: config eval_prompts_file)")
    p.add_argument("--out-dir", type=str, default="outputs", help="Directory for eval_*.txt")
    p.add_argument(
        "--no-coherent",
        action="store_true",
        help="Use Config temperature / top_p / top_k (looser) instead of chat_coherent_*",
    )
    p.add_argument("--greedy", action="store_true", help="Argmax decoding (ignores --no-coherent sampling knobs)")
    p.add_argument("--max-new", type=int, default=None, help="Override max new tokens")
    args = p.parse_args()

    pp: Path = default_prompts_path() if args.prompts is None else Path(args.prompts)
    use_coherent = not args.no_coherent and not args.greedy
    out = run_fixed_prompt_eval(
        Path(args.checkpoint),
        prompts_path=pp,
        out_dir=Path(args.out_dir),
        coherent=use_coherent,
        greedy=args.greedy,
        max_new=args.max_new,
    )
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
