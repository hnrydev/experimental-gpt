"""
Load a trained checkpoint and sample new text (char or BPE tokens).

Unlike training, we do not have "correct" next characters: we take the model's
predicted distribution at the last position, sample one id, append it, and repeat.
That is "autoregressive" generation. Temperature changes how random vs focused
sampling is; with `--greedy` we take the argmax each step (no randomness).
"""

import argparse
from pathlib import Path

import torch

# Default checkpoint names (fast / full, char / BPE) for error messages
_CKPT_NAMES = (
    "baby_gpt_fast.pt",
    "baby_gpt_fast_bpe.pt",
    "baby_gpt.pt",
    "baby_gpt_bpe.pt",
)

from config import Config
from local_text_fix import local_sensible_postprocess
from model import GPT
from utils import decode, maps_from_vocab, maybe_torch_compile, text_to_tensor


def clamp_prompt_to_block(prompt: str, block_size: int) -> str:
    """
    The model can only see `block_size` characters. Training and inference
    use the *tail* of a long prompt — UI must strip the same prefix.
    """
    if len(prompt) > block_size:
        return prompt[-block_size:]
    return prompt


def effective_prompt_string(prompt: str, block_size: int, bpe_tok: object = None) -> str:
    """
    The prefix the model *actually* sees, as a string (for chat strip). For BPE, this is
    the decode of the last `block_size` *tokens* (not characters).
    """
    if bpe_tok is None:
        return clamp_prompt_to_block(prompt, block_size)
    from bpe_text import decode_from_ids, encode_to_ids

    ids = encode_to_ids(bpe_tok, prompt)
    if len(ids) > block_size:
        ids = ids[-block_size:]
    return decode_from_ids(bpe_tok, ids)


def _sampling_from_config():
    return {
        "temperature": getattr(Config, "temperature", 0.5),
        "top_k": getattr(Config, "top_k", 32),
        "top_p": getattr(Config, "top_p", 0.88),
        "repetition_penalty": getattr(Config, "repetition_penalty", 1.2),
        "repetition_context_len": getattr(Config, "repetition_context_len", 20),
    }


def _no_repeat_ngram_size(bpe_tok: object | None) -> int:
    """Char: block duplicate 4-grams; BPE: n-gram repeat block in token space (set 0 in config to disable)."""
    if bpe_tok is not None:
        return int(getattr(Config, "decode_no_repeat_ngram_bpe", 4))
    return int(getattr(Config, "decode_no_repeat_ngram_size", 4))


def load_trained_gpt(ck_path: str | Path, device: str | None = None):
    """
    Load weights and tokenizer from train.py.

    Returns ``(model, stoi, itos, bpe_tok)``. For BPE models, ``stoi``/``itos`` are
    None and ``bpe_tok`` is set; for character models, ``bpe_tok`` is None.
    """
    device = device or Config.device
    ck_path = Path(ck_path)
    try:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ck_path, map_location=device)
    h = ck["hparams"]
    mtype = ck.get("model_type", "char")

    bpe_tok = None
    stoi, itos = None, None
    if mtype == "bpe":
        try:
            from bpe_text import load_bpe_tokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "This checkpoint is BPE. Install: pip install tokenizers (see requirements.txt)."
            ) from e
        bpe_tok = load_bpe_tokenizer(ck_path)
        vocab_size = int(ck.get("bpe_vocab_size") or bpe_tok.get_vocab_size())
    else:
        vocab = ck.get("vocab")
        if not vocab:
            raise ValueError("Checkpoint is missing 'vocab' (not a BPE or old char checkpoint).")
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
    model = maybe_torch_compile(model)
    return model, stoi, itos, bpe_tok


def missing_checkpoint_message(ck_path: Path) -> str:
    here = ck_path.parent.resolve()
    known = " | ".join(str(here / n) for n in _CKPT_NAMES)
    return (
        f"No file at {ck_path}.\n"
        "Train first: `python train.py` (char) or `$env:BABY_GPT_BPE='1'; python train.py` (BPE). "
        f"Point --checkpoint at the matching .pt (e.g. {known})."
    )


def generate_continuation(
    model: GPT,
    stoi: dict | None,
    itos: dict | None,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    repetition_context_len: int | None = None,
    greedy: bool = False,
    bpe_tok: object = None,
) -> str:
    """Autoregressive sampling; return full string (clamped prefix + new tokens). BPE: `max_new` is *tokens*."""
    if bpe_tok is not None:
        from bpe_text import encode_to_ids

        ids = encode_to_ids(bpe_tok, prompt)
        if len(ids) > model.block_size:
            ids = ids[-model.block_size :]
        idx = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
    else:
        assert stoi is not None and itos is not None
        for ch in prompt:
            if ch not in stoi:
                raise ValueError(f"Character {ch!r} is not in the training vocabulary.")
        prompt = clamp_prompt_to_block(prompt, model.block_size)
        idx = text_to_tensor(prompt, stoi, device=device).view(1, -1)
    base = _sampling_from_config()
    rep = repetition_penalty if repetition_penalty is not None else base["repetition_penalty"]
    rcl = repetition_context_len if repetition_context_len is not None else base["repetition_context_len"]
    tk = top_k if top_k is not None else base["top_k"]
    tp = top_p if top_p is not None else base["top_p"]
    nrg = _no_repeat_ngram_size(bpe_tok)
    with torch.no_grad():
        if greedy:
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=None,
                top_p=None,
                repetition_penalty=rep,
                repetition_context_len=rcl,
                greedy=True,
                no_repeat_ngram_size=nrg,
            )
        else:
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=tk,
                top_p=tp,
                repetition_penalty=rep,
                repetition_context_len=rcl,
                greedy=False,
                no_repeat_ngram_size=nrg,
            )
    # Post-process only the *new* continuation, not the prompt prefix. Running
    # light_surface_english on the full string capitalized the first char of the prompt
    # ("what" -> "What"), breaking startswith(p_eff) in chat and looked like an echo.
    seq = out[0]
    n0 = int(idx.shape[1])
    all_ids = seq.tolist()
    if bpe_tok is not None:
        from bpe_text import decode_from_ids

        prefix_text = decode_from_ids(bpe_tok, all_ids[:n0])
        new_text = decode_from_ids(bpe_tok, all_ids[n0:])
    else:
        prefix_text = decode(all_ids[:n0], itos)
        new_text = decode(all_ids[n0:], itos)
    mcr = int(getattr(Config, "local_max_char_run", 3))
    surf = bool(getattr(Config, "local_surface_english", False))
    gram = bool(getattr(Config, "local_grammar_tweaks", True))
    new_f = local_sensible_postprocess(
        new_text,
        max_same=mcr,
        trim_dup_tail=True,
        surface_english=surf,
        grammar_tweaks=gram,
    )
    return prefix_text + new_f


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prompt",
        type=str,
        default="The ",
        help="Input prefix. Char checkpoint: only characters from training. BPE: any UTF-8 (ByteLevel).",
    )
    p.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="How many new tokens to generate: characters for char checkpoints, subword steps for BPE (default: config).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature (default: config). Lower = more focused, higher = more random.",
    )
    p.add_argument("--checkpoint", type=str, default=Config.checkpoint, help="Path to the .pt file from train.py.")
    p.add_argument("--top-k", type=int, default=None, help="Limit sampling to top-k chars (default: config).")
    p.add_argument("--top-p", type=float, default=None, help="Nucleus sampling p (default: config).")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Down-weight recently seen chars; >1 reduces stutter (default: config).",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Argmax next character each step (no sampling). Ignores temperature / top-k / top-p.",
    )
    args = p.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_file():
        raise SystemExit(missing_checkpoint_message(ck_path))

    max_new = args.max_new if args.max_new is not None else Config.max_new_tokens
    temp = args.temperature if args.temperature is not None else Config.temperature
    device = Config.device

    model, stoi, itos, bpe = load_trained_gpt(ck_path, device=device)
    try:
        text = generate_continuation(
            model,
            stoi,
            itos,
            args.prompt,
            max_new,
            temp,
            device,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            greedy=args.greedy,
            bpe_tok=bpe,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    print(text)


if __name__ == "__main__":
    main()
