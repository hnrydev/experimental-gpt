"""
Train the GPT on the corpus: character tokens by default, or BPE (subword) with BABY_GPT_BPE=1.

Environment:
  MAX_ITERS           — override step count
  BABY_GPT_BPE=1     — BPE mode (install: pip install tokenizers)
  SAMPLE_EVERY=N     — if N>0, every N steps print a short greedy sample (local only)
  SAMPLE_PROMPT=...  — prefix for the sample (default: "The ")
  RUN_EVAL=1         — after save, run sample_eval (fixed prompts -> outputs/eval_samples_*.txt)
"""

import os
import random
from pathlib import Path

import torch
from config import Config
from model import GPT
from utils import create_vocab, load_text, text_to_tensor

torch.manual_seed(Config.seed)
random.seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)


def get_batch(data, device):
    n = data.shape[0] - Config.block_size
    if n <= 0:
        raise SystemExit("Corpus too short: need at least block_size + 1 token.")
    ix = torch.randint(n, (Config.batch_size,))
    b = int(Config.block_size)
    # int() avoids device/index edge cases with 0-d CPU tensors and CUDA `data`
    x = torch.stack([data[int(i) : int(i) + b] for i in ix])
    y = torch.stack([data[int(i) + 1 : int(i) + b + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, device):
    model.eval()
    losses = []
    for _ in range(Config.eval_batches):
        x, y = get_batch(data, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def _print_train_sample(
    model: GPT,
    stoi: dict | None,
    itos: dict | None,
    bpe,
    device: str,
    step: int,
    prompt: str,
    n_tok: int,
) -> None:
    from generate import _sampling_from_config
    from utils import decode

    try:
        if bpe is not None:
            from bpe_text import encode_to_ids

            ids = encode_to_ids(bpe, prompt)
            if len(ids) > model.block_size:
                ids = ids[-model.block_size :]
            idx = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
        else:
            assert stoi is not None and itos is not None
            p = prompt[-model.block_size :] if len(prompt) > model.block_size else prompt
            t = [stoi[c] for c in p]
            idx = torch.tensor(t, dtype=torch.long, device=device).view(1, -1)
        rdef = _sampling_from_config()
        nrg = int(getattr(Config, "decode_no_repeat_ngram_bpe", 3)) if bpe is not None else int(
            getattr(Config, "decode_no_repeat_ngram_size", 4)
        )
        out = model.generate(
            idx,
            max_new_tokens=n_tok,
            temperature=0.6,
            top_k=20,
            top_p=0.9,
            repetition_penalty=rdef.get("repetition_penalty", 1.2),
            repetition_context_len=rdef.get("repetition_context_len", 20),
            greedy=True,
            no_repeat_ngram_size=nrg,
        )
        raw = out[0].tolist()
        text = bpe.decode(raw) if bpe is not None else decode(raw, itos)
        s = text.replace("\n", " ")[:220]
        more = "..." if len(s) == 220 else ""
        print(f"  [sample @ {step}] {s!r}{more}")
    except Exception as e:  # pragma: no cover
        print(f"  [sample @ {step}] (skip) {e}")


def main() -> None:
    text = load_text()

    bpe_tok = None
    vocab_str: str = ""

    if getattr(Config, "use_bpe", False):
        try:
            from bpe_text import encode_to_ids, save_bpe_tokenizer, train_bpe_tokenizer
        except ImportError as e:
            raise SystemExit("BABY_GPT_BPE=1 needs: pip install tokenizers (see requirements.txt).") from e
        out_ck = Path(Config.checkpoint)
        print("Training BPE tokenizer (local, once per run)...")
        bpe_tok = train_bpe_tokenizer(
            text,
            vocab_size=int(getattr(Config, "bpe_vocab_size", 2048)),
        )
        tok_path = save_bpe_tokenizer(bpe_tok, out_ck)
        n_vocab = bpe_tok.get_vocab_size()
        print(f"BPE vocabulary size: {n_vocab} | {tok_path.name}")
        all_ids: list[int] = []
        for i in range(0, len(text), 1_000_000):
            all_ids.extend(encode_to_ids(bpe_tok, text[i : i + 1_000_000]))
        data = torch.tensor(all_ids, dtype=torch.long, device=Config.device)
        vocab_size = n_vocab
        stoi, itos = None, None
    else:
        ns, stoi, itos, vocabulary = create_vocab(text)
        if ns < 2:
            raise SystemExit("Need at least 2 unique characters in the corpus (longer or richer text).")
        vocab_size = ns
        vocab_str = vocabulary
        data = text_to_tensor(text, stoi, device=Config.device)

    n_tok = data.shape[0]
    n_starts = n_tok - Config.block_size
    print(f"Tokenized length: {n_tok:,} | start positions: {n_starts:,} | mode: {'BPE' if bpe_tok else 'char'}")

    model = GPT(
        vocab_size=vocab_size,
        block_size=Config.block_size,
        n_embd=Config.n_embd,
        n_head=Config.n_head,
        n_layer=Config.n_layer,
        dropout=Config.dropout,
    )
    model = model.to(Config.device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    max_iters = int(os.environ.get("MAX_ITERS", str(Config.max_iters)))
    eta_min = float(getattr(Config, "min_learning_rate", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_iters, eta_min=eta_min
    )
    sample_every = int(os.environ.get("SAMPLE_EVERY", "0").strip() or 0)
    sample_prompt = os.environ.get("SAMPLE_PROMPT", "The ")

    for step in range(max_iters):
        x, y = get_batch(data, Config.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (step + 1) % Config.eval_interval == 0 or step == 0:
            l = estimate_loss(model, data, Config.device)
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step+1:5d} | loss {l:.4f} | batch {loss.item():.4f} | lr {lr:.2e}"
            )
        if sample_every > 0 and (step + 1) % sample_every == 0:
            _print_train_sample(
                model, stoi, itos, bpe_tok, Config.device, step + 1, sample_prompt, 64
            )

    final = estimate_loss(model, data, Config.device)
    print(f"final loss (avg over eval batches): {final:.4f}")

    out = Path(Config.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    hparams = {
        "block_size": Config.block_size,
        "n_embd": Config.n_embd,
        "n_head": Config.n_head,
        "n_layer": Config.n_layer,
        "dropout": Config.dropout,
    }
    if bpe_tok is not None:
        save_obj = {
            "state_dict": model.state_dict(),
            "hparams": hparams,
            "model_type": "bpe",
            "bpe_vocab_size": bpe_tok.get_vocab_size(),
        }
    else:
        save_obj = {
            "state_dict": model.state_dict(),
            "vocab": vocab_str,
            "hparams": hparams,
            "model_type": "char",
        }
    torch.save(save_obj, out)
    print(f"Saved {out} ({'BPE + .tokenizer.json' if bpe_tok else 'char'})")

    if os.environ.get("RUN_EVAL", "").strip().lower() in ("1", "true", "yes"):
        try:
            from sample_eval import run_fixed_prompt_eval

            pfile = Path(getattr(Config, "eval_prompts_file", "data/eval_prompts.txt"))
            w = run_fixed_prompt_eval(out, prompts_path=pfile)
            print(f"RUN_EVAL: wrote sample eval -> {w.resolve()}")
        except Exception as e:  # pragma: no cover
            print(f"RUN_EVAL failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
