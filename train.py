"""
Train the GPT on the corpus: character tokens by default, or BPE (subword) with BABY_GPT_BPE=1.

Environment:
  MAX_ITERS           — override step count
  BABY_GPT_BPE=1     — BPE mode (install: pip install tokenizers)
  SAMPLE_EVERY=N     — if N>0, every N steps print a short greedy sample (local only)
  SAMPLE_PROMPT=...  — prefix for the sample (default: "The ")
  RUN_EVAL=1         — after save, run sample_eval (fixed prompts -> outputs/eval_samples_*.txt)
  VAL_FRACTION=0     — disable train/val split (else uses Config.val_fraction, last slice = val)
  EARLY_STOP_PATIENCE=0  — with val, stop after this many val evals without val improvement; 0 = off
  EARLY_STOP_MIN_DELTA=0  — new val best only if l_va drops by this much
  LR_WARMUP_ITERS   — override linear ramp before cosine (0 = no warmup; default from Config)
  BABY_GPT_NUM_THREADS  — e.g. 4 to cap CPU BLAS/OMP threads (ThinkPad)
  BABY_GPT_COMPILE=1  — torch.compile the model (PyTorch 2+; may speed CPU train/infer)
"""

import os
import random
from pathlib import Path

import torch
from config import Config
from model import GPT
from utils import create_vocab, load_text, maybe_torch_compile, text_to_tensor

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
        nrg = int(getattr(Config, "decode_no_repeat_ngram_bpe", 4)) if bpe is not None else int(
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


def _maybe_set_cpu_threads() -> None:
    raw = os.environ.get("BABY_GPT_NUM_THREADS", "").strip()
    if not raw:
        return
    try:
        n = int(raw)
    except ValueError:
        return
    if n < 1:
        return
    before = torch.get_num_threads()
    torch.set_num_threads(n)
    print(f"BABY_GPT_NUM_THREADS: OpenMP/BLAS threads {before} -> {n}")


def main() -> None:
    _maybe_set_cpu_threads()
    text = load_text()

    bpe_tok = None
    vocab_str: str = ""

    if getattr(Config, "use_bpe", False):
        try:
            from bpe_text import encode_to_ids, save_bpe_tokenizer, train_bpe_tokenizer
        except ImportError as e:
            raise SystemExit("BABY_GPT_BPE=1 needs: pip install tokenizers (see requirements.txt).") from e
        out_ck = Path(Config.checkpoint)
        n_char = len(text)
        print(
            f"Training BPE tokenizer (local, once per run) on {n_char:,} characters...",
            flush=True,
        )
        bpe_tok = train_bpe_tokenizer(
            text,
            vocab_size=int(getattr(Config, "bpe_vocab_size", 2048)),
        )
        tok_path = save_bpe_tokenizer(bpe_tok, out_ck)
        n_vocab = bpe_tok.get_vocab_size()
        print(f"BPE vocabulary size: {n_vocab} | {tok_path.name}", flush=True)
        all_ids: list[int] = []
        step = 1_000_000
        n_chunks = (n_char + step - 1) // step
        for ci, i in enumerate(range(0, n_char, step)):
            print(
                f"  BPE encode chunk {ci + 1}/{n_chunks} (chars {i:,}…{min(i + step, n_char):,})",
                flush=True,
            )
            all_ids.extend(encode_to_ids(bpe_tok, text[i : i + step]))
        print("  Building token tensor on device…", flush=True)
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
    b = int(Config.block_size)
    val_fr = float(os.environ.get("VAL_FRACTION", str(getattr(Config, "val_fraction", 0.0))))
    if val_fr > 0.001 and n_tok * val_fr > b * 2 + 2000:
        n_tr = int(n_tok * (1.0 - val_fr))
        n_tr = max(n_tr, b + Config.batch_size + 2)
        if n_tok - n_tr <= b + 2:
            train_data, val_data = data, data
            use_val = False
            print("Corpus: val split skipped (tail too short after cut).")
        else:
            train_data = data[:n_tr].contiguous()
            val_data = data[n_tr:].contiguous()
            use_val = True
            print(
                f"Train/val split: {n_tr:,} train / {n_tok - n_tr:,} val tokens "
                f"(~{val_fr*100:.0f}% held out; VAL_FRACTION=0 to disable)"
            )
    else:
        train_data, val_data = data, data
        use_val = False
        if val_fr > 0.001:
            print("val_fraction set but corpus too small for a split; using all tokens for training + loss.")

    n_starts = int(train_data.shape[0]) - b
    print(
        f"Tokenized length: {n_tok:,} | train start positions: {n_starts:,} | mode: "
        f"{'BPE' if bpe_tok else 'char'}"
    )
    if use_val:
        n_vs = int(val_data.shape[0]) - b
        print(f"  val start positions: {n_vs:,} (val loss = generalization on held-out tail)")

    model = GPT(
        vocab_size=vocab_size,
        block_size=Config.block_size,
        n_embd=Config.n_embd,
        n_head=Config.n_head,
        n_layer=Config.n_layer,
        dropout=Config.dropout,
    )
    model = model.to(Config.device)
    model = maybe_torch_compile(model)

    opt = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    max_iters = int(os.environ.get("MAX_ITERS", str(Config.max_iters)))
    eta_min = float(getattr(Config, "min_learning_rate", 1e-5))
    warm = int(
        float(
            (os.environ.get("LR_WARMUP_ITERS", str(getattr(Config, "lr_warmup_iters", 0)))).strip()
        )
    )
    warm = max(0, min(warm, max(0, max_iters - 1)))
    if warm > 0 and max_iters - warm > 0:
        wsched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warm
        )
        csched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_iters - warm, eta_min=eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, [wsched, csched], milestones=[warm]
        )
        print(
            f"LR schedule: linear warmup {warm} iters, then cosine (T_max={max_iters - warm} -> {eta_min:g})"
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_iters, eta_min=eta_min
        )
        print(f"LR schedule: cosine to {eta_min:g} (T_max={max_iters}, no warmup)")
    sample_every = int(os.environ.get("SAMPLE_EVERY", "0").strip() or 0)
    sample_prompt = os.environ.get("SAMPLE_PROMPT", "The ")

    es_p = int(
        float(os.environ.get("EARLY_STOP_PATIENCE", str(getattr(Config, "early_stop_patience", 0))).strip())
    )
    es_mind = float(
        os.environ.get(
            "EARLY_STOP_MIN_DELTA", str(getattr(Config, "early_stop_min_delta", 0.0))
        ).strip()
    )
    do_early = use_val and es_p > 0
    best_state: dict | None = None
    if do_early:
        best_val = float("inf")
        val_no_improve = 0
        print(
            f"Early stop: val patience {es_p} evals (min_delta={es_mind}) — "
            "checkpoint = best val so far, not the last step."
        )

    for step in range(max_iters):
        x, y = get_batch(train_data, Config.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (step + 1) % Config.eval_interval == 0 or step == 0:
            l_tr = estimate_loss(model, train_data, Config.device)
            lr = scheduler.get_last_lr()[0]
            if use_val:
                l_va = estimate_loss(model, val_data, Config.device)
                if do_early:
                    if l_va < best_val - es_mind:
                        best_val = l_va
                        val_no_improve = 0
                        best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                    else:
                        val_no_improve += 1
                print(
                    f"step {step+1:5d} | val {l_va:.4f} | train {l_tr:.4f} | "
                    f"batch {loss.item():.4f} | lr {lr:.2e}"
                )
                if do_early and val_no_improve >= es_p:
                    print(
                        f"  Early stop: no val gain for {es_p} evals; best val {best_val:.4f}."
                    )
                    break
            else:
                print(
                    f"step {step+1:5d} | loss {l_tr:.4f} | batch {loss.item():.4f} | lr {lr:.2e}"
                )
        if sample_every > 0 and (step + 1) % sample_every == 0:
            _print_train_sample(
                model, stoi, itos, bpe_tok, Config.device, step + 1, sample_prompt, 64
            )

    if do_early and best_state is not None:
        model.load_state_dict(best_state)
        print("Restored val-best weights for save.")

    l_tr = estimate_loss(model, train_data, Config.device)
    if use_val:
        l_va = estimate_loss(model, val_data, Config.device)
        print(f"final: val {l_va:.4f} | train {l_tr:.4f}  (if val > train, overfitting; if both high, underfit or hard data)")
    else:
        print(f"final loss (avg over eval batches): {l_tr:.4f}")

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
