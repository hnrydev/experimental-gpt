"""
Train the char-level GPT on the text in Config.dataset.

Pipeline in plain terms:
  1) Read file → string.
  2) Build char vocabulary → map string to a long 1D tensor of ids.
  3) Sample random windows: input x is `block_size` chars; target y is the same
     window shifted by one (each position predicts the *next* character).
  4) Minimize cross-entropy of predicted next-char vs true next char (AdamW).
  5) Save weights + vocab + architecture fields so generate.py can reload.

Environment: set MAX_ITERS to override Config.max_iters without editing files
(e.g. quick smoke tests).
"""

import os
import random
from pathlib import Path

import torch
from config import Config
from model import GPT
from utils import create_vocab, load_text, text_to_tensor

# Reproducibility: same seed → same random batches and init (given same PyTorch version).
torch.manual_seed(Config.seed)
random.seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)


def get_batch(data, device):
    """
    Build one training batch.

    `data` is the full corpus as a 1D tensor of token ids (length = num characters).
    We pick `batch_size` random *start indices* `i`; each row:
      x = data[i : i + block_size]
      y = data[i + 1 : i + block_size + 1]  # shifted: model at t predicts y[t] == data[i+t+1]

    So for each position in x, the correct "next token" label is the matching
    position in y — classic next-token (here next-character) modeling.
    """
    n = data.shape[0] - Config.block_size
    if n <= 0:
        raise SystemExit("Text too short: need at least block_size + 1 characters.")
    # Random starting positions so the model sees many parts of the file over time.
    ix = torch.randint(n, (Config.batch_size,))
    x = torch.stack([data[i : i + Config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + Config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, device):
    """Average loss over several random batches (no gradient; for monitoring)."""
    model.eval()  # e.g. turns off dropout for a fair loss readout
    losses = []
    n = data.shape[0] - Config.block_size
    for _ in range(Config.eval_batches):
        x, y = get_batch(data, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()  # back to training mode (dropout on)
    return sum(losses) / len(losses)


def main():
    text = load_text()
    vocab_size, stoi, _itos, vocab = create_vocab(text)
    if vocab_size < 2:
        raise SystemExit("Need at least 2 unique characters in the corpus (longer or richer text).")

    # Start on CPU then move to device — small tensors; avoids device mismatch bugs while building.
    data = text_to_tensor(text, stoi, device="cpu")
    n_tokens = data.numel()
    n_starts = n_tokens - Config.block_size
    print(f"Possible start positions (train window count): {n_starts}")

    model = GPT(
        vocab_size=vocab_size,
        block_size=Config.block_size,
        n_embd=Config.n_embd,
        n_head=Config.n_head,
        n_layer=Config.n_layer,
        dropout=Config.dropout,
    )
    model = model.to(Config.device)
    data = data.to(Config.device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    max_iters = int(os.environ.get("MAX_ITERS", str(Config.max_iters)))
    eta_min = float(getattr(Config, "min_learning_rate", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_iters, eta_min=eta_min
    )

    for step in range(max_iters):
        x, y = get_batch(data, Config.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip global norm of gradients (stops rare huge updates if loss spikes).
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (step + 1) % Config.eval_interval == 0 or step == 0:
            l = estimate_loss(model, data, Config.device)
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step+1:5d} | loss {l:.4f} | sample {loss.item():.4f} | lr {lr:.2e}"
            )

    final = estimate_loss(model, data, Config.device)
    print(f"final loss (avg over eval batches): {final:.4f}")

    out = Path(Config.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoint must include vocabulary + hparams: raw state_dict alone is not enough to sample.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "hparams": {
                "block_size": Config.block_size,
                "n_embd": Config.n_embd,
                "n_head": Config.n_head,
                "n_layer": Config.n_layer,
                "dropout": Config.dropout,
            },
        },
        out,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
