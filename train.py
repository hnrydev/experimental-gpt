"""Train the char-level GPT on data/input.txt."""

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
        raise SystemExit("Text too short: need at least block_size + 1 characters.")
    ix = torch.randint(n, (Config.batch_size,))
    x = torch.stack([data[i : i + Config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + Config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, device):
    model.eval()
    losses = []
    n = data.shape[0] - Config.block_size
    for _ in range(Config.eval_batches):
        x, y = get_batch(data, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    text = load_text()
    vocab_size, stoi, _itos, vocab = create_vocab(text)
    if vocab_size < 2:
        raise SystemExit("Need at least 2 unique characters in the corpus (longer or richer text).")

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
    for step in range(max_iters):
        x, y = get_batch(data, Config.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % Config.eval_interval == 0 or step == 0:
            l = estimate_loss(model, data, Config.device)
            print(f"step {step+1:5d} | loss {l:.4f} | sample loss {loss.item():.4f}")

    final = estimate_loss(model, data, Config.device)
    print(f"final loss (avg over eval batches): {final:.4f}")

    out = Path(Config.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
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
