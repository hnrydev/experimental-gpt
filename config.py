"""
Central place for paths and hyperparameters.

Set environment variable BABY_GPT_FAST=1 for a **small** model that trains many
more steps in ~30 minutes on a CPU laptop (see README). Default = larger, slower
run when you have hours or a GPU.
"""

import os
import torch

_fast = os.environ.get("BABY_GPT_FAST", "").strip().lower() in ("1", "true", "yes")

if _fast:

    class Config:
        """Small model + shorter context: many steps per minute on a typical laptop CPU."""

        dataset = "data/input.txt"
        checkpoint = "models/baby_gpt_fast.pt"  # separate file so you do not overwrite a long run
        seed = 42
        clean_corpus = True

        batch_size = 32
        block_size = 64  # cheaper attention than 128
        max_iters = 500  # override with MAX_ITERS; ~400–800 often fits a 30 min budget on CPU
        eval_interval = 50
        eval_batches = 15
        learning_rate = 3e-4
        min_learning_rate = 1e-5
        weight_decay = 0.1

        n_embd = 96
        n_head = 4  # 96 / 4 = 24 per head
        n_layer = 3
        dropout = 0.1

        max_new_tokens = 300
        temperature = 0.6  # a bit sharper for a small model

        device = "cuda" if torch.cuda.is_available() else "cpu"

else:

    class Config:
        """Defaults for a multi-million-character corpus (slower; use a GPU or overnight CPU)."""

        dataset = "data/input.txt"
        checkpoint = "models/baby_gpt.pt"
        seed = 42
        clean_corpus = True

        batch_size = 24
        block_size = 128
        max_iters = 8000
        eval_interval = 800
        eval_batches = 30
        learning_rate = 2e-4
        min_learning_rate = 1e-5
        weight_decay = 0.1

        n_embd = 256
        n_head = 8
        n_layer = 6
        dropout = 0.1

        max_new_tokens = 300
        temperature = 0.65

        device = "cuda" if torch.cuda.is_available() else "cpu"
