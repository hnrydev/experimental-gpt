import torch


class Config:
    """Training, model, and paths."""

    # data
    dataset = "data/input.txt"
    checkpoint = "models/baby_gpt.pt"
    seed = 42

    # training
    batch_size = 32
    block_size = 64
    max_iters = 3000
    eval_interval = 300
    eval_batches = 20
    learning_rate = 3e-4
    weight_decay = 0.1

    # model
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.1

    # generation (default for generate.py)
    max_new_tokens = 200
    temperature = 0.8

    device = "cuda" if torch.cuda.is_available() else "cpu"
