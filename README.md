# baby-gpt

Minimal **char-level** GPT in PyTorch: config, data utils, small causal Transformer, `train.py`, and `generate.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Checkpoint: `models/baby_gpt.pt` (see `config.py`). Quick test: `MAX_ITERS=100 python train.py` (PowerShell: `$env:MAX_ITERS="100"; python train.py`).

## Sample

```bash
python generate.py --prompt "The " --max-new 200
```

## Data

Put plain text in `data/input.txt` (path in `config.py`).
