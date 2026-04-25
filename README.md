# baby-gpt

Minimal **char-level** GPT in PyTorch: config, data utils, small causal Transformer, `train.py`, `generate.py`, and interactive `chat.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

1. **Put text** in `data/input.txt` (the model only learns from this file).
2. **Train** (writes a checkpoint — path depends on profile, see below):

```bash
python train.py
```

Quick dry run: `MAX_ITERS=100 python train.py` (PowerShell: `$env:MAX_ITERS="100"; python train.py`).

### ~30 minutes on a laptop (CPU, no GPU)

You cannot run the *full* large model for thousands of steps in 30 minutes on a typical **Lenovo / CPU-only** machine — but you *can* fit a **smaller** model and **hundreds of steps**, which is enough to see clearer text than a 3-step smoke test.

1. Set the **fast profile** (smaller width/depth, `block_size` 64, saves `models/baby_gpt_fast.pt`).
2. Set **step count** from a quick timing run: `$env:BABY_GPT_FAST="1"; $env:MAX_ITERS="10"; python train.py` and note seconds ÷ 10 = seconds per step. For **~30 minutes**, `MAX_ITERS ≈ floor(1800 / seconds_per_step)` (often **~250–500** on a laptop CPU — **yours will vary**).

PowerShell (example):

```powershell
$env:BABY_GPT_FAST="1"
$env:MAX_ITERS="400"
python train.py
```

**Chat / generate after a fast train** — use the same profile so the default checkpoint path matches:

```powershell
$env:BABY_GPT_FAST="1"
python chat.py
```

Or explicitly: `python chat.py --checkpoint models/baby_gpt_fast.pt`

**Realistic expectation:** output will not match a big GPU run or ChatGPT; you are trading **quality for time**. For the **big** model + long runs, unset `BABY_GPT_FAST` and use a GPU or run overnight.

3. **Generate one continuation** (prompt must use characters that appear in `data/input.txt`):

```bash
python generate.py --prompt "Part " --max-new 200
```

4. **Interactive “chat”** (really: repeated text continuation — not a real dialogue model):

```bash
python chat.py
```

If a character was never in training, you’ll get an error for that line.

## Data

**More text → better statistics** (especially for “grammar-like” surface patterns). Put plain UTF-8 in `data/input.txt` (path in `config.py`).

### Download a large public-domain corpus (recommended)

This repo includes `scripts/fetch_corpus.py`, which pulls several English novels from **Project Gutenberg** (polite delays + User-Agent). Only use if that matches how you want to source text.

```bash
python scripts/fetch_corpus.py
```

Optional: cap size (characters) for faster experiments:

```bash
python scripts/fetch_corpus.py --max-chars 1500000
```

An existing `data/input.txt` is copied to `data/input.txt.bak` before overwrite.

Training defaults in `config.py` assume a **large** corpus (`block_size` 128, deeper model). If you only have a **tiny** file, lower `block_size` / `n_layer` / `n_embd` or training may be slow for little gain.

### Copy-paste your own books

You can paste econometrics or other material into `data/input.txt`, but **copyright** is your responsibility (keep local copies out of git if unsure). `utils.clean_text` (enabled via `Config.clean_corpus`) normalizes whitespace and Unicode from messy PDF exports.
