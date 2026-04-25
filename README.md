# baby-gpt

Minimal **GPT in PyTorch** (character tokens by default, or **local BPE** subwords for much better “word” quality), data utils, small causal Transformer, `train.py`, `generate.py`, and `chat.py`. **No paid API;** `tokenizers` is the only extra dependency for BPE (see `requirements.txt`).

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

1. **Corpus:** training loads, in order, the four primers, then grammar + general-examples (each **twice** to up-weight clean text), then optional **`data/wikipedia_corpus.txt`** (skipped until you generate it; see below), then **`data/input.txt`**. Edit `Config.corpus_files` in `config.py` to add files or change order. **After changing the corpus, retrain**; an old checkpoint’s vocabulary can miss new characters.
2. **Train** (writes a checkpoint — path depends on profile, see below):

```bash
python train.py
```

Quick dry run: `MAX_ITERS=100 python train.py` (PowerShell: `$env:MAX_ITERS="100"; python train.py`).

**Train/val (generalization readout):** The last `val_fraction` of the **token** stream (default **5%** in `config.py`) is **held out**; batches use only the prefix, and the step log can show `val` and `train` loss. **Val > train** often means overfitting; **both high** means the task is still hard (underfit, tiny data, or hard distribution). Set `VAL_FRACTION=0` to disable the split. If the **tail** of the corpus is not like the head (e.g. one long section at the end), this sequential split is imperfect—more diverse text still helps more than a perfect train/val line.

**Early stopping (optional, small data):** Set `EARLY_STOP_PATIENCE` to a number of **val eval intervals** (e.g. `8` with the default 50-step interval ≈ 400 steps without a val win). Training stops if val does not beat the running best (by at least `EARLY_STOP_MIN_DELTA`, default 0) that many times in a row, and the saved checkpoint is the **lowest val** so far, not the last step. Patience 0 = disabled (default).

### Qualitative eval (grammar, sensible text)

Val loss is not enough to compare runs if your goal is **readable** output. This repo includes **`data/eval_prompts.txt`** (one prompt per line; `#` comments allowed) and **`sample_eval.py`**, which writes continuations to **`outputs/eval_samples_*.txt`** using the same “coherent”-style decoding as `chat.py --coherent` by default (tighter sampling for stability).

```bash
python sample_eval.py
# or:  python sample_eval.py --checkpoint models/baby_gpt_fast_bpe.pt
# or:  python sample_eval.py --no-coherent   # use config temperature / top_p
# or:  python sample_eval.py --greedy
```

After training, run an eval in one step (optional):

```powershell
$env:RUN_EVAL="1"; python train.py
```

### Best “communication” without an API (recommended)

**BPE (ByteLevel, trained on your corpus, fully local)** predicts **subword** tokens, not raw characters—usually the largest quality jump you can get from this repo alone. It uses a **different checkpoint** (`models/baby_gpt_fast_bpe.pt` in the fast profile) and a sidecar `*.tokenizer.json` next to it. Longer default run (3000 fast steps) and larger `block_size` in **token** space are set in `config.py` when BPE is on.

```powershell
$env:BABY_GPT_FAST="1"
$env:BABY_GPT_BPE="1"
$env:MAX_ITERS="3000"   # or more while loss is falling
$env:SAMPLE_EVERY="500" # optional: print a short greedy sample during training
python train.py
$env:BABY_GPT_BPE="1"   # so Config.default checkpoint matches
python chat.py
```

**Char mode** (no BPE) stays the default if you do not set `BABY_GPT_BPE=1`. Old `models/baby_gpt_fast.pt` checkpoints are unchanged.

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

**Refining output (what actually helps):** (1) **`BABY_GPT_BPE=1` + retrain** — subword tokens (see above). (2) **Train longer** while loss falls (override `MAX_ITERS`). (3) **Chat defaults are strict** (low temp, top-*p*/*k*, short `max_new`, high repetition penalty — see `config.py` `chat_*`); use **`--coherent`** for strictest, **`--greedy`** for argmax, **`--looser`** if you want wider sampling. (4) **Bigger / non–fast** model and overnight runs if you can. **Reality check:** this is still a **base LM** (continuation), not a full chat or instruction system; decoding can only reduce noise, not install understanding.

**Sensible text without a paid API:** after sampling, `local_text_fix` runs on the **new continuation only** (not your prompt), so the chat UI can strip the prefix correctly — otherwise sentence-capitalization on the full string could break prefix matching and look like an “echo.” It **caps** insane repeats, trims `?!.` / newlines, optional **tail dedupe**, and optional **`light_surface_english`** on the new part only. **Decoding** uses **top-k**, **nucleus (top-p)**, and **repetition penalty** (`config.py`); chat defaults are tuned a bit to reduce copying your line verbatim.

**Long lines in chat** — the model only conditions on the **last** `block_size` **tokens** (BPE) or **characters** (char mode). `chat.py` uses the same effective prefix for stripping the continuation.

**Expectations:** polish helps **readability**, not deep reasoning; a weak LM may still look odd—**retrain** for real gains. On a **CPU budget**, output will not match a large GPU run or ChatGPT; unset `BABY_GPT_FAST` and use a GPU or overnight runs for the bigger model.

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

This repo includes `scripts/fetch_corpus.py`, which pulls from **Project Gutenberg** (polite delays + User-Agent). The default list mixes **novels** with **essays, drama, science, and philosophy** so the character model sees more *registers* and topics than a single genre. That helps surface statistics across styles; it does **not** turn the model into a general reasoner (scale and task design still dominate).

```bash
python scripts/fetch_corpus.py
```

`--books diverse` fetches only the non-fiction / drama / essay batch; `--books novels` fetches only the long-fiction batch; default is **all** (fiction first, then diverse). Optional: cap size (characters) for faster experiments:

```bash
python scripts/fetch_corpus.py --max-chars 1500000
```

An existing `data/input.txt` is copied to `data/input.txt.bak` before overwrite.

### Modern encyclopedic text (Wikipedia)

For **clear, contemporary written English** (at the cost of sounding encyclopedic), run **`scripts/fetch_wikipedia_corpus.py`**. It pulls plain-text extracts from the **MediaWiki API**—default is **Simple English Wikipedia** (shorter, simpler sentences, often a good match for a small LM). Use `--mode en` for full English Wikipedia (longer, denser).

Text is **CC BY-SA 4.0**; if you publish a derived dataset or model, provide appropriate **attribution** and understand **share-alike** obligations. The script adds a short license header in the file.

```bash
python scripts/fetch_wikipedia_corpus.py
python scripts/fetch_wikipedia_corpus.py --mode en --max-chars 2000000
```

This writes **`data/wikipedia_corpus.txt`** (listed in `config.py` but **gitignored** so clones stay small). The script includes **large curated title lists** (hundreds of Simple English seeds, ~90+ extra English articles); you get as many as fit under **`--max-chars`** (default **2,500,000**). One HTTP request per title, with a short delay, per Wikimedia access norms.

Training defaults in `config.py` assume a **large** corpus (`block_size` 128, deeper model). If you only have a **tiny** file, lower `block_size` / `n_layer` / `n_embd` or training may be slow for little gain.

### Copy-paste your own books

You can paste econometrics or other material into `data/input.txt`, but **copyright** is your responsibility (keep local copies out of git if unsure). `utils.clean_text` (enabled via `Config.clean_corpus`) normalizes whitespace and Unicode from messy PDF exports.
