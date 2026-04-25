# Data + training plan (best-results recipe)

This is the end-to-end checklist for **mixing broad text (coverage) with curated text (signal)** on a small CPU-friendly LM. Nothing here auto-runs; follow in order and retrain after you change data or `bpe_vocab_size`.

## What “best” means here

- **Long Gutenberg (and similar):** vocabulary, rhythm, story-like English — good *breadth*.
- **Curated files (primers, grammar, your own clean examples):** high quality per byte — good *target style* and grammar.
- **Two Wikipedia registers:** Simple English = short, clear sentences; English Wikipedia = denser, more formal/encyclopedic — good *diversity* if you use both.

The model is tiny: **quality and variety beat raw size alone**.

---

## Phase 1 — Curated text (highest signal per line)

1. Open and, if you can, **extend** (with your own or public-domain prose):
   - `data/grammar_clarity_corpus.txt`
   - `data/general_english_examples.txt`
   - `data/modern_language_primer.txt`, `data/short_form_primer.txt`, `data/world_fields_primer.txt`
2. Prefer **short, correct, modern** sentences; one idea per line is fine.
3. `config.py` already **up-weights** grammar + general English (listed **three times** in `corpus_files`). To emphasize them more, duplicate those paths again (same pattern as now) — at the cost of less relative weight on `input.txt`.

**Do not** paste copyrighted text you do not have rights to use.

---

## Phase 2 — Project Gutenberg (`data/input.txt`)

1. Read `scripts/fetch_corpus.py` (robot policy, book lists).
2. From the repo root, fetch or refresh narrative + mixed registers, e.g.:

   ```bash
   python scripts/fetch_corpus.py --max-chars 3000000
   ```

   Use `--books all` (default) or `novels` / `diverse` as documented in the script.
3. Larger `max-chars` → longer training; on a ThinkPad, pick a cap that still finishes in the time you have.

This file is the main **long-form** slice; it should not be the *only* slice (primers + grammar exist for a reason).

---

## Phase 3 — Wikipedia (optional second register)

**Licensing:** Wikipedia text is **CC BY-SA 4.0**. If you redistribute the corpus or a model substantially based on it, review attribution and share-alike. The fetch script docstring links the license.

1. **Simple English** (default; good for small LMs), e.g.:

   ```bash
   python scripts/fetch_wikipedia_corpus.py --out data/wikipedia_corpus.txt
   ```

2. **English Wikipedia** (different style — add as a *second* file, not a replace):

   ```bash
   python scripts/fetch_wikipedia_corpus.py --mode en --max-chars 1500000 --out data/wikipedia_en_corpus.txt
   ```

   Adjust `--max-chars` to your patience and train time.

3. **Config:** `corpus_files` in `config.py` already includes `data/wikipedia_en_corpus.txt` after Simple Wikipedia. **Create the file** with the command above; until it exists, training prints one **skip** warning and continues without it. To avoid the warning until you are ready, remove that line from `corpus_files` temporarily.

---

## Phase 4 — Check `corpus_files` order

Order matters: **primers and grammar first**, then optional wikis, then **large** `input.txt` last (see comments in `config.py`).

After edits, the training log should list all files you expect under “Loaded … file(s): …”.

---

## Phase 5 — Train (BPE + fast profile on laptop)

1. **BPE** is recommended (`BABY_GPT_BPE=1`). Changing `bpe_vocab_size` in `config.py` means a **new** tokenizer and checkpoint — plan a full retrain, not an old-weights resume.
2. Use **val** and optional **early stopping** to avoid overtraining on small data (see main `README.md`).
3. Example **PowerShell** (tune `MAX_ITERS` and `EARLY_STOP_PATIENCE`):

   ```powershell
   $env:PYTHONUNBUFFERED="1"   # live log lines (esp. BPE encode + long runs)
   $env:BABY_GPT_FAST="1"
   $env:BABY_GPT_BPE="1"
   $env:MAX_ITERS="4000"
   $env:EARLY_STOP_PATIENCE="8"
   # optional: $env:BABY_GPT_NUM_THREADS="4"
   # optional: $env:BABY_GPT_COMPILE="1"
   # If `LR_WARMUP_ITERS` was set in the shell, it overrides config; remove with:
   # Remove-Item Env:LR_WARMUP_ITERS -ErrorAction SilentlyContinue
   python train.py
   ```

4. If you use `RUN_EVAL=1`, training will also run `sample_eval` at the end.

---

## Phase 6 — Evaluate (quality, not just loss)

1. **Quantitative:** watch **train vs val** in the log; `val >> train` suggests overfitting.
2. **Qualitative:**

   ```powershell
   $env:BABY_GPT_FAST="1"
   $env:BABY_GPT_BPE="1"
   # Coherent-style decoding is the default; use --no-coherent for looser sampling
   python sample_eval.py
   ```

3. **Interactive:** prefer `python chat.py --coherent` for readable text; `--greedy` is strict argmax and can loop on a weak model.

Read `outputs/` and fix data or decoding before chasing bigger `MAX_ITERS` alone.

---

## Phase 7 — Iterate

| Symptom | Try |
|--------|-----|
| Repetitive or 19c-only flavor | More modern lines in primers/grammar; more `diverse` Gutenberg |
| Thin vocabulary | More Gutenberg / larger `--max-chars` |
| Garbled subwords | BPE on; ensure long enough training; check val |
| Overfitting (val bad) | Stronger regularization, shorter runs, or more diverse data |
| Good loss, ugly text | Tighten `--coherent` decoding; add targeted curated lines |

When you add or remove **files** or change **BPE vocab size**, **retrain** from a new run; old checkpoints are not automatically “updated.”

---

## One-page checklist

- [ ] Curated / grammar / primers reviewed or extended  
- [ ] `python scripts/fetch_corpus.py` with a sensible `--max-chars`  
- [ ] `python scripts/fetch_wikipedia_corpus.py` → `wikipedia_corpus.txt`  
- [ ] (Optional) `fetch_wikipedia_corpus.py --mode en` → `wikipedia_en_corpus.txt` (path already in `config.py`)  
- [ ] `corpus_files` order and log line verified  
- [ ] `$env:BABY_GPT_FAST="1"; $env:BABY_GPT_BPE="1"; python train.py` (+ `MAX_ITERS` / `EARLY_STOP_PATIENCE`)  
- [ ] `python sample_eval.py` (default coherent) and/or `python chat.py --coherent`  

This matches the “**broad + curated + two registers**” strategy described in the project chat; there is no separate magic source — execution is the plan.
