"""
Central place for paths and hyperparameters.

Set environment variable BABY_GPT_FAST=1 for a **small** model that trains many
more steps in ~30 minutes on a CPU laptop (see README). Default = larger, slower
run when you have hours or a GPU.
"""

import os
import torch

_fast = os.environ.get("BABY_GPT_FAST", "").strip().lower() in ("1", "true", "yes")
_use_bpe = os.environ.get("BABY_GPT_BPE", "").strip().lower() in ("1", "true", "yes")

if _fast:

    class Config:
        """Small model + shorter context: many steps per minute on a typical laptop CPU."""

        use_bpe = _use_bpe
        bpe_vocab_size = 2048
        # Subword (BPE) in *token* space: larger `block_size` in tokens, separate checkpoint
        # so char runs are not broken. Set BABY_GPT_BPE=1; requires: pip install tokenizers
        checkpoint = "models/baby_gpt_fast_bpe.pt" if _use_bpe else "models/baby_gpt_fast.pt"
        dataset = "data/input.txt"  # primary narrative/large text; still used if corpus_files is absent
        # Training order: world map, short Q&A/turns (stops the LM from *only* knowing 19c blocks),
        # then main text (Gutenberg, etc. — set by fetch_corpus or your own file).
        corpus_files = (
            "data/world_fields_primer.txt",
            "data/short_form_primer.txt",
            "data/modern_language_primer.txt",
            "data/grammar_clarity_corpus.txt",
            "data/input.txt",
        )
        seed = 42
        clean_corpus = True

        batch_size = 32
        block_size = 128 if _use_bpe else 64
        # BPE: longer run by default; char: quick smoke. Override with MAX_ITERS.
        max_iters = 3000 if _use_bpe else 500
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
        temperature = 0.45
        top_k = 32
        top_p = 0.88
        repetition_penalty = 1.22
        repetition_context_len = 24
        # N-gram blocking in `model.generate` (0 = off). Tames loops like Cycycy / token stutter.
        decode_no_repeat_ngram_size = 4
        decode_no_repeat_ngram_bpe = 3
        # Local collapse of char runs — always on in generate.py (see local_text_fix.py)
        local_max_char_run = 3
        # Cosmetic only: spaces, word "i" -> "I", cap after . ! ? — not grammar repair
        local_surface_english = True
        # Chat only: suffix + looser sampling so the model is less likely to copy your line verbatim
        chat_prompt_suffix = "\n"
        # Tighter chat decoding to reduce collapse / verbatim copy (tune in config, not a chat model).
        chat_temperature = 0.64
        chat_top_p = 0.9
        chat_repetition_penalty = 1.62
        chat_repetition_context_len = 48
        chat_safety_max_new = 200
        # Tighter sampling for `python chat.py --coherent` (less “creative” garbage).
        chat_coherent_temperature = 0.55
        chat_coherent_top_p = 0.85
        chat_coherent_top_k = 20
        chat_coherent_repetition_penalty = 1.55
        chat_coherent_max_new = 200
        # Shorter default for `chat.py --greedy` (argmax can loop or ramble in long runs).
        chat_greedy_max_new = 180

        device = "cuda" if torch.cuda.is_available() else "cpu"

else:

    class Config:
        """Defaults for a multi-million-character corpus (slower; use a GPU or overnight CPU)."""

        use_bpe = _use_bpe
        bpe_vocab_size = 4096
        checkpoint = "models/baby_gpt_bpe.pt" if _use_bpe else "models/baby_gpt.pt"
        dataset = "data/input.txt"
        corpus_files = (
            "data/world_fields_primer.txt",
            "data/short_form_primer.txt",
            "data/modern_language_primer.txt",
            "data/grammar_clarity_corpus.txt",
            "data/input.txt",
        )
        seed = 42
        clean_corpus = True

        batch_size = 24
        block_size = 256 if _use_bpe else 128
        max_iters = 10000 if _use_bpe else 8000
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
        temperature = 0.5
        top_k = 32
        top_p = 0.88
        repetition_penalty = 1.22
        repetition_context_len = 24
        decode_no_repeat_ngram_size = 4
        decode_no_repeat_ngram_bpe = 3
        local_max_char_run = 3
        local_surface_english = True
        chat_prompt_suffix = "\n"
        chat_temperature = 0.64
        chat_top_p = 0.9
        chat_repetition_penalty = 1.62
        chat_repetition_context_len = 48
        chat_safety_max_new = 220
        chat_coherent_temperature = 0.55
        chat_coherent_top_p = 0.85
        chat_coherent_top_k = 20
        chat_coherent_repetition_penalty = 1.55
        chat_coherent_max_new = 220
        chat_greedy_max_new = 200

        device = "cuda" if torch.cuda.is_available() else "cpu"
