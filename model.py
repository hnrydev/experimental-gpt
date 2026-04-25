"""
Minimal GPT-style decoder: token + position embeddings, causal self-attention,
feedforward MLPs, and a language modeling head.

"Causal" means each position may only attend to itself and earlier positions —
the model cannot peek at future characters when predicting the next one, which
matches how we train (predict next char from prefix). This is the same family
of architecture as large language models, just tiny and character-level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (lower-triangular) mask.

    Intuition: each position builds a weighted mix of value vectors from other
    positions, where the weights ("attention scores") come from how well query
    keys match. The mask forces position t to ignore positions > t so the
    representation at t only uses the past — required for next-token prediction.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # each head operates in a smaller subspace
        # One linear that projects to query, key, value stacked (3 * n_embd total).
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)  # mix heads back together
        self.dropout = nn.Dropout(dropout)
        # Not a learnable parameter: fixed mask of ones below diagonal, zeros above.
        # Shape (1,1, L, L) so it broadcasts over batch and heads.
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape to (B, n_head, T, head_dim) for parallel per-head attention.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention: (Q K^T) / sqrt(dk) — scale keeps softmax stable.
        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        # Where mask is 0, set score to -inf so softmax becomes ~0 (no future peeking).
        att = att.masked_fill(self.causal[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # weights over past positions sum to 1
        att = self.dropout(att)
        y = att @ v  # weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    """
    Position-wise feedforward block: up-project, GELU, down-project.

    This is the "memory" / non-linear mix after attention; four times width is
    a common Transformer choice (see "FFN dim" in papers).
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # smooth ReLU-like activation; standard in GPT-2 style models
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One Transformer layer: pre-norm attention, residual, pre-norm MLP, residual.

    "Pre-norm" means LayerNorm is applied before each sublayer (stable training).
    Residuals let gradients flow and let the block learn small adjustments.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Full model: token embedding + position embedding, stack of blocks, LM head.

    Weight tying: `lm_head` shares weights with `wte` (token embedding). Fewer
    parameters; empirically works well (same idea as in GPT-2).
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        # wte: "which character" -> vector. wpe: "which position in window" -> vector.
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # Linear map from hidden state to logits over the vocabulary (unnormalized scores).
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying (output projection = input lookup transpose)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token ids. Optional targets: (B, T) next-token ids for training.

        Returns logits (B, T, vocab_size) and optional cross-entropy loss vs targets.
        At each position, the model predicts the *next* character in targets.
        """
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab) vs (B*T,) for token-level cross-entropy.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        top_k = min(max(top_k, 1), logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        out = logits.clone()
        out[out < v[:, [-1]]] = float("-inf")
        return out

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumprobs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        return logits.masked_fill(remove, float("-inf"))

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, idx: torch.Tensor, penalty: float, context_len: int
    ) -> torch.Tensor:
        if penalty <= 1.0:
            return logits
        recent = idx[0, -context_len:].unique()
        out = logits.clone()
        for t in recent:
            out[0, t] /= penalty
        return out

    @staticmethod
    def _apply_no_repeat_ngram_mask(
        logits: torch.Tensor, idx: torch.Tensor, n: int, logits_unmasked: torch.Tensor
    ) -> torch.Tensor:
        """
        Bans a next token if the n-gram it would complete has already appeared (see
        NoRepeatNGramLogitsProcessor in e.g. summarization decoding). If every
        logit is masked, fall back to `logits_unmasked` (after repetition penalty).
        """
        if n <= 0 or idx.shape[1] < n - 1:
            return logits
        seq = idx[0].tolist()
        lseq = len(seq)
        ngrams: set[tuple] = set()
        for i in range(lseq - n + 1):
            ngrams.add(tuple(seq[i : i + n]))
        prefix = tuple(seq[-(n - 1) :])
        out = logits.clone()
        for c in range(logits.size(-1)):
            ngram = prefix + (c,)
            if len(ngram) == n and ngram in ngrams:
                out[0, c] = float("-inf")
        if not bool(torch.isfinite(out[0]).any()):
            return logits_unmasked
        return out

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_len: int = 12,
        greedy: bool = False,
        no_repeat_ngram_size: int = 0,
    ):
        """
        Autoregressive generation: multinomial by default, or **greedy** (argmax) for
        the most likely next character at each step. Greedy has no sampling noise; on
        a weak char-LM it can still look odd, but it often beats random pseudo-words.

        ``no_repeat_ngram_size > 0`` blocks completion of n-grams already seen
        in the current sequence (reduces stutter; common in long-form decoding).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = self._apply_repetition_penalty(
                logits, idx, repetition_penalty, repetition_context_len
            )
            logits_unmasked = logits.clone()
            if no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram_mask(
                    logits, idx, int(no_repeat_ngram_size), logits_unmasked
                )
            if greedy:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    logits = self._apply_top_k(logits, int(top_k))
                if top_p is not None and 0.0 < top_p < 1.0:
                    logits = self._apply_top_p(logits, float(top_p))
                probs = F.softmax(logits, dim=-1)
                if torch.isnan(probs).any():
                    probs = F.softmax(logits.nan_to_num(nan=0.0, neginf=0.0), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
