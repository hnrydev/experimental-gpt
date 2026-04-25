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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressive sampling: repeatedly append one sampled character.

        We only feed the last `block_size` tokens to stay within training context.
        temperature scales logits: lower = more greedy, higher = more diverse.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # only predict from last time step
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # one sample from distribution
            idx = torch.cat((idx, next_id), dim=1)
        return idx
