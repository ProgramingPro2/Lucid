"""GPT-2-scale fluent writer decoder, optionally conditioned on a concept vector.

Phase 1: plain next-token LM (learns word-level fluency).
Phase 2: conditioned on a concept vector (the JEPA reasoner's decision) so the
writer can be "aimed" at the intended next sentence.

Modern best practices baked in: RoPE rotary positions, RMSNorm, SwiGLU FFN,
weight-tying between input embedding and LM head.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def _precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    return freqs


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., head_dim); cos/sin: (seq, head_dim/2) applied per half (RoPE)."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        freqs = _precompute_rope(rope_len, self.head_dim)
        self.register_buffer("_cos", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("_sin", freqs.sin().unsqueeze(0).unsqueeze(0))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos = self._cos[:, :, :t]
        sin = self._sin[:, :, :t]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal
        attn = F.softmax(attn, dim=-1)
        out = self.drop(attn) @ v
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_len: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, rope_len, dropout)
        self.ffn = SwiGLU(dim, dim * ff_mult)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))
        return x


class FluWriter(nn.Module):
    """Autoregressive writer head. Optional concept conditioning (Phase 2)."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        ff_mult: int = 4,
        max_seq_len: int = 256,
        concept_dim: int = 512,
        use_concept: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.use_concept = use_concept
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, max_seq_len + 8, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying: share input embedding and output head.
        self.tok_embed.weight = self.lm_head.weight
        if use_concept:
            self.concept_proj = nn.Linear(concept_dim, dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        concept: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """input_ids: (B, T); concept: (B, concept_dim) optional; returns logits or loss."""
        b, t = input_ids.shape
        h = self.tok_embed(input_ids)
        if self.use_concept and concept is not None:
            c = self.concept_proj(concept).unsqueeze(1)  # (B, 1, dim)
            h = torch.cat([c, h], dim=1)  # (B, T+1, dim)
        else:
            start = torch.zeros(b, 1, self.dim, device=input_ids.device)
            h = torch.cat([start, h], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)
        logits = self.lm_head(h[:, 1:])  # (B, T, vocab)
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return loss
        return logits

    @torch.no_grad()
    def generate(
        self,
        concept: torch.Tensor,
        max_tokens: int,
        bos_id: int,
        eos_id: int | None = None,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Sample token ids from a concept vector (Phase-2 conditioned decode)."""
        self.eval()
        dev = concept.device
        batch = concept.size(0)
        generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=dev)
        for _ in range(max_tokens):
            logits = self.forward(generated, concept)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    kth = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return generated

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
