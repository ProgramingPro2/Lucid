"""Ranking / retrieval reasoner over paragraph concepts.

Maps a window of past paragraph concept vectors to an L2-normalized key, then
scores candidate next-paragraph vectors by cosine (InfoNCE / M3-aligned CE).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


@dataclass
class RankingReasonerConfig:
    """Small causal transformer (~1-4M params at concept_dim=512)."""

    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 32
    temperature: float = 0.07


class RankingReasoner(nn.Module):
    """in_proj -> causal TransformerEncoder -> out key vector (L2-normalized).

    Same shape as ConceptPredictor, but the training objective is ranking
    (cosine against candidate paragraph vectors), not point regression.
    """

    def __init__(self, concept_dim: int, cfg: RankingReasonerConfig | None = None) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.cfg = cfg or RankingReasonerConfig()
        c = self.cfg
        self.in_proj = nn.Linear(concept_dim, c.d_model)
        self.pos = _PositionalEncoding(c.d_model, c.max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_model * c.ff_mult,
            dropout=c.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=c.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(c.d_model)
        self.out_proj = nn.Linear(c.d_model, concept_dim)

    def forward(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """context: (B, T, dim) -> key: (B, dim) L2-normalized."""
        seq_len = context.size(1)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=context.device),
            diagonal=1,
        )
        h = self.in_proj(context)
        h = self.pos(h)
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        last = self._last_real_token(h, pad_mask)
        return F.normalize(self.out_proj(last), dim=-1)

    @staticmethod
    def _last_real_token(h: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if pad_mask is None:
            return h[:, -1]
        lengths = (~pad_mask).sum(dim=1)
        idx = (lengths - 1).clamp_min(0)
        return h[torch.arange(h.size(0), device=h.device), idx]

    def score(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """Cosine logits: (B, K) = query @ candidates^T / temp."""
        temp = self.cfg.temperature if temperature is None else temperature
        q = F.normalize(query, dim=-1)
        c = F.normalize(candidates, dim=-1)
        return torch.bmm(c, q.unsqueeze(-1)).squeeze(-1) / max(temp, 1e-8)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ranking_loss(
    query: torch.Tensor,
    candidates: torch.Tensor,
    n_positives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-positive InfoNCE / soft CE over in-document candidates.

    ``candidates[:, 0]`` is always the true next paragraph. Additional leading
    slots (up to ``n_positives[i]``) are following-paragraph positives. Negatives
    fill the rest. Loss = -log(sum_exp(pos) / sum_exp(all)).
    """
    q = F.normalize(query, dim=-1)
    c = F.normalize(candidates, dim=-1)
    logits = torch.bmm(c, q.unsqueeze(-1)).squeeze(-1) / max(temperature, 1e-8)  # (B, K)
    # log-sum-exp multi-positive
    # mask: True for positive slots
    B, K = logits.shape
    pos_mask = torch.zeros(B, K, dtype=torch.bool, device=logits.device)
    for i in range(B):
        n_pos = int(n_positives[i].item())
        n_pos = max(1, min(n_pos, K))
        pos_mask[i, :n_pos] = True
    # Numerically stable: L = logsumexp(all) - logsumexp(pos)
    lse_all = torch.logsumexp(logits, dim=-1)
    neg_inf = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~pos_mask, neg_inf)
    lse_pos = torch.logsumexp(pos_logits, dim=-1)
    return (lse_all - lse_pos).mean()


def ranking_ce_hard(
    query: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Single-positive CE with label 0 (true next) — M3-aligned hard target."""
    q = F.normalize(query, dim=-1)
    c = F.normalize(candidates, dim=-1)
    logits = torch.bmm(c, q.unsqueeze(-1)).squeeze(-1) / max(temperature, 1e-8)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
