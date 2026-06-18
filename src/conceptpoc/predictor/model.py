"""Concept predictor: a causal transformer over sequences of concept vectors.

Unlike a token LM, the input and output live in the continuous concept space.
The model reads a sequence of concept vectors and predicts the next one. Output
vectors are L2-normalized so they live on the same unit sphere as the codec's
encodings, which keeps cosine similarity meaningful for decoding and rewards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PredictorConfig


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


class ConceptPredictor(nn.Module):
    def __init__(self, concept_dim: int, cfg: PredictorConfig) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.cfg = cfg
        self.in_proj = nn.Linear(concept_dim, cfg.d_model)
        self.pos = _PositionalEncoding(cfg.d_model, cfg.max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, concept_dim)

    def forward(
        self, context: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """context: (B, T, dim). Returns the predicted next vector (B, dim).

        ``pad_mask`` is a boolean key-padding mask (True=pad) for right-padded
        sequences. The prediction is read from each sequence's last real token.
        """
        seq_len = context.size(1)
        # Boolean causal mask (True = disallowed) keeps the mask type consistent
        # with the boolean key-padding mask.
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=context.device),
            diagonal=1,
        )
        h = self.pos(self.in_proj(context))
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        last = self._last_real_token(h, pad_mask)
        pred = self.out_proj(last)
        return F.normalize(pred, dim=-1)

    @staticmethod
    def _last_real_token(
        h: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if pad_mask is None:
            return h[:, -1]
        # Right-padded: last real index is (number of real tokens) - 1.
        lengths = (~pad_mask).sum(dim=1)
        idx = (lengths - 1).clamp_min(0)
        return h[torch.arange(h.size(0), device=h.device), idx]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class LossOutput:
    total: torch.Tensor
    mse: torch.Tensor
    cosine: torch.Tensor


def predictor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> LossOutput:
    target = F.normalize(target, dim=-1)
    mse = F.mse_loss(pred, target)
    cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return LossOutput(total=total, mse=mse, cosine=cosine)
