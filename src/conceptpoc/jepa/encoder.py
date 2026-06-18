"""Text encoder: tokens -> a single L2-normalized concept vector."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import JepaConfig
from .modules import PositionalEncoding


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: JepaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_tokens + 2)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.enc_heads,
            dim_feedforward=cfg.d_model * cfg.enc_ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.enc_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.to_concept = nn.Linear(cfg.d_model, cfg.dim)

    def forward(
        self, ids: torch.Tensor, pad_mask: torch.Tensor, return_raw: bool = False
    ):
        """ids: (B, L), pad_mask: (B, L) True=pad.

        Returns the L2-normalized concept vector (B, dim). If ``return_raw`` is
        set, also returns the pre-normalization projection, which is what the
        variance/covariance anti-collapse terms operate on (normalized vectors
        have per-dimension std capped near 1/sqrt(dim), making margin-based
        VICReg meaningless on them).
        """
        h = self.pos(self.embed(ids))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        raw = self.to_concept(pooled)
        z = F.normalize(raw, dim=-1)
        if return_raw:
            return z, raw
        return z
