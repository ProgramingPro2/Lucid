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


class ContextTransformer(nn.Module):
    """Shared causal transformer trunk over concept-vector sequences."""

    def __init__(self, concept_dim: int, cfg: PredictorConfig) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.cfg = cfg
        self.in_proj = nn.Linear(concept_dim, cfg.d_model)
        self.pos = _PositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.segment_emb = (
            nn.Embedding(2, cfg.d_model) if cfg.use_segment_embedding else None
        )
        self.theme_proj = (
            nn.Linear(concept_dim, cfg.d_model) if cfg.use_theme else None
        )
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

    def forward(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-position hidden states (B, T, d_model)."""
        seq_len = context.size(1)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=context.device),
            diagonal=1,
        )
        h = self.in_proj(context)
        if segment_ids is not None and self.segment_emb is not None:
            h = h + self.segment_emb(segment_ids)
        if theme_vec is not None and self.theme_proj is not None:
            h = h + self.theme_proj(theme_vec).unsqueeze(1)
        h = self.pos(h)
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.norm(h)

    @staticmethod
    def last_real_token(
        h: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if pad_mask is None:
            return h[:, -1]
        lengths = (~pad_mask).sum(dim=1)
        idx = (lengths - 1).clamp_min(0)
        return h[torch.arange(h.size(0), device=h.device), idx]


class ConceptPredictor(nn.Module):
    """Causal transformer over concept vectors (point-regression head).

    Optional conditioning (E10), enabled per-field via ``PredictorConfig`` and
    fully backward compatible: when no ``segment_ids``/``theme_vec`` are passed
    the extra modules contribute nothing, so ``forward`` returns exactly the
    same output as before. A learned segment embedding distinguishes
    instruction (0) from response (1) positions, and an optional global theme
    vector is added to every position.
    """

    def __init__(self, concept_dim: int, cfg: PredictorConfig) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.cfg = cfg
        self.trunk = ContextTransformer(concept_dim, cfg)
        self.out_proj = nn.Linear(cfg.d_model, concept_dim)

    def encode_context(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.trunk(context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec)

    def forward(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """context: (B, T, dim). Returns the predicted next vector (B, dim).

        ``pad_mask`` is a boolean key-padding mask (True=pad) for right-padded
        sequences. The prediction is read from each sequence's last real token.

        Optional conditioning (no effect when ``None``):
        - ``segment_ids`` (B, T) long tensor of 0/1 role ids.
        - ``theme_vec`` (B, dim) global theme added to every position.
        ``return_sequence`` returns a per-position prediction (B, T, dim)
        instead of only the last real token; used by teacher-forced training.
        """
        h = self.encode_context(
            context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec
        )
        if return_sequence:
            return F.normalize(self.out_proj(h), dim=-1)
        last = ContextTransformer.last_real_token(h, pad_mask)
        pred = self.out_proj(last)
        return F.normalize(pred, dim=-1)

    @staticmethod
    def _last_real_token(
        h: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return ContextTransformer.last_real_token(h, pad_mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def segment_emb(self):
        """Backward-compatible access to the trunk's segment embedding."""
        return self.trunk.segment_emb

    @property
    def theme_proj(self):
        """Backward-compatible access to the trunk's theme projection."""
        return self.trunk.theme_proj


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


def remap_legacy_predictor_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map pre-trunk checkpoints (``in_proj.*``) to ``trunk.in_proj.*``."""
    if not any(k.startswith("in_proj.") for k in state):
        return state
    if any(k.startswith("trunk.") for k in state):
        return state
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("out_proj."):
            out[key] = value
        else:
            out[f"trunk.{key}"] = value
    return out
