"""Delta predictor: forward JEPA that predicts change-direction, not destination.

Reuses the causal-transformer trunk from :mod:`..predictor.model`. The semantic
difference: input and output are concept-*deltas* (normalized change directions)
instead of absolute concept vectors. This is the "oscillator" fix -- text
bounces, so predict how it swings rather than where it lands.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import PredictorConfig
from .model import ConceptPredictor, predictor_loss


class DeltaPredictor(ConceptPredictor):
    """Causal transformer over concept-delta windows; predicts next delta.

    Input  : (B, T, dim) window of normalized change directions
    Output : (B, dim) predicted next change direction (normalized)

    The trunk is identical to ``ConceptPredictor``; only the semantic of the
    vectors changes. Output is already L2-normalized.
    """

    def __init__(self, concept_dim: int, cfg: PredictorConfig) -> None:
        super().__init__(concept_dim, cfg)

    def forward(self, context, pad_mask=None, *, return_sequence=False, **kw):
        # identical behavior; subclass to keep the delta semantics explicit and
        # allow later delta-specific heads without touching the base model.
        return super().forward(context, pad_mask, return_sequence=return_sequence, **kw)


def delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.0,
    cosine_weight: float = 1.0,
):
    """Loss for delta prediction. Deltas are directions -> cosine is the signal;
    MSE on normalized directions is small and not very informative, so default
    to cosine. Normalize the target defensively.
    """
    return predictor_loss(pred, target, mse_weight=mse_weight, cosine_weight=cosine_weight)


def delta_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine between predicted and target change directions (the gate)."""
    return F.cosine_similarity(pred, F.normalize(target, dim=-1), dim=-1).mean()


def apply_delta(current: torch.Tensor, delta: torch.Tensor, step: float = 1.0) -> torch.Tensor:
    """current + step*delta, renormalized onto the unit sphere (the decode path)."""
    nxt = current + step * delta
    return F.normalize(nxt, dim=-1)
