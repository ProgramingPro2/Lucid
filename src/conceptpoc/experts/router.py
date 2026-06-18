"""Gating router over multiple concept-predictor experts."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..predictor.model import ConceptPredictor


class GatingRouter(nn.Module):
    """Maps a context summary to a soft distribution over experts."""

    def __init__(self, concept_dim: int, n_experts: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(concept_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, context: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if pad_mask is not None:
            keep = (~pad_mask).float().unsqueeze(-1)
            summary = (context * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        else:
            summary = context.mean(1)
        return F.softmax(self.net(summary), dim=-1)


class MixtureOfPredictors(nn.Module):
    def __init__(self, experts: list[ConceptPredictor], router: GatingRouter) -> None:
        super().__init__()
        if not experts:
            raise ValueError("need at least one expert")
        self.experts = nn.ModuleList(experts)
        self.router = router

    def forward(
        self, context: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        weights = self.router(context, pad_mask)  # (B, E)
        preds = torch.stack(
            [e(context, pad_mask) for e in self.experts], dim=1
        )  # (B, E, dim)
        blended = (weights.unsqueeze(-1) * preds).sum(dim=1)
        return F.normalize(blended, dim=-1)

    @torch.no_grad()
    def routing_weights(
        self, context: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.router(context, pad_mask)
