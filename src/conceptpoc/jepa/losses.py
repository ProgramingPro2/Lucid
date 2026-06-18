"""Loss terms for joint training.

- ``reconstruction_loss``: token cross-entropy (pad ignored) + token accuracy.
  This is the anchor that makes collapse impossible: a constant concept vector
  cannot reconstruct distinct sentences.
- ``prediction_loss``: cosine + MSE of the predicted next vector against a
  detached (stop-grad) target.
- ``variance_loss`` / ``covariance_loss``: VICReg-style anti-collapse pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ReconOutput:
    loss: torch.Tensor
    accuracy: float


def reconstruction_loss(
    logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int
) -> ReconOutput:
    """logits: (B, L, V); target_ids: (B, L)."""
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        target_ids.reshape(-1),
        ignore_index=pad_id,
    )
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = target_ids != pad_id
        correct = ((pred == target_ids) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / max(1, total)
    return ReconOutput(loss=loss, accuracy=acc)


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = F.normalize(target.detach(), dim=-1)
    cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    mse = F.mse_loss(pred, target)
    return cosine + mse


def variance_loss(z: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge that keeps each concept dimension's batch std above ``margin``."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(margin - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Push off-diagonal covariances toward zero (decorrelate dimensions)."""
    b, d = z.shape
    if b < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (b - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


def mean_dim_std(z: torch.Tensor) -> float:
    """Collapse monitor: mean per-dimension std across the batch."""
    with torch.no_grad():
        return float(torch.sqrt(z.var(dim=0, unbiased=False) + 1e-8).mean().item())
