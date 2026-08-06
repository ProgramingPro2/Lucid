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


def uniformity_loss(z, t=2.0, exclude_mask=None):
    """exclude_mask: (N, N) bool, True = drop this pair from the repulsion sum."""
    z = F.normalize(z, dim=-1)
    n = z.size(0)
    if n < 2:
        return z.new_zeros(())
    sq = torch.cdist(z, z, p=2).pow(2)                     # (N, N)
    keep = torch.triu(torch.ones(n, n, dtype=torch.bool, device=z.device), diagonal=1)
    if exclude_mask is not None:
        keep = keep & ~exclude_mask
    vals = sq[keep]
    if vals.numel() == 0:
        return z.new_zeros(())
    return torch.logsumexp(-t * vals, dim=0) - torch.log(vals.new_tensor(float(vals.numel())))


def adjacent_infonce(z, rows, slots, temperature=0.07):
    """InfoNCE: positive = same window's next slot; negatives = all rows from OTHER windows.
    z: (N, dim) normalized online vectors (valid rows)
    rows: (N,) long window index; slots: (N,) long slot index.
    Anchors with no valid successor are skipped.
    Same-window non-positive columns are masked to -inf (never negatives)."""
    n = z.size(0)
    if n < 2:
        return z.new_zeros(())
    device = z.device
    rows = rows.long()
    slots = slots.long()
    max_row = int(rows.max().item())
    max_slot = int(slots.max().item())
    # lookup[r, s] = flat index j of (row=r, slot=s), else -1
    lookup = torch.full(
        (max_row + 1, max_slot + 2), -1, dtype=torch.long, device=device
    )
    lookup[rows, slots] = torch.arange(n, device=device)
    pos_cols = lookup[rows, slots + 1]  # (N,) successor flat index or -1
    anchors = (pos_cols >= 0).nonzero(as_tuple=False).squeeze(-1)
    if anchors.numel() == 0:
        return z.new_zeros(())
    pos_cols = pos_cols[anchors]
    logits = (z[anchors] @ z.T) / temperature  # (A, N)
    same_window = rows[anchors].unsqueeze(1) == rows.unsqueeze(0)
    mask = same_window.clone()
    mask[torch.arange(anchors.numel(), device=device), pos_cols] = False
    logits = logits.masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, pos_cols)


def mean_pairwise_cosine(z: torch.Tensor, max_rows: int = 2048) -> float:
    """Anisotropy monitor: mean cosine between distinct normalized vectors."""
    with torch.no_grad():
        z = F.normalize(z, dim=-1)
        if z.size(0) < 2:
            return float("nan")
        if z.size(0) > max_rows:
            z = z[:max_rows]
        sims = z @ z.T
        n = z.size(0)
        off = (sims.sum() - n) / (n * (n - 1))
        return float(off.item())


def mean_dim_std(z: torch.Tensor) -> float:
    """Collapse monitor: mean per-dimension std across the batch."""
    with torch.no_grad():
        return float(torch.sqrt(z.var(dim=0, unbiased=False) + 1e-8).mean().item())
