"""Discrete codebook for the E2 latent-emission arm.

A continuous vector is replaced by the nearest code (top-1). The straight-through
estimator copies the answer-loss gradient onto that continuous vector. The table
itself moves by exponential moving average toward the vectors that selected each
code.

There is no target index anywhere in this module. A commitment term, when its
coefficient is nonzero, pulls the continuous vector toward whichever code it
already chose. That anchors geometry. It does not say what the code should mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Quantized:
    """One top-1 quantisation."""

    z: torch.Tensor
    indices: torch.Tensor
    commitment: torch.Tensor

    @property
    def n_used(self) -> int:
        if self.indices.numel() == 0:
            return 0
        return int(torch.unique(self.indices).numel())


def usage_stats(indices: torch.Tensor, n_codes: int) -> dict:
    """How the codes were used. Perplexity is 1 when every pick is the same code."""
    if n_codes < 1:
        raise ValueError("n_codes must be positive")
    flat = indices.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    if flat.numel() == 0:
        return {"n_used": 0, "perplexity": 0.0, "dead_fraction": 1.0, "max_fraction": 0.0}
    if int(flat.min()) < 0 or int(flat.max()) >= n_codes:
        raise ValueError("code index out of range")
    counts = torch.bincount(flat, minlength=n_codes).to(dtype=torch.float64)
    total = float(counts.sum())
    used = int((counts > 0).sum())
    p = counts / total
    positive = p[p > 0]
    perplexity = float(torch.exp(-(positive * positive.log()).sum()))
    return {
        "n_used": used,
        "perplexity": perplexity,
        "dead_fraction": float((counts == 0).sum()) / float(n_codes),
        "max_fraction": float(counts.max() / total),
    }


def is_degenerate(indices: torch.Tensor, n_codes: int, *, min_used: int = 2) -> bool:
    """True when the picks collapse onto fewer than ``min_used`` codes."""
    return usage_stats(indices, n_codes)["n_used"] < min_used


class VectorQuantizer(nn.Module):
    """Top-1 vector quantiser with straight-through gradients and EMA updates.

    ``n_codes`` is ``2 ** bits`` for the dose axis (1, 8 or 16 bits). ``dim`` is
    the code width, not the backbone width: a learned projection places the code
    at one position. The embedding table does not receive autograd; EMA is the
    only update, so a loss cannot assign a code a label by backpropagating into
    the table.
    """

    def __init__(
        self,
        n_codes: int,
        dim: int,
        *,
        commitment_beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if n_codes < 2:
            raise ValueError("a codebook needs at least 2 codes so 'unused' is possible")
        if dim < 1:
            raise ValueError("dim must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.n_codes = int(n_codes)
        self.dim = int(dim)
        self.commitment_beta = float(commitment_beta)
        self.decay = float(decay)
        self.eps = float(eps)

        self.embedding = nn.Embedding(self.n_codes, self.dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("ema_count", torch.ones(self.n_codes))
        self.register_buffer("ema_sum", self.embedding.weight.detach().clone())

    def forward(self, z_e: torch.Tensor, *, update: bool | None = None) -> Quantized:
        """Quantise ``z_e`` (shape ``(..., dim)``).

        The returned ``z`` equals the chosen code on the forward pass and equals
        ``z_e`` on the backward pass (straight-through).
        """
        if z_e.shape[-1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {z_e.shape[-1]}")
        flat = z_e.reshape(-1, self.dim)
        # Distances in fp32. A bf16 argmin is a coin flip between nearby codes.
        flat_f = flat.float()
        weight = self.embedding.weight.float()
        # ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2. Argmin over codes.
        distance = (
            flat_f.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_f @ weight.t()
            + weight.pow(2).sum(dim=1)
        )
        indices = distance.argmin(dim=1)
        z_q = weight.index_select(0, indices).to(dtype=z_e.dtype).view_as(z_e)
        # Straight-through: forward value is z_q, gradient w.r.t. z_e is I.
        z_st = z_e + (z_q - z_e).detach()
        commitment = F.mse_loss(z_e, z_q.detach())
        do_update = self.training if update is None else bool(update)
        if do_update:
            self.ema_update(flat.detach(), indices.detach())
        return Quantized(z=z_st, indices=indices.view(z_e.shape[:-1]), commitment=commitment)

    @torch.no_grad()
    def ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Move each chosen code toward the vectors that selected it."""
        flat = flat.reshape(-1, self.dim).to(dtype=self.ema_sum.dtype)
        indices = indices.reshape(-1).to(dtype=torch.long)
        if flat.shape[0] != indices.shape[0]:
            raise ValueError("EMA update received mismatched vectors and indices")
        one_hot = F.one_hot(indices, self.n_codes).to(dtype=flat.dtype)
        count = one_hot.sum(dim=0)
        summed = one_hot.t() @ flat
        decay = self.decay
        self.ema_count.mul_(decay).add_(count, alpha=1.0 - decay)
        self.ema_sum.mul_(decay).add_(summed, alpha=1.0 - decay)
        n = float(self.ema_count.sum().clamp_min(self.eps))
        # Laplace smoothing so an unused code keeps a finite vector.
        smoothed = (self.ema_count + self.eps) / (n + self.n_codes * self.eps) * n
        self.embedding.weight.copy_(self.ema_sum / smoothed.unsqueeze(1))

    def commitment_loss(self, quantized: Quantized) -> torch.Tensor:
        """Geometry anchor. Zero when ``commitment_beta`` is zero."""
        return quantized.commitment * self.commitment_beta
