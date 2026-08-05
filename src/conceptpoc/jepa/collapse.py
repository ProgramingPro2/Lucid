"""Anti-collapse vs predictability: test utilities and metrics.

The core tension in JEPA text prediction: anti-collapse losses (VICReg-ish)
push all embeddings apart, which destroys the local correlation between
consecutive sentences (we measure ~0.05 cosine). These functions let us
quantify BOTH axes so any proposed regularizer can be judged on:
  - collapse (batch variance, effective rank)   -- want HIGH
  - predictability (adjacent-sentence cosine)   -- want HIGH
A good regularizer keeps variance high WITHOUT flattening forward correlation.

Based on: VICReg (arXiv:2105.04906), LeJEPA/SIGReg (arXiv:2511.08544),
and the JEPA-in-language paradox (arXiv:2607.23531).
"""
from __future__ import annotations

import torch
import numpy as np


def batch_variance(z: torch.Tensor) -> float:
    """Mean per-dimension variance across the batch. High (~1) = healthy, ~0 = collapse."""
    return float(z.var(dim=0).mean())


def effective_rank(z: torch.Tensor) -> float:
    """Shannon effective rank of the covariance. Collapsed -> ~1. Spread -> ~dim."""
    zc = z - z.mean(dim=0, keepdim=True)
    # stable: use SVD singular values of the batch
    s = torch.linalg.svdvals(zc.float())
    p = s / (s.sum() + 1e-9)
    ent = -(p * torch.log(p + 1e-9)).sum()
    return float(torch.exp(ent))


def isotropy(z: torch.Tensor) -> float:
    """Ratio of min to max singular value of covariance (1.0 = perfectly isotropic)."""
    zc = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(zc.float())
    return float((s.min() / (s.max() + 1e-9)))


def adjacent_cosine(v0: np.ndarray, v1: np.ndarray) -> float:
    """Mean cosine between consecutive sentence vectors (the predictability axis)."""
    n0 = np.linalg.norm(v0, axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    dot = (v0 * v1).sum(-1) / (n0 * n1 + 1e-9)
    return float(dot.mean())


# ---------------------------------------------------------------------------
# Candidate anti-collapse regularizers (research: we want one that spreads
# LOCALLY so it keeps correlation). All return a loss scalar given a batch.
# ---------------------------------------------------------------------------

def vicreg_regularizer(z: torch.Tensor, var_weight: float = 25.0,
                       cov_weight: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Classic VICReg variance+covariance (global, isotropic — kills correlation)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(torch.relu(1.0 - std))
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.size(0) - 1)
    off = cov - torch.diag(cov.diagonal())
    cov_loss = (off ** 2).sum() / z.size(1)
    return var_weight * var_loss + cov_weight * cov_loss


def context_relative_vicreg(z: torch.Tensor, ctx_mask: torch.Tensor):
    """PROPOSED: spread embeddings relative to their context, not globally.

    The pathology of VICReg: it pushes ALL vectors apart globally. This variant
    computes variance/complexity per *context group* (rows with the same
    context_mask value) so different contexts can share latent structure while
    still avoiding collapse within a context. Returns (loss, kept_correlation).
    """
    raise NotImplementedError("research candidate — filled in by experiments")


def sigreg_isotropic(z: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

    Enforces the *provably optimal* isotropic-Gaussian embedding distribution so
    collapse is prevented by construction, using random-direction projections and
    a characteristic-function (Epps-Pulley) goodness-of-fit. Single weight.
    Approximated here with an empirical moment-matching proxy for the test.
    """
    zc = z - z.mean(dim=0, keepdim=True)
    # Random projections of the (centered) batch.
    rng = torch.Generator(device=z.device).manual_seed(0)
    proj = torch.randn(z.size(1), n_proj, generator=rng, device=z.device)
    proj = proj / proj.norm(dim=0, keepdim=True)
    p = zc @ proj  # (B, n_proj)
    # Match each projected dim to N(0,1): penalize (var-1)^2 + (skew)^2 + (kurt-3)^2
    var = p.var(dim=0)
    var_loss = ((var - 1.0) ** 2).mean()
    skew = (p ** 3).mean(dim=0) / (var + 1e-9) ** 1.5
    kurt = (p ** 4).mean(dim=0) / (var + 1e-9) ** 2 - 3.0
    shape_loss = (skew ** 2).mean() + (kurt ** 2).mean()
    return var_loss + 0.5 * shape_loss
