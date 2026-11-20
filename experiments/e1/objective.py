"""Objectives for E1: JEPA prediction plus anti-collapse terms.

F10 is the reason the variance/covariance terms are not optional: a collapsed
space inflates the prediction cosine 8x. A metric that can be gamed by
degeneracy is not a metric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
    delta: int,
    mask: torch.Tensor | None = None,
    *,
    use_cosine: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Predict the teacher's state at t+delta from the student's state at t.

    Both tensors are (B, T, D). Returns ``(loss, diagnostics)``. If the sequence
    is too short for the requested horizon the loss is a zero tensor with
    ``n_positions = 0``; callers must check that rather than silently training
    on nothing.
    """
    if student_states.shape != teacher_states.shape:
        raise ValueError("student and teacher must have the same shape")
    if delta < 1:
        raise ValueError("delta must be >= 1")

    b, t, d = student_states.shape
    if t <= delta:
        zero = student_states.sum() * 0.0
        return zero, {"n_positions": 0, "mean": 0.0, "delta": delta}

    s = student_states[:, : t - delta, :]
    tgt = teacher_states[:, delta:, :]

    if use_cosine:
        per_pos = 1.0 - F.cosine_similarity(s, tgt.detach(), dim=-1)
    else:
        per_pos = F.mse_loss(s, tgt.detach(), reduction="none").mean(dim=-1)

    if mask is not None:
        if mask.shape[:2] != (b, t):
            raise ValueError("mask must be (B, T)")
        m = mask[:, : t - delta].to(per_pos.dtype)
        denom = m.sum().clamp_min(1.0)
        loss = (per_pos * m).sum() / denom
        n_pos = int(m.sum().item())
    else:
        loss = per_pos.mean()
        n_pos = b * (t - delta)

    return loss, {"n_positions": n_pos, "mean": float(loss.detach()), "delta": delta}


def vicreg_terms(
    x: torch.Tensor,
    eps: float = 1e-4,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge and covariance penalty (VICReg, Bardes et al. 2022).

    ``var`` penalises any feature dimension whose std falls below 1.
    ``cov`` penalises correlation between feature dimensions. Together they make
    collapse expensive, which is the property the F10 ablation lacked.

    The covariance term is computed on the **correlation** matrix, not the raw
    covariance. LLM hidden states are anisotropic and their raw covariance
    entries scale with the state magnitude: on Qwen3-1.7B this made the penalty
    ~1.3e8 and swamped both the language-model and prediction losses. Dividing
    each dimension by its own std makes the penalty dimensionless and bounded,
    so the same weight means the same thing across models and layers.

    ``mask`` (B, T) selects real positions. Padding is a constant vector, so
    including it inflates the variance term and makes the penalty depend on
    sequence length rather than on representation quality.
    """
    if mask is not None:
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must match the leading dimensions of x")
        flat = x[mask.bool()]
    elif x.ndim != 2:
        flat = x.reshape(-1, x.size(-1))
    else:
        flat = x
    n, d = flat.shape
    if n < 2:
        z = flat.sum() * 0.0
        return z, z

    centred = flat - flat.mean(dim=0, keepdim=True)
    var_d = centred.var(dim=0, unbiased=False)
    # A *relative* eps keeps the correlation term exactly scale-invariant: the
    # reference scales with the data, so multiplying the states by k changes
    # both the variance and the eps by k^2 and leaves ``std / centred`` alone.
    ref = var_d.mean().clamp_min(1e-12)
    std = torch.sqrt(var_d + eps * ref)
    var = torch.relu(1.0 - std).mean()

    # Per-dimension normalisation turns the covariance into a correlation
    # matrix, whose off-diagonal entries lie in [-1, 1]. A collapsed dimension
    # has ``centred == 0`` and contributes zero here; it is caught by ``var``.
    normed = centred / std
    corr = (normed.T @ normed) / max(n - 1, 1)
    off = corr - torch.diag_embed(torch.diagonal(corr))
    cov_pen = off.pow(2).sum() / (d * max(d - 1, 1))
    return var, cov_pen


def combined_loss(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
    delta: int,
    *,
    mask: torch.Tensor | None = None,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
    use_cosine: bool = True,
) -> tuple[torch.Tensor, dict]:
    """JEPA prediction + VICReg regularisers.

    The prediction term is computed on **whitened** states: raw cosine is
    dominated by a shared anisotropic direction (cos ~0.97 before any training),
    so it would measure nearly nothing. The regularisers are computed on the raw
    student states, where the variance hinge is meaningful; ``vicreg_terms``
    already makes the covariance term scale-invariant.
    """
    from .block import whiten

    s_white = whiten(student_states, mask=mask)
    t_white = whiten(teacher_states, mask=mask)
    pred, diag = jepa_loss(
        s_white, t_white, delta, mask, use_cosine=use_cosine
    )
    var, cov = vicreg_terms(student_states, mask=mask)
    total = pred + lambda_var * var + lambda_cov * cov
    diag.update(
        {
            "pred": float(pred.detach()),
            "var": float(var.detach()),
            "cov": float(cov.detach()),
            "total": float(total.detach()),
        }
    )
    return total, diag


def context_gain(
    nll_true: torch.Tensor, nll_shuffled: torch.Tensor
) -> torch.Tensor:
    """How much the model gains from real context vs shuffled context.

    This is E1's primary endpoint. It is immune to the "+54 nats" confound
    because raw knowledge cancels: both terms are the same model on the same
    targets.
    """
    if nll_true.shape != nll_shuffled.shape:
        raise ValueError("nll tensors must align")
    return nll_shuffled - nll_true
