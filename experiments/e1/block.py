"""The inserted JEPA block (experiment E1).

Three properties matter more than the architecture itself, and all are tested:

1. **Zero-init output.** With ``zero_init_output=True`` the block contributes
   exactly nothing at step 0, so arms A / B / E are bit-identical functions of
   their inputs before training. Without this the "inserted block" is
   confounded with initialisation noise (docs/PROTOCOL.md §10).

2. **The block returns its contribution, not the residual sum.** Ablation means
   zeroing that contribution. If ``forward`` returned ``x + delta`` there would
   be no way to ablate the block without also deleting the stream it reads.

3. **The attention is causal.** The block is inserted into a causal LM and the
   LM loss reads every position. A bidirectional block makes that loss depend
   on future tokens, so the leak is shared by every arm that carries the block
   and the contrast measures the leak rather than the objective.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class JEPABlock(nn.Module):
    """A pre-norm transformer block whose output projection can be zeroed."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.out = nn.Linear(d_model, d_model)
        if zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        # P3 (necessity) is measured by zeroing the block's *contribution* at
        # evaluation. This flag is toggled by the harness, not trained.
        self._ablated = False

    def set_ablation(self, ablated: bool) -> None:
        """Zero the block contribution for evaluation (P3)."""
        self._ablated = bool(ablated)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the block's *contribution* to the residual stream.

        The mask is causal. The block sits inside a causal LM, and the LM loss
        reads every position. A bidirectional block lets the loss at position
        ``t`` depend on tokens after ``t``, so the thing being optimised is not
        the language-model loss the protocol names, and every arm that carries
        the block (A, B, E, N) is contaminated together.
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"expected (B,T,{self.d_model}), got {tuple(x.shape)}")
        if self._ablated:
            # Exactly zero, so ``apply_to`` is the identity. Using ``x * 0``
            # keeps the autograd graph intact for any caller that differentiates.
            return x * 0.0

        h = self.ln1(x)
        # ``nn.MultiheadAttention`` uses True to mean "ignore this key". Without
        # this, a padded batch lets every real token attend to padding, and the
        # block's contribution depends on batch composition. Padding is folded
        # into the same additive mask as causality so a key cannot be visible
        # through one path and hidden through the other.
        key_padding = None
        if attention_mask is not None:
            if attention_mask.shape != x.shape[:2]:
                raise ValueError("attention_mask must be (B, T)")
            key_padding = attention_mask == 0
        attn_mask = combined_attn_mask(
            h.size(1), h.dtype, h.device, key_padding, self.attn.num_heads
        )
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = x + attn_out
        h = h + self.mlp(self.ln2(h))
        return self.out(h)

    def apply_to(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The block in situ: residual plus contribution."""
        return x + self.forward(x, attention_mask)


class EMATeacher(nn.Module):
    """Slow copy of a module used as a prediction target, never differentiated."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = copy.deepcopy(module)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, student: nn.Module, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        for t, s in zip(self.module.parameters(), student.parameters()):
            t.mul_(momentum).add_(s.detach(), alpha=1.0 - momentum)
        for tb, sb in zip(self.module.buffers(), student.buffers()):
            tb.copy_(sb)


def causal_attn_mask(
    seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive causal mask: 0 on and below the diagonal, ``-inf`` above it.

    ``-inf`` is required. A large finite penalty only down-weights the future,
    and the language-model loss still moves when a future token changes. The
    dtype matches the activations the mask is added to; an fp32 mask against
    a bf16 score is a silent cast, and the wrong one is not bitwise causal.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    if seq_len == 1:
        return mask
    upper = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    return mask.masked_fill(upper, -torch.inf)


def combined_attn_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    key_padding: torch.Tensor | None,
    n_heads: int,
) -> torch.Tensor:
    """Causal mask, with padding keys set to ``-inf`` for every query.

    With no padding this is the ``(T, T)`` causal mask, broadcast across the
    batch. With padding it is ``(B * n_heads, T, T)``, which is the 3D shape
    ``F.multi_head_attention_forward`` accepts, so a padding key cannot stay
    visible through one head and hidden through another.

    A query whose every legal key is padding would otherwise be an all-``-inf``
    row. Softmax of that row is NaN, and a NaN times the zero-init output
    projection is still NaN, which would break the step-0 identity on padded
    batches. Those rows attend only to themselves. That does not open the
    future: the diagonal is position ``t`` attending to token ``t``.
    """
    mask = causal_attn_mask(seq_len, dtype, device)
    if key_padding is None:
        return mask
    if key_padding.ndim != 2 or key_padding.size(-1) != seq_len:
        raise ValueError(
            f"key_padding must be (B, {seq_len}), got {tuple(key_padding.shape)}"
        )
    if n_heads < 1:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    mask = mask.unsqueeze(0).expand(key_padding.size(0), -1, -1).clone()
    mask = mask.masked_fill(key_padding.unsqueeze(1), -torch.inf)
    finite = torch.isfinite(mask)
    all_masked = ~finite.any(dim=-1, keepdim=True)
    if bool(all_masked.any()):
        eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
        mask = mask.masked_fill(eye & all_masked, 0.0)
    # One copy per head. ``expand`` shares storage; the attention kernel wants
    # a real (B * n_heads, T, T) tensor.
    return (
        mask.unsqueeze(1)
        .expand(-1, n_heads, -1, -1)
        .reshape(-1, seq_len, seq_len)
        .contiguous()
    )


def whiten(
    x: torch.Tensor,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Centre and scale per feature dimension using batch statistics.

    LLM hidden states are anisotropic (a few high-norm dimensions dominate), so
    a raw MSE target is mostly a norm term. Whitening makes the prediction loss
    about direction, which is what "content" means here.

    Statistics are computed over real positions only when ``mask`` is given.
    Padding would otherwise shift the mean and shrink the std in a way that
    depends on the longest sequence in the batch.
    """
    if mask is None:
        mean = x.mean(dim=tuple(range(x.ndim - 1)), keepdim=True)
        std = x.std(dim=tuple(range(x.ndim - 1)), keepdim=True, unbiased=False)
        return (x - mean) / (std + eps)

    if mask.shape != x.shape[:2]:
        raise ValueError("mask must match the leading dimensions of x")
    real = x[mask.bool()]
    mean = real.mean(dim=0, keepdim=True)
    std = real.std(dim=0, keepdim=True, unbiased=False)
    out = (x - mean) / (std + eps)
    # Keep padded positions at zero so downstream sums are unaffected.
    return out * mask.unsqueeze(-1).to(out.dtype)
