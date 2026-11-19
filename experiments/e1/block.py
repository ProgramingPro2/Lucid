"""The inserted JEPA block (experiment E1).

Two properties matter more than the architecture itself, and both are tested:

1. **Zero-init output.** With ``zero_init_output=True`` the block contributes
   exactly nothing at step 0, so arms A / B / E are bit-identical functions of
   their inputs before training. Without this the "inserted block" is
   confounded with initialisation noise (docs/PROTOCOL.md §10).

2. **The block returns its contribution, not the residual sum.** Ablation means
   zeroing that contribution. If ``forward`` returned ``x + delta`` there would
   be no way to ablate the block without also deleting the stream it reads.
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

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the block's *contribution* to the residual stream."""
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"expected (B,T,{self.d_model}), got {tuple(x.shape)}")

        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
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


def whiten(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Centre and scale per feature dimension using batch statistics.

    LLM hidden states are anisotropic (a few high-norm dimensions dominate), so
    a raw MSE target is mostly a norm term. Whitening makes the prediction loss
    about direction, which is what "content" means here.
    """
    mean = x.mean(dim=tuple(range(x.ndim - 1)), keepdim=True)
    std = x.std(dim=tuple(range(x.ndim - 1)), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)
