"""Latent trajectory planner: iterative continuous-thought planning.

Coconut-style (arXiv:2412.06769): rather than writing "reasoning tokens", the
planner *iterates a latent state* in concept space, then decodes that refined
latent into a sequence of K future concept vectors (the plan). The writer then
realizes the plan into text, so many "planning steps" happen before any token
is committed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryPlanner(nn.Module):
    """Given (B, C, dim) context vectors, emit (B, K, dim) planned trajectory.

    Design:
    - ctx_proj  : compress context into a latent start state h0
    - latent block (a small transformer) refines the hidden state
    - latent_recur: recurrent cell iterating the "continuous thought" in latent
      space -- this is the Coconut-style planning loop
    - traj_head : decode final hidden -> (B, K, dim)
    """

    def __init__(self, dim: int = 512, C: int = 8, K: int = 5,
                 n_layers: int = 4, n_heads: int = 8, latent_iters: int = 3,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.dim, self.C, self.K = dim, C, K
        self.ctx_proj = nn.Linear(dim, dim)
        self.latent_iters = latent_iters
        # Recurrent continuous-thought cell.
        self.latent_recur = nn.GRUCell(dim, dim)
        self.norm_latent = nn.LayerNorm(dim)
        # Tiny transformer to refine the latent thought at each step.
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4 * dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.latent_blocks = nn.ModuleList([layer for _ in range(n_layers)])
        # Decode refined latent into K trajectory vectors.
        self.traj_head = nn.Linear(dim, K * dim)
        self.norm_out = nn.LayerNorm(dim)
        # Init stability: keep head near-zero so start of training is mild.
        nn.init.zeros_(self.traj_head.weight)
        nn.init.zeros_(self.traj_head.bias)
        self.ema: torch.Tensor | None = None

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """context (B, C, dim) -> (B, dim) latent start state."""
        # mean-pool then project
        pooled = context.mean(dim=1)
        return self.ctx_proj(pooled)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context (B, C, dim) -> trajectory (B, K, dim)."""
        b = context.size(0)
        h: torch.Tensor = self.encode_context(context)  # (B, dim)
        # Iterate the "continuous thought" in latent space.
        for _ in range(self.latent_iters):
            h = self.latent_recur(self.norm_latent(h), h)
            # transformer refinement (batched as (B, 1, dim) through a block)
            h_emb = h.unsqueeze(1)
            for blk in self.latent_blocks:
                h_emb = blk(h_emb)
            h = h_emb.squeeze(1)
        # Decode to trajectory.
        traj = self.traj_head(self.norm_out(h))
        traj = traj.view(b, self.K, self.dim)
        return traj

    @torch.no_grad()
    def plan(self, context: torch.Tensor, iters: int | None = None) -> torch.Tensor:
        """Explicit planning loop (test-time) with optional deeper search."""
        if iters is None:
            iters = self.latent_iters
        self.eval()
        b = context.size(0)
        h = self.encode_context(context)
        for _ in range(iters):
            h = self.latent_recur(self.norm_latent(h), h)
            h_emb = h.unsqueeze(1)
            for blk in self.latent_blocks:
                h_emb = blk(h_emb)
            h = h_emb.squeeze(1)
        traj = self.traj_head(self.norm_out(h)).view(b, self.K, self.dim)
        return traj

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
