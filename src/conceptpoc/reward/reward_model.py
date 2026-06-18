"""Reward model over concept trajectories.

Maps a sequence of generated concept vectors to a scalar reward. Trained with a
Bradley-Terry pairwise objective on preference pairs: the chosen continuation
should score higher than the rejected one. Operating directly on concept vectors
(rather than text) keeps it cheap and aligned with the space the predictor
optimizes in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .preference import PreferencePair


class RewardModel(nn.Module):
    def __init__(self, concept_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(concept_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, vectors: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """vectors: (B, T, dim) -> scalar reward (B,). Mean-pools over time."""
        if mask is not None:
            keep = (~mask).float().unsqueeze(-1)
            pooled = (vectors * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        else:
            pooled = vectors.mean(1)
        return self.net(pooled).squeeze(-1)

    @torch.no_grad()
    def score_vectors(self, vectors: np.ndarray, device: str = "cpu") -> float:
        self.eval()
        t = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).unsqueeze(0).to(device)
        return float(self(t).item())


@dataclass
class RewardTrainHistory:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)


def _pad_stack(arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(a.shape[0] for a in arrays)
    dim = arrays[0].shape[1]
    out = torch.zeros(len(arrays), max_len, dim, dtype=torch.float32)
    mask = torch.ones(len(arrays), max_len, dtype=torch.bool)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = torch.from_numpy(np.asarray(a, dtype=np.float32))
        mask[i, : a.shape[0]] = False
    return out, mask


def train_reward_model(
    pairs: list[PreferencePair],
    concept_dim: int,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[RewardModel, RewardTrainHistory]:
    if not pairs:
        raise ValueError("no preference pairs to train on")
    torch.manual_seed(seed)
    model = RewardModel(concept_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    chosen, c_mask = _pad_stack([p.chosen.vectors for p in pairs])
    rejected, r_mask = _pad_stack([p.rejected.vectors for p in pairs])
    chosen, c_mask = chosen.to(device), c_mask.to(device)
    rejected, r_mask = rejected.to(device), r_mask.to(device)

    history = RewardTrainHistory()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        rc = model(chosen, c_mask)
        rr = model(rejected, r_mask)
        loss = -F.logsigmoid(rc - rr).mean()
        loss.backward()
        opt.step()
        history.loss.append(float(loss.item()))
        history.accuracy.append(float((rc > rr).float().mean().item()))
    return model, history
