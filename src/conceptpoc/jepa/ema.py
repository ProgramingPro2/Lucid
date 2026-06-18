"""Exponential moving average target encoder.

Produces prediction targets that the online encoder cannot game. The EMA copy
is never trained by gradients; it only tracks the online encoder slowly, which
(with stop-grad on the target) is the BYOL/JEPA recipe for avoiding collapse.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class EmaEncoder(nn.Module):
    def __init__(self, online: nn.Module, momentum: float = 0.99) -> None:
        super().__init__()
        self.momentum = momentum
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        m = self.momentum
        for tp, op in zip(self.target.parameters(), online.parameters()):
            tp.mul_(m).add_(op.detach(), alpha=1.0 - m)
        for tb, ob in zip(self.target.buffers(), online.buffers()):
            tb.copy_(ob)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.target(*args, **kwargs).detach()
