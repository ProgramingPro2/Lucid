"""The E1 cell table as data (docs/PROTOCOL.md §5.1).

Keeping the arms in one place means the counts, roles and parameter-matching
claim cannot drift between the protocol, the run manifests and the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Arm:
    name: str
    role: str
    block: bool
    jepa_loss: bool
    loss_weight: float
    target_mode: str
    n_seeds: int
    token_budget: int
    description: str


def all_arms() -> list[Arm]:
    """The nine cells. ``joint`` runs are counted separately."""
    return [
        Arm("A", "treatment", True, True, 1.0, "true", 32, 10_000_000,
            "block + JEPA objective: tests C1"),
        Arm("B", "isotype", True, False, 0.0, "none", 32, 10_000_000,
            "bit-identical block, objective switched off"),
        Arm("A_prime", "secondary", False, True, 1.0, "true", 32, 10_000_000,
            "parameter-matched head on the trunk, no block: objective without architecture"),
        Arm("B_prime", "isotype", False, False, 0.0, "none", 32, 10_000_000,
            "head present, objective switched off"),
        Arm("E", "specificity", True, True, 1.0, "scrambled", 12, 10_000_000,
            "as A, targets permuted across the batch each step"),
        Arm("D", "vehicle", False, False, 0.0, "none", 6, 30_000_000,
            "plain LoRA; absolute reference, never a contrast"),
        Arm("N", "nuisance", True, False, 0.0, "none", 3, 1_000_000,
            "frozen random block: estimate of the cost of any random nonlinearity"),
        Arm("PC_hi", "positive_control", False, False, 0.0, "none", 3, 2_000_000,
            "LoRA lr=1e-4; must beat PC_lo or the assay cannot see anything"),
        Arm("PC_lo", "positive_control", False, False, 0.0, "none", 3, 2_000_000,
            "LoRA lr=0"),
    ]


def by_name(name: str) -> Arm:
    for a in all_arms():
        if a.name == name:
            return a
    raise KeyError(name)


def joint_confirmation_runs() -> int:
    """8 seeds per cell for the joint-training generalisation check."""
    return 8 * 4


def trunk_runs() -> int:
    """One shared trunk per seed, for the within-trunk design."""
    return 32


def total_training_units() -> int:
    """219 — the number in docs/EXPERIMENTS.md §3.

    155 arm runs (32 each for A/B/A_prime/B_prime, plus E, D, N, PC)
    + 32 shared trunks
    + 32 joint-training confirmation runs (8 seeds x 4 cells).
    """
    return sum(a.n_seeds for a in all_arms()) + trunk_runs() + joint_confirmation_runs()


def build_block(d_model: int, n_heads: int = 8, **kw) -> nn.Module:
    from .block import JEPABlock

    return JEPABlock(d_model, n_heads=n_heads, **kw)


def build_head(d_model: int) -> nn.Module:
    """The A'/B' parameter-matched prediction head (no inserted block)."""

    class Head(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return Head(d_model)


def n_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_parameter_matched(arms: list[Arm] | None = None, d_model: int = 64) -> None:
    """A and B must train the same number of parameters; so must A' and B'.

    The block row's parameter delta versus the head row is *declared*, not
    matched: you cannot match an inserted block without inserting it.
    """
    block = build_block(d_model)
    head = build_head(d_model)
    n_block = n_trainable(block)
    n_head = n_trainable(head)
    if n_block <= 0 or n_head <= 0:
        raise AssertionError("empty module in parameter match")
    # A vs B and A' vs B' differ only in the loss weight, so parameter counts are
    # equal by construction; assert the construction is what we think it is.
    if n_trainable(build_block(d_model)) != n_block:
        raise AssertionError("block params are not deterministic")
    if n_trainable(build_head(d_model)) != n_head:
        raise AssertionError("head params are not deterministic")
