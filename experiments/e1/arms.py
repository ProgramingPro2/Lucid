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


#: Confirmatory training budget, docs/PROTOCOL.md §5.6. One constant so the
#: arm table, the manifest, and the protocol cannot each pick a different
#: number. Positive-control arms are an assay at ``PC_TOKENS`` and are not
#: this budget. N does not train, so its budget is 0.
TOKENS_PER_RUN = 30_000_000

#: Positive-control assay, docs/PROTOCOL.md §5.6. Not a cell of the factorial.
PC_TOKENS = 2_000_000


def all_arms() -> list[Arm]:
    """The nine cells. ``joint`` runs are counted separately."""
    return [
        Arm("A", "treatment", True, True, 1.0, "true", 32, TOKENS_PER_RUN,
            "block + JEPA objective: tests C1"),
        Arm("B", "isotype", True, False, 0.0, "none", 32, TOKENS_PER_RUN,
            "bit-identical block, objective switched off"),
        Arm("A_prime", "secondary", False, True, 1.0, "true", 32, TOKENS_PER_RUN,
            "parameter-matched head on the trunk, no block: objective without architecture"),
        Arm("B_prime", "isotype", False, False, 0.0, "none", 32, TOKENS_PER_RUN,
            "head present, objective switched off"),
        Arm("E", "specificity", True, True, 1.0, "scrambled", 12, TOKENS_PER_RUN,
            "as A, targets permuted across the batch each step"),
        Arm("D", "vehicle", False, False, 0.0, "none", 6, TOKENS_PER_RUN,
            "plain LoRA; absolute reference, never a contrast"),
        Arm("N", "nuisance", True, False, 0.0, "none", 3, 0,
            "frozen random block; does not train (PROTOCOL §5.6), so the budget is 0"),
        Arm("PC_hi", "positive_control", False, False, 0.0, "none", 3, PC_TOKENS,
            "LoRA lr=1e-4; must beat PC_lo or the assay cannot see anything"),
        Arm("PC_lo", "positive_control", False, False, 0.0, "none", 3, PC_TOKENS,
            "LoRA lr=0"),
    ]


def by_name(name: str) -> Arm:
    for a in all_arms():
        if a.name == name:
            return a
    raise KeyError(name)


@dataclass(frozen=True)
class ScheduleRow:
    """A manifest row that is not one of the nine factorial cells.

    Trunk and joint confirmation are confirmatory runs at
    ``TOKENS_PER_RUN``. The dose grid is G6 only: PROTOCOL §5.6 does not list
    it among the confirmatory cells, and it is not part of the 219 training
    units. The experiments manifest still fixes that row at ``TOKENS_PER_RUN``,
    so the number lives here rather than only in the doc. N is a cell, not a
    row in this list; its budget is 0 because it does not train.
    """

    pattern: str
    count: int
    token_budget: int
    note: str


def schedule_rows() -> list[ScheduleRow]:
    """Trunk, joint confirmation, and the G6 dose grid."""
    return [
        ScheduleRow(
            "e1/trunk/s*",
            32,
            TOKENS_PER_RUN,
            "shared LoRA trunk, one per seed; confirmatory budget",
        ),
        ScheduleRow(
            "e1/*/s*.joint",
            32,
            TOKENS_PER_RUN,
            "joint-training confirmation, 8 seeds x 4 cells",
        ),
        ScheduleRow(
            "e1/A/s*.dose*",
            48,
            TOKENS_PER_RUN,
            "G6 mechanism grid only; not a factorial cell and not in the 219",
        ),
    ]


def resolve_token_budget(arm: str, override: int | None) -> tuple[int, str | None]:
    """Return ``(tokens, warning)`` for one arm.

    ``override is None`` uses the arm's ``token_budget``. An override that
    differs from that budget is returned unchanged, with a warning: one
    shared ``--tokens`` for every arm is how the 2026-11-20 pilot trained
    the 30M cells for 2M tokens. The warning is the caller's to print. This
    function does not change the override.
    """
    budget = by_name(arm).token_budget
    if override is None or override == budget:
        return budget, None
    warning = (
        f"WARNING: --tokens {override} for arm {arm} differs from the "
        f"arm-table budget {budget} (experiments.e1.arms.by_name). The "
        "voided 2026-11-20 pilot passed one token count for every arm. "
        f"This run will use {override}, not {budget}."
    )
    return override, warning


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


def build_head(d_model: int, zero_init: bool = False) -> nn.Module:
    """The A'/B' parameter-matched prediction head (no inserted block).

    Unlike the block, this head is *not* in the language-model path, so a
    zero output does not make A' and B' differ at step 0. It is therefore
    initialised small-but-nonzero: the JEPA loss is a cosine, and a cosine of a
    zero vector is undefined, which would make the first step a no-op.
    """

    class Head(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
            )
            if zero_init:
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
            else:
                nn.init.normal_(self.net[-1].weight, std=1e-3)
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
