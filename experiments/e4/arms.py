"""E4-S1 arm inventory.

The counts are the manifest in ``docs/EXPERIMENTS.md`` §5, which is the same
design as ``docs/PROTOCOL.md`` §7.1. A scaled run (the smoke) uses the same six
arms with a smaller *n*. A G3 call requires the manifest counts.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pre-registered confirmatory size. Do not derive a "nearby" design and call
# it G3.
MANIFEST_ITEMS = 192
MANIFEST_FITS = 3
MANIFEST_TEXT = 96
MANIFEST_NULL = 192

#: Latent-channel arms. The text prompt is shared; only the draft changes.
LATENT_ARMS: tuple[str, ...] = ("true", "shuf", "none", "dose50")

#: The six protocol rows, in the order §7.1 lists them.
PROTOCOL_ARM_NAMES: tuple[str, ...] = (
    "true",
    "shuf",
    "none",
    "dose50",
    "textpc",
    "null",
)


@dataclass(frozen=True)
class ArmCount:
    """One protocol row.

    ``textpc`` and ``null`` each carry two conditions (true/random, a/b).
    ``n_fits`` is 1 for the null pair: it is a decode noise floor, not an
    adapter replicate (``docs/EXPERIMENTS.md`` §5, ``e4s1/null/{a,b}/i*``).
    """

    arm: str
    role: str
    n_items: int
    n_fits: int
    n_conditions: int = 1

    @property
    def generations(self) -> int:
        return self.n_items * self.n_fits * self.n_conditions


def arm_plan(
    *,
    n_items: int = MANIFEST_ITEMS,
    n_fits: int = MANIFEST_FITS,
    n_text: int = MANIFEST_TEXT,
    n_null: int = MANIFEST_NULL,
) -> list[ArmCount]:
    """Return the six arms and the generation counts each one schedules."""
    if n_text > n_items:
        raise ValueError("text-prefix subset cannot exceed the item count")
    if n_items < 2 or n_null < 2 or n_text < 2 or n_fits < 1:
        raise ValueError("need at least 2 items, 2 text items, 2 null items, 1 fit")
    return [
        ArmCount("true", "treatment", n_items, n_fits),
        ArmCount("shuf", "necessity control", n_items, n_fits),
        ArmCount("none", "format control", n_items, n_fits),
        ArmCount("dose50", "dose", n_items, n_fits),
        ArmCount("textpc", "positive control", n_text, n_fits, n_conditions=2),
        ArmCount("null", "noise floor", n_null, 1, n_conditions=2),
    ]


def manifest_plan() -> list[ArmCount]:
    """The confirmatory inventory. 4×3×192 + 2×3×96 + 2×192 = 3264 generations."""
    return arm_plan()


def plan_counts(plan: list[ArmCount]) -> dict[str, int]:
    """Flat counts keyed the way the manifest writes run ids.

    ``textpc`` splits into ``textpc_true`` / ``textpc_rand``. ``null`` splits
    into ``null_a`` / ``null_b``. Latent arms are already one condition each.
    """
    out: dict[str, int] = {}
    for arm in plan:
        if arm.arm == "textpc":
            per = arm.n_items * arm.n_fits
            out["textpc_true"] = per
            out["textpc_rand"] = per
        elif arm.arm == "null":
            out["null_a"] = arm.n_items
            out["null_b"] = arm.n_items
        else:
            out[arm.arm] = arm.generations
    out["adapter_fits"] = next(a.n_fits for a in plan if a.arm == "true")
    out["generations"] = sum(a.generations for a in plan)
    return out
