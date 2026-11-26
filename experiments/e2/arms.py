"""E2 arm table.

The manifest (docs/EXPERIMENTS.md §4) is 189 runs::

    12 seeds × 3 families × {A, Btext, Bdepth, E}  = 144
    9 seeds of C on F1                                 = 9
    4 seeds × 9 dose cells on F1                       = 36

PROTOCOL §6.3 prints C as *n* = 3 and the dose row as "6 each". Those phrases
do not add up to the manifest's 189. This table follows the manifest arithmetic
(189). It does not invent a third sample size.

λ and the codebook width are fixed here, before any trained result. Arm A uses
the middle charge and 8-bit codes. Arm C is the same gate with λ = 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Charge on each text token the gate emits inside the reasoning span.
# Fixed a priori. Units are nats added to the answer cross-entropy.
LAMBDA_TOKEN: dict[str, float] = {
    "zero": 0.0,
    "low": 0.01,
    "mid": 0.10,
    "high": 1.0,
}

VQ_BITS: tuple[int, ...] = (1, 8, 16)

# Answer text is the same length budget on every arm (C3: equal output length).
ANSWER_BUDGET = 24

# Explicit text CoT budget. F48: every family has saturated by 1024 new tokens.
COT_BUDGET = 1024

# Hard cap on latent tokens. The gate may emit fewer, including zero.
CAPACITY = 16

# Code width. The backbone is 2048-d; the code is a gist that a projection
# places at one position. 16-bit codes at this width are 8 MiB in bf16.
CODE_DIM = 64

# Geometry anchor. Not a content label. See experiments/e2/DESIGN.md.
COMMITMENT_BETA = 0.25

MATCH_TOLERANCE = 0.05

FAMILIES: tuple[str, ...] = ("F1", "F2", "F3")
CORE_ARMS: tuple[str, ...] = ("A", "Btext", "Bdepth", "E")
CORE_SEEDS = 12
C_SEEDS = 9
DOSE_SEEDS = 4


def family_key(family: str) -> str:
    """``f1`` / ``F1`` → ``F1``."""
    key = str(family).strip().upper()
    if key not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    return key


def family_code(family: str) -> str:
    """``F1`` → ``f1``, the generator's name."""
    return family_key(family).lower()


@dataclass(frozen=True)
class ArmSpec:
    """One cell of the manifest. ``depth_repeats`` is the FLOP-match knob."""

    name: str
    kind: str                 # latent | text | depth | vehicle
    lambda_name: str
    lambda_token: float
    vq_bits: int
    n_codes: int
    capacity: int
    depth_repeats: int
    cot_tokens: int
    answer_tokens: int
    commitment_beta: float
    allow_text: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _codes(bits: int) -> int:
    if bits not in VQ_BITS:
        raise ValueError(f"VQ bits must be one of {VQ_BITS}, got {bits}")
    return 1 << int(bits)


def dose_names() -> list[str]:
    """``dose-llow-vq1`` … nine cells, λ × bits."""
    return [f"dose-l{lam}-vq{bits}" for lam in ("low", "mid", "high") for bits in VQ_BITS]


def arm_spec(
    name: str,
    *,
    depth_repeats: int = 1,
    capacity: int | None = None,
    cot_tokens: int = COT_BUDGET,
    answer_tokens: int = ANSWER_BUDGET,
) -> ArmSpec:
    """Resolve a manifest arm name to a fixed specification.

    ``depth_repeats`` and ``capacity`` are the only knobs a pilot may lower.
    λ, bits, and the answer budget stay on the pre-registered values. Passing
    ``capacity`` above the manifest cap is rejected.
    """
    if depth_repeats < 1:
        raise ValueError("depth_repeats must be >= 1")
    cap = CAPACITY if capacity is None else int(capacity)
    if not 1 <= cap <= CAPACITY:
        raise ValueError(f"capacity must be in [1, {CAPACITY}], got {cap}")

    key = str(name)
    if key == "A":
        lam, bits, kind = "mid", 8, "latent"
    elif key == "C":
        lam, bits, kind = "zero", 8, "latent"
    elif key == "Btext":
        lam, bits, kind = "zero", 8, "text"
    elif key == "Bdepth":
        lam, bits, kind = "zero", 8, "depth"
    elif key == "E":
        lam, bits, kind = "zero", 8, "vehicle"
    elif key.startswith("dose-l"):
        kind = "latent"
        try:
            rest = key[len("dose-l"):]
            lam, bits_s = rest.split("-vq")
            bits = int(bits_s)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"bad dose arm {key!r}") from exc
        if lam not in ("low", "mid", "high"):
            raise ValueError(f"dose λ must be low|mid|high, got {lam!r}")
    else:
        raise ValueError(f"unknown arm {key!r}")

    return ArmSpec(
        name=key,
        kind=kind,
        lambda_name=lam,
        lambda_token=LAMBDA_TOKEN[lam],
        vq_bits=bits,
        n_codes=_codes(bits),
        capacity=cap,
        depth_repeats=int(depth_repeats),
        cot_tokens=int(cot_tokens),
        answer_tokens=int(answer_tokens),
        commitment_beta=COMMITMENT_BETA if kind == "latent" else 0.0,
        allow_text=kind == "latent",
    )


def run_id(family: str, arm: str, seed: int) -> str:
    """``e2/F2/A/s09``. Seeds are zero-padded the way the manifest example is."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    # Confirm the arm name parses. Dose and core arms only.
    arm_spec(arm)
    return f"e2/{family_key(family)}/{arm}/s{int(seed):02d}"


def manifest_run_ids() -> list[str]:
    """Every run id in the 189-run manifest, in a stable order."""
    ids: list[str] = []
    for family in FAMILIES:
        for arm in CORE_ARMS:
            for seed in range(CORE_SEEDS):
                ids.append(run_id(family, arm, seed))
    for seed in range(C_SEEDS):
        ids.append(run_id("F1", "C", seed))
    for arm in dose_names():
        for seed in range(DOSE_SEEDS):
            ids.append(run_id("F1", arm, seed))
    if len(ids) != 189:
        raise RuntimeError(f"manifest arithmetic drifted: {len(ids)} != 189")
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate run id in the manifest")
    return ids
