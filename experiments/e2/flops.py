"""Matched-FLOP check for E2.

The comparison is valid only when three things are true at once:

1. The measured ratio ``FLOPs(A) / FLOPs(B_text)`` is inside 5% of 1.
2. The uncertainty interval on that ratio is also inside 5% of 1.
3. A is not cheaper than B_text. A cheaper latent arm cannot be reported as a
   win: that is the F14 strawman (a stronger method against a weaker baseline),
   and PROTOCOL §6.4 voids a FLOP mismatch above 5%.

Wall-clock is a second measurement of the same runs. If it and the FLOP
counter disagree by more than 5% relative, the comparison is not certified.
A formula that was never run is not a measurement.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from .arms import COT_BUDGET, MATCH_TOLERANCE


@dataclass(frozen=True)
class MatchVerdict:
    flops_a: float
    flops_b: float
    ratio: float
    ratio_se: float
    ratio_low: float
    ratio_high: float
    matched: bool
    a_cheaper: bool
    comparison_valid: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def ratio_se(mean_a: float, se_a: float, mean_b: float, se_b: float) -> float:
    """Delta-method standard error of ``mean_a / mean_b``."""
    if mean_b <= 0:
        raise ValueError("denominator must be positive")
    if se_a < 0 or se_b < 0:
        raise ValueError("standard errors must be non-negative")
    var = (se_a / mean_b) ** 2 + (mean_a * se_b / mean_b**2) ** 2
    return float(var**0.5)


def judge_flop_ratio(
    flops_a: float,
    flops_b: float,
    *,
    se: float = 0.0,
    tolerance: float = MATCH_TOLERANCE,
    z: float = 1.96,
) -> MatchVerdict:
    """Judge one ratio. ``se`` is the standard error of the ratio, not of a mean.

    ``comparison_valid`` is the win-eligibility flag. It is false when the point
    estimate is outside ``tolerance``, when the uncertainty interval leaves the
    tolerance band, or when A is strictly cheaper than B_text.
    """
    if flops_b <= 0 or flops_a < 0:
        raise ValueError(f"FLOPs must be non-negative and B_text > 0, got {flops_a}, {flops_b}")
    if se < 0:
        raise ValueError("se must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    ratio = float(flops_a) / float(flops_b)
    half = float(z) * float(se)
    lo = ratio - half
    hi = ratio + half
    a_cheaper = ratio < 1.0
    matched = abs(ratio - 1.0) <= float(tolerance) + 1e-12
    interval_inside = lo >= 1.0 - float(tolerance) - 1e-12 and hi <= 1.0 + float(tolerance) + 1e-12
    if a_cheaper and not matched:
        reason = "FLOP ratio is outside 5% and the latent arm is cheaper than B_text"
        valid = False
    elif a_cheaper:
        reason = (
            "latent arm is cheaper than B_text; refusing a win "
            "(F14 strawman, even though the point estimate is inside 5%)"
        )
        valid = False
    elif not matched:
        reason = "FLOP ratio is outside 5% of 1"
        valid = False
    elif not interval_inside:
        reason = "FLOP ratio is inside 5% at the point estimate but its uncertainty interval is not"
        valid = False
    else:
        reason = "matched within 5% and the latent arm is not cheaper than B_text"
        valid = True
    return MatchVerdict(
        flops_a=float(flops_a),
        flops_b=float(flops_b),
        ratio=ratio,
        ratio_se=float(se),
        ratio_low=lo,
        ratio_high=hi,
        matched=matched,
        a_cheaper=a_cheaper,
        comparison_valid=valid,
        reason=reason,
    )


@dataclass(frozen=True)
class Measured:
    """Repeated measurements of one callable."""

    flops_mean: float
    flops_se: float
    wall_mean_s: float
    wall_se_s: float
    repeats: int

    def as_dict(self) -> dict:
        return asdict(self)


def _se(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return float((var / len(values)) ** 0.5)


def measure_calls(fn, *, repeats: int = 3, warmup: int = 1) -> Measured:
    """Time ``fn`` and count FLOPs with PyTorch's flop counter.

    The counter misses some kernels. Callers must compare it with the
    wall-clock ratio before treating either number as the match.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    from torch.utils.flop_counter import FlopCounterMode

    for _ in range(max(0, warmup)):
        fn()
    flops: list[float] = []
    walls: list[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with FlopCounterMode(display=False) as counter:
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
        flops.append(float(counter.get_total_flops()))
    return Measured(
        flops_mean=sum(flops) / len(flops),
        flops_se=_se(flops),
        wall_mean_s=sum(walls) / len(walls),
        wall_se_s=_se(walls),
        repeats=repeats,
    )


@dataclass(frozen=True)
class PairVerdict:
    """FLOP-counter verdict and wall-clock verdict of the same pair of callables."""

    flops: MatchVerdict
    wall: MatchVerdict
    disagree: bool
    comparison_valid: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "flops": self.flops.as_dict(),
            "wall": self.wall.as_dict(),
            "disagree": self.disagree,
            "comparison_valid": self.comparison_valid,
            "reason": self.reason,
        }


def judge_pair(a: Measured, b: Measured, *, tolerance: float = MATCH_TOLERANCE) -> PairVerdict:
    """Judge A against B_text from two instruments.

    Disagreement means the two ratios differ by more than ``tolerance``
    relative to the FLOP-counter ratio. A disagreement voids the comparison
    even when one instrument, alone, would pass.
    """
    if a.flops_mean < 0 or b.flops_mean <= 0:
        raise ValueError("measured FLOPs are not usable")
    if a.wall_mean_s < 0 or b.wall_mean_s <= 0:
        raise ValueError("measured wall-clock is not usable")
    flop = judge_flop_ratio(
        a.flops_mean,
        b.flops_mean,
        se=ratio_se(a.flops_mean, a.flops_se, b.flops_mean, b.flops_se),
        tolerance=tolerance,
    )
    wall = judge_flop_ratio(
        a.wall_mean_s,
        b.wall_mean_s,
        se=ratio_se(a.wall_mean_s, a.wall_se_s, b.wall_mean_s, b.wall_se_s),
        tolerance=tolerance,
    )
    scale = max(abs(flop.ratio), 1e-12)
    disagree = abs(flop.ratio - wall.ratio) / scale > tolerance
    if disagree:
        return PairVerdict(
            flops=flop,
            wall=wall,
            disagree=True,
            comparison_valid=False,
            reason=(
                "FLOP counter and wall-clock disagree by more than 5% relative; "
                "the match is not certified"
            ),
        )
    valid = flop.comparison_valid and wall.comparison_valid
    reason = flop.reason if valid else f"not certified: flops [{flop.reason}]; wall [{wall.reason}]"
    if valid:
        reason = flop.reason
    return PairVerdict(
        flops=flop,
        wall=wall,
        disagree=False,
        comparison_valid=valid,
        reason=reason,
    )


def plan_depth_repeats(
    flops_per_latent_step: float,
    flops_per_text_token: float,
    *,
    capacity: int,
    cot_tokens: int = COT_BUDGET,
    tolerance: float = MATCH_TOLERANCE,
) -> dict:
    """Find an integer depth multiplier that matches a ``cot_tokens`` text budget.

    One latent step at ``depth_repeats=1`` costs ``flops_per_latent_step``.
    B_text costs ``cot_tokens * flops_per_text_token``. The search prefers a
    ratio in ``[1, 1 + tolerance]`` so A is not cheaper.

    ``fits`` is false when no integer lands in that window. The returned
    ``repeats`` is then the closest integer, and the caller must not treat the
    plan as a valid match.
    """
    if flops_per_latent_step <= 0 or flops_per_text_token <= 0:
        raise ValueError("per-step FLOPs must be positive")
    if capacity < 1 or cot_tokens < 1:
        raise ValueError("capacity and cot_tokens must be positive")
    unit = float(capacity) * float(flops_per_latent_step)
    denom = float(cot_tokens) * float(flops_per_text_token)
    # Aim at the middle of the win-eligible band: [1, 1+tolerance].
    target = (1.0 + 0.5 * float(tolerance)) * denom
    guess = max(1, int(round(target / unit)))
    window = range(max(1, guess - 5), guess + 6)

    def _ratio(repeats: int) -> float:
        return (repeats * unit) / denom

    fits: list[int] = []
    for repeats in window:
        verdict = judge_flop_ratio(repeats * unit, denom, tolerance=tolerance)
        if verdict.comparison_valid:
            fits.append(repeats)
    if fits:
        # Smallest eligible repeats: extra depth is the control, not a goal.
        chosen = min(fits)
        verdict = judge_flop_ratio(chosen * unit, denom, tolerance=tolerance)
        return {
            "repeats": chosen,
            "ratio": verdict.ratio,
            "fits": True,
            "flops_a": chosen * unit,
            "flops_b": denom,
            "reason": verdict.reason,
        }
    chosen = guess
    verdict = judge_flop_ratio(chosen * unit, denom, tolerance=tolerance)
    return {
        "repeats": chosen,
        "ratio": _ratio(chosen),
        "fits": False,
        "flops_a": chosen * unit,
        "flops_b": denom,
        "reason": "no integer depth_repeats lands in the win-eligible band: " + verdict.reason,
    }


def project_gpu_hours(
    seconds_per_step: float,
    *,
    steps_per_run: int,
    n_runs: int,
    scale: float = 1.0,
) -> dict:
    """Project wall-clock from a measured step time.

    ``scale`` is the honest multiplier from the measured configuration to the
    configuration being projected (capacity, depth, sequence length). It is
    applied linearly and reported, not hidden inside the hour count.
    """
    if seconds_per_step < 0 or steps_per_run < 0 or n_runs < 0 or scale < 0:
        raise ValueError("projection inputs must be non-negative")
    seconds = float(seconds_per_step) * int(steps_per_run) * int(n_runs) * float(scale)
    hours = seconds / 3600.0
    window_h = 17.5
    return {
        "seconds_per_step_measured": float(seconds_per_step),
        "scale": float(scale),
        "steps_per_run": int(steps_per_run),
        "n_runs": int(n_runs),
        "gpu_hours": hours,
        "windows_17_5h": hours / window_h if window_h else float("inf"),
        "fits_one_17_5h_window": hours <= window_h,
        "fits_55h_envelope": hours <= 55.0,
    }
