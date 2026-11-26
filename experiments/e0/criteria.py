"""G0 decision rule.

The four criteria and their thresholds live here and nowhere else. The report
copies :func:`thresholds`, so a write-up cannot drift from the code, and the
code cannot drift from docs/PROTOCOL.md §3:

* null half-width ≤ 0.015, on the null-pair difference
* dose monotone
* 100-vs-0 lower bound > 0.03
* text-prefix lower bound > 0.03

A non-finite score is VOID, not a pass. That is the F27 failure (a fluency
gate printed PASS at ``mean_nll = Infinity``). ``absolute_gate_is_valid`` is
the F38 guard: a threshold that does not clear the measured null cannot pass,
because the null would already satisfy it.

N2 (docs/PROTOCOL.md §1): the null pair must have a CI that contains zero and
is not a point. A zero-variance greedy pair has a zero-width CI at any *n*,
and a half-width check alone would pass it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as st

from experiments.e4.necessity import absolute_gate_is_valid
from experiments.protocol.stats import Summary, mean_ci, paired_summary, required_n

# docs/PROTOCOL.md §3, G0 row. Do not restate these in the report by hand.
NULL_HALF_WIDTH_MAX = 0.015
DOSE_100_VS_0_LB_MIN = 0.03
TEXT_PREFIX_LB_MIN = 0.03
# A strict point-estimate ladder fails with probability 1/2 on any flat
# segment, at every n. The tolerance is the same number as the null
# half-width, so the two criteria cannot be edited apart.
DOSE_DROP_TOLERANCE = NULL_HALF_WIDTH_MAX

PASS = "PASS"
FAIL = "FAIL"
VOID = "VOID"
GO = "GO"
NO_GO = "NO-GO"

# Published aggregates in artifacts/e0/rehearsal.json (n = 14). The artifact
# is gitignored, so the gate math carries the figures it was computed from.
REHEARSAL_N = 14
REHEARSAL_COS_LADDER = (0.5489, 0.5771, 0.5735)
REHEARSAL_MARGIN_LADDER = (0.0099, 0.1035, 0.1313)
REHEARSAL_SHUFFLED_MARGIN_CI = (-0.0983, 0.1181)
REHEARSAL_NECESSITY_CI = (-0.0289, 0.2718)
REHEARSAL_TRUE_COS = 0.5735
REHEARSAL_ZERO_COS = 0.6567
REHEARSAL_TRUE_MARGIN = 0.1313
REHEARSAL_ZERO_MARGIN = 0.1641


def thresholds() -> dict:
    """The only copy of the G0 numbers the report is allowed to use."""
    return {
        "null_half_width_max": NULL_HALF_WIDTH_MAX,
        "dose_drop_tolerance": DOSE_DROP_TOLERANCE,
        "dose_100_vs_0_lb_min": DOSE_100_VS_0_LB_MIN,
        "text_prefix_lb_min": TEXT_PREFIX_LB_MIN,
    }


@dataclass(frozen=True)
class AssayScores:
    """Per-item protocol margins. Dose arrays are aligned. Null arrays are aligned."""

    dose0: np.ndarray
    dose50: np.ndarray
    dose100: np.ndarray
    text_true: np.ndarray
    text_rand: np.ndarray
    zero: np.ndarray
    null_a: np.ndarray
    null_b: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "dose0": np.asarray(self.dose0, dtype=float),
            "dose50": np.asarray(self.dose50, dtype=float),
            "dose100": np.asarray(self.dose100, dtype=float),
            "text_true": np.asarray(self.text_true, dtype=float),
            "text_rand": np.asarray(self.text_rand, dtype=float),
            "zero": np.asarray(self.zero, dtype=float),
            "null_a": np.asarray(self.null_a, dtype=float),
            "null_b": np.asarray(self.null_b, dtype=float),
        }


def _summary_dict(s: Summary) -> dict:
    return {
        "n": s.n,
        "mean": s.mean,
        "sd": s.sd,
        "se": s.se,
        "ci_low": s.ci_low,
        "ci_high": s.ci_high,
        "half_width": (s.ci_high - s.ci_low) / 2.0,
    }


def _finite(x: np.ndarray) -> bool:
    return bool(x.size) and bool(np.isfinite(x).all())


def _criterion(name: str, verdict: str, *, fail_when: str, **detail) -> dict:
    return {"name": name, "verdict": verdict, "fail_when": fail_when, **detail}


def half_width_at(sd: float, n: int, level: float = 0.95) -> float:
    """Half-width of :func:`mean_ci` for a sample with this SD and size.

    The sample is built so its sample SD equals ``sd`` exactly, then the
    interval is the one the gate uses. A closed form that skipped ``mean_ci``
    could drift from the criterion.
    """
    if n < 2:
        return float("inf")
    if sd < 0:
        raise ValueError("sd must be non-negative")
    if sd == 0:
        return 0.0
    base = np.arange(n, dtype=float)
    base -= base.mean()
    sample = base / base.std(ddof=1) * sd
    summary = mean_ci(sample, level=level)
    return (summary.ci_high - summary.ci_low) / 2.0


def n_for_half_width(
    sd: float, half_width: float = NULL_HALF_WIDTH_MAX, level: float = 0.95
) -> int:
    """Smallest n whose ``mean_ci`` half-width is at most ``half_width``."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    if sd <= 0:
        return 2
    z = float(st.norm.ppf(0.5 + level / 2.0))
    guess = max(4, int(np.ceil((z * sd / half_width) ** 2)))
    hi = guess
    while half_width_at(sd, hi, level) > half_width:
        hi *= 2
        if hi > 5_000_000:
            raise RuntimeError("half-width not reachable at a practical n")
    lo = 2
    while lo < hi:
        mid = (lo + hi) // 2
        if half_width_at(sd, mid, level) <= half_width:
            hi = mid
        else:
            lo = mid + 1
    return lo


def sd_from_ci(ci_low: float, ci_high: float, n: int, level: float = 0.95) -> float:
    """Invert the t interval ``mean_ci`` prints, for a published CI."""
    if n < 2:
        raise ValueError("n must be at least 2")
    half = (ci_high - ci_low) / 2.0
    crit = float(st.t.ppf(0.5 + level / 2.0, df=n - 1))
    return half * np.sqrt(n) / crit


def rehearsal_arm_sd() -> float:
    """SD of the rehearsal's shuffled-arm margin, from its published CI."""
    lo, hi = REHEARSAL_SHUFFLED_MARGIN_CI
    return sd_from_ci(lo, hi, REHEARSAL_N)


def rehearsal_dose_contrast_sd() -> float:
    """SD of true-margin minus shuffled-margin, from the published necessity CI."""
    lo, hi = REHEARSAL_NECESSITY_CI
    return sd_from_ci(lo, hi, REHEARSAL_N)


def sample_size_plan(arm_sd: float | None = None) -> dict:
    """Item counts the null-width criterion implies at the rehearsal SD.

    The implemented null is the paired difference of two disjoint replicates.
    Independent replicates have SD ``arm_sd * sqrt(2)``. ``required_pairs`` is
    the n that puts that difference's ``mean_ci`` half-width at 0.015.
    ``required_documents`` is twice that, because the two replicates do not
    share items. ``single_arm_items`` is the weaker reading (half-width of one
    arm's mean). Both are reported; the preflight uses the pair.
    """
    sd = float(rehearsal_arm_sd() if arm_sd is None else arm_sd)
    pair_sd = sd * np.sqrt(2.0)
    pairs = n_for_half_width(pair_sd, NULL_HALF_WIDTH_MAX)
    single = n_for_half_width(sd, NULL_HALF_WIDTH_MAX)
    contrast_sd = rehearsal_dose_contrast_sd()
    return {
        "arm_sd": sd,
        "pair_sd": pair_sd,
        "required_pairs": pairs,
        "required_documents": pairs * 2,
        "single_arm_items": single,
        "half_width_at_protocol_n_256_pair": half_width_at(pair_sd, 256),
        "half_width_at_protocol_n_256_arm": half_width_at(sd, 256),
        "dose_contrast_sd": contrast_sd,
        "required_n_power_mpid_0.03": required_n(contrast_sd, DOSE_100_VS_0_LB_MIN),
        "required_n_to_detect_rehearsal_dip": required_n(
            sd, abs(REHEARSAL_COS_LADDER[2] - REHEARSAL_COS_LADDER[1])
        ),
        "thresholds": thresholds(),
    }


def diagnose_rehearsal() -> dict:
    """Why the published dose ladder [0.5489, 0.5771, 0.5735] is not monotone.

    The dip is on absolute cosine to the target. The protocol margin on the
    same run rises at both steps. The dip is a small fraction of the SE of
    the mean at n=14. ``slerp`` at dose 0.5 is the spherical midpoint, so
    argument order cannot move it. A strict point-estimate monotone check on
    a flat upper step fails about half the time at every n; more items do
    not make a true zero step look strictly increasing.
    """
    cos = REHEARSAL_COS_LADDER
    margin = REHEARSAL_MARGIN_LADDER
    dip = cos[1] - cos[2]
    sd = rehearsal_arm_sd()
    se = sd / np.sqrt(REHEARSAL_N)
    plan = sample_size_plan(sd)
    return {
        "n": REHEARSAL_N,
        "cos_ladder": list(cos),
        "cos_steps": [cos[1] - cos[0], cos[2] - cos[1]],
        "cos_dip": dip,
        "margin_ladder": list(margin),
        "margin_monotone": margin[0] <= margin[1] <= margin[2],
        "arm_sd": sd,
        "se_of_mean": se,
        "dip_over_se": dip / se,
        "dip_within_dose_tolerance": dip <= DOSE_DROP_TOLERANCE,
        "n_to_detect_dip_as_an_effect": plan["required_n_to_detect_rehearsal_dip"],
        "zero_minus_true_cos": REHEARSAL_ZERO_COS - REHEARSAL_TRUE_COS,
        "zero_minus_true_margin": REHEARSAL_ZERO_MARGIN - REHEARSAL_TRUE_MARGIN,
        "cause": "underpowered_absolute_cosine",
        "slerp": (
            "The rehearsal's only slerp call was the 50% midpoint. "
            "dose_latent(shuffled, true, 0.5) matches slerp(true, shuffled, 0.5). "
            "The endpoints of the published ladder were the raw vectors, not slerp."
        ),
    }


def _null_criterion(null_a: np.ndarray, null_b: np.ndarray) -> tuple[dict, Summary | None]:
    fail_when = (
        "half-width > 0.015, or the CI excludes 0, or the CI is a single point "
        "(N2: a null that does not vary)"
    )
    if null_a.shape != null_b.shape or null_a.size < 2:
        return (
            _criterion(
                "null_half_width",
                VOID,
                fail_when=fail_when,
                reason="null pair needs at least 2 aligned items",
            ),
            None,
        )
    if not (_finite(null_a) and _finite(null_b)):
        return (
            _criterion(
                "null_half_width",
                VOID,
                fail_when=fail_when,
                reason="non-finite null score",
            ),
            None,
        )
    summary = paired_summary(null_a, null_b)
    hw = (summary.ci_high - summary.ci_low) / 2.0
    contains_zero = bool(summary.ci_low <= 0.0 <= summary.ci_high)
    is_point = bool(summary.sd == 0.0 or summary.ci_high <= summary.ci_low)
    ok = bool(hw <= NULL_HALF_WIDTH_MAX and contains_zero and not is_point)
    return (
        _criterion(
            "null_half_width",
            PASS if ok else FAIL,
            fail_when=fail_when,
            **_summary_dict(summary),
            contains_zero=contains_zero,
            is_point=is_point,
            threshold=NULL_HALF_WIDTH_MAX,
        ),
        summary,
    )


def _monotone_criterion(means: list[float]) -> dict:
    fail_when = (
        f"an adjacent dose mean drops by more than {DOSE_DROP_TOLERANCE} "
        "(the null half-width). Equality is monotone. A drop inside the "
        "tolerance is sampling noise on the scale the null criterion claims "
        "to resolve"
    )
    if any(not np.isfinite(m) for m in means):
        return _criterion(
            "dose_monotone", VOID, fail_when=fail_when, reason="non-finite dose mean"
        )
    drops = [means[i] - means[i + 1] for i in range(len(means) - 1)]
    worst = max(drops) if drops else 0.0
    ok = bool(worst <= DOSE_DROP_TOLERANCE)
    return _criterion(
        "dose_monotone",
        PASS if ok else FAIL,
        fail_when=fail_when,
        means=means,
        drops=drops,
        worst_drop=worst,
        tolerance=DOSE_DROP_TOLERANCE,
        strict_monotone=bool(all(d <= 0.0 for d in drops)),
    )


def _lb_criterion(
    name: str,
    treatment: np.ndarray,
    control: np.ndarray,
    threshold: float,
    null_mean: float | None,
    *,
    fail_when: str,
) -> dict:
    if treatment.shape != control.shape or treatment.size < 2:
        return _criterion(name, VOID, fail_when=fail_when, reason="need at least 2 aligned items")
    if not (_finite(treatment) and _finite(control)):
        return _criterion(name, VOID, fail_when=fail_when, reason="non-finite score")
    if null_mean is None or not np.isfinite(null_mean):
        return _criterion(name, VOID, fail_when=fail_when, reason="null mean was not measured")
    # F38: a bar that sits at or below the measured null is already cleared
    # by doing nothing. That is VOID, not a pass and not a treatment failure.
    if not absolute_gate_is_valid(threshold, float(null_mean)):
        return _criterion(
            name,
            VOID,
            fail_when=fail_when,
            reason="threshold does not clear the measured null mean",
            threshold=threshold,
            null_mean=float(null_mean),
        )
    summary = paired_summary(treatment, control)
    ok = bool(summary.ci_low > threshold)
    return _criterion(
        name,
        PASS if ok else FAIL,
        fail_when=fail_when,
        **_summary_dict(summary),
        threshold=threshold,
        null_mean=float(null_mean),
        threshold_clears_null=True,
    )


def evaluate(scores: AssayScores) -> dict:
    """Score the four G0 criteria. Overall GO only if every one PASSes."""
    arrays = scores.as_arrays()
    null, null_summary = _null_criterion(arrays["null_a"], arrays["null_b"])
    null_mean = None if null_summary is None else null_summary.mean

    if arrays["dose0"].shape != arrays["dose50"].shape or arrays["dose0"].shape != arrays["dose100"].shape:
        raise ValueError("dose columns must align")

    dose_means: list[float] = []
    dose_void = False
    for key in ("dose0", "dose50", "dose100"):
        col = arrays[key]
        if not _finite(col) or col.size == 0:
            dose_void = True
            break
        dose_means.append(float(col.mean()))
    if dose_void:
        monotone = _criterion(
            "dose_monotone",
            VOID,
            fail_when=_monotone_criterion([0.0, 0.0, 0.0])["fail_when"],
            reason="non-finite or empty dose column",
        )
    else:
        monotone = _monotone_criterion(dose_means)

    dose_lb = _lb_criterion(
        "dose_100_vs_0",
        arrays["dose100"],
        arrays["dose0"],
        DOSE_100_VS_0_LB_MIN,
        null_mean,
        fail_when=(
            "paired lower bound of (dose 1 − dose 0) is not > 0.03, "
            "or 0.03 does not clear the measured null-pair mean"
        ),
    )
    text_lb = _lb_criterion(
        "text_prefix",
        arrays["text_true"],
        arrays["text_rand"],
        TEXT_PREFIX_LB_MIN,
        null_mean,
        fail_when=(
            "paired lower bound of (true text-prefix − random text-prefix) "
            "is not > 0.03, or 0.03 does not clear the measured null-pair mean"
        ),
    )
    criteria = [null, monotone, dose_lb, text_lb]
    verdicts = [c["verdict"] for c in criteria]
    if any(v == VOID for v in verdicts):
        overall = VOID
    elif all(v == PASS for v in verdicts):
        overall = GO
    else:
        overall = NO_GO

    format_control = _format_control(arrays["dose100"], arrays["zero"])
    return {
        "thresholds": thresholds(),
        "criteria": criteria,
        "verdict": overall,
        "format_control": format_control,
        "n_dose": int(arrays["dose0"].size),
        "n_null_pairs": int(arrays["null_a"].size),
        "n_text": int(arrays["text_true"].size),
    }


def _format_control(true_margin: np.ndarray, zero_margin: np.ndarray) -> dict:
    """True latent minus the no-latent arm. Not one of the four gates.

    F42: on the rehearsal the zero arm beat the true latent (cosine 0.657 vs
    0.574, margin 0.164 vs 0.131). E4's PASS rule says the no-draft arm must
    not match. G0's four criteria do not say that. A GO here can coexist
    with a zero arm that wins. The flag is reported so that cannot hide.
    """
    note = (
        "Not a G0 gate. Protocol §3 does not require the latent arm to beat "
        "the zero-prefix arm. Protocol §7.3 does, for E4. F42 is this gap."
    )
    if true_margin.shape != zero_margin.shape or true_margin.size < 2:
        return {"status": VOID, "reason": "true and zero margins do not align", "note": note}
    if not (_finite(true_margin) and _finite(zero_margin)):
        return {"status": VOID, "reason": "non-finite format-control score", "note": note}
    summary = paired_summary(true_margin, zero_margin)
    return {
        "status": "reported",
        "note": note,
        **_summary_dict(summary),
        "zero_beats_true": bool(summary.mean < 0.0),
        "true_beats_zero_lb": bool(summary.ci_low > 0.0),
    }


def effect_detected(treatment, control, threshold: float = DOSE_100_VS_0_LB_MIN) -> str:
    """Detection rule used by the false-positive and false-negative checks.

    The null mean passed to the F38 guard is 0: this helper is about whether
    the contrast itself clears the threshold, which is the question a synthetic
    zero effect and a synthetic known effect are there to answer.
    """
    result = _lb_criterion(
        "effect",
        np.asarray(treatment, dtype=float),
        np.asarray(control, dtype=float),
        threshold,
        0.0,
        fail_when="lower bound does not clear the threshold",
    )
    return result["verdict"]
