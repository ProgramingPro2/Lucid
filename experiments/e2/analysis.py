"""E2 endpoints: two-level bootstrap, responders, necessity, decorative path.

The primary estimand is the unweighted mean across F1, F2 and F3 of the paired
difference A − B_text. Resampling draws runs first and items inside the drawn
run second. Pooling items across runs would shrink the interval by roughly
sqrt(items per run), which is the pseudoreplication the protocol forbids.

A seed is a responder only when it emits latents on at least 10% of positions
and zeroing those latents removes at least 25% of that seed's own A − E effect.
PASS requires the Clopper–Pearson lower bound on the responder proportion to
be at least 0.60, the necessity fraction to be at least 0.50, and the paired
difference's lower bound to be above 0, all at a certified FLOP match.

The decorative-path check is the F23/F24 failure: a gate that never emits, or
a gate whose ablation does not hurt, is not a latent arm.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as st

from experiments.protocol.stats import Summary

from .flops import MatchVerdict

FAMILIES: tuple[str, ...] = ("f1", "f2", "f3")
RESPONDER_EMISSION = 0.10
RESPONDER_DROP = 0.25
NECESSITY_PASS = 0.50
RESPONDER_PASS = 0.60
# Arm C counts as ignored below the responder emission bar, or when ablation
# removes less than the per-seed drop the responder rule itself requires.
C_IGNORED_EMISSION = RESPONDER_EMISSION
C_IGNORED_NECESSITY = RESPONDER_DROP


def exact_binomial_ci(k: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    """Clopper–Pearson interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= int(k) <= int(n):
        raise ValueError(f"k={k} is outside 0..{n}")
    alpha = 1.0 - float(level)
    lo = 0.0 if k == 0 else float(st.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(st.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def _as_runs(runs: Sequence[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for run in runs:
        arr = np.asarray(run, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("a run has no items")
        if not np.isfinite(arr).all():
            raise ValueError("non-finite item score (VOID)")
        out.append(arr)
    if len(out) < 1:
        raise ValueError("need at least one run")
    return out


def two_level_bootstrap(
    runs: Sequence[np.ndarray],
    *,
    n_boot: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> Summary:
    """Mean of run-means, with runs resampled and items resampled inside them.

    The point estimate gives every run the same weight, whatever its length.
    An item-weighted mean would let a long run dominate a short one.
    """
    grouped = _as_runs(runs)
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    run_means = np.array([g.mean() for g in grouped], dtype=float)
    point = float(run_means.mean())
    n_runs = len(grouped)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_runs, size=n_runs)
        acc = 0.0
        for j in pick:
            items = grouped[int(j)]
            take = rng.integers(0, items.size, size=items.size)
            acc += float(items[take].mean())
        boots[b] = acc / n_runs
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    sd = float(run_means.std(ddof=1)) if n_runs > 1 else 0.0
    return Summary(
        n=int(sum(g.size for g in grouped)),
        mean=point,
        sd=sd,
        se=float(np.std(boots, ddof=1)),
        ci_low=float(lo),
        ci_high=float(hi),
        level=float(level),
    )


def two_level_paired(
    treatment: Sequence[np.ndarray],
    control: Sequence[np.ndarray],
    *,
    n_boot: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> Summary:
    """Paired item differences, then the same two-level bootstrap.

    Run ``i`` of ``treatment`` and run ``i`` of ``control`` are the same items
    in the same order. Resampling an item uses that item's difference, never
    a treatment item against a different control item.
    """
    a = _as_runs(treatment)
    b = _as_runs(control)
    if len(a) != len(b):
        raise ValueError("paired arms must have the same runs")
    diffs: list[np.ndarray] = []
    for left, right in zip(a, b):
        if left.shape != right.shape:
            raise ValueError("paired runs must have the same items in the same order")
        diffs.append(left - right)
    return two_level_bootstrap(diffs, n_boot=n_boot, seed=seed, level=level)


def unweighted_family_mean(by_family: dict[str, float]) -> float:
    """Mean of the three family estimates. Missing a family is an error."""
    missing = [f for f in FAMILIES if f not in by_family]
    if missing:
        raise ValueError(f"unweighted mean needs every family, missing {missing}")
    values = [float(by_family[f]) for f in FAMILIES]
    if not np.isfinite(values).all():
        raise ValueError("non-finite family mean")
    return float(np.mean(values))


def two_level_family_contrast(
    by_family: dict[str, Sequence[np.ndarray]],
    *,
    n_boot: int = 4000,
    seed: int = 0,
    level: float = 0.95,
) -> Summary:
    """Unweighted mean of per-family two-level means, one bootstrap for the mean.

    ``by_family[f]`` is a list of runs. Each run is that family's item scores
    for one seed. Families are resampled independently (a seed's F1 run is not
    tied to its F2 run inside the bootstrap). The point estimate still weights
    the three families equally even when item counts differ.
    """
    missing = [f for f in FAMILIES if f not in by_family]
    if missing:
        raise ValueError(f"contrast needs every family, missing {missing}")
    grouped = {f: _as_runs(by_family[f]) for f in FAMILIES}
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    point = unweighted_family_mean({f: float(np.mean([g.mean() for g in grouped[f]])) for f in FAMILIES})
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        family_means = []
        for f in FAMILIES:
            runs = grouped[f]
            pick = rng.integers(0, len(runs), size=len(runs))
            acc = 0.0
            for j in pick:
                items = runs[int(j)]
                take = rng.integers(0, items.size, size=items.size)
                acc += float(items[take].mean())
            family_means.append(acc / len(runs))
        boots[b] = float(np.mean(family_means))
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return Summary(
        n=int(sum(g.size for runs in grouped.values() for g in runs)),
        mean=point,
        sd=float(np.std([float(np.mean([g.mean() for g in grouped[f]])) for f in FAMILIES], ddof=1)),
        se=float(np.std(boots, ddof=1)),
        ci_low=float(lo),
        ci_high=float(hi),
        level=float(level),
    )


def necessity_fraction(acc_arm: float, acc_ablated: float, acc_vehicle: float) -> float:
    """Share of the arm-versus-vehicle effect removed by zeroing the latents.

    If the arm does not beat the vehicle, there is no effect to explain and the
    fraction is 0. A negative drop (ablation helps) is also 0: the channel was
    not load-bearing.
    """
    effect = float(acc_arm) - float(acc_vehicle)
    drop = float(acc_arm) - float(acc_ablated)
    if not np.isfinite([effect, drop]).all():
        raise ValueError("non-finite accuracy")
    if effect <= 1e-12 or drop <= 0:
        return 0.0
    return float(drop / effect)


def is_responder(
    emission_rate: float,
    acc_arm: float,
    acc_vehicle: float,
    acc_ablated: float,
    *,
    min_emission: float = RESPONDER_EMISSION,
    min_drop: float = RESPONDER_DROP,
) -> bool:
    """PROTOCOL §6.4 responder rule for one seed."""
    if float(emission_rate) < float(min_emission):
        return False
    effect = float(acc_arm) - float(acc_vehicle)
    if effect <= 0:
        return False
    drop = float(acc_arm) - float(acc_ablated)
    return drop >= float(min_drop) * effect


def responder_summary(flags: Sequence[bool], *, level: float = 0.95) -> dict:
    """Exact binomial interval on the responder proportion."""
    bits = [bool(f) for f in flags]
    if not bits:
        raise ValueError("no seeds")
    k = int(sum(bits))
    n = len(bits)
    lo, hi = exact_binomial_ci(k, n, level=level)
    return {
        "k": k,
        "n": n,
        "proportion": k / n,
        "ci_low": lo,
        "ci_high": hi,
        "pass_bound": lo >= RESPONDER_PASS,
    }


def decorative_report(
    emission_rate: float,
    acc_arm: float,
    acc_ablated: float,
) -> dict:
    """Detect an auxiliary channel that is not doing the work (F23/F24).

    ``no_latent_emitted`` is the hard failure: accuracy then comes from the
    text path, and the gate is decoration. ``ablation_does_not_hurt`` is the
    same failure with the channel present: zeroing it does not drop accuracy,
    which is what the identity and zero arms did in F23 and F24.
    """
    rate = float(emission_rate)
    drop = float(acc_arm) - float(acc_ablated)
    if not np.isfinite([rate, drop]).all():
        raise ValueError("non-finite decorative inputs")
    flags: list[str] = []
    if rate <= 0.0:
        flags.append("no_latent_emitted")
    elif rate < RESPONDER_EMISSION:
        flags.append("emission_below_10pct")
    if rate > 0.0 and drop <= 0.0:
        flags.append("ablation_does_not_hurt")
    return {
        "decorative": bool(flags),
        "flags": flags,
        "emission_rate": rate,
        "ablation_drop": drop,
    }


def c_is_ignored(emission_rate: float, necessity: float) -> bool:
    """Arm C (λ = 0) is ignored when it fails the emission bar or the drop bar.

    PROTOCOL §6.4 voids the experiment when C is *not* ignored. A free latent
    that is load-bearing means the charge is not what the design thinks it is.
    """
    return float(emission_rate) < C_IGNORED_EMISSION or float(necessity) < C_IGNORED_NECESSITY


def g2b_verdict(
    *,
    diff_ci_low: float,
    responder_ci_low: float,
    necessity: float,
    flop: MatchVerdict,
    b_text_minus_e: float,
    c_ignored: bool | None,
    capability_void: bool = False,
) -> dict:
    """Apply PROTOCOL §6.4. Any void condition preempts PASS and FAIL.

    The written PASS line is ``A − B_text`` lower bound > 0, not lower bound
    above the +0.05 MPID. The MPID is reported by the caller; it is not
    substituted for that line.
    """
    reasons: list[str] = []
    if capability_void:
        reasons.append("backbone failed the §6.3.1 capability screen")
    if not flop.comparison_valid:
        reasons.append(flop.reason)
    if float(b_text_minus_e) < 0.10:
        reasons.append("B_text − E < 0.10; the assay cannot see the effect")
    if c_ignored is False:
        reasons.append("arm C (λ = 0) was not ignored")
    if reasons:
        return {"verdict": "VOID", "reasons": reasons}
    passed = (
        float(diff_ci_low) > 0.0
        and float(responder_ci_low) >= RESPONDER_PASS
        and float(necessity) >= NECESSITY_PASS
    )
    fail_reasons: list[str] = []
    if not (float(diff_ci_low) > 0.0):
        fail_reasons.append("A − B_text lower bound is not above 0")
    if not (float(responder_ci_low) >= RESPONDER_PASS):
        fail_reasons.append("responder proportion lower bound is below 0.60")
    if not (float(necessity) >= NECESSITY_PASS):
        fail_reasons.append("necessity is below 50%")
    return {
        "verdict": "PASS" if passed else "FAIL",
        "reasons": fail_reasons,
    }
