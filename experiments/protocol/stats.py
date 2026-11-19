"""Statistics for the 2026-11 protocol.

Every confirmatory claim in ``docs/PROTOCOL.md`` is decided by comparing a
confidence interval against a pre-registered MPID. This module implements that
comparison and the resampling needed to respect nesting, so no analysis script
re-implements it differently.

Design notes
------------
- Intervals are **paired** where the design is paired (same seed, same item).
- Where observations nest inside documents, the interval is a **cluster
  bootstrap**: resample documents, not paragraphs.
- ``required_n`` uses the closed form for a one-sample/paired design, then
  applies an inflation factor for small-pilot uncertainty and t-degrees of
  freedom.

Run ``python -m experiments.protocol.stats`` for the worked numbers used in
docs/PROTOCOL.md §5.4 and §7.4.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as st


# --------------------------------------------------------------------------
# Basic summaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """A point estimate with a confidence interval."""

    n: int
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    level: float

    def as_dict(self) -> dict:
        return asdict(self)


def paired_differences(treatment, control) -> np.ndarray:
    """Per-unit differences for a paired design.

    ``treatment`` and ``control`` must be aligned: index *i* is the same seed
    or the same item in both arms.
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(control, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("empty paired arrays")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite value in paired arrays (VOID)")
    return a - b


def mean_ci(values, level: float = 0.95) -> Summary:
    """Mean with a t-based confidence interval. Fails loudly on non-finite input."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("empty sample")
    if not np.isfinite(x).all():
        raise ValueError("non-finite value in sample (VOID)")
    n = int(x.size)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else float("nan")
    if n > 1:
        crit = float(st.t.ppf(0.5 + level / 2, df=n - 1))
    else:
        crit = float("nan")
    return Summary(
        n=n,
        mean=mean,
        sd=sd,
        se=se,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        level=level,
    )


def paired_summary(treatment, control, level: float = 0.95) -> Summary:
    """Paired mean difference with a t-based interval."""
    return mean_ci(paired_differences(treatment, control), level=level)


# --------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------


def cluster_bootstrap_ci(
    values,
    clusters,
    *,
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Summary:
    """Percentile bootstrap resampling **clusters**, not observations.

    Use this whenever observations nest (paragraphs inside documents, items
    inside runs). Resampling observations would shrink the interval by roughly
    ``sqrt(items_per_cluster)`` — the pseudoreplication error in docs/EXPLAINER.md.
    """
    x = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    if x.shape[0] != c.shape[0]:
        raise ValueError("values and clusters must align")
    if not np.isfinite(x).all():
        raise ValueError("non-finite value (VOID)")

    uniq, inverse = np.unique(c, return_inverse=True)
    n_clusters = uniq.size
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters for a bootstrap")

    # Group indices by cluster once, then resample clusters with replacement.
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(n_clusters + 1))
    groups = [order[bounds[i] : bounds[i + 1]] for i in range(n_clusters)]

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[j] for j in pick])
        means[b] = x[idx].mean()

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return Summary(
        n=int(x.size),
        mean=float(x.mean()),
        sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
        se=float(np.std(means, ddof=1)),
        ci_low=float(lo),
        ci_high=float(hi),
        level=level,
    )


# --------------------------------------------------------------------------
# Power and sample size
# --------------------------------------------------------------------------


def mde(sd: float, n: int, *, alpha: float = 0.05, power: float = 0.90) -> float:
    """Minimum detectable effect for a one-sided paired test."""
    if sd <= 0:
        return 0.0
    if n < 2:
        return float("inf")
    z_alpha = float(st.norm.ppf(1.0 - alpha))
    z_power = float(st.norm.ppf(power))
    return (z_alpha + z_power) * sd / np.sqrt(n)


def required_n(
    sd: float,
    mpid: float,
    *,
    alpha: float = 0.05,
    power: float = 0.90,
    pilot_inflation: float = 1.0,
    floor: int = 1,
    cap: int | None = None,
) -> int:
    """Sample size for a one-sided paired test.

    ``pilot_inflation`` is the upper confidence factor on an SD estimated from a
    small pilot: 1.29 for 10 seeds, 2.1 for 3 seeds. Applying it is what stops a
    noisy pilot from under-sizing the main experiment.
    """
    if mpid <= 0:
        raise ValueError("mpid must be positive")
    if sd <= 0:
        return floor
    eff_sd = sd * pilot_inflation
    z_alpha = float(st.norm.ppf(1.0 - alpha))
    z_power = float(st.norm.ppf(power))
    n = int(np.ceil(((z_alpha + z_power) * eff_sd / mpid) ** 2))
    n = max(n, floor)
    if cap is not None:
        n = min(n, cap)
    return n


def se_from_variance_components(
    sigma_item: float, sigma_run: float, n_items: int, n_runs: int
) -> float:
    """SE of a necessity contrast averaged over items within runs.

    ``SE = sqrt(sigma_item^2 / (R*I) + sigma_run^2 / R)`` — the decomposition
    used to justify 10 refiner seeds in docs/PROTOCOL.md §7.4.
    """
    if n_items < 1 or n_runs < 1:
        raise ValueError("n_items and n_runs must be positive")
    var = sigma_item**2 / (n_items * n_runs) + sigma_run**2 / n_runs
    return float(np.sqrt(var))


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
VOID = "VOID"


def decide(
    ci: Summary | tuple[float, float],
    mpid: float,
    *,
    voided: bool = False,
    void_reason: str | None = None,
) -> dict:
    """Apply the pre-registered decision rule.

    PASS          : CI lower bound > MPID
    FAIL          : CI upper bound < MPID
    INCONCLUSIVE  : CI straddles the MPID
    VOID          : a qualification check failed, so no answer exists
    """
    if voided:
        return {"verdict": VOID, "reason": void_reason or "qualification check failed"}

    if isinstance(ci, Summary):
        lo, hi = ci.ci_low, ci.ci_high
    else:
        lo, hi = ci

    if not (np.isfinite(lo) and np.isfinite(hi)):
        return {"verdict": VOID, "reason": "non-finite interval"}

    if lo > mpid:
        verdict = PASS
    elif hi < mpid:
        verdict = FAIL
    else:
        verdict = INCONCLUSIVE
    return {"verdict": verdict, "ci_low": lo, "ci_high": hi, "mpid": mpid}


def holm(pvalues, alpha: float = 0.05) -> list[dict]:
    """Holm–Bonferroni step-down across a family.

    Thresholds are ``alpha / (m - k)`` for the k-th smallest p-value. Rejection
    is step-down: once a test fails, every larger p-value fails too, but each
    test still reports its own threshold (do not overwrite them on the break).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    out = [{"p": float(v), "threshold": float("nan"), "reject": False} for v in p]

    stop = False
    for rank, idx in enumerate(order):
        thr = alpha / (m - rank)
        out[idx]["threshold"] = thr
        if not stop and p[idx] <= thr:
            out[idx]["reject"] = True
        else:
            stop = True
    return out


# --------------------------------------------------------------------------
# Worked numbers
# --------------------------------------------------------------------------


def _worked_examples() -> None:
    print("E1 sample size  (alpha=0.025 one-sided, power=0.90, pilot 10 seeds)")
    print(f"{'SD_pilot':>10} {'n_main':>8} {'MDE at cap 64':>16}")
    for sd in (0.008, 0.010, 0.015, 0.020, 0.030):
        n = required_n(sd, 0.0112, alpha=0.025, pilot_inflation=1.29, floor=24, cap=64)
        print(f"{sd:>10.3f} {n:>8d} {mde(sd, n, alpha=0.025):>16.4f}")

    print("\nE4-S2 seeds    (sigma_item=0.09, sigma_run=0.03, 128 items)")
    print(f"{'R':>4} {'SE':>8} {'MDE a=0.05':>12} {'MDE a=0.025':>13}")
    for r in (5, 8, 10, 12, 16):
        se = se_from_variance_components(0.09, 0.03, 128, r)
        print(f"{r:>4d} {se:>8.4f} {2.8*se:>12.4f} {3.0*se:>13.4f}")


if __name__ == "__main__":
    _worked_examples()
