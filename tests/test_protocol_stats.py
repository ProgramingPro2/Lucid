"""Tests for the frozen protocol statistics (docs/PROTOCOL.md §2, §5.4, §7.4).

These tests pin the analysis primitives, not the analysis scripts. A change
that makes one of them fail is a protocol amendment unless it is fixing a real
defect, and the distinction has to be argued in the open.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.protocol.stats import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    VOID,
    cluster_bootstrap_ci,
    decide,
    holm,
    mde,
    mean_ci,
    paired_differences,
    paired_summary,
    required_n,
    se_from_variance_components,
)


# --------------------------------------------------------------------------
# Paired summaries
# --------------------------------------------------------------------------


def test_paired_summary_recovers_constant_difference_exactly():
    """A planted constant difference has sd 0, so the CI must be a point."""
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treatment = control + 0.5

    summary = paired_summary(treatment, control)

    assert summary.n == 5
    assert summary.mean == pytest.approx(0.5)
    assert summary.sd == 0.0
    assert summary.se == 0.0
    assert summary.ci_low == pytest.approx(0.5)
    assert summary.ci_high == pytest.approx(0.5)


def test_paired_summary_rejects_non_finite_input():
    """NaN anywhere is a VOID, never a missing value to be dropped."""
    with pytest.raises(ValueError, match="non-finite"):
        paired_summary([1.0, np.inf, 2.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        paired_summary([1.0, 2.0, 3.0], [0.0, np.nan, 0.0])


def test_mean_ci_rejects_non_finite_input_and_empty_sample():
    with pytest.raises(ValueError, match="non-finite"):
        mean_ci([0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="empty"):
        mean_ci([])


def test_paired_differences_requires_alignment_and_finite_values():
    with pytest.raises(ValueError, match="align"):
        paired_differences([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="non-finite"):
        paired_differences([1.0, np.inf], [1.0, 2.0])


# --------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------


def _nested_values() -> tuple[np.ndarray, np.ndarray]:
    """Ten clusters with real between-cluster spread and little within."""
    rng = np.random.default_rng(11)
    cluster_means = np.array(
        [-0.50, -0.30, -0.10, 0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
    )
    per_cluster = 25
    values = np.concatenate(
        [mean + rng.normal(0.0, 0.01, per_cluster) for mean in cluster_means]
    )
    clusters = np.repeat(np.arange(cluster_means.size), per_cluster)
    return values, clusters


def test_cluster_bootstrap_is_wider_than_naive_mean_ci():
    """Resampling documents, not paragraphs, must not pseudoreplicate."""
    values, clusters = _nested_values()

    boot = cluster_bootstrap_ci(values, clusters, n_boot=4_000, seed=123)
    naive = mean_ci(values)

    boot_width = boot.ci_high - boot.ci_low
    naive_width = naive.ci_high - naive.ci_low

    assert boot_width > naive_width
    assert boot.mean == pytest.approx(float(values.mean()))
    assert boot.n == values.size


def test_cluster_bootstrap_is_reproducible_for_a_fixed_seed():
    values, clusters = _nested_values()
    a = cluster_bootstrap_ci(values, clusters, n_boot=1_000, seed=7)
    b = cluster_bootstrap_ci(values, clusters, n_boot=1_000, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_cluster_bootstrap_requires_at_least_two_clusters():
    with pytest.raises(ValueError, match="at least 2 clusters"):
        cluster_bootstrap_ci([1.0, 2.0, 3.0], ["doc-1", "doc-1", "doc-1"])


# --------------------------------------------------------------------------
# Power and sample size
# --------------------------------------------------------------------------


def test_required_n_is_monotone_in_mpid_and_sd():
    mpids = [0.004, 0.008, 0.0112, 0.020, 0.040]
    ns = [required_n(0.015, mpid, alpha=0.025, pilot_inflation=1.29) for mpid in mpids]
    assert all(later <= earlier for earlier, later in zip(ns, ns[1:]))

    sds = [0.004, 0.008, 0.012, 0.015, 0.020, 0.030]
    ns = [required_n(sd, 0.0112, alpha=0.025, pilot_inflation=1.29) for sd in sds]
    assert all(later >= earlier for earlier, later in zip(ns, ns[1:]))


def test_required_n_respects_floor_and_cap():
    assert required_n(0.0, 0.0112, floor=24, cap=64) == 24
    assert required_n(1.0, 0.0112, floor=24, cap=64) == 64


def test_required_n_e1_case_matches_protocol_table():
    """docs/PROTOCOL.md §5.4: sd=0.015, MPID=0.0112 -> n_main = 32.

    The closed form is ceil(((z_{1-alpha/2} + z_{power}) * inflation * sd / mpid)^2).
    With alpha=0.025 and power=0.90 that is (1.95996 + 1.28155) * 1.29 = 4.1815,
    so (4.1815 * 0.015 / 0.0112)^2 = 31.36 -> 32. An earlier draft of the doc
    said 40; the doc was corrected, the code was always right.
    """
    n = required_n(
        0.015,
        0.0112,
        alpha=0.025,
        pilot_inflation=1.29,
        floor=24,
        cap=64,
    )
    assert n == 32


def test_mde_is_finite_and_decreases_with_n():
    values = [mde(0.015, n, alpha=0.025) for n in (10, 20, 40, 64)]
    assert all(np.isfinite(value) for value in values)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))


def test_se_from_variance_components_matches_the_documented_formula():
    sigma_item, sigma_run, n_items, n_runs = 0.09, 0.03, 128, 10
    expected = np.sqrt(
        sigma_item**2 / (n_items * n_runs) + sigma_run**2 / n_runs
    )
    assert se_from_variance_components(
        sigma_item, sigma_run, n_items, n_runs
    ) == pytest.approx(expected)
    assert se_from_variance_components(0.09, 0.03, 128, 20) < se_from_variance_components(
        0.09, 0.03, 128, 10
    )
    with pytest.raises(ValueError):
        se_from_variance_components(0.09, 0.03, 0, 10)


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------


def test_decide_pass_requires_lower_bound_above_mpid():
    assert decide((0.06, 0.10), 0.05)["verdict"] == PASS


def test_decide_fail_when_entire_interval_below_mpid():
    assert decide((0.00, 0.04), 0.05)["verdict"] == FAIL


def test_decide_inconclusive_when_interval_straddles_mpid():
    assert decide((0.01, 0.09), 0.05)["verdict"] == INCONCLUSIVE
    # Exactly at the MPID is not "above" it.
    assert decide((0.05, 0.09), 0.05)["verdict"] == INCONCLUSIVE


def test_decide_void_paths():
    voided = decide((0.06, 0.10), 0.05, voided=True, void_reason="nan in eval")
    assert voided["verdict"] == VOID
    assert voided["reason"] == "nan in eval"
    assert decide((float("nan"), 0.10), 0.05)["verdict"] == VOID
    assert decide((0.06, float("inf")), 0.05)["verdict"] == VOID


def test_decide_accepts_a_summary_object():
    summary = paired_summary([0.10, 0.11, 0.12], [0.0, 0.0, 0.0])
    assert decide(summary, 0.05)["verdict"] == PASS


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------


def test_holm_known_pvalues():
    """Holm thresholds are 0.0167, 0.025, 0.05, so all three reject.

    [True, False, False] would be the *Bonferroni* answer (0.05/3 = 0.0167);
    Holm is step-down and less conservative, which is the whole point of using
    it over Bonferroni in a confirmatory family.
    """
    rejects = [entry["reject"] for entry in holm([0.01, 0.02, 0.03], 0.05)]
    assert rejects == [True, True, True]


def test_holm_borderline_pair():
    rejects = [entry["reject"] for entry in holm([0.01, 0.04], 0.05)]
    assert rejects == [True, True]


def test_holm_thresholds_are_step_down():
    entries = holm([0.001, 0.02, 0.04], 0.05)
    assert [entry["threshold"] for entry in entries] == pytest.approx(
        [0.05 / 3, 0.05 / 2, 0.05]
    )
