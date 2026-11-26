"""G0 assay: the criteria have to be able to fail.

A gate that cannot fail is not a gate (FINDINGS F27, F38). Each of the four
protocol criteria has an input that FAILs it, and the null has the extra N2
failure of a point CI. The synthetic dry run is plumbing: a decorative
channel is a NO-GO, not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.e0.assay import build_dry_report, write_report
from experiments.e0.collect import collect_scores, text_prefix
from experiments.e0.criteria import (
    DOSE_DROP_TOLERANCE,
    NULL_HALF_WIDTH_MAX,
    AssayScores,
    diagnose_rehearsal,
    effect_detected,
    evaluate,
    half_width_at,
    n_for_half_width,
    rehearsal_arm_sd,
    sample_size_plan,
    thresholds,
)
from experiments.e0.dose import dose_latent, slerp, unit
from experiments.e0.items import (
    CorpusTooSmall,
    items_from_records,
    load_records,
    require_items,
    split_section,
)
from experiments.e0.synthetic import bow_embed, echoing_generate
from experiments.protocol.stats import FAIL, PASS, VOID, mean_ci, paired_summary


def _base_scores(**overrides) -> AssayScores:
    """A passing assay, so a test can break exactly one criterion."""
    rng = np.random.default_rng(0)
    n = 400
    noise = rng.normal(0.0, 0.01, n)
    null = rng.normal(0.0, 0.02, n)
    scores = dict(
        dose0=noise,
        dose50=0.08 + noise,
        dose100=0.16 + noise,
        text_true=0.30 + rng.normal(0.0, 0.01, n),
        text_rand=rng.normal(0.0, 0.01, n),
        zero=noise,
        null_a=null,
        null_b=rng.normal(0.0, 0.02, n),
    )
    scores.update(overrides)
    return AssayScores(**scores)


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["criteria"]}


# --------------------------------------------------------------------------
# Dose geometry
# --------------------------------------------------------------------------


def test_dose_endpoints_are_the_shuffled_and_true_drafts():
    shuffled = np.array([1.0, 0.2, -0.4])
    true = np.array([-0.3, 1.0, 0.5])
    assert np.allclose(dose_latent(shuffled, true, 0.0), unit(shuffled))
    assert np.allclose(dose_latent(shuffled, true, 1.0), unit(true))


def test_rehearsal_midpoint_call_matches_dose_half():
    """slerp(true, shuffled, 0.5) is the midpoint. Argument order did not move it."""
    shuffled = np.array([0.2, 0.9, 0.1, -0.4])
    true = np.array([0.8, -0.1, 0.4, 0.3])
    assert np.allclose(slerp(true, shuffled, 0.5), dose_latent(shuffled, true, 0.5))


def test_dose_cosine_to_true_is_monotone():
    rng = np.random.default_rng(4)
    for _ in range(8):
        shuffled = rng.normal(size=16)
        true = rng.normal(size=16)
        # Include an obtuse pair: the short arc still has to be monotone.
        if float(np.dot(unit(shuffled), unit(true))) > 0:
            true = -true
        doses = np.linspace(0.0, 1.0, 11)
        cosines = [
            float(dose_latent(shuffled, true, t) @ unit(true)) for t in doses
        ]
        assert cosines[0] < cosines[-1]
        assert all(b >= a - 1e-8 for a, b in zip(cosines, cosines[1:]))


def test_slerp_rejects_zero_antipodal_and_out_of_range():
    a = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="zero"):
        dose_latent(np.array([0.0, 0.0]), a, 0.5)
    with pytest.raises(ValueError, match="antipodal"):
        slerp(np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 0.5)
    with pytest.raises(ValueError, match="0, 1"):
        dose_latent(a, np.array([0.0, 1.0]), 1.5)


# --------------------------------------------------------------------------
# Each criterion can fail. A criterion that cannot fail is not a criterion.
# --------------------------------------------------------------------------


def test_null_fails_when_the_half_width_exceeds_0_015():
    diff = np.linspace(-1.0, 1.0, 12)
    report = evaluate(_base_scores(null_a=diff, null_b=np.zeros(12)))
    got = _by_name(report)["null_half_width"]
    assert got["verdict"] == FAIL
    assert got["half_width"] > NULL_HALF_WIDTH_MAX
    assert got["contains_zero"] is True
    assert report["verdict"] != "GO"


def test_null_fails_on_a_point_ci_even_though_the_width_is_inside_the_cap():
    """N2. A zero-variance pair has half-width 0, which a width-only gate would pass."""
    same = np.linspace(0.0, 0.0, 30)
    report = evaluate(_base_scores(null_a=same, null_b=same))
    got = _by_name(report)["null_half_width"]
    assert got["is_point"] is True
    assert got["half_width"] <= NULL_HALF_WIDTH_MAX
    assert got["contains_zero"] is True
    assert got["verdict"] == FAIL


def test_null_fails_when_the_ci_excludes_zero():
    rng = np.random.default_rng(1)
    report = evaluate(
        _base_scores(
            null_a=rng.normal(0.20, 0.01, 400),
            null_b=rng.normal(0.00, 0.01, 400),
        )
    )
    got = _by_name(report)["null_half_width"]
    assert got["contains_zero"] is False
    assert got["is_point"] is False
    assert got["verdict"] == FAIL


def test_null_passes_when_precise_zero_centered_and_not_a_point():
    rng = np.random.default_rng(2)
    report = evaluate(
        _base_scores(
            null_a=rng.normal(0.0, 0.02, 800),
            null_b=rng.normal(0.0, 0.02, 800),
        )
    )
    got = _by_name(report)["null_half_width"]
    assert got["verdict"] == PASS
    assert got["contains_zero"] is True
    assert got["is_point"] is False
    assert got["half_width"] <= NULL_HALF_WIDTH_MAX


def test_dose_monotone_fails_on_a_real_drop():
    n = 40
    report = evaluate(
        _base_scores(
            dose0=np.full(n, 0.0),
            dose50=np.full(n, 0.20),
            dose100=np.full(n, 0.0),
        )
    )
    got = _by_name(report)["dose_monotone"]
    assert got["worst_drop"] > DOSE_DROP_TOLERANCE
    assert got["verdict"] == FAIL


def test_dose_monotone_allows_a_drop_inside_the_null_tolerance():
    n = 40
    report = evaluate(
        _base_scores(
            dose0=np.full(n, 0.0),
            dose50=np.full(n, 0.10),
            dose100=np.full(n, 0.10 - 0.010),
        )
    )
    got = _by_name(report)["dose_monotone"]
    assert got["strict_monotone"] is False
    assert got["worst_drop"] <= DOSE_DROP_TOLERANCE
    assert got["verdict"] == PASS


def test_dose_monotone_fails_just_past_the_tolerance():
    n = 40
    report = evaluate(
        _base_scores(
            dose0=np.full(n, 0.0),
            dose50=np.full(n, 0.10),
            dose100=np.full(n, 0.10 - (DOSE_DROP_TOLERANCE + 0.001)),
        )
    )
    assert _by_name(report)["dose_monotone"]["verdict"] == FAIL


def test_dose_100_vs_0_fails_on_zero_and_on_a_negative_effect():
    n = 80
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, n)
    zero = evaluate(_base_scores(dose0=base, dose50=base, dose100=base))
    assert _by_name(zero)["dose_100_vs_0"]["verdict"] == FAIL
    negative = evaluate(
        _base_scores(dose0=base, dose50=base, dose100=base - 0.10)
    )
    assert _by_name(negative)["dose_100_vs_0"]["verdict"] == FAIL
    assert _by_name(negative)["dose_100_vs_0"]["mean"] < 0


def test_dose_100_vs_0_passes_on_a_known_effect():
    report = evaluate(_base_scores())
    got = _by_name(report)["dose_100_vs_0"]
    assert got["ci_low"] > got["threshold"]
    assert got["verdict"] == PASS


def test_text_prefix_fails_on_a_zero_effect_and_passes_on_a_known_effect():
    n = 80
    rng = np.random.default_rng(5)
    same = rng.normal(0.05, 0.01, n)
    blind = evaluate(_base_scores(text_true=same, text_rand=same + rng.normal(0, 0.01, n)))
    assert _by_name(blind)["text_prefix"]["verdict"] == FAIL
    seen = evaluate(_base_scores())
    assert _by_name(seen)["text_prefix"]["verdict"] == PASS


def test_non_finite_score_is_void_and_does_not_pass():
    broken = np.ones(400)
    broken[3] = np.inf
    report = evaluate(_base_scores(dose100=broken))
    assert _by_name(report)["dose_100_vs_0"]["verdict"] == VOID
    assert report["verdict"] == VOID
    assert report["verdict"] != PASS
    nan = np.ones(400)
    nan[0] = np.nan
    report = evaluate(_base_scores(text_true=nan))
    assert _by_name(report)["text_prefix"]["verdict"] == VOID
    assert report["verdict"] == VOID


def test_threshold_at_or_below_the_null_cannot_pass():
    """F38, via absolute_gate_is_valid. A bar the null already clears is VOID."""
    rng = np.random.default_rng(6)
    n = 400
    # Null-pair mean sits above 0.03, and the dose contrast would otherwise pass.
    report = evaluate(
        _base_scores(
            null_a=rng.normal(0.10, 0.005, n),
            null_b=rng.normal(0.00, 0.005, n),
        )
    )
    dose = _by_name(report)["dose_100_vs_0"]
    text = _by_name(report)["text_prefix"]
    assert dose["verdict"] == VOID
    assert text["verdict"] == VOID
    assert "null" in dose["reason"]
    assert report["verdict"] == VOID


def test_zero_effect_ci_contains_zero_and_is_not_detected():
    rng = np.random.default_rng(7)
    control = rng.normal(0.0, 0.05, 300)
    delta = rng.normal(0.0, 0.02, 300)
    delta -= delta.mean()
    treatment = control + delta
    summary = paired_summary(treatment, control)
    assert summary.mean == pytest.approx(0.0, abs=1e-12)
    assert summary.sd > 0
    assert summary.ci_low < 0.0 < summary.ci_high
    assert effect_detected(treatment, control) == FAIL


def test_known_effect_is_detected():
    rng = np.random.default_rng(8)
    control = rng.normal(0.0, 0.05, 300)
    treatment = control + 0.25 + rng.normal(0.0, 0.01, 300)
    summary = paired_summary(treatment, control)
    assert summary.ci_low > 0.03
    assert effect_detected(treatment, control) == PASS


def test_four_gates_can_pass_while_the_zero_arm_wins():
    """The protocol hole F42 fell through. This is a GO with zero beating true."""
    rng = np.random.default_rng(9)
    n = 400
    noise = rng.normal(0.0, 0.01, n)
    report = evaluate(
        _base_scores(
            dose0=noise,
            dose50=0.08 + noise,
            dose100=0.16 + noise,
            zero=0.40 + noise,
        )
    )
    assert report["verdict"] == "GO"
    assert report["format_control"]["zero_beats_true"] is True
    assert report["format_control"]["true_beats_zero_lb"] is False
    assert "Not a G0 gate" in report["format_control"]["note"]


def test_report_thresholds_are_the_protocol_constants():
    report = evaluate(_base_scores())
    assert report["thresholds"] == thresholds()
    assert report["thresholds"]["null_half_width_max"] == 0.015
    assert report["thresholds"]["dose_100_vs_0_lb_min"] == 0.03
    assert report["thresholds"]["text_prefix_lb_min"] == 0.03
    assert report["thresholds"]["dose_drop_tolerance"] == report["thresholds"]["null_half_width_max"]
    assert [c["name"] for c in report["criteria"]] == [
        "null_half_width",
        "dose_monotone",
        "dose_100_vs_0",
        "text_prefix",
    ]


# --------------------------------------------------------------------------
# Rehearsal diagnosis and the sample-size plan
# --------------------------------------------------------------------------


def test_rehearsal_dip_is_absolute_cosine_noise_and_the_margin_is_monotone():
    diag = diagnose_rehearsal()
    assert diag["cause"] == "underpowered_absolute_cosine"
    assert diag["margin_monotone"] is True
    assert diag["cos_steps"][1] < 0
    assert diag["cos_dip"] == pytest.approx(0.0036, abs=1e-6)
    assert diag["dip_within_dose_tolerance"] is True
    assert diag["dip_over_se"] < 0.2
    assert diag["n_to_detect_dip_as_an_effect"] > 10_000
    assert diag["zero_minus_true_cos"] > 0.05
    assert diag["zero_minus_true_margin"] > 0


def test_rehearsal_constants_match_the_artifact_when_present():
    path = Path("artifacts/e0/rehearsal.json")
    if not path.exists():
        pytest.skip("rehearsal artifact is local and gitignored")
    payload = json.loads(path.read_text())
    diag = diagnose_rehearsal()
    assert payload["dose_ladder_cos_true"] == diag["cos_ladder"]
    assert payload["n_items"] == diag["n"]
    assert payload["dose_monotone"] is False


def test_required_n_is_the_mean_ci_half_width_and_the_corpus_plan_is_larger_than_14():
    sd = rehearsal_arm_sd()
    # Recovered SD, pushed back through mean_ci, reproduces the published width.
    hw = half_width_at(sd, 14)
    published = (0.1181 - (-0.0983)) / 2
    assert hw == pytest.approx(published, rel=1e-6)
    plan = sample_size_plan()
    n = plan["required_pairs"]
    assert half_width_at(plan["pair_sd"], n) <= NULL_HALF_WIDTH_MAX
    assert half_width_at(plan["pair_sd"], n - 1) > NULL_HALF_WIDTH_MAX
    assert plan["required_documents"] == plan["required_pairs"] * 2
    assert plan["required_documents"] > 14
    assert plan["single_arm_items"] > 14
    assert plan["half_width_at_protocol_n_256_pair"] > NULL_HALF_WIDTH_MAX
    # The search really called mean_ci: a hand-rolled normal interval would
    # disagree at small n, and this pins the t interval.
    summary = mean_ci(np.linspace(-sd, sd, plan["single_arm_items"]))
    assert summary.n == plan["single_arm_items"]


def test_n_for_half_width_matches_mean_ci_on_a_small_sd():
    n = n_for_half_width(0.05, 0.015)
    assert half_width_at(0.05, n) <= 0.015
    assert half_width_at(0.05, n - 1) > 0.015


# --------------------------------------------------------------------------
# Corpus items
# --------------------------------------------------------------------------


def _record(doc_id: str, *, license_name="cc-by", fulltext="ok", date="2026-02-01", words=700):
    body = " ".join(f"{doc_id[-4:]}w{k}" for k in range(words))
    return {
        "doc_id": doc_id,
        "source": "biorxiv",
        "title": doc_id,
        "text": body,
        "sections": {"results": body},
        "license": license_name,
        "date": date,
        "meta": {"fulltext": fulltext},
    }


def test_items_are_document_disjoint_and_the_target_is_held_out():
    records = [_record(f"biorxiv:10.1/{i:04d}") for i in range(12)]
    items, census = items_from_records(records, null_seed=0)
    assert census["n_items"] == 12
    assert len({item.doc_id for item in items}) == 12
    ids = [item.doc_id for item in items]
    for item in items:
        assert item.partner_index != ids.index(item.doc_id)
        partner = items[item.partner_index]
        assert partner.doc_id != item.doc_id
        assert item.prompt not in item.target
        assert item.target not in item.prompt
        assert split_section(item.prompt + " " + item.target) is not None
    splits = {item.null_split for item in items}
    assert splits <= {"a", "b"}
    assert sum(item.null_split == "a" for item in items) == 6
    assert sum(item.null_split == "b" for item in items) == 6


def test_items_drop_non_ok_fulltext_bad_licence_early_dates_and_duplicates():
    records = [
        _record("biorxiv:10.1/0001"),
        _record("biorxiv:10.1/0002", fulltext="abstract_only"),
        _record("biorxiv:10.1/0003", license_name="cc-by-nc-4.0"),
        _record("biorxiv:10.1/0004", date="2025-06-15"),
        _record("biorxiv:10.1/0001"),
        _record("biorxiv:10.1/0005", words=20),
    ]
    items, census = items_from_records(records)
    assert [item.doc_id for item in items] == ["biorxiv:10.1/0001"]
    assert census["drops"]["not_fulltext"] == 1
    assert census["drops"]["licence"] == 1
    assert census["drops"]["date"] == 1
    assert census["drops"]["duplicate"] == 1
    assert census["drops"]["short"] == 1


def test_load_records_skips_a_truncated_tail_and_raises_on_a_torn_middle(tmp_path: Path):
    good = json.dumps(_record("biorxiv:10.1/0001"))
    tail = tmp_path / "tail.jsonl"
    tail.write_text(good + "\n" + "{\"doc_id\": \"torn")
    rows, skipped = load_records(tail)
    assert skipped == 1
    assert len(rows) == 1

    middle = tmp_path / "middle.jsonl"
    middle.write_text(good + "\n" + "{\"torn\": \n" + good + "\n")
    with pytest.raises(json.JSONDecodeError):
        load_records(middle)


def test_corpus_too_small_fails_loudly_and_fourteen_items_are_not_enough():
    plan = sample_size_plan()
    with pytest.raises(CorpusTooSmall) as exc:
        require_items(14, plan["required_documents"], measure_anyway=False)
    assert exc.value.have == 14
    assert exc.value.need == plan["required_documents"]
    require_items(14, plan["required_documents"], measure_anyway=True)
    require_items(plan["required_documents"], plan["required_documents"], measure_anyway=False)


# --------------------------------------------------------------------------
# End-to-end CPU path
# --------------------------------------------------------------------------


def test_collect_uses_dose_geometry_and_a_decorative_channel_is_flat():
    records = [_record(f"biorxiv:10.2/{i:04d}") for i in range(8)]
    items, _ = items_from_records(records, null_seed=1)
    scores = collect_scores(
        items,
        echoing_generate,
        lambda text: bow_embed(text, dim=16),
        dose_seeds=(3,),
        zero_seed=4,
        text_seed=5,
        null_seed_a=6,
        null_seed_b=7,
    )
    # The echo ignores the latent, so the three doses are the same strings.
    assert np.allclose(scores.dose0, scores.dose100)
    assert np.allclose(scores.dose50, scores.dose100)
    # The readout still separates a true prefix from a random document.
    assert float(scores.text_true.mean()) > float(scores.text_rand.mean()) + 0.2
    # Null pairs are disjoint items, so the contrast is not a copy.
    assert scores.null_a.shape == scores.null_b.shape
    assert not np.allclose(scores.null_a, scores.null_b)


def test_cpu_dry_run_evaluates_four_criteria_and_does_not_go():
    # Seed 2 keeps the synthetic null-pair mean under 0.03, so the dose gate
    # FAILs. Seed 0's null mean sits above 0.03 and the F38 guard VOIDs it.
    report = build_dry_report(n_items=48, dim=32, null_seed=2)
    assert report["mode"] == "cpu-dry-run"
    assert report["not_g0"] is True
    assert report["synthetic_channel"] == "decorative"
    assert report["verdict"] == "NO-GO"
    named = _by_name(report)
    assert set(named) == {"null_half_width", "dose_monotone", "dose_100_vs_0", "text_prefix"}
    # Decorative latent: the dose contrast is a point at zero and must fail.
    assert named["dose_100_vs_0"]["verdict"] == FAIL
    assert named["dose_100_vs_0"]["mean"] == pytest.approx(0.0, abs=1e-9)
    # Readout control can pass while the channel fails. That is F42's shape.
    assert named["text_prefix"]["verdict"] == PASS
    # n=24 pairs cannot buy a 0.015 half-width out of item noise.
    assert named["null_half_width"]["verdict"] == FAIL
    assert named["null_half_width"]["is_point"] is False
    assert named["null_half_width"]["contains_zero"] is True
    assert report["planning_n_met"] is False
    assert report["item_census"]["n_items"] == 48
    assert len(report["criteria"]) == 4


def test_noisy_null_mean_above_the_threshold_voids_rather_than_passing():
    """Seed 0 is the dry-run draw whose null mean already clears 0.03."""
    report = build_dry_report(n_items=48, dim=32, null_seed=0)
    assert report["verdict"] == VOID
    dose = _by_name(report)["dose_100_vs_0"]
    assert dose["verdict"] == VOID
    assert dose["null_mean"] > 0.03


def test_dry_run_report_json_has_finite_verdicts(tmp_path: Path):
    report = build_dry_report(n_items=16, dim=16, null_seed=2)
    out = tmp_path / "g0_dry_run.json"
    write_report(report, out)
    loaded = json.loads(out.read_text())
    assert loaded["verdict"] == "NO-GO"
    assert loaded["thresholds"]["null_half_width_max"] == 0.015
    names = [c["name"] for c in loaded["criteria"]]
    assert names == ["null_half_width", "dose_monotone", "dose_100_vs_0", "text_prefix"]
    # allow_nan=False is the F27 guard on the file itself.
    dirty = {"value": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        write_report(dirty, tmp_path / "bad.json")


def test_text_prefix_helper_cuts_on_a_word():
    text = "alpha beta gamma delta " * 30
    got = text_prefix(text, 0.7)
    assert " " not in got[-1:] or got.endswith("delta")
    assert not got.endswith(" ")
    assert len(got) < len(text.strip())
