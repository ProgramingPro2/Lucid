"""Tests for the E2 capability-screen aggregation (no model, no GPU)."""

from __future__ import annotations

from experiments.e2.evaluate import summarise


def row(family: str, mode: str, correct: bool, marker: bool = True) -> dict:
    return {
        "id": "x",
        "family": family,
        "mode": mode,
        "gold": "g",
        "correct": correct,
        "has_marker": marker,
        "chars": 10 if marker else 300,
        "text": "",
    }


def test_summarise_detects_large_cot_gain():
    rows = [row("f1", "direct", i == 0) for i in range(4)]
    rows += [row("f1", "cot", True) for _ in range(4)]
    s = summarise(rows)
    f1 = s["by_family"]["f1"]
    assert f1["direct"]["accuracy"] == 0.25
    assert f1["cot"]["accuracy"] == 1.0
    assert f1["cot_minus_direct"]["mean"] == 0.75
    assert s["overall"]["capability_gate"] == "PASS"


def test_summarise_flags_void_when_effect_is_small():
    rows = [row("f2", "direct", i == 0) for i in range(4)]
    rows += [row("f2", "cot", i == 0) for i in range(4)]
    s = summarise(rows)
    assert s["by_family"]["f2"]["cot_minus_direct"]["mean"] == 0.0
    assert s["overall"]["capability_gate"] == "VOID"


def test_summarise_counts_marker_rate():
    rows = [row("f3", "direct", True, marker=True) for _ in range(2)]
    rows += [row("f3", "direct", False, marker=False) for _ in range(2)]
    rows += [row("f3", "cot", False, marker=False) for _ in range(4)]
    s = summarise(rows)
    assert s["by_family"]["f3"]["direct"]["marker_rate"] == 0.5
    assert s["by_family"]["f3"]["cot"]["marker_rate"] == 0.0


def test_summarise_overall_is_mean_of_families_not_items():
    # F1 has a full gain, F3 has none. The overall accuracy mean is across all
    # item-mode observations, but the gate uses the paired difference.
    rows = [row("f1", "direct", False) for _ in range(2)]
    rows += [row("f1", "cot", True) for _ in range(2)]
    rows += [row("f3", "direct", True) for _ in range(2)]
    rows += [row("f3", "cot", True) for _ in range(2)]
    s = summarise(rows)
    assert s["overall"]["n_items"] == 4
    assert abs(s["overall"]["cot_minus_direct"]["mean"] - 0.5) < 1e-9
