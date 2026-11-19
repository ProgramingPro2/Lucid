"""Tests for the E2 CoT budget-curve aggregation (no model, no GPU)."""

from __future__ import annotations

from experiments.e2.budget_curve import summarise_budgets


def row(mode: str, correct: bool, *, budget: int | None = None, chars: int = 10):
    r = {
        "family": "f1",
        "mode": mode,
        "correct": correct,
        "has_marker": True,
        "chars": chars,
    }
    if budget is not None:
        r["budget"] = budget
    return r


def test_curve_reports_each_budget_against_shared_direct_arm():
    rows = [row("direct", i == 0) for i in range(4)]
    rows += [row("cot", i < 2, budget=128) for i in range(4)]
    rows += [row("cot", True, budget=512) for i in range(4)]

    out = summarise_budgets(rows, [128, 512])["f1"]
    assert out["direct"]["accuracy"] == 0.25
    assert out["cot"]["128"]["accuracy"] == 0.5
    assert out["cot"]["128"]["cot_minus_direct"]["mean"] == 0.25
    assert out["cot"]["512"]["accuracy"] == 1.0
    assert out["cot"]["512"]["cot_minus_direct"]["mean"] == 0.75
    assert out["n"] == 4


def test_curve_omits_budgets_with_no_rows():
    rows = [row("direct", True) for _ in range(2)]
    rows += [row("cot", True, budget=256) for _ in range(2)]
    out = summarise_budgets(rows, [128, 256])["f1"]
    assert "128" not in out["cot"]
    assert "256" in out["cot"]


def test_curve_keeps_families_separate():
    rows = [row("direct", True) for _ in range(2)]
    rows += [row("cot", True, budget=128) for _ in range(2)]
    rows.append(
        {
            "family": "f3",
            "mode": "direct",
            "correct": False,
            "has_marker": True,
            "chars": 10,
        }
    )
    rows.append(
        {
            "family": "f3",
            "mode": "cot",
            "correct": True,
            "has_marker": True,
            "chars": 10,
            "budget": 128,
        }
    )
    out = summarise_budgets(rows, [128])
    assert out["f1"]["direct"]["accuracy"] == 1.0
    assert out["f3"]["direct"]["accuracy"] == 0.0
    assert out["f3"]["cot"]["128"]["cot_minus_direct"]["mean"] == 1.0
