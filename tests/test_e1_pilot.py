"""Tests for the E1 pilot sample-size report.

The number printed next to a pilot SD is an estimate from a within-design
paired SD. It is not a run-level variance, and it is not ``required_n`` with
the default inflation of 1 and floor of 1 — that combination is what printed
``required n = 1`` for the 2026-11-20 pilot.
"""

from __future__ import annotations

import json

import pytest

from experiments.e1.pilot import summarise
from experiments.protocol.stats import required_n


def _write(tmp_path, arm: str, seed: int, gain: float) -> None:
    payload = {
        "seed": seed,
        "eval": {"context_gain_mean": gain, "nll_true": 2.0},
    }
    (tmp_path / f"{arm}_seed{seed}.json").write_text(json.dumps(payload))


def test_summarise_floors_a_small_within_design_sd_at_24(tmp_path):
    gains = {
        0: (1.0, 1.0),
        1: (1.0, 1.0),
        2: (1.0, 1.0),
        3: (1.001, 1.0),
    }
    for seed, (gain_a, gain_b) in gains.items():
        _write(tmp_path, "A", seed, gain_a)
        _write(tmp_path, "B", seed, gain_b)

    summary = summarise(
        tmp_path, {"A": [0, 1, 2, 3], "B": [0, 1, 2, 3]}, mpid=0.0112
    )
    row = summary["A_minus_B"]
    assert row["n_pairs"] == 4
    assert row["pilot_sd_inflation"] == pytest.approx(1.73, abs=0.01)
    assert "not a run-level variance" in row["variance_note"]
    # The uninflated, unfloored call under-sizes this SD. The report must not.
    assert required_n(row["sd"], 0.0112) == 1
    assert row["estimated_n_within_design"] == 24


def test_summarise_caps_a_large_within_design_sd_at_64(tmp_path):
    for seed in range(4):
        _write(tmp_path, "A", seed, 0.0 if seed % 2 == 0 else 1.0)
        _write(tmp_path, "B", seed, 1.0 if seed % 2 == 0 else 0.0)

    summary = summarise(
        tmp_path, {"A": [0, 1, 2, 3], "B": [0, 1, 2, 3]}, mpid=0.0112
    )
    assert summary["A_minus_B"]["estimated_n_within_design"] == 64
