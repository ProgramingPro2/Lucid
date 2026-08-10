"""Pure-math unit tests for Stage-B reward pieces (Fable §6.4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.pivot.reward_stage_b import (  # noqa: E402
    digit_to_unit,
    expected_digit_from_logits,
    f_ramp,
    group_advantages,
    hard_collapse_gate,
    longest_verbatim_loop,
    max_ngram_count,
    distinct_n,
    tokenize_words,
)


def test_f_ramp_bounds():
    assert f_ramp(0.0) == 0.0
    assert f_ramp(0.35) == 0.0
    assert abs(f_ramp(0.485) - 0.5) < 1e-6
    assert f_ramp(0.62) == 1.0
    assert f_ramp(0.90) == 1.0


def test_group_advantages_zero_mean():
    a = group_advantages([0.1, 0.2, 0.3, 0.4])
    assert abs(sum(a)) < 1e-6
    assert abs(a[0] + a[3]) < 1e-6


def test_hard_gate_short():
    g, flags, _ = hard_collapse_gate("too short")
    assert g == 0
    assert flags.short


def test_hard_gate_repetition():
    toks = (["the", "cat", "sat", "mat"] * 20)
    text = " ".join(toks)
    g, flags, stats = hard_collapse_gate(text)
    assert g == 0
    assert flags.fourgram or flags.distinct2 or flags.loop


def test_hard_gate_clean_paragraph():
    text = (
        "The floury baker is a medium-sized cicada found along the coast. "
        "Individuals vary in size depending on local rainfall patterns. "
        "Coastal regions with higher rainfall tend to produce larger adults. "
        "Naturalists described the species in the nineteenth century from museum specimens. "
        "Its wingspan approaches ten centimeters in the wettest habitats."
    )
    g, flags, stats = hard_collapse_gate(text)
    assert g == 1
    assert not flags.tripped
    assert stats["n_tokens"] >= 40


def test_distinct2_and_loop_helpers():
    toks = tokenize_words("a b a b a b a b")
    assert distinct_n(toks, 2) <= 0.5
    loop = longest_verbatim_loop(["x"] * 40)
    assert loop >= 30
    assert max_ngram_count(["a", "b", "c", "d"] * 5, 4) >= 4


def test_logit_digit_expectation():
    # Fake vocab logits: make token ids 10..14 correspond to digits 1..5
    logits = torch.full((100,), -100.0)
    # Prefer digit 4 (id 13)
    logits[13] = 10.0
    e = expected_digit_from_logits(logits, [10, 11, 12, 13, 14])
    assert abs(e - 4.0) < 0.05
    assert abs(digit_to_unit(e) - 0.75) < 0.02


def test_list_gate():
    text = "- one item here\n- two item here\n- three item here\n" + ("word " * 50)
    g, flags, _ = hard_collapse_gate(text)
    assert g == 0
    assert flags.list_fmt
