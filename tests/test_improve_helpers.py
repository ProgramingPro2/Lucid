from __future__ import annotations

from conceptpoc.cli import _pick_winner, _spread_sample


def test_spread_sample_returns_all_when_n_exceeds_total():
    seq = list(range(5))
    assert _spread_sample(seq, 10) == seq
    assert _spread_sample(seq, 5) == seq


def test_spread_sample_non_positive_returns_all():
    seq = list(range(5))
    assert _spread_sample(seq, 0) == seq


def test_spread_sample_spreads_across_full_range():
    seq = list(range(100))
    out = _spread_sample(seq, 8)
    assert len(out) == 8
    # Endpoints are included and the picks are non-decreasing and unique.
    assert out[0] == 0
    assert out[-1] == 99
    assert out == sorted(set(out))
    # Spread (not a contiguous head slice like the old [:8]).
    assert out[-1] - out[0] == 99


def test_spread_sample_single():
    assert _spread_sample(list(range(10)), 1) == [0]


def test_pick_winner_prefers_raft_only_on_strict_gain():
    assert _pick_winner({"sft": 0.40, "raft": 0.45}) == "raft"
    assert _pick_winner({"sft": 0.45, "raft": 0.40}) == "sft"
    # Ties keep the established SFT model.
    assert _pick_winner({"sft": 0.42, "raft": 0.42}) == "sft"
