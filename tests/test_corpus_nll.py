"""Tests for the corpus NLL scorer (windowing only; no model, no GPU)."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from experiments.corpus.nll import score_document, token_windows


def test_short_document_is_one_window():
    ids = list(range(10))
    assert token_windows(ids, length=32, n_windows=4) == [ids]


def test_long_document_windows_cover_start_and_end():
    ids = list(range(1000))
    windows = token_windows(ids, length=100, n_windows=4)
    assert len(windows) == 4
    assert all(len(w) == 100 for w in windows)
    assert windows[0][0] == 0
    assert windows[-1][-1] == 999
    assert all(windows[i][-1] < windows[i + 1][0] + 1 for i in range(3))


def test_windows_do_not_repeat_start_positions():
    ids = list(range(12))
    windows = token_windows(ids, length=10, n_windows=8)
    assert len(windows) == 3
    assert len({w[0] for w in windows}) == 3
    assert all(len(w) == 10 for w in windows)


def test_windows_are_deterministic():
    ids = list(range(500))
    a = token_windows(ids, length=64, n_windows=3)
    b = token_windows(ids, length=64, n_windows=3)
    assert a == b


def test_score_document_is_token_weighted():
    ids = list(range(10))
    # One window of 10 ids -> 9 predicted tokens.
    assert score_document(ids, lambda w: 2.0, length=32) == pytest.approx(2.0)


def test_score_document_averages_windows_by_token_count():
    ids = list(range(10))
    calls: list[int] = []

    def loss(w):
        calls.append(len(w))
        return float(len(w))

    # length=5 gives two windows of 5 ids (4 predicted tokens each).
    value = score_document(ids, loss, length=5, n_windows=2)
    assert calls == [5, 5]
    assert value == pytest.approx(5.0)


def test_score_document_rejects_a_single_token():
    assert math.isnan(score_document([1], lambda w: 1.0, length=4))


def test_build_accepts_the_scorer_envelope(tmp_path):
    from experiments.corpus.build import build

    today = date.today().isoformat()
    rows = [
        {
            "doc_id": f"d{i}",
            "source": "biorxiv",
            "title": "t",
            "text": f"Running prose about cohort {i} and receptor {i}. " * 100,
            "sections": {
                "methods": f"Running prose about cohort {i} and receptor {i}. " * 100
            },
            "license": "cc-by",
            "date": today,
            "meta": {"fulltext": "ok"},
        }
        for i in range(2)
    ]
    docs = tmp_path / "docs.jsonl"
    docs.write_text("\n".join(json.dumps(r) for r in rows))
    nll = tmp_path / "nll.json"
    nll.write_text(json.dumps({"model": "m", "nll": {"d0": 2.0, "d1": 2.5}}))

    result = build(
        source="biorxiv",
        cutoff=(date.today() - timedelta(days=60)).isoformat(),
        out=tmp_path / "out",
        docs_jsonl=docs,
        nll_json=nll,
        dry_run=True,
    )
    assert result["card"]["documents_retained"] == 2


def test_build_names_the_missing_nll_document(tmp_path):
    from experiments.corpus.build import build

    rows = [
        {
            "doc_id": "missing",
            "source": "biorxiv",
            "title": "t",
            "text": "Running prose about a cohort. " * 100,
            "sections": {"methods": "Running prose about a cohort. " * 100},
            "license": "cc-by",
            "date": date.today().isoformat(),
            "meta": {"fulltext": "ok"},
        }
    ]
    docs = tmp_path / "docs.jsonl"
    docs.write_text("\n".join(json.dumps(r) for r in rows))
    nll = tmp_path / "nll.json"
    nll.write_text(json.dumps({"nll": {}}))
    with pytest.raises(KeyError, match="missing"):
        build(
            source="biorxiv",
            cutoff=(date.today() - timedelta(days=60)).isoformat(),
            out=tmp_path / "out",
            docs_jsonl=docs,
            nll_json=nll,
            dry_run=True,
        )
