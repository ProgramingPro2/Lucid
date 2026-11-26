"""G-1d disjointness and G-1e reproducibility.

The growth test is the one the shuffle failed: adding a document, including
one whose id sorts before the others, must not move an existing assignment.
The hash tests build from the same inputs twice, change one token, and
reorder rows.
"""

from __future__ import annotations

import json

import pytest

from experiments.corpus.build import build
from experiments.corpus.sources import Document
from experiments.corpus.split import (
    SPLIT_MODULUS,
    _fraction_cuts,
    split_documents,
    token_stream_sha256,
    verify_clean_rebuild,
    verify_document_disjoint,
)
from experiments.corpus.split import main as split_main

_STRATA = ("abstract", "introduction", "methods", "results", "discussion")


def _doc(doc_id: str, text: str = "alpha beta gamma", stratum: str = "methods") -> Document:
    return Document(
        doc_id=doc_id,
        source="biorxiv",
        title="t",
        text=text,
        sections={"methods": text},
        license="cc-by",
        date="2025-07-15",
        meta={"stratum": stratum, "fulltext": "ok"},
    )


def _ids(n: int) -> list[Document]:
    return [
        _doc(f"d{i:04d}", text=f"unique prose {i} about a receptor", stratum=_STRATA[i % 5])
        for i in range(n)
    ]


def _row(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "source": "biorxiv",
        "title": "t",
        "text": text,
        "sections": {"methods": text},
        "license": "cc-by",
        "date": "2025-07-15",
        "meta": {"stratum": "methods", "fulltext": "ok"},
    }


def _write(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tok(text: str) -> list[int]:
    return [len(word) for word in text.split()]


_ROWS = [
    _row("b", "methods prose about a receptor pathway in a cohort"),
    _row("a", "introduction prose about transcription and a membrane"),
    _row("d", "results prose about a randomised dose and an endpoint"),
    _row("c", "discussion prose about kinetics and a tissue culture"),
]


# --------------------------------------------------------------------------
# G-1d
# --------------------------------------------------------------------------


def test_default_fraction_cuts_are_exact_integers():
    assert SPLIT_MODULUS == 10_000_000
    assert _fraction_cuts((0.8, 0.1, 0.1)) == (8_000_000, 9_000_000)


def test_assignment_stable_when_corpus_grows():
    """Adding a document, even one that sorts first, does not move the others."""
    docs = _ids(40)
    before = split_documents(docs, seed=0)
    added = _doc("aaa-sorts-first", text="a brand new document", stratum="results")
    after = split_documents([added] + list(reversed(docs)), seed=0)
    for name, ids in before.items():
        for doc_id in ids:
            assert doc_id in after[name]
    homes = [name for name, ids in after.items() if "aaa-sorts-first" in ids]
    assert len(homes) == 1
    verify_document_disjoint(after)


def test_assignment_independent_of_input_order():
    docs = _ids(30)
    assert split_documents(docs, seed=5) == split_documents(list(reversed(docs)), seed=5)


def test_disjoint_with_duplicate_looking_ids():
    """Prefix-like ids are distinct documents and each lands in one split."""
    ids = [
        "doc",
        "doc:1",
        "doc:10",
        "doc:1:0",
        "10.1101/2025.01.01",
        "10.1101/2025.01.011",
        "10.1101/2025.01.01.1",
        "prefix",
        "prefix-extra",
    ]
    docs = [
        _doc(doc_id, text=f"body {i} differs", stratum=_STRATA[i % 5])
        for i, doc_id in enumerate(ids)
    ]
    splits = split_documents(docs, seed=11)
    verify_document_disjoint(splits)
    flat = [doc_id for name in ("train", "dev", "test") for doc_id in splits[name]]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat))
    # A prefix check would have treated some of these as the same document.
    assert "doc" in flat and "doc:1" in flat and "doc:10" in flat


def test_verify_raises_when_an_id_is_repeated():
    with pytest.raises(ValueError, match="appears twice"):
        verify_document_disjoint({"train": ["doc", "doc"], "dev": [], "test": []})
    with pytest.raises(ValueError, match="both"):
        verify_document_disjoint(
            {"train": ["doc:1"], "dev": ["doc:10"], "test": ["doc:1"]}
        )
    with pytest.raises(ValueError, match="appears twice"):
        split_documents([_doc("same", text="one"), _doc("same", text="two")])


# --------------------------------------------------------------------------
# G-1e
# --------------------------------------------------------------------------


def test_hash_builds_twice_from_the_same_inputs(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write(path, _ROWS)
    first = build(
        source="biorxiv",
        cutoff="2025-04-29",
        out=tmp_path / "a",
        docs_jsonl=path,
        seed=0,
    )
    second = build(
        source="biorxiv",
        cutoff="2025-04-29",
        out=tmp_path / "b",
        docs_jsonl=path,
        seed=0,
    )
    assert first["card"]["token_stream_sha256"] == second["card"]["token_stream_sha256"]
    assert first["card"]["documents_retained"] == 4
    assert first["card"]["split_counts"] == second["card"]["split_counts"]


def test_hash_changes_when_one_token_changes():
    docs = [_doc("a", "one two three"), _doc("b", "four five")]
    original = token_stream_sha256(docs, _tok)

    def flipped(text: str) -> list[int]:
        ids = _tok(text)
        if text == "one two three":
            ids[-1] = ids[-1] + 1
        return ids

    assert token_stream_sha256(docs, flipped) != original
    # Same tokens under another id are a different stream: the boundary is hashed.
    moved = [_doc("a", "one two three"), _doc("c", "four five")]
    assert token_stream_sha256(moved, _tok) != original


def test_hash_ignores_input_row_order(tmp_path):
    forward = tmp_path / "forward.jsonl"
    backward = tmp_path / "backward.jsonl"
    _write(forward, _ROWS)
    _write(backward, list(reversed(_ROWS)))
    kwargs = {"source": "biorxiv", "cutoff": "2025-04-29", "seed": 0}
    a = build(out=tmp_path / "a", docs_jsonl=forward, **kwargs)
    b = build(out=tmp_path / "b", docs_jsonl=backward, **kwargs)
    assert a["card"]["token_stream_sha256"] == b["card"]["token_stream_sha256"]
    assert a["card"]["split_counts"] == b["card"]["split_counts"]

    docs = [_doc("b", "zz"), _doc("a", "yy")]
    assert token_stream_sha256(docs, _tok) == token_stream_sha256(list(reversed(docs)), _tok)


def test_hash_ignores_section_dict_order_and_rejects_floats():
    left = _doc("a", "same body")
    right = _doc("a", "same body")
    left.sections = {"results": "r", "abstract": "a"}
    right.sections = {"abstract": "a", "results": "r"}
    assert token_stream_sha256([left], _tok) == token_stream_sha256([right], _tok)
    with pytest.raises(TypeError, match="non-bool int"):
        token_stream_sha256([left], lambda _text: [1.0])


def test_bounded_rebuild_matches_and_states_its_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write(path, _ROWS)
    report = verify_clean_rebuild(
        path,
        tmp_path / "out-a",
        tmp_path / "out-b",
        cutoff="2025-04-29",
        limit=2,
        seed=0,
    )
    assert report["equal"] is True
    assert report["hash_a"] == report["hash_b"]
    assert report["documents_harvested_a"] == 2
    assert "full retained corpus" in report["does_not_prove"]
    assert "historical card" in report["does_not_prove"]
    code = split_main(
        [
            "verify-rebuild",
            "--jsonl",
            str(path),
            "--cutoff",
            "2025-04-29",
            "--limit",
            "2",
            "--out-a",
            str(tmp_path / "cli-a"),
            "--out-b",
            str(tmp_path / "cli-b"),
        ]
    )
    assert code == 0
