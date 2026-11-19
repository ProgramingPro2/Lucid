"""Tests for the corpus pipeline (docs/CORPUS.md §4, §5).

Every test uses ``FakeClient`` or hand-built documents. No network.
"""

from __future__ import annotations

import json
import re

import pytest

from experiments.corpus.build import build
from experiments.corpus.pipeline import (
    CARD_KEYS,
    build_card,
    clean_text,
    deduplicate,
    drop_low_nll,
    filter_by_date,
    filter_by_license,
    split_documents,
    token_stream_sha256,
)
from experiments.corpus.sources import Document, normalise_license


def doc(
    doc_id: str,
    *,
    date: str = "2025-06-01",
    license: str = "CC BY 4.0",
    text: str = "Some running prose about a study.",
    sections: dict | None = None,
    stratum: str = "methods",
) -> Document:
    return Document(
        doc_id=doc_id,
        source="fake",
        title="t",
        text=text,
        sections=sections if sections is not None else {"methods": text},
        license=license,
        date=date,
        meta={"stratum": stratum},
    )


# --------------------------------------------------------------------------
# date
# --------------------------------------------------------------------------


def test_filter_by_date_excludes_boundary_exactly():
    cutoff = "2025-01-01"
    docs = [
        doc("a", date="2025-01-30"),
        doc("b", date="2025-01-31"),  # cutoff + 30 days: excluded
        doc("c", date="2025-02-01"),
    ]
    kept = filter_by_date(docs, cutoff, grace_days=30)
    assert [d.doc_id for d in kept] == ["c"]


def test_filter_by_date_grace_is_configurable():
    docs = [doc("a", date="2025-01-20"), doc("b", date="2025-01-25")]
    assert [d.doc_id for d in filter_by_date(docs, "2025-01-01", grace_days=10)] == [
        "a",
        "b",
    ]
    assert [d.doc_id for d in filter_by_date(docs, "2025-01-01", grace_days=20)] == [
        "b",
    ]


# --------------------------------------------------------------------------
# licence
# --------------------------------------------------------------------------


def test_licence_filter_is_per_item():
    docs = [
        doc("keep", license="CC BY 4.0"),
        doc("drop", license="CC BY-NC 4.0"),
        doc("also_keep", license="CC0"),
    ]
    kept, dropped = filter_by_license(docs, {"cc-by", "cc-by-4.0", "cc0", "public-domain"})
    assert {d.doc_id for d in kept} == {"keep", "also_keep"}
    assert [d[0] for d in dropped] == ["drop"]
    assert "licence:cc-by-nc-4.0" in dropped[0][1]


def test_normalise_license_variants():
    assert normalise_license("CC BY 4.0") == "cc-by-4.0"
    assert normalise_license("cc-by") == "cc-by"
    assert normalise_license("CC0 1.0") == "cc0"
    assert normalise_license("public domain") == "public-domain"
    # NC/ND are classified so the card can count them, but they are not allowed.
    assert normalise_license("CC BY-NC-ND 4.0") == "cc-by-nc-nd-4.0"
    assert normalise_license("CC BY-NC-ND 4.0") not in {"cc-by", "cc-by-4.0", "cc0"}
    assert normalise_license(None) == "unknown"


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


def test_clean_text_removes_backmatter_and_captions():
    raw = (
        "Methods\nWe measured systolic pressure in 412 participants.\n\n"
        "Figure 1. A caption that must not survive.\n\n"
        "References\n1. Smith et al. 2020. A paper.\n2. Jones et al. 2021."
    )
    out = clean_text(doc("a", text=raw, sections={"methods": raw}))
    assert "412 participants" in out.text
    assert "Figure 1" not in out.text
    assert "Smith et al" not in out.text
    assert out.sections["methods"].startswith("Methods")


def test_clean_text_drops_acknowledgements():
    raw = "Results\nWe found a thing.\n\nAcknowledgements\nWe thank the funders."
    out = clean_text(doc("a", text=raw))
    assert "found a thing" in out.text
    assert "funders" not in out.text


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


def test_deduplicate_keeps_earliest_of_a_near_identical_pair():
    body = (
        "The study recruited four hundred participants and measured blood "
        "pressure over twelve weeks using a mixed effects model."
    )
    early = doc("early", date="2025-03-01", text=body)
    late = doc("late", date="2025-04-01", text=body)
    different = doc(
        "different",
        date="2025-05-01",
        text="A completely unrelated report on soil carbon and fungal networks.",
    )
    kept = deduplicate([late, different, early], threshold=0.95)
    ids = {d.doc_id for d in kept}
    assert "early" in ids
    assert "late" not in ids
    assert "different" in ids


def test_deduplicate_is_order_independent():
    body = "Repeated text about a kinase assay with six replicates and controls."
    a = doc("a", date="2025-03-01", text=body)
    b = doc("b", date="2025-04-01", text=body)
    assert [d.doc_id for d in deduplicate([a, b])] == [
        d.doc_id for d in deduplicate([b, a])
    ]


# --------------------------------------------------------------------------
# NLL screen
# --------------------------------------------------------------------------


def test_drop_low_nll_removes_bottom_decile():
    docs = [doc(f"d{i:03d}") for i in range(100)]
    nll = {d.doc_id: i / 100.0 for i, d in enumerate(docs)}
    kept, dropped = drop_low_nll(docs, lambda d: nll[d.doc_id], quantile=0.10)
    assert len(dropped) == 10
    assert len(kept) == 90
    assert max(nll[d.doc_id] for d in dropped) < min(nll[d.doc_id] for d in kept)


def test_drop_low_nll_is_deterministic_on_ties():
    docs = [doc(f"d{i}") for i in range(20)]
    kept1, dropped1 = drop_low_nll(docs, lambda d: 0.0, quantile=0.10)
    kept2, dropped2 = drop_low_nll(docs, lambda d: 0.0, quantile=0.10)
    assert [d.doc_id for d in dropped1] == [d.doc_id for d in dropped2]


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def _many_docs(n: int = 1000) -> list[Document]:
    strata = ["abstract", "introduction", "methods", "results", "discussion"]
    return [
        doc(f"d{i:04d}", stratum=strata[i % len(strata)], date="2025-06-01")
        for i in range(n)
    ]


def test_split_proportions_within_two_percent():
    docs = _many_docs(1000)
    splits = split_documents(docs, seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == 1000
    assert abs(len(splits["train"]) / total - 0.8) <= 0.02
    assert abs(len(splits["dev"]) / total - 0.1) <= 0.02
    assert abs(len(splits["test"]) / total - 0.1) <= 0.02


def test_split_has_no_shared_documents():
    docs = _many_docs(300)
    splits = split_documents(docs, seed=7)
    seen: set[str] = set()
    for ids in splits.values():
        assert not (seen & set(ids))
        seen |= set(ids)
    assert seen == {d.doc_id for d in docs}


def test_split_is_deterministic():
    docs = _many_docs(200)
    a = split_documents(docs, seed=3)
    b = split_documents(docs, seed=3)
    assert a == b
    c = split_documents(docs, seed=4)
    assert a != c


def test_split_is_stratified():
    docs = _many_docs(1000)
    splits = split_documents(docs, seed=0)
    by_id = {d.doc_id: d.meta["stratum"] for d in docs}
    for name, ids in splits.items():
        counts = {s: sum(by_id[i] == s for i in ids) for s in set(by_id.values())}
        # Every stratum should appear in every split.
        assert all(c > 0 for c in counts.values()), (name, counts)


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def test_token_stream_hash_is_stable_and_sensitive():
    def tok(text: str) -> list[int]:
        return [len(w) for w in text.split()]

    docs = [doc("a", text="one two three"), doc("b", text="four five")]
    h1 = token_stream_sha256(docs, tok)
    h2 = token_stream_sha256(list(reversed(docs)), tok)
    assert h1 == h2
    assert re.fullmatch(r"[0-9a-f]{64}", h1)

    changed = [doc("a", text="one two three four"), doc("b", text="four five")]
    assert token_stream_sha256(changed, tok) != h1


# --------------------------------------------------------------------------
# card + full dry run
# --------------------------------------------------------------------------


def test_build_card_contains_all_section_9_keys():
    docs = _many_docs(40)
    splits = split_documents(docs, seed=0)
    card = build_card(docs, splits, {"date": 3}, "2025-01-01")
    for key in CARD_KEYS:
        assert key in card
    assert card["documents_retained"] == 40
    assert card["recorded_cutoff"] == "2025-01-01"
    assert sum(card["split_counts"].values()) == 40


def test_full_dry_run_produces_documents_and_hash(tmp_path):
    result = build(
        source="fake",
        cutoff="2025-01-01",
        out=tmp_path,
        dry_run=True,
        max_docs=60,
    )
    assert result["manifest"]["documents_retained"] >= 30
    assert re.fullmatch(r"[0-9a-f]{64}", result["manifest"]["token_stream_sha256"])
    assert (tmp_path / "card.json").exists()
    assert (tmp_path / "manifest.json").exists()

    card = json.loads((tmp_path / "card.json").read_text())
    for key in CARD_KEYS:
        assert key in card


def test_dry_run_is_reproducible(tmp_path):
    a = build(source="fake", cutoff="2025-01-01", out=tmp_path / "a", dry_run=True)
    b = build(source="fake", cutoff="2025-01-01", out=tmp_path / "b", dry_run=True)
    assert a["manifest"]["token_stream_sha256"] == b["manifest"]["token_stream_sha256"]
