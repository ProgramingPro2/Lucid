"""Tests for the corpus pipeline (docs/CORPUS.md §4, §5).

Every test uses ``FakeClient`` or hand-built documents. No network.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest

from experiments.corpus.build import DEGRADED_WARNING, build
from experiments.corpus.pipeline import (
    CARD_KEYS,
    build_card,
    clean_text,
    deduplicate,
    drop_low_nll,
    filter_by_date,
    filter_by_license,
    fulltext_status_reason,
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


# --------------------------------------------------------------------------
# G-1c fulltext status. The old default kept a non-ok row whenever
# --require-fulltext was left off. No existing test asserted that default;
# these tests pin the new one. tests/test_corpus_harvest.py still calls
# build(..., require_fulltext=True) and is not edited here.
# --------------------------------------------------------------------------


def _row(
    doc_id: str,
    *,
    fulltext: str | None,
    license: str = "cc-by",
    text: str | None = None,
    sections: dict | None = None,
    posted: str | None = None,
) -> dict:
    body = text if text is not None else ("Running prose about a receptor pathway. " * 80)
    return {
        "doc_id": doc_id,
        "source": "biorxiv",
        "title": "t",
        "text": body,
        "sections": sections if sections is not None else {"methods": body},
        "license": license,
        "date": posted or date.today().isoformat(),
        "meta": {} if fulltext is None else {"fulltext": fulltext},
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_fulltext_status_buckets_the_exception_class():
    failed = doc("e")
    failed.source = "biorxiv"
    failed.meta["fulltext"] = (
        "error:TransientRequestError: transient failure after retries"
    )
    assert fulltext_status_reason(failed) == "error:TransientRequestError"
    abstract = doc("a")
    abstract.source = "biorxiv"
    abstract.meta["fulltext"] = "abstract_only"
    assert fulltext_status_reason(abstract) == "abstract_only"
    # A synthetic row has no harvest status and is not a failed fetch.
    assert fulltext_status_reason(doc("fake")) is None
    bare = doc("missing")
    bare.source = "biorxiv"
    assert fulltext_status_reason(bare) == "missing"


def test_default_build_drops_non_ok_fulltext_independent_of_licence(tmp_path):
    """cc-by error and abstract rows must not survive, and a bad-licence
    abstract is counted as abstract_only rather than only as a licence drop.

    The error row is longer than 2,000 characters, so the length floor is
    not what excludes it. require_fulltext is left at its default, off.
    """
    long_error = "Running prose about a failed fetch. " * 80  # > 2000 chars
    assert len(long_error) > 2000
    rows = [
        _row("ok", fulltext="ok"),
        _row("error_cc_by", fulltext="error:RuntimeError", text=long_error),
        _row("abstract_cc_by", fulltext="abstract_only"),
        _row("abstract_nc", fulltext="abstract_only", license="CC BY-NC 4.0"),
        _row("nosection", fulltext="ok", sections={}, text="prose " * 40),
    ]
    path = tmp_path / "docs.jsonl"
    _write_jsonl(path, rows)
    result = build(
        source="biorxiv",
        cutoff="2025-01-01",
        out=tmp_path / "out",
        docs_jsonl=path,
        dry_run=True,
    )
    assert result["drops"]["not_fulltext"] == 3
    assert result["drops"]["no_complete_section"] == 1
    assert result["drops"].get("too_short", 0) == 0
    assert result["drops"].get("licence:cc-by-nc-4.0", 0) == 0
    assert result["card"]["fulltext_status_counts"] == {
        "abstract_only": 2,
        "error:RuntimeError": 1,
    }
    assert result["card"]["documents_retained"] == 1
    assert result["card"]["g1c_compliant"] is True
    assert result["card"]["g1c_verdict"].startswith("G-1c pass")


def test_allow_degraded_keeps_the_error_and_fails_g1c(tmp_path, capsys):
    rows = [
        _row("ok", fulltext="ok"),
        _row(
            "error_cc_by",
            fulltext="error:RuntimeError",
            text="A different failed-fetch abstract about mosquito bitter taste. " * 40,
        ),
    ]
    path = tmp_path / "docs.jsonl"
    _write_jsonl(path, rows)
    result = build(
        source="biorxiv",
        cutoff="2025-01-01",
        out=tmp_path / "out",
        docs_jsonl=path,
        dry_run=True,
        allow_degraded=True,
    )
    err = capsys.readouterr().err
    assert DEGRADED_WARNING in err
    assert "NOT G-1c compliant" in err
    assert result["drops"].get("not_fulltext", 0) == 0
    assert result["card"]["fulltext_status_counts"] == {"error:RuntimeError": 1}
    assert result["card"]["documents_retained"] == 2
    assert result["card"]["g1c_compliant"] is False
    assert result["card"]["g1c_verdict"].startswith("NOT G-1c compliant")
    written = json.loads((tmp_path / "out" / "card.json").read_text())
    assert written["g1c_compliant"] is False
    assert "NOT G-1c compliant" in written["g1c_verdict"]


# --------------------------------------------------------------------------
# Cutoff provenance. An unrecorded cutoff used to be a placeholder the build
# would accept as soon as it parsed as... it didn't parse, but nothing said
# the card had to carry a defensible date. These tests make _pending_ and an
# absent cutoff unshippable, and they stop 2025-06-01 being labelled as the
# model card.
# --------------------------------------------------------------------------


def test_build_refuses_an_unrecorded_cutoff(tmp_path):
    from experiments.corpus.build import main
    from experiments.corpus.sources import UnrecordedCutoff

    with pytest.raises(UnrecordedCutoff, match="unrecorded"):
        build(source="fake", cutoff="_pending_", out=tmp_path / "a", dry_run=True)
    with pytest.raises(UnrecordedCutoff, match="absent"):
        build(source="fake", cutoff=None, out=tmp_path / "b", dry_run=True)
    with pytest.raises(UnrecordedCutoff, match="unrecorded"):
        build(source="fake", cutoff="", out=tmp_path / "c", dry_run=True)
    with pytest.raises(UnrecordedCutoff, match="unrecorded"):
        build(source="fake", cutoff="pending", out=tmp_path / "d", dry_run=True)

    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({"cutoff": "_pending_"}))
    with pytest.raises(UnrecordedCutoff):
        build(source="fake", cutoff=str(pending), out=tmp_path / "e", dry_run=True)

    absent = tmp_path / "absent.json"
    absent.write_text("{}")
    with pytest.raises(UnrecordedCutoff):
        build(source="fake", cutoff=str(absent), out=tmp_path / "f", dry_run=True)

    null = tmp_path / "null.json"
    null.write_text('{"cutoff": null}')
    with pytest.raises(UnrecordedCutoff):
        build(source="fake", cutoff=str(null), out=tmp_path / "g", dry_run=True)

    # No card is written when the build refuses.
    assert not (tmp_path / "a" / "card.json").exists()
    code = main(
        [
            "--source",
            "fake",
            "--cutoff",
            "_pending_",
            "--dry-run",
            "--out",
            str(tmp_path / "cli"),
        ]
    )
    assert code == 2
    assert not (tmp_path / "cli" / "card.json").exists()


def test_programme_cutoff_records_provenance_and_evidence(tmp_path):
    from experiments.corpus.sources import (
        PROGRAMME_CUTOFF,
        first_retained_posting,
    )

    assert first_retained_posting("2025-04-29") == "2025-05-30"
    result = build(
        source="fake",
        cutoff=PROGRAMME_CUTOFF,
        out=tmp_path,
        dry_run=True,
    )
    card = result["card"]
    assert card["recorded_cutoff"] == "2025-04-29"
    assert card["base_model"] == "Qwen/Qwen3-1.7B-Base"
    assert card["cutoff_compliant"] is True
    provenance = card["cutoff_provenance"]
    assert provenance["kind"] == "release-date-proxy"
    assert provenance["model_card_cutoff"] is None
    assert provenance["release_date"] == "2025-04-29"
    assert provenance["weights_uploaded"] == "2025-04-28T05:05:55Z"
    assert provenance["programme_first_retained_posting"] == "2025-05-30"
    assert provenance["revision_policy"] == "upward-only"
    assert "2025-04-29" in card["cutoff_justification"]
    assert "never earlier" in card["cutoff_justification"]
    blob = json.dumps(card["cutoff_evidence"])
    assert "36 trillion tokens" in blob
    assert "no README.md" in blob
    assert "2505.09388" in blob
    assert "2025.04.29" in blob


def test_june_cutoff_is_an_upward_revision_not_the_model_card(tmp_path):
    result = build(source="fake", cutoff="2025-06-01", out=tmp_path, dry_run=True)
    card = result["card"]
    assert card["recorded_cutoff"] == "2025-06-01"
    assert card["cutoff_provenance"]["kind"] == "upward-revision"
    assert card["cutoff_provenance"]["model_card_cutoff"] is None
    assert card["cutoff_compliant"] is True
    assert "not a model-card cutoff" in card["cutoff_justification"]


def test_cutoff_earlier_than_the_proxy_is_not_compliant(tmp_path):
    result = build(source="fake", cutoff="2025-01-01", out=tmp_path, dry_run=True)
    card = result["card"]
    assert card["recorded_cutoff"] == "2025-01-01"
    assert card["cutoff_compliant"] is False
    assert card["cutoff_provenance"]["kind"] == "earlier-than-proxy"
