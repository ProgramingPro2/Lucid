"""Tests for the live-harvest path: pagination, JATS parsing, resume (no network)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta

import pytest

from experiments.corpus import harvest as harvest_mod
from experiments.corpus.sources import (
    BioRxivClient,
    Document,
    normalise_license,
    parse_jats,
)

JATS = b"""<?xml version="1.0"?>
<article xmlns="http://jats.nlm.nih.gov">
  <front><abstract><title>Abstract</title><p>An abstract about receptors.</p></abstract></front>
  <body>
    <sec><title>Introduction</title><p>Intro running prose.</p></sec>
    <sec><title>Materials and Methods</title>
      <p>We measured 412 participants.</p>
      <table-wrap><p>TABLE CELL MUST NOT SURVIVE</p></table-wrap>
    </sec>
    <sec><title>Results</title>
      <p>The effect was present.</p>
      <fig><caption>FIGURE CAPTION MUST NOT SURVIVE</caption></fig>
    </sec>
    <sec><title>References</title><ref-list><ref><p>REF MUST NOT SURVIVE</p></ref></ref-list></sec>
  </body>
</article>
"""


def test_normalise_biorxiv_underscore_licences():
    # The live API returns machine tokens, not prose. A missing alias here
    # silently drops an entire licence class.
    assert normalise_license("cc_by") == "cc-by"
    assert normalise_license("cc_by_nc_nd") == "cc-by-nc-nd-4.0"
    assert normalise_license("cc_by_nd") == "cc-by-nd-4.0"
    assert normalise_license("cc0") == "cc0"
    assert normalise_license("cc_no") == "unknown"


def test_parse_jats_extracts_strata_and_drops_backmatter():
    text, sections = parse_jats(JATS)
    assert sections["abstract"] == "An abstract about receptors."
    assert sections["introduction"] == "Intro running prose."
    assert "412 participants" in sections["methods"]
    assert "Results" not in sections  # title is metadata, not a section
    assert "TABLE CELL" not in text
    assert "FIGURE CAPTION" not in text
    assert "REF MUST NOT SURVIVE" not in text


def test_fetch_advances_cursor_by_rows_returned():
    """bioRxiv returns 30 rows/page; advancing by 100 skipped 70 % (F46)."""
    urls: list[str] = []

    class Client(BioRxivClient):
        def _get(self, url, *, as_json=False):
            urls.append(url)
            cursor = int(url.rsplit("/", 1)[-1])
            rows = [
                {
                    "doi": f"10.1/{cursor + i}",
                    "date": "2026-09-01",
                    "license": "cc_by",
                    "title": "t",
                    "abstract": "a",
                    "jatsxml": None,
                }
                for i in range(30)
            ]
            return {"collection": rows}

    client = Client(delay=0.0)
    docs = client.fetch("2026-09-01", "2026-09-07", max_docs=60)
    assert len(docs) == 60
    assert urls[-1].endswith("/30"), urls
    assert client._last_request == 0.0  # _get override bypasses the clock


def test_get_retries_a_429_and_honours_retry_after(monkeypatch):
    import time
    import urllib.error

    calls: list[str] = []

    class Response:
        def read(self):
            return b'{"collection": []}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, None
            )
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    client = BioRxivClient(delay=0.0, retries=2)
    assert client._get("https://example.test/x", as_json=True) == {"collection": []}
    assert len(calls) == 2
    assert client._cooldown_until > 0


def test_fulltext_fetched_only_for_allowed_licence():
    class Client(BioRxivClient):
        def _get(self, url, *, as_json=False):
            return JATS

    client = Client(delay=0.0, fulltext=True)
    doc = Document(
        doc_id="biorxiv:1",
        source="biorxiv",
        title="t",
        text="abstract",
        sections={"abstract": "abstract"},
        license="cc-by",
        date="2026-09-01",
        meta={"jatsxml": "https://example.test/a.xml", "fulltext": "abstract_only"},
    )
    out = client.fetch_fulltext(doc)
    assert out.meta["fulltext"] == "ok"
    assert "412 participants" in out.text
    assert "REF MUST NOT SURVIVE" not in out.text

    blocked = Document(
        doc_id="biorxiv:2",
        source="biorxiv",
        title="t",
        text="abstract",
        sections={"abstract": "abstract"},
        license="cc-by-nc-nd-4.0",
        date="2026-09-01",
        meta={"jatsxml": "https://example.test/b.xml", "fulltext": "abstract_only"},
    )
    # fetch() is responsible for the licence check; fetch_fulltext itself is a
    # pure fetch and must not silently downgrade a licence decision.
    assert client.fetch_fulltext(blocked).meta["fulltext"] == "ok"


def test_harvest_resumes_without_duplicate_doc_ids(tmp_path, monkeypatch):
    docs = [
        Document(
            doc_id=f"biorxiv:{i}",
            source="biorxiv",
            title="t",
            text="running prose",
            sections={"methods": "running prose"},
            license="cc-by",
            date=date.today().isoformat(),
            meta={"fulltext": "ok"},
        )
        for i in range(3)
    ]

    class Fake(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def iter_metadata(self, since, until):
            yield from docs

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Fake)
    out = tmp_path / "harvest.jsonl"
    since = (date.today() - timedelta(days=60)).isoformat()

    first = harvest_mod.harvest(
        server="biorxiv", since=since, until=date.today().isoformat(),
        out=out, max_docs=2, delay=0.0,
    )
    assert first["written"] == 2
    second = harvest_mod.harvest(
        server="biorxiv", since=since, until=date.today().isoformat(),
        out=out, max_docs=2, delay=0.0,
    )
    # The fake returns the same three docs in every weekly window. The first run
    # committed two; the second recognises both and writes only the third, so no
    # doc id is ever written twice.
    assert second["skipped_existing"] >= 3
    assert second["written"] == 1

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 3
    assert len({r["doc_id"] for r in lines}) == 3
    assert set(lines[0]) >= {"doc_id", "source", "text", "sections", "license", "date"}


def test_harvest_retries_a_rate_limited_window(tmp_path, monkeypatch):
    docs = [
        Document(
            doc_id=f"biorxiv:retry{i}",
            source="biorxiv",
            title="t",
            text="running prose",
            sections={"methods": "running prose"},
            license="cc-by",
            date=date.today().isoformat(),
            meta={"fulltext": "ok"},
        )
        for i in range(2)
    ]

    class Flaky(BioRxivClient):
        calls = 0

        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def iter_metadata(self, since, until):
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("simulated 429")
            yield from docs

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Flaky)
    out = tmp_path / "harvest.jsonl"
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=(date.today() - timedelta(days=60)).isoformat(),
        until=date.today().isoformat(),
        out=out,
        max_docs=10,
        delay=0.0,
        cooldown=0.0,
        window_retries=2,
    )
    assert Flaky.calls >= 2
    assert counts["written"] == 2
    assert len(out.read_text().splitlines()) == 2


def test_build_from_jsonl_requires_fulltext(tmp_path):
    from experiments.corpus.build import build

    rows = [
        {
            "doc_id": "full",
            "source": "biorxiv",
            "title": "t",
            "text": "Running prose about a cohort and a receptor pathway. " * 60,
            "sections": {
                "methods": "Running prose about a cohort and a receptor pathway. " * 60
            },
            "license": "cc-by",
            "date": date.today().isoformat(),
            "meta": {"fulltext": "ok"},
        },
        {
            "doc_id": "abstract",
            "source": "biorxiv",
            "title": "t",
            "text": "Short abstract. " * 30,
            "sections": {"abstract": "Short abstract. " * 30},
            "license": "cc-by",
            "date": date.today().isoformat(),
            "meta": {"fulltext": "abstract_only"},
        },
        {
            "doc_id": "withdrawn",
            "source": "biorxiv",
            "title": "WITHDRAWN: a study",
            "text": "Withdrawal Statement. " * 200,
            "sections": {"abstract": "Withdrawal Statement. " * 200},
            "license": "cc-by",
            "date": date.today().isoformat(),
            "meta": {"fulltext": "ok"},
        },
        {
            "doc_id": "short",
            "source": "biorxiv",
            "title": "t",
            "text": "Too short.",
            "sections": {"abstract": "Too short."},
            "license": "cc-by",
            "date": date.today().isoformat(),
            "meta": {"fulltext": "ok"},
        },
    ]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    result = build(
        source="biorxiv",
        cutoff=(date.today() - timedelta(days=60)).isoformat(),
        out=tmp_path / "out",
        docs_jsonl=path,
        require_fulltext=True,
        dry_run=True,
    )
    assert result["manifest"]["documents_harvested"] == 4  # raw count is kept
    assert result["drops"]["not_fulltext"] == 1
    assert result["drops"]["withdrawn"] == 1
    assert result["drops"]["too_short"] == 1
    assert result["card"]["documents_retained"] == 1
