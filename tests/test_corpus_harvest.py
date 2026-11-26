"""Tests for the live-harvest path: pagination, JATS parsing, resume (no network)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta

import pytest

from experiments.corpus import harvest as harvest_mod
from experiments.corpus.sources import (
    LICENSE_ALLOWLIST,
    BioRxivClient,
    Document,
    HostRefused,
    TransientRequestError,
    backoff_seconds,
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
    # ``cc_no`` is the token whose JATS says all rights reserved. It must not
    # be inferred into the allowlist.
    assert normalise_license("cc_no") == "unknown"
    assert normalise_license("cc_no") not in LICENSE_ALLOWLIST


def test_normalise_observed_cc0_ng_is_public_domain():
    # Live details rows use ``cc0_ng``. The JATS license href on those papers
    # is https://creativecommons.org/publicdomain/zero/1.0/ and the statement
    # says the preprint was placed in the Public Domain.
    assert normalise_license("cc0_ng") == "cc0"
    assert normalise_license("cc0_ng") in LICENSE_ALLOWLIST


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


def _one_window() -> tuple[str, str]:
    """A since/until pair that the grace rule collapses to a single window."""
    return "2026-01-01", "2026-02-02"


def test_backoff_is_bounded_and_jittered(monkeypatch):
    import random

    draws: list[tuple[float, float]] = []

    def uniform(low, high):
        draws.append((low, high))
        return (low + high) / 2

    monkeypatch.setattr(random, "uniform", uniform)
    assert backoff_seconds(0, base=2.0, cap=60.0) == 1.5
    assert draws[-1] == (1.0, 2.0)
    assert backoff_seconds(10, base=2.0, cap=60.0) == 45.0
    assert draws[-1] == (30.0, 60.0)
    assert backoff_seconds(3, base=5.0, cap=0.0) == 0.0


def test_user_agent_names_a_real_contact():
    assert "mailto:research@talontodo.com" in BioRxivClient.UA
    assert "contact: local" not in BioRxivClient.UA


def test_get_does_not_retry_a_host_refusal(monkeypatch):
    import io
    import time
    import urllib.error

    calls: list[str] = []

    def fake_urlopen(req, timeout):
        calls.append(req.get_header("User-agent"))
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {}, io.BytesIO(b"Request blocked")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = BioRxivClient(delay=0.0, retries=4)
    with pytest.raises(HostRefused) as exc_info:
        client._get("https://example.test/x", as_json=True)
    assert len(calls) == 1
    assert "mailto:research@talontodo.com" in calls[0]
    message = str(exc_info.value)
    assert "403" in message
    assert "blocked" in message.lower()


def test_get_retries_a_transient_503_and_reports_the_status(monkeypatch):
    import io
    import time
    import urllib.error

    calls: list[int] = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(
            req.full_url,
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b"unavailable"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = BioRxivClient(delay=0.0, retries=3)
    with pytest.raises(TransientRequestError) as exc_info:
        client._get("https://example.test/x", as_json=True)
    assert len(calls) == 3
    assert exc_info.value.status == 503
    assert "503" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_repeated_connection_failures_are_a_refusal(monkeypatch):
    import time
    import urllib.error

    calls: list[int] = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.URLError("connection reset by peer")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = BioRxivClient(delay=0.0, retries=3)
    with pytest.raises(HostRefused) as exc_info:
        client._get("https://example.test/x")
    assert len(calls) == 3
    assert "connection reset" in str(exc_info.value)


def test_get_does_not_treat_a_404_as_a_host_block(monkeypatch):
    import io
    import time
    import urllib.error

    calls: list[int] = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", {}, io.BytesIO(b"missing")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = BioRxivClient(delay=0.0, retries=4)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        client._get("https://example.test/missing.xml")
    assert len(calls) == 1
    assert exc_info.value.code == 404


def test_fetch_fulltext_records_the_error_and_propagates_refusal():
    class Failing(BioRxivClient):
        def _get(self, url, *, as_json=False):
            raise TransientRequestError(url, 503, "upstream unavailable")

    doc = Document(
        doc_id="biorxiv:err",
        source="biorxiv",
        title="t",
        text="abstract",
        sections={"abstract": "abstract"},
        license="cc-by",
        date="2026-09-01",
        meta={"jatsxml": "https://example.test/a.xml", "fulltext": "abstract_only"},
    )
    out = Failing(delay=0.0).fetch_fulltext(doc)
    assert out.meta["fulltext"].startswith("error:")
    assert "503" in out.meta["fulltext"]
    assert out.meta["fulltext"] != "abstract_only"

    class Refusing(BioRxivClient):
        def _get(self, url, *, as_json=False):
            raise HostRefused(url, 403, "Request blocked")

    with pytest.raises(HostRefused):
        Refusing(delay=0.0).fetch_fulltext(doc)


def test_harvest_resumes_a_window_from_the_next_page(tmp_path, monkeypatch):
    """A failure on page 2 must not re-fetch pages 0 and 1, or drop a document."""
    state = {"failed": False, "urls": []}

    class Client(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def _get(self, url, *, as_json=False):
            if not as_json:
                return JATS
            state["urls"].append(url)
            cursor = int(url.rsplit("/", 1)[-1])
            if cursor == 4 and not state["failed"]:
                state["failed"] = True
                raise TransientRequestError(url, 503, "simulated upstream 503")
            if cursor >= 6:
                return {"collection": []}
            rows = [
                {
                    "doi": f"10.1/{cursor + i}",
                    "date": "2026-02-01",
                    "license": "cc_by",
                    "title": "t",
                    "abstract": "a",
                    "jatsxml": f"https://example.test/{cursor + i}.xml",
                }
                for i in range(2)
            ]
            return {"collection": rows}

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Client)
    out = tmp_path / "harvest.jsonl"
    since, until = _one_window()
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=since,
        until=until,
        out=out,
        max_docs=20,
        delay=0.0,
        cooldown=0.0,
        window_retries=2,
    )
    cursors = [int(url.rsplit("/", 1)[-1]) for url in state["urls"]]
    assert cursors == [0, 2, 4, 4, 6], cursors
    assert counts["written"] == 6
    assert counts["fulltext_ok"] == 6
    assert "aborted" not in counts
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    ids = [row["doc_id"] for row in lines]
    assert len(ids) == len(set(ids)) == 6
    assert set(ids) == {f"biorxiv:10.1/{i}" for i in range(6)}
    saved = json.loads(harvest_mod.cursor_path_for(out).read_text())
    assert saved
    assert max(saved.values()) >= 4


def test_harvest_stops_on_host_refused_without_burning_retries(tmp_path, monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []
    monkeypatch.setattr(harvest_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    class Client(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def iter_metadata(self, since, until):
            calls["n"] += 1
            raise HostRefused(
                "https://api.biorxiv.org/details/biorxiv/x", 403, "Request blocked"
            )

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Client)
    # Two windows would be walked if a refusal were retried like a transient error.
    counts = harvest_mod.harvest(
        server="biorxiv",
        since="2026-01-01",
        until="2026-02-15",
        out=tmp_path / "harvest.jsonl",
        max_docs=50,
        delay=0.0,
        cooldown=120.0,
        window_retries=3,
    )
    assert calls["n"] == 1
    assert counts["aborted"] == "host_refused"
    assert counts["written"] == 0
    assert sleeps == []


def test_harvest_stops_when_a_window_makes_no_progress(tmp_path, monkeypatch):
    calls = {"n": 0}

    class Client(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def iter_metadata(self, since, until):
            calls["n"] += 1
            raise TransientRequestError(
                "https://api.biorxiv.org/details/biorxiv/x", 503, "unavailable"
            )

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Client)
    counts = harvest_mod.harvest(
        server="biorxiv",
        since="2026-01-01",
        until="2026-02-15",
        out=tmp_path / "harvest.jsonl",
        max_docs=50,
        delay=0.0,
        cooldown=0.0,
        window_retries=2,
    )
    assert calls["n"] == 2
    assert counts["aborted"] == "no_progress"
    assert counts["written"] == 0


def test_harvest_counts_fulltext_failures_separately(tmp_path, monkeypatch):
    docs = [
        Document(
            doc_id="biorxiv:allowed-error",
            source="biorxiv",
            title="t",
            text="abstract",
            sections={"abstract": "abstract"},
            license="cc-by",
            date="2026-02-01",
            meta={"jatsxml": "https://example.test/a.xml", "fulltext": "abstract_only"},
        ),
        Document(
            doc_id="biorxiv:excluded",
            source="biorxiv",
            title="t",
            text="abstract",
            sections={"abstract": "abstract"},
            license="cc-by-nc-nd-4.0",
            date="2026-02-01",
            meta={"jatsxml": "https://example.test/b.xml", "fulltext": "abstract_only"},
        ),
        Document(
            doc_id="biorxiv:no-link",
            source="biorxiv",
            title="t",
            text="abstract",
            sections={"abstract": "abstract"},
            license="cc-by",
            date="2026-02-01",
            meta={"jatsxml": None, "fulltext": "abstract_only"},
        ),
    ]

    class Client(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def iter_metadata(self, since, until):
            yield from docs

        def fetch_fulltext(self, doc):
            doc.meta["fulltext"] = "error:TimeoutError:timed out"
            return doc

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Client)
    since, until = _one_window()
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=since,
        until=until,
        out=tmp_path / "harvest.jsonl",
        max_docs=10,
        delay=0.0,
        cooldown=0.0,
    )
    assert counts["fulltext_error"] == 1
    assert counts["fulltext_error_reasons"] == {
        "error:TimeoutError:timed out": 1
    }
    assert counts["licence_excluded"] == 1
    assert counts["no_jats_link"] == 1
    assert counts["abstract_only"] == 0
    assert counts["fulltext_ok"] == 0
    lines = [json.loads(line) for line in (tmp_path / "harvest.jsonl").read_text().splitlines()]
    by_id = {row["doc_id"]: row["meta"]["fulltext"] for row in lines}
    assert by_id["biorxiv:allowed-error"].startswith("error:")
    assert by_id["biorxiv:excluded"] == "licence_excluded"
    assert by_id["biorxiv:no-link"] == "no_jats_link"


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


class _BytesResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.status = 200

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _silence_sleep(monkeypatch):
    import time

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(float(seconds)))
    return sleeps


def _window_rows():
    return [
        {
            "doi": "10.1/enet",
            "date": "2026-02-01",
            "license": "cc_by",
            "title": "t",
            "abstract": "abstract prose",
            "jatsxml": "https://www.biorxiv.org/content/early/2026/02/01/10.1.enet.source.xml",
        },
        {
            "doi": "10.1/next",
            "date": "2026-02-01",
            "license": "cc_by_nc_nd",
            "title": "t",
            "abstract": "abstract prose",
            "jatsxml": "https://www.biorxiv.org/content/early/2026/02/01/10.1.next.source.xml",
        },
    ]


def _install_split_client(monkeypatch, *, retries: int, jats_urlopen):
    """Metadata answers locally; JATS goes through ``jats_urlopen``."""

    class Client(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            kwargs["retries"] = retries
            super().__init__(**kwargs)

    def fake_urlopen(req, timeout):
        url = req.full_url
        if "/details/" in url:
            cursor = int(url.rsplit("/", 1)[-1])
            if cursor > 0:
                return _BytesResponse(b'{"collection": []}')
            return _BytesResponse(json.dumps({"collection": _window_rows()}).encode())
        return jats_urlopen(req, timeout)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(harvest_mod, "BioRxivClient", Client)


def test_transport_failures_retry_then_stay_transient(monkeypatch):
    """Errno 101/104, timeouts and DNS are not a permanent host refusal."""
    import socket
    import urllib.error

    reasons = [
        OSError(101, "Network is unreachable"),
        OSError(104, "Connection reset by peer"),
        TimeoutError("The read operation timed out"),
        socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
    ]
    for reason in reasons:
        calls: list[int] = []

        def fake_urlopen(req, timeout, reason=reason):
            calls.append(1)
            raise urllib.error.URLError(reason)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        _silence_sleep(monkeypatch)
        client = BioRxivClient(delay=0.0, retries=3)
        with pytest.raises(TransientRequestError) as exc_info:
            client._get("https://www.biorxiv.org/content/x.source.xml")
        assert len(calls) == 3
        assert not isinstance(exc_info.value, HostRefused)
        assert exc_info.value.status is None


def test_harvest_enetunreach_on_jats_is_a_fulltext_error(tmp_path, monkeypatch):
    """Errno 101 on one JATS fetch must not abort the window."""
    import urllib.error

    jats_calls: list[str] = []

    def jats_urlopen(req, timeout):
        jats_calls.append(req.full_url)
        raise urllib.error.URLError(OSError(101, "Network is unreachable"))

    _silence_sleep(monkeypatch)
    _install_split_client(monkeypatch, retries=3, jats_urlopen=jats_urlopen)
    since, until = _one_window()
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=since,
        until=until,
        out=tmp_path / "harvest.jsonl",
        max_docs=10,
        delay=0.0,
        cooldown=0.0,
        window_retries=3,
    )
    assert "aborted" not in counts
    assert counts["written"] == 2
    assert counts["fulltext_error"] == 1
    assert counts["licence_excluded"] == 1
    assert len(jats_calls) == 3  # retried, then skipped; the next paper was not fetched
    lines = [json.loads(line) for line in (tmp_path / "harvest.jsonl").read_text().splitlines()]
    by_id = {row["doc_id"]: row for row in lines}
    status = by_id["biorxiv:10.1/enet"]["meta"]["fulltext"]
    assert status.startswith("error:")
    assert "101" in status or "unreachable" in status.lower()
    assert by_id["biorxiv:10.1/next"]["meta"]["fulltext"] == "licence_excluded"


def test_harvest_jats_http_403_still_aborts(tmp_path, monkeypatch):
    import io
    import urllib.error

    jats_calls: list[str] = []

    def jats_urlopen(req, timeout):
        jats_calls.append(req.full_url)
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {}, io.BytesIO(b"Request blocked")
        )

    _silence_sleep(monkeypatch)
    _install_split_client(monkeypatch, retries=4, jats_urlopen=jats_urlopen)
    since, until = _one_window()
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=since,
        until=until,
        out=tmp_path / "harvest.jsonl",
        max_docs=10,
        delay=0.0,
        cooldown=120.0,
        window_retries=3,
    )
    assert counts["aborted"] == "host_refused"
    assert counts["written"] == 0
    assert len(jats_calls) == 1
    assert (tmp_path / "harvest.jsonl").read_text() == ""


def test_sustained_429_aborts_only_after_the_cooldown_budget(tmp_path, monkeypatch):
    import io
    import urllib.error

    jats_calls: list[str] = []
    sleeps = _silence_sleep(monkeypatch)

    def jats_urlopen(req, timeout):
        jats_calls.append(req.full_url)
        raise urllib.error.HTTPError(
            req.full_url,
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b"slow down"),
        )

    _install_split_client(monkeypatch, retries=3, jats_urlopen=jats_urlopen)
    since, until = _one_window()
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=since,
        until=until,
        out=tmp_path / "harvest.jsonl",
        max_docs=10,
        delay=0.0,
        cooldown=120.0,
        window_retries=3,
    )
    # One 429 is not an abort: the client cools down and uses every retry.
    assert len(jats_calls) == 3
    # Cooldown is 60s; a patched sleep does not advance the clock, so the
    # recorded wait is a few microseconds under 60.
    assert max(sleeps) >= 59.0
    assert counts["aborted"] == "host_refused"
    assert counts["written"] == 0


def test_a_429_that_clears_does_not_abort(tmp_path, monkeypatch):
    import io
    import urllib.error

    jats_calls: list[str] = []
    sleeps = _silence_sleep(monkeypatch)

    def jats_urlopen(req, timeout):
        jats_calls.append(req.full_url)
        if len(jats_calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                io.BytesIO(b"slow down"),
            )
        return _BytesResponse(JATS)

    _install_split_client(monkeypatch, retries=4, jats_urlopen=jats_urlopen)
    since, until = _one_window()
    counts = harvest_mod.harvest(
        server="biorxiv",
        since=since,
        until=until,
        out=tmp_path / "harvest.jsonl",
        max_docs=10,
        delay=0.0,
        cooldown=0.0,
        window_retries=2,
    )
    assert "aborted" not in counts
    assert len(jats_calls) == 2
    assert max(sleeps) >= 59.0
    assert counts["fulltext_ok"] == 1
    assert counts["written"] == 2


def _record(doc_id: str) -> str:
    return json.dumps(
        {
            "doc_id": doc_id,
            "source": "biorxiv",
            "title": "t",
            "text": "body",
            "sections": {"introduction": "body"},
            "license": "cc-by",
            "date": "2025-07-01",
            "meta": {},
        }
    )


def test_load_documents_skips_only_a_malformed_final_line(tmp_path):
    """A truncated tail must not fail a build. An earlier bad line must."""
    path = tmp_path / "docs.jsonl"
    first = _record("biorxiv:10.1/a")
    second = _record("biorxiv:10.1/b")
    path.write_bytes(first.encode() + b"\n" + second.encode() + b"\n" + b'{"doc_id": "torn')
    docs = harvest_mod.load_documents(path)
    assert [d.doc_id for d in docs] == ["biorxiv:10.1/a", "biorxiv:10.1/b"]

    # The same truncation with its own newline is still the final line.
    terminated = tmp_path / "terminated.jsonl"
    terminated.write_text(first + "\n" + '{"doc_id": "torn"\n', encoding="utf-8")
    docs = harvest_mod.load_documents(terminated)
    assert [d.doc_id for d in docs] == ["biorxiv:10.1/a"]

    interior = tmp_path / "interior.jsonl"
    interior.write_text('{"doc_id": "torn"\n' + second + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        harvest_mod.load_documents(interior)

    both = tmp_path / "both.jsonl"
    both.write_bytes(b'{"broken"\n' + second.encode() + b"\n" + b'{"also": "torn')
    with pytest.raises(json.JSONDecodeError):
        harvest_mod.load_documents(both)

    clean = tmp_path / "clean.jsonl"
    clean.write_text(first + "\n\n" + second + "\n", encoding="utf-8")
    docs = harvest_mod.load_documents(clean)
    assert [d.doc_id for d in docs] == ["biorxiv:10.1/a", "biorxiv:10.1/b"]


def test_boundary_day_is_outside_the_fetch_set(tmp_path, monkeypatch):
    """The date-gate boundary is not requested. The next day is.

    cutoff 2025-06-01 plus 30 days is 2025-07-01, and the date gate keeps
    postings strictly after that day. A document posted on 2025-07-01 used
    to be fetched and then dropped. The window now opens on 2025-07-02.
    """
    from experiments.corpus.sources import cutoff_plus_grace

    requested: list[tuple[str, str]] = []

    class Recording(BioRxivClient):
        def __init__(self, **kwargs):
            kwargs["delay"] = 0.0
            super().__init__(**kwargs)

        def iter_metadata(self, since, until):
            requested.append((since, until))
            yield from ()

    monkeypatch.setattr(harvest_mod, "BioRxivClient", Recording)
    start = harvest_mod.harvest_window_start("2025-06-01")
    assert start == cutoff_plus_grace("2025-06-01") + timedelta(days=1)
    assert start.isoformat() == "2025-07-02"
    # The proxy's first retained posting is unchanged: 2025-04-29 + 30 + 1.
    assert harvest_mod.harvest_window_start("2025-04-29").isoformat() == "2025-05-30"

    harvest_mod.harvest(
        server="biorxiv",
        since="2025-06-01",
        until="2025-07-09",
        out=tmp_path / "aligned.jsonl",
        max_docs=10,
        delay=0.0,
        window_retries=1,
    )
    assert requested == [
        ("2025-07-02", "2025-07-08"),
        ("2025-07-09", "2025-07-09"),
    ]
    assert not any(lo <= "2025-07-01" <= hi for lo, hi in requested)
    assert any(lo <= "2025-07-02" <= hi for lo, hi in requested)

    requested.clear()
    harvest_mod.harvest(
        server="biorxiv",
        since="2025-06-01",
        until="2025-07-01",
        out=tmp_path / "boundary_only.jsonl",
        max_docs=10,
        delay=0.0,
        window_retries=1,
    )
    assert requested == []
