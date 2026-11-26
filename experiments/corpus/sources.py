"""Corpus sources and the licence/cutoff rules (docs/CORPUS.md §2).

Every network call is isolated in a ``_http_get`` method so that the pipeline can
be tested with ``FakeClient`` and can be dry-run with no connectivity at all.
That separation is the point: a test that needs the internet is not a test of
the pipeline, it is a test of the internet.

Nothing here may infer a licence from a repository default. The licence field is
per item, and an item we cannot classify is dropped and counted (§2, §8).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# Licences we may keep. Anything else — including NC and ND — is dropped.
LICENSE_ALLOWLIST: frozenset[str] = frozenset(
    {"cc-by", "cc-by-4.0", "cc0", "public-domain"}
)

_LICENSE_ALIASES = {
    "cc by": "cc-by",
    "cc-by": "cc-by",
    "cc by 4.0": "cc-by-4.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc-by 4.0": "cc-by-4.0",
    # bioRxiv/medRxiv report machine tokens such as ``cc_by_nc_nd``. Normalise
    # underscores before lookup and list the resulting space-separated forms;
    # missing one silently drops a whole licence class, so this is tested.
    "cc by 4": "cc-by-4.0",
    "cc by nc": "cc-by-nc-4.0",
    "cc by nc 4.0": "cc-by-nc-4.0",
    # NC / ND are classified (not "unknown") so the card can count why they were
    # dropped; they are still outside LICENSE_ALLOWLIST.
    "cc by-nc 4.0": "cc-by-nc-4.0",
    "cc-by-nc-4.0": "cc-by-nc-4.0",
    "cc by nc nd": "cc-by-nc-nd-4.0",
    "cc by nc nd 4.0": "cc-by-nc-nd-4.0",
    "cc by-nc-nd 4.0": "cc-by-nc-nd-4.0",
    "cc-by-nc-nd-4.0": "cc-by-nc-nd-4.0",
    "cc by nd": "cc-by-nd-4.0",
    "cc by nd 4.0": "cc-by-nd-4.0",
    "cc by-nd 4.0": "cc-by-nd-4.0",
    "cc-by-nd-4.0": "cc-by-nd-4.0",
    "cc0": "cc0",
    "cc0 1.0": "cc0",
    "cc0 1": "cc0",
    "cc-0": "cc0",
    # Live details payloads use ``cc0_ng``. The JATS license href on those
    # papers is creativecommons.org/publicdomain/zero/1.0/ and the statement
    # says the preprint was placed in the Public Domain. Underscores are
    # turned into spaces before this lookup. ``cc_no`` is not an alias: the
    # JATS for that token says all rights reserved, so it stays "unknown".
    "cc0 ng": "cc0",
    "public domain": "public-domain",
    "public-domain": "public-domain",
    "publicdomain": "public-domain",
    "us government work": "public-domain",
}


def normalise_license(value: str | None) -> str:
    """Map a free-text licence string to a canonical token, or ``"unknown"``.

    The mapping is deliberately small and exact. A fuzzy match here would be a
    licence violation with a green tick next to it.
    """
    if not value:
        return "unknown"
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    s = s.removeprefix("http://").removeprefix("https://")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in _LICENSE_ALIASES:
        return _LICENSE_ALIASES[s]
    # Strip a trailing version for the common "CC BY 4.0" shape only.
    base = re.sub(r"\s+v?\d+(\.\d+)*$", "", s)
    return _LICENSE_ALIASES.get(base, "unknown")


@dataclass
class Document:
    """One intact document, with its section structure preserved.

    ``sections`` is ``{stratum: text}`` and is what makes the stratified endpoint
    possible; ``text`` is the concatenation used for token streams. Keeping both
    means a cleaned document is never re-segmented by a heuristic downstream.
    """

    doc_id: str
    source: str
    title: str
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    license: str = "unknown"
    date: str = "1970-01-01"
    meta: dict = field(default_factory=dict)

    def parsed_date(self) -> date:
        return date.fromisoformat(self.date[:10])


# --------------------------------------------------------------------------
# Cutoff
# --------------------------------------------------------------------------


def load_cutoff(path_or_date: str | Path) -> str:
    """Read the model's recorded training cutoff.

    Accepts either an ISO date or a JSON file with a ``"cutoff"`` key. It never
    guesses: docs/CORPUS.md step 1 requires the value to be read verbatim from the
    model card, because "recent" is not a date test.
    """
    p = Path(path_or_date)
    if p.suffix.lower() == ".json" and p.exists():
        payload = json.loads(p.read_text())
        if "cutoff" not in payload:
            raise KeyError(f"{p} has no 'cutoff' key")
        value = payload["cutoff"]
    else:
        value = str(path_or_date)

    try:
        date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(
            f"cutoff must be an ISO date or a JSON file with 'cutoff'; got {value!r}"
        ) from exc
    return str(value)[:10]


def cutoff_plus_grace(cutoff: str, grace_days: int = 30) -> date:
    """The first date a document may have: strictly after cutoff + grace."""
    return date.fromisoformat(cutoff[:10]) + timedelta(days=grace_days)


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


class FakeClient:
    """A source that returns a fixed list. Used by tests and ``--dry-run``."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = list(documents)

    def fetch(self, **_: object) -> list[Document]:
        return [Document(**{**d.__dict__}) for d in self.documents]


_SECTION_STRATA: tuple[tuple[str, str], ...] = (
    ("introduction", "introduction"),
    ("background", "introduction"),
    ("method", "methods"),
    ("material", "methods"),
    ("result", "results"),
    ("discussion", "discussion"),
    ("conclusion", "discussion"),
)


def _xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _xml_text(node, *, skip_titles: bool = True) -> str:
    """All descendant text except tables, figures, refs and formulae.

    Section and abstract titles are metadata, and concatenating them into the
    prose produces artefacts like ``AbstractApoptosis has been…``. They are
    skipped by default; :func:`parse_jats` reads a title explicitly with
    ``skip_titles=False``.
    """
    import xml.etree.ElementTree as ET

    drop = {"table-wrap", "table", "fig", "ref-list", "ref", "disp-formula"}
    if skip_titles:
        drop.add("title")
    parts: list[str] = []

    def walk(n) -> None:
        if _xml_local(n.tag) in drop:
            return
        if n.text:
            parts.append(n.text)
        for child in n:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    if isinstance(node, ET.Element):
        walk(node)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def backoff_seconds(attempt: int, *, base: float = 2.0, cap: float = 60.0) -> float:
    """Bounded exponential backoff with jitter.

    ``attempt`` is 0-based. The sleep is drawn from the upper half of the
    exponential ceiling so a retry waits, but never longer than ``cap``.
    A non-positive cap disables the wait (tests pass ``cooldown=0``).
    """
    import random

    if cap <= 0:
        return 0.0
    ceiling = min(cap, base * (2**attempt))
    if ceiling <= 0:
        return 0.0
    return random.uniform(ceiling / 2.0, ceiling)


def _error_snippet(exc: BaseException) -> str:
    """Short, single-line body or message from a failed request."""
    data: bytes | str | None = None
    read = getattr(exc, "read", None)
    if callable(read):
        try:
            data = read()
        except Exception:  # noqa: BLE001 - a missing body must not hide the status
            data = None
    if isinstance(data, bytes):
        text = data[:240].decode("utf-8", errors="replace")
    elif isinstance(data, str):
        text = data[:240]
    else:
        text = str(exc)
    return " ".join(text.split())[:240]


_BLOCK_MARKERS = (
    "access denied",
    "request blocked",
    "captcha",
    "cloudflare",
    "attention required",
)


def _looks_like_block(snippet: str) -> bool:
    text = snippet.lower()
    return any(marker in text for marker in _BLOCK_MARKERS)


# Routing and path failures. These are not a host deciding to refuse the
# client: ENETUNREACH (101), ECONNRESET (104), timeouts, unreachable peers.
# ECONNREFUSED (111) is intentionally absent; a bare URLError with no errno
# stays a refusal so an unclassified reset is not retried for hours.
_TRANSIENT_ERRNOS = frozenset({100, 101, 102, 103, 104, 110, 113})


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """The exception plus a URLError's ``reason`` and ``__cause__``, once each."""
    import urllib.error

    chain: list[BaseException] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        nxt: BaseException | None = None
        if isinstance(cur, urllib.error.URLError) and not isinstance(
            cur, urllib.error.HTTPError
        ):
            reason = getattr(cur, "reason", None)
            if isinstance(reason, BaseException):
                nxt = reason
        if nxt is None:
            cause = cur.__cause__
            if isinstance(cause, BaseException):
                nxt = cause
        cur = nxt
    return chain


def _is_transient_transport(exc: BaseException) -> bool:
    """True for routing, reset, timeout and DNS failures.

    HTTP statuses are never transport failures. A URLError whose reason is a
    plain string (no errno) is not either: that shape is the unclassified
    refusal the retry budget must not absorb.
    """
    import socket
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return False
    chain = _exception_chain(exc)
    for item in chain:
        if isinstance(item, TimeoutError):
            return True
        if isinstance(item, socket.gaierror):
            return True
        if isinstance(item, OSError) and item.errno in _TRANSIENT_ERRNOS:
            return True
    text = " ".join(str(item).lower() for item in chain)
    if "timed out" in text or "timeout" in text:
        return True
    if "network is unreachable" in text or "[errno 101]" in text or "[errno 104]" in text:
        return True
    if (
        "name or service not known" in text
        or "temporary failure in name resolution" in text
        or "nodename nor servname" in text
    ):
        return True
    return False


class HostRefused(RuntimeError):
    """The host refused this client. Retrying extends the block."""

    def __init__(self, url: str, status: int | None, detail: str) -> None:
        self.url = url
        self.status = status
        self.detail = detail
        super().__init__(f"host refused {url} (status={status}): {detail}")


class TransientRequestError(RuntimeError):
    """A retryable failure exhausted its attempts.

    Covers transport errors (routing, reset, timeout, DNS) and HTTP 5xx.
    A 429 that is still failing after the cooldown budget is ``HostRefused``,
    not this error: the host asked the client to stop.
    """

    def __init__(self, url: str, status: int | None, detail: str) -> None:
        self.url = url
        self.status = status
        self.detail = detail
        super().__init__(
            f"transient failure after retries for {url} (status={status}): {detail}"
        )


def _failure_label(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:180]


def parse_jats(xml_bytes: bytes) -> tuple[str, dict[str, str]]:
    """Extract running prose and section strata from a JATS article.

    Returns ``(text, sections)``. Section titles are mapped onto the same five
    strata the pipeline already uses; unknown or missing titles go to ``other``.
    The abstract is kept separately, because it is a different register and the
    stratified endpoint needs to be able to exclude it.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    sections: dict[str, list[str]] = {}

    for abstract in root.iter():
        if _xml_local(abstract.tag) == "abstract":
            body = _xml_text(abstract)
            if body:
                sections.setdefault("abstract", []).append(body)

    for sec in root.iter():
        if _xml_local(sec.tag) != "sec":
            continue
        title = ""
        for child in sec:
            if _xml_local(child.tag) == "title":
                title = _xml_text(child, skip_titles=False).lower()
                break
        stratum = "other"
        for needle, name in _SECTION_STRATA:
            if needle in title:
                stratum = name
                break
        paragraphs = [
            _xml_text(child)
            for child in sec
            if _xml_local(child.tag) in {"p", "sec"}
        ]
        body = " ".join(p for p in paragraphs if p)
        if body:
            sections.setdefault(stratum, []).append(body)

    joined = {k: "\n\n".join(v) for k, v in sections.items() if v}
    text = "\n\n".join(joined.values())
    return text, joined


class BioRxivClient:
    """bioRxiv / medRxiv details API, queried by posting date.

    The details endpoint returns metadata and the abstract. Full text is a
    separate JATS XML request to the per-paper ``jatsxml`` link, which is fetched
    only when the per-item licence allows it. Every request goes through
    ``_get``, so tests substitute a client without touching the network.

    The API paginates 30 records per page. The cursor is an offset, so it must
    advance by the number of rows actually returned — advancing by a fixed 100
    silently skips 70 % of the corpus (found by live harvest, 2026-11-18).
    """

    BASE = "https://api.biorxiv.org/details"
    UA = "LucidResearch/0.1 (academic corpus build; mailto:research@talontodo.com)"

    def __init__(
        self,
        server: str = "biorxiv",
        timeout: float = 45.0,
        delay: float = 3.0,
        retries: int = 6,
        fulltext: bool = False,
    ) -> None:
        if server not in {"biorxiv", "medrxiv"}:
            raise ValueError("server must be 'biorxiv' or 'medrxiv'")
        self.server = server
        self.timeout = timeout
        self.delay = delay
        self.retries = retries
        self.fulltext = fulltext
        self._last_request = 0.0
        # A 429 is a host-level request to stop, not a per-URL hiccup. Once one
        # arrives, every subsequent request waits out this cooldown.
        self._cooldown_until = 0.0
        # Set by the harvester so a retried window continues mid-interval.
        self.start_cursor = 0
        self.on_page_complete = None

    def _get(self, url: str, *, as_json: bool = False):
        import time
        import urllib.error
        import urllib.request

        last: Exception | None = None
        last_status: int | None = None
        saw_http_transient = False
        for attempt in range(self.retries):
            now = time.monotonic()
            wait = max(
                self.delay - (now - self._last_request),
                self._cooldown_until - now,
            )
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.UA,
                        "Accept": "application/json" if as_json else "application/xml,text/xml,*/*",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    self._last_request = time.monotonic()
                    payload = resp.read()
                    status = getattr(resp, "status", 200)
                if as_json:
                    try:
                        return json.loads(payload.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        snippet = payload[:240].decode("utf-8", errors="replace")
                        if _looks_like_block(snippet):
                            raise HostRefused(url, status, snippet) from exc
                        raise TransientRequestError(
                            url, status, f"not json: {snippet}"
                        ) from exc
                return payload
            except (HostRefused, TransientRequestError):
                raise
            except urllib.error.HTTPError as exc:
                last = exc
                last_status = exc.code
                snippet = _error_snippet(exc)
                if exc.code in {401, 403, 451}:
                    raise HostRefused(url, exc.code, snippet) from exc
                if exc.code == 429:
                    saw_http_transient = True
                    header = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        retry_after = float(header) if header else 0.0
                    except (TypeError, ValueError):
                        retry_after = 0.0
                    # The host said stop. Back off for at least a minute even if
                    # it gave no header; retrying immediately extends the ban.
                    cooldown_for = max(retry_after, 60.0)
                    self._cooldown_until = time.monotonic() + cooldown_for
                    if attempt + 1 < self.retries:
                        print(
                            f"[biorxiv] HTTP 429; cooling down {cooldown_for:.0f}s "
                            f"before retry {attempt + 2}/{self.retries} {url}",
                            flush=True,
                        )
                elif exc.code in {500, 502, 503, 504}:
                    if _looks_like_block(snippet):
                        raise HostRefused(url, exc.code, snippet) from exc
                    saw_http_transient = True
                else:
                    # A per-paper 404 must not be treated as a host-wide block.
                    raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                last_status = None
            if attempt + 1 < self.retries:
                time.sleep(backoff_seconds(attempt, base=2.0, cap=30.0))
        detail = _error_snippet(last) if last is not None else "unknown"
        # Routing, reset, timeout and DNS are properties of the path. They must
        # not abort a multi-hour harvest; the caller records the document and
        # moves on. A 429 that survived every cooldown is a sustained block.
        if last is not None and _is_transient_transport(last):
            raise TransientRequestError(url, last_status, detail) from last
        if saw_http_transient:
            if last_status == 429:
                raise HostRefused(url, last_status, detail) from last
            raise TransientRequestError(url, last_status, detail) from last
        raise HostRefused(url, last_status, detail) from last

    def _metadata(self, row: dict) -> Document:
        text = row.get("abstract") or ""
        return Document(
            doc_id=f"{self.server}:{row.get('doi', '')}",
            source=self.server,
            title=row.get("title", ""),
            text=text,
            sections={"abstract": text} if text else {},
            license=normalise_license(row.get("license")),
            date=(row.get("date") or "1970-01-01")[:10],
            meta={
                "doi": row.get("doi"),
                "version": row.get("version"),
                "jatsxml": row.get("jatsxml"),
                "license_raw": row.get("license"),
                "fulltext": "abstract_only",
            },
        )

    def fetch_fulltext(self, doc: Document) -> Document:
        """Attach JATS full text to an allowed-licence document.

        A failure is recorded on the document rather than raised: one blocked
        paper must not abort a multi-hour harvest. The build can then drop
        abstract-only documents explicitly (``--require-fulltext``).
        """
        url = doc.meta.get("jatsxml")
        if not url:
            doc.meta["fulltext"] = "no_jats_link"
            return doc
        try:
            xml_bytes = self._get(url)
            text, sections = parse_jats(xml_bytes)
        except HostRefused:
            # A refusal is host-wide. Recording it as a per-paper miss would
            # keep the harvest hammering the same block.
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            doc.meta["fulltext"] = f"error:{_failure_label(exc)}"
            return doc
        if not text.strip():
            doc.meta["fulltext"] = "empty"
            return doc
        doc.text = text
        doc.sections = sections
        doc.meta["fulltext"] = "ok"
        doc.meta["fulltext_chars"] = len(text)
        return doc

    def iter_metadata(self, since_date: str, until_date: str):
        """Yield metadata documents page by page, without fetching full text.

        Streaming matters: a weekly window is ~1,400 papers, and waiting for the
        whole window before writing means a failure loses an hour of work. The
        harvester consumes this generator and commits each document immediately.
        """
        cursor = int(self.start_cursor or 0)
        while True:
            url = f"{self.BASE}/{self.server}/{since_date}/{until_date}/{cursor}"
            payload = self._get(url, as_json=True)
            rows = payload.get("collection", [])
            if not rows:
                break
            for row in rows:
                yield self._metadata(row)
            cursor += len(rows)
            self.start_cursor = cursor
            # Runs only once the consumer has pulled every row of this page,
            # and before the next page is requested. A failure on the next
            # page therefore resumes here rather than at cursor 0.
            hook = self.on_page_complete
            if hook is not None:
                hook(cursor)

    def fetch(
        self, since_date: str, until_date: str, max_docs: int = 1000
    ) -> list[Document]:
        out: list[Document] = []
        for doc in self.iter_metadata(since_date, until_date):
            if (
                self.fulltext
                and doc.license in LICENSE_ALLOWLIST
                and doc.meta.get("jatsxml")
            ):
                doc = self.fetch_fulltext(doc)
            out.append(doc)
            if len(out) >= max_docs:
                break
        return out


class FdaLabelClient:
    """openFDA drug-label endpoint, keyed on ``effective_time``.

    US Government work, so the licence is public domain; that is recorded per
    document rather than assumed at the repository level.
    """

    BASE = "https://api.fda.gov/drug/label.json"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def _http_get(self, url: str) -> dict:
        import urllib.request

        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def fetch(self, max_docs: int = 1500) -> list[Document]:
        out: list[Document] = []
        skip = 0
        page = 100
        while len(out) < max_docs:
            url = f"{self.BASE}?limit={page}&skip={skip}"
            try:
                payload = self._http_get(url)
            except Exception:  # noqa: BLE001 - stop on API error, keep what we have
                break
            rows = payload.get("results", [])
            if not rows:
                break
            for row in rows:
                sections = {
                    k: " ".join(v)
                    for k, v in row.items()
                    if k
                    in {
                        "indications_and_usage",
                        "dosage_and_administration",
                        "warnings_and_cautions",
                        "adverse_reactions",
                        "clinical_pharmacology",
                    }
                    and isinstance(v, list)
                }
                text = "\n\n".join(sections.values())
                effective = row.get("effective_time", "")
                iso = (
                    f"{effective[:4]}-{effective[4:6]}-{effective[6:8]}"
                    if len(effective) >= 8
                    else "1970-01-01"
                )
                out.append(
                    Document(
                        doc_id=f"fda:{row.get('id', row.get('set_id', ''))}",
                        source="fda",
                        title=(row.get("openfda", {}).get("brand_name") or [""])[0],
                        text=text,
                        sections=sections,
                        license="public-domain",
                        date=iso,
                        meta={"effective_time": effective},
                    )
                )
                if len(out) >= max_docs:
                    break
            skip += page
        return out
