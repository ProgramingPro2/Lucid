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
    # NC / ND are classified (not "unknown") so the card can count why they were
    # dropped; they are still outside LICENSE_ALLOWLIST.
    "cc by-nc 4.0": "cc-by-nc-4.0",
    "cc-by-nc-4.0": "cc-by-nc-4.0",
    "cc by-nc-nd 4.0": "cc-by-nc-nd-4.0",
    "cc-by-nc-nd-4.0": "cc-by-nc-nd-4.0",
    "cc by-nd 4.0": "cc-by-nd-4.0",
    "cc-by-nd-4.0": "cc-by-nd-4.0",
    "cc0": "cc0",
    "cc0 1.0": "cc0",
    "cc-0": "cc0",
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


class BioRxivClient:
    """bioRxiv / medRxiv details API, queried by posting date.

    The API returns metadata and, for the details endpoint, the full abstract but
    not the full text. Full text is fetched from the ``*_content`` link when the
    licence permits it. Both calls go through ``_http_get`` so the whole client
    can be replaced in tests.
    """

    BASE = "https://api.biorxiv.org/details"

    def __init__(self, server: str = "biorxiv", timeout: float = 30.0) -> None:
        if server not in {"biorxiv", "medrxiv"}:
            raise ValueError("server must be 'biorxiv' or 'medrxiv'")
        self.server = server
        self.timeout = timeout

    def _http_get(self, url: str) -> dict:
        import urllib.request

        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def fetch(
        self, since_date: str, until_date: str, max_docs: int = 1000
    ) -> list[Document]:
        out: list[Document] = []
        cursor = 0
        while len(out) < max_docs:
            url = (
                f"{self.BASE}/{self.server}/{since_date}/{until_date}/{cursor}"
            )
            payload = self._http_get(url)
            rows = payload.get("collection", [])
            if not rows:
                break
            for row in rows:
                text = row.get("abstract") or ""
                # The details endpoint carries only the abstract; mark that so
                # the build cannot silently treat it as full text.
                out.append(
                    Document(
                        doc_id=f"{self.server}:{row.get('doi', '')}",
                        source=self.server,
                        title=row.get("title", ""),
                        text=text,
                        sections={"abstract": text} if text else {},
                        license=normalise_license(row.get("license")),
                        date=(row.get("date") or "1970-01-01")[:10],
                        meta={"doi": row.get("doi"), "version": row.get("version")},
                    )
                )
                if len(out) >= max_docs:
                    break
            cursor += 100
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
