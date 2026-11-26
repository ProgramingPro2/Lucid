"""Sealed FDA second register (docs/CORPUS.md gate G-1f).

The primary corpus is post-cutoff biology preprints. This module fetches a
bounded second register of prescribing-information labels so the secondary
endpoint in docs/PROTOCOL.md §5.3 — cross-register context gain — can be a
paired difference on a genuinely different distribution.

The register is sealed the way the primary split is specified and the way the
primary build has not yet implemented: an 80/10/10 document split, written
once, whose held-out document ids are stored as SHA-256 hashes. A later
training run calls :func:`training_conflict` and must get an empty list.

Network policy matches the bioRxiv harvester. ``OpenFdaLabelClient`` sends
every request through ``BioRxivClient._get``, so the User-Agent, the delay,
the 429 cooldown, and the transport-versus-refusal split are the ones already
tested. A transient failure retries inside that client and then skips the
page. A 401, 403, 451, or a 429 that survived its cooldown aborts the run.
An openFDA 404 means the page has no matches; it ends the harvest.

Nothing here loads a model. Token counts for the real register use the local
Qwen tokenizer only, and only when a caller asks for it.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import random
import re
import sys
import time
import urllib.error
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

from experiments.corpus import harvest as harvest_mod
from experiments.corpus.pipeline import (
    filter_by_date,
    filter_by_license,
    filter_complete_sections,
    split_documents,
)
from experiments.corpus.sources import (
    LICENSE_ALLOWLIST,
    PROGRAMME_CUTOFF,
    BioRxivClient,
    Document,
    HostRefused,
    TransientRequestError,
    normalise_license,
)
from experiments.protocol.stats import mde, required_n

# --------------------------------------------------------------------------
# Licence. Verified 2026-11-26 by reading the URL the live API puts in
# meta.license, plus the terms page that URL cites. Drug-label documentation
# does not carry the third-party warning the terms require for a CC0
# exemption. GMDN is the only named exemption on the licence page, and this
# register does not request it.
# --------------------------------------------------------------------------

LICENSE_URL = "https://open.fda.gov/license/"
TERMS_URL = "https://open.fda.gov/terms/"
LABEL_DOCS_URL = "https://open.fda.gov/apis/drug/label/"
LICENSE_RETRIEVED = "2026-11-26"
LICENSE_EXACT = "Creative Commons CC0 1.0 Universal"
# ``normalise_license("CC0 1.0")`` is ``cc0``, which is on the allowlist.
# The exact dedication above is stored beside it; the machine field has to
# be a string the shared normaliser accepts, or the document is "unknown".
LICENSE_MACHINE = "CC0 1.0"
LICENSE_QUOTE = (
    "As noted in the terms of service, unless otherwise noted, the content, "
    "data, documentation, code, and related materials on openFDA is public "
    "domain and made available with a Creative Commons CC0 1.0 Universal "
    "dedication. Under CC0, FDA has dedicated the work to the public domain "
    "by waiving all rights to the work worldwide under copyright law, "
    "including all related and neighboring rights, to the extent allowed by "
    "law. You can copy, modify, distribute and perform the work, even for "
    "commercial purposes, all without asking permission."
)
LICENSE_EXEMPTION_QUOTE = (
    "In the future, if we post data that may not be covered under the terms "
    "listed above, we will specify the data (or specifc fields inside a "
    "dataset) that is not covered and the copyright status of that work here."
)
# CC0 does not waive a sponsor's own copyright. The terms say such works are
# marked. A per-item mark is recorded as unknown rather than relabelled CC0.
LICENSE_CAVEAT = (
    "FDA's CC0 dedication waives FDA's rights in unmarked openFDA content. "
    "It does not waive patent, trademark, or third-party rights, and the "
    "CC0 considerations on the licence page say so. Prescribing information "
    "is sponsor-submitted SPL (21 CFR 207); the drug-label documentation "
    "says FDA reformats it and does not alter the content. That documentation "
    "does not contain the warning the terms require where CC0 does not apply "
    "('This data is not in the public domain. Third party copy rights may "
    "apply.'). Records that themselves contain a copyright reservation are "
    "not given the CC0 token."
)

_KNOWN_LICENSE_URLS = frozenset(
    {
        "https://open.fda.gov/license/",
        "https://open.fda.gov/license",
        "http://open.fda.gov/license/",
        "http://open.fda.gov/license",
    }
)

# Running prose of a prescribing-information label. ``*_table`` fields and
# product-data elements are tables or inventory and are not read.
PROSE_FIELDS: tuple[str, ...] = (
    "indications_and_usage",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    "boxed_warning",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "clinical_pharmacology",
    "mechanism_of_action",
    "pharmacodynamics",
    "pharmacokinetics",
    "clinical_studies",
    "description",
    "overdosage",
    "information_for_patients",
)

# E1 TrainConfig defaults (experiments/e1/train.py). Imported by value so this
# module does not import the E1 package, whose ``__init__`` loads torch.
EVAL_PREFIX = 384
EVAL_TARGET = 128
# 10% held-out (the primary split) times E1's 64 evaluation items.
USABLE_TARGET = 640
FETCH_CAP = 2000
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_DOCS = 720
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SEED = 20261126
ITEM_SEED = 20261126
# Protocol §5.4 planning values. ``required_n`` is called with these.
MPID = 0.0112
PLANNING_SD = 0.015
PROTOCOL_ALPHA = 0.025
PROTOCOL_POWER = 0.90
PROTOCOL_INFLATION = 1.29

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")
_COPYRIGHT = re.compile(
    r"all rights reserved"
    r"|this data is not in the public domain"
    r"|third party copy ?rights may apply"
    r"|©"
    r"|\bcopyright\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z]{2,}")
_DOSAGE = re.compile(
    r"\b(?:mg|mcg|ml|tablet|tablets|capsule|capsules|contraindicated|dosage)\b",
    re.IGNORECASE,
)
_SCHOLARLY = re.compile(
    r"\bet al\.?|\bp\s*[<=>]|\bhypothesis\b|\bmethods\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "the", "of", "and", "to", "in", "a", "for", "with", "on", "by", "or",
        "as", "is", "are", "be", "this", "that", "from", "at", "an", "it",
        "its", "was", "were", "these", "than", "not", "may", "can", "which",
        "their", "been", "or", "into", "such", "also", "have", "has", "had",
        "but", "we", "our", "they", "them", "his", "her",
    }
)
# Defined here because the primary corpus has no cross-register rule.
# Document-pair content-word Jaccard must clear both the gap between the two
# within-register figures and this absolute margin. Marker rates must separate
# by MARKER_GAP in the expected direction. Disagreement is "mixed".
CONTENT_JACCARD_MARGIN = 0.02
MARKER_GAP = 0.25
_SEAL_BODY_KEYS = (
    "blocked_duplicate_sha256",
    "created",
    "dev_id_sha256",
    "fractions",
    "held_out_split",
    "seed",
    "test_id_sha256",
    "train_id_sha256",
)


class SealError(RuntimeError):
    """The seal file is missing, corrupt, or no longer matches its hash."""


def min_tokens_for_item(prefix: int = EVAL_PREFIX, target: int = EVAL_TARGET) -> int:
    """Tokens an intact document needs before it can host one paired item.

    ``eval_items`` keeps a document when ``len(ids) >= prefix + target + 1``.
    """
    return prefix + target + 1


def id_sha256(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()


def strip_markup(value: str) -> str:
    text = _TAG.sub(" ", value)
    text = html.unescape(text).replace("\xa0", " ")
    text = _WS.sub(" ", text)
    return _NL.sub("\n\n", text).strip()


def copyright_mark(text: str) -> str | None:
    found = _COPYRIGHT.search(text or "")
    if found is None:
        return None
    return found.group(0)


def effective_to_iso(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) < 8:
        return "1970-01-01"
    iso = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    try:
        date.fromisoformat(iso)
    except ValueError:
        return "1970-01-01"
    return iso


def _first(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def _section_text(row: dict, field: str) -> str:
    raw = row.get(field)
    if isinstance(raw, list):
        parts = [part for part in raw if isinstance(part, str)]
        text = "\n\n".join(parts)
    elif isinstance(raw, str):
        text = raw
    else:
        return ""
    return strip_markup(text)


def row_to_document(row: dict, response_meta: dict | None) -> Document:
    """One label version. The licence is the response's, unless the text marks a reservation.

    openFDA does not put a licence field on the result. Every response carries
    ``meta.license``. A URL we have not read is ``unknown``. A copyright
    reservation in the prose is ``unknown`` as well: the repository dedication
    does not outvote the item.
    """
    sections: dict[str, str] = {}
    for field_name in PROSE_FIELDS:
        text = _section_text(row, field_name)
        if text:
            sections[field_name] = text
    body = "\n\n".join(sections[name] for name in PROSE_FIELDS if name in sections)
    meta_in = dict(response_meta or {})
    license_url = str(meta_in.get("license") or "")
    mark = copyright_mark(body)
    if mark:
        license_field = mark
        scope = "per-item-copyright-mark"
        redistributable = False
        quote = ""
    elif license_url in _KNOWN_LICENSE_URLS:
        license_field = LICENSE_MACHINE
        scope = "openfda-response-meta"
        redistributable = True
        quote = LICENSE_QUOTE
    else:
        license_field = "unknown"
        scope = "unverified-response-licence"
        redistributable = False
        quote = ""

    openfda = row.get("openfda") if isinstance(row.get("openfda"), dict) else {}
    product_type = _first(openfda.get("product_type")) or "unspecified"
    label_id = str(row.get("id") or "")
    set_id = str(row.get("set_id") or "")
    effective = str(row.get("effective_time") or "")
    if label_id:
        doc_id = f"fda:{label_id}"
    else:
        doc_id = f"fda:{set_id}:{effective or 'undated'}"
    title = _first(openfda.get("brand_name")) or _first(openfda.get("generic_name"))
    try:
        version = int(row.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    return Document(
        doc_id=doc_id,
        source="fda",
        title=title,
        text=body,
        sections=sections,
        license=license_field,
        date=effective_to_iso(effective),
        meta={
            "set_id": set_id,
            "version": version,
            "effective_time": effective,
            "product_type": product_type,
            "stratum": product_type,
            "application_number": _first(openfda.get("application_number")),
            "license_url": license_url,
            "license_terms_url": str(meta_in.get("terms") or ""),
            "license_exact": LICENSE_EXACT if redistributable else license_field,
            "license_quote": quote,
            "license_scope": scope,
            "license_retrieved": LICENSE_RETRIEVED,
            "redistributable": redistributable,
            "fulltext": "ok" if body else "empty",
            "api_last_updated": meta_in.get("last_updated"),
        },
    )


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class OpenFdaLabelClient:
    """Paged openFDA drug-label client.

    ``fetch_page`` is the only method the harvester calls, and ``_transport._get``
    is the only method that touches the network. Tests replace ``fetch_page``
    or patch ``urlopen``.
    """

    BASE = "https://api.fda.gov/drug/label.json"
    UA = BioRxivClient.UA

    def __init__(
        self,
        *,
        delay: float = 3.0,
        retries: int = 6,
        timeout: float = 120.0,
    ) -> None:
        self._transport = BioRxivClient(delay=delay, retries=retries, timeout=timeout)

    def fetch_page(self, search: str, skip: int, limit: int) -> dict:
        url = self.BASE + "?" + _query({"search": search, "skip": skip, "limit": limit})
        try:
            payload = self._transport._get(url, as_json=True)
        except urllib.error.HTTPError as exc:
            # openFDA uses 404 for "No matches found", including a skip past
            # the end. That is the end of the register, not a refusal.
            if exc.code == 404:
                return {"results": [], "meta": {"http_status": 404}}
            raise
        if not isinstance(payload, dict):
            raise TransientRequestError(url, 200, "response was not a JSON object")
        if "results" not in payload and payload.get("error"):
            detail = str(payload.get("error"))[:240]
            raise TransientRequestError(url, None, detail)
        return payload


def _query(params: dict) -> str:
    from urllib.parse import urlencode

    return urlencode({key: str(value) for key, value in params.items()})


def build_search(since: date, until: date) -> str:
    """Prescribing-information labels whose ``effective_time`` is inside the window.

    ``clinical_pharmacology`` is required so the sample is full prescribing
    information rather than a short listing. The window is the 2026 label
    versions docs/CORPUS.md names.
    """
    return (
        f"effective_time:[{since.strftime('%Y%m%d')} TO {until.strftime('%Y%m%d')}] "
        "AND _exists_:indications_and_usage AND _exists_:clinical_pharmacology"
    )


# --------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------


def harvest_sidecar(out: Path) -> Path:
    return out.with_name(out.name + ".harvest.json")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def resolve_search(out: Path, since: date, until: date, resume: bool) -> str:
    """Keep the query stable across a resume. A new day must not move the cursor."""
    if resume:
        stored = _load_json(harvest_sidecar(out)).get("search")
        if isinstance(stored, str) and stored.strip():
            return stored
    return build_search(since, until)


def _empty_counts() -> dict:
    return {
        "fetched": 0,
        "written": 0,
        "skipped_existing": 0,
        "pages": 0,
        "pages_skipped": 0,
        "page_failures": [],
        "licence_counts": {},
        "license_url": None,
        "terms_url": None,
        "api_last_updated": None,
        "disclaimer": None,
        "search": "",
        "end": None,
    }


def harvest_fda(
    *,
    out: Path,
    search: str,
    max_docs: int,
    page_size: int = DEFAULT_PAGE_SIZE,
    delay: float = 3.0,
    retries: int = 6,
    resume: bool = True,
    client: OpenFdaLabelClient | None = None,
    max_consecutive_transient: int = 3,
) -> dict:
    """Append labels to ``out``. The cursor sidecar is the next ``skip`` to request.

    The cursor advances by the number of rows consumed, not by a fixed page
    size. A transient page is retried inside the client and then skipped; the
    failure is recorded. ``max_consecutive_transient`` is the budget for a
    dead path: past it the run aborts instead of walking the whole register
    as holes. A refusal aborts immediately and does not advance the cursor.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    client = client or OpenFdaLabelClient(delay=delay, retries=retries)
    done = harvest_mod._load_done(out) if resume else set()
    cursors_path = harvest_mod.cursor_path_for(out)
    cursors = harvest_mod._load_cursors(cursors_path) if resume else {}
    key = "openfda|" + hashlib.sha256(search.encode("utf-8")).hexdigest()[:16]
    skip = int(cursors.get(key, 0) or 0)
    previous = _load_json(harvest_sidecar(out)) if resume else {}
    counts = _empty_counts()
    counts["search"] = search
    counts["pages_skipped"] = int(previous.get("pages_skipped") or 0)
    prior_failures = previous.get("page_failures")
    if isinstance(prior_failures, list):
        counts["page_failures"] = list(prior_failures)[:50]
    consecutive = 0
    print(
        f"[fda] delay={delay}s retries={retries} page={page_size} "
        f"ua={OpenFdaLabelClient.UA} skip={skip}",
        flush=True,
    )
    started = time.time()
    with out.open("a", encoding="utf-8") as fh:
        while counts["written"] < max_docs:
            try:
                payload = client.fetch_page(search, skip, page_size)
            except HostRefused as exc:
                counts["aborted"] = "host_refused"
                counts["abort_detail"] = str(exc)[:400]
                print(f"[fda] host refused; stopping: {exc}", flush=True)
                break
            except TransientRequestError as exc:
                consecutive += 1
                counts["pages_skipped"] += 1
                counts["page_failures"].append({"skip": skip, "error": str(exc)[:300]})
                counts["page_failures"] = counts["page_failures"][-50:]
                print(
                    f"[fda] transient page at skip {skip}; skipping it "
                    f"({consecutive} consecutive): {exc}",
                    flush=True,
                )
                if consecutive >= max_consecutive_transient:
                    counts["aborted"] = "transient_budget"
                    counts["abort_detail"] = str(exc)[:400]
                    print("[fda] transient budget exhausted; stopping", flush=True)
                    break
                skip += page_size
                cursors[key] = skip
                if resume:
                    harvest_mod._flush_cursors(cursors_path, cursors)
                continue
            except Exception as exc:  # noqa: BLE001 - recorded, then we stop
                counts["aborted"] = "unclassified"
                counts["abort_detail"] = f"{type(exc).__name__}: {exc}"[:400]
                print(f"[fda] unclassified failure; stopping: {exc}", flush=True)
                break

            consecutive = 0
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            if counts["license_url"] is None and meta.get("license"):
                counts["license_url"] = meta.get("license")
                counts["terms_url"] = meta.get("terms")
                counts["api_last_updated"] = meta.get("last_updated")
                counts["disclaimer"] = meta.get("disclaimer")
            rows = payload.get("results") or []
            counts["pages"] += 1
            if not rows:
                counts["end"] = "empty"
                cursors[key] = skip
                if resume:
                    harvest_mod._flush_cursors(cursors_path, cursors)
                break
            consumed = 0
            for row in rows:
                consumed += 1
                if not isinstance(row, dict):
                    continue
                if counts["written"] >= max_docs:
                    consumed -= 1
                    break
                doc = row_to_document(row, meta)
                counts["fetched"] += 1
                if doc.doc_id in done:
                    counts["skipped_existing"] += 1
                    continue
                fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
                fh.flush()
                done.add(doc.doc_id)
                counts["written"] += 1
                canon = normalise_license(doc.license)
                bucket = counts["licence_counts"]
                bucket[canon] = bucket.get(canon, 0) + 1
                if counts["written"] % 100 == 0:
                    elapsed = max(1e-9, time.time() - started)
                    print(
                        f"[fda] written={counts['written']} "
                        f"{counts['written'] / elapsed * 60:.1f}/min",
                        flush=True,
                    )
            skip += consumed
            cursors[key] = skip
            if resume:
                harvest_mod._flush_cursors(cursors_path, cursors)
            if consumed < len(rows):
                break
    _atomic_json(harvest_sidecar(out), counts)
    return counts


def load_register(path: str | Path) -> list[Document]:
    """Read the JSONL. A torn final line is skipped; an earlier bad line raises.

    This is ``harvest.load_documents``. The final-line rule is the one the
    bioRxiv harvest learned from a kill mid-flush.
    """
    return harvest_mod.load_documents(path)


# --------------------------------------------------------------------------
# Retain, dedupe, seal
# --------------------------------------------------------------------------


def _version_key(doc: Document) -> tuple:
    try:
        version = int(doc.meta.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    return (doc.date, version, doc.doc_id)


def dedupe_labels(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """One document per ``set_id``: the latest ``effective_time``, then version.

    Other versions of a sealed ``set_id`` are blocked from training. They are
    near-copies, and training on them would leak the held-out label.
    """
    best: dict[str, Document] = {}
    for doc in docs:
        key = str(doc.meta.get("set_id") or doc.doc_id)
        current = best.get(key)
        if current is None or _version_key(doc) > _version_key(current):
            best[key] = doc
    kept_ids = {doc.doc_id for doc in best.values()}
    dropped = [doc for doc in docs if doc.doc_id not in kept_ids]
    kept = sorted(best.values(), key=lambda doc: doc.doc_id)
    return kept, dropped


def _canonical_seal_bytes(body: dict) -> bytes:
    payload = {key: body[key] for key in _SEAL_BODY_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_sha256(body: dict) -> str:
    return hashlib.sha256(_canonical_seal_bytes(body)).hexdigest()


def verify_seal(payload: dict) -> None:
    missing = [key for key in _SEAL_BODY_KEYS if key not in payload]
    if missing or "seal_sha256" not in payload:
        raise SealError(f"seal is missing keys: {missing or ['seal_sha256']}")
    expected = seal_sha256(payload)
    if expected != payload["seal_sha256"]:
        raise SealError("seal hash does not match the held-out id list")
    parts = [
        payload["train_id_sha256"],
        payload["dev_id_sha256"],
        payload["test_id_sha256"],
    ]
    flat = [item for part in parts for item in part]
    if len(flat) != len(set(flat)):
        raise SealError("a document hash appears in more than one split")


def _split_hashes(docs: list[Document], seed: int) -> dict[str, list[str]]:
    splits = split_documents(docs, SPLIT_FRACTIONS, seed=seed)
    return {
        name: sorted(id_sha256(doc_id) for doc_id in splits[name])
        for name in ("train", "dev", "test")
    }


def _blocked_duplicate_hashes(
    dropped: list[Document], kept: list[Document], test_hashes: set[str]
) -> list[str]:
    set_of_hash: dict[str, str] = {}
    for doc in kept:
        set_of_hash[str(doc.meta.get("set_id") or doc.doc_id)] = id_sha256(doc.doc_id)
    blocked: list[str] = []
    for doc in dropped:
        key = str(doc.meta.get("set_id") or doc.doc_id)
        if set_of_hash.get(key) in test_hashes:
            blocked.append(id_sha256(doc.doc_id))
    return sorted(set(blocked))


def write_seal_once(
    docs: list[Document],
    dropped: list[Document],
    path: Path,
    *,
    seed: int = SPLIT_SEED,
    created: str = LICENSE_RETRIEVED,
) -> tuple[dict, bool]:
    """Write the seal if it is absent. An existing seal is verified and returned.

    The file is never rewritten. A corrupt existing file raises :class:`SealError`
    rather than being replaced: redrawing the held-out split is the failure
    mode docs/CORPUS.md §8 forbids.
    """
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_seal(existing)
        return existing, False
    if len(docs) < 2:
        raise SealError("need at least two retained documents to seal a split")
    hashes = _split_hashes(docs, seed)
    body = {
        "blocked_duplicate_sha256": _blocked_duplicate_hashes(
            dropped, docs, set(hashes["test"])
        ),
        "created": created,
        "dev_id_sha256": hashes["dev"],
        "fractions": list(SPLIT_FRACTIONS),
        "held_out_split": "test",
        "seed": seed,
        "test_id_sha256": hashes["test"],
        "train_id_sha256": hashes["train"],
    }
    covered = set(hashes["train"]) | set(hashes["dev"]) | set(hashes["test"])
    if covered != {id_sha256(doc.doc_id) for doc in docs}:
        raise SealError("split does not cover the retained documents")
    payload = dict(body)
    payload["seal_sha256"] = seal_sha256(body)
    payload["n_retained"] = len(docs)
    payload["n_held_out"] = len(hashes["test"])
    payload["n_blocked_duplicates"] = len(body["blocked_duplicate_sha256"])
    verify_seal(payload)
    _atomic_json(path, payload)
    return payload, True


def load_seal(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_seal(payload)
    return payload


def training_conflict(train_ids: list[str], seal: dict) -> list[str]:
    """Document ids that a training run must not see.

    The banned set is the held-out split plus near-duplicate versions of those
    labels. Dev is not in this set: it may be used for tuning, and it is not
    the one-look split.
    """
    verify_seal(seal)
    banned = set(seal["test_id_sha256"]) | set(seal["blocked_duplicate_sha256"])
    return [doc_id for doc_id in train_ids if id_sha256(doc_id) in banned]


def _docs_in(docs: list[Document], hashes: list[str]) -> list[Document]:
    wanted = set(hashes)
    chosen = [doc for doc in docs if id_sha256(doc.doc_id) in wanted]
    return sorted(chosen, key=lambda doc: doc.doc_id)


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------


def _e1_data():
    """Load ``experiments/e1/data.py`` without importing ``experiments.e1``.

    The package ``__init__`` imports the JEPA block, which imports torch. This
    register must not touch the GPU, and item construction only needs
    ``PackedCorpus.eval_items``.
    """
    name = "lucid_e1_data_standalone"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "e1" / "data.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _find_donor(prefix: list[int], token_docs: list[list[int]], own: int) -> int | None:
    if not prefix:
        return None
    first = prefix[0]
    width = len(prefix)
    for index, doc in enumerate(token_docs):
        if index == own:
            continue
        start = 0
        while True:
            try:
                found = doc.index(first, start)
            except ValueError:
                break
            if doc[found : found + width] == prefix:
                return index
            start = found + 1
    return None


def build_cross_register_items(
    sealed_docs: list[Document],
    id_by_doc: dict[str, list[int]],
    *,
    prefix: int = EVAL_PREFIX,
    target: int = EVAL_TARGET,
    seed: int = ITEM_SEED,
) -> dict:
    """True-context versus shuffled-context pairs on the sealed documents.

    One item per eligible document. The target span is the same token ids in
    both conditions; the prefix is a contiguous span from a different sealed
    document. That identity is what makes the difference a use-of-context
    measure. The shuffled prefix is checked, not assumed: a donor that cannot
    be found in another document fails the report.
    """
    minimum = min_tokens_for_item(prefix, target)
    eligible_docs: list[Document] = []
    eligible_ids: list[list[int]] = []
    for doc in sorted(sealed_docs, key=lambda item: item.doc_id):
        ids = id_by_doc.get(doc.doc_id)
        if ids is None:
            continue
        if len(ids) >= minimum:
            eligible_docs.append(doc)
            eligible_ids.append(list(ids))
    report: dict = {
        "prefix": prefix,
        "target": target,
        "seed": seed,
        "n_sealed_documents": len(sealed_docs),
        "n_eligible": len(eligible_docs),
        "n_items": 0,
        "one_item_per_eligible_document": False,
        "targets_identical": False,
        "shuffled_prefix_document_disjoint": False,
        "nll_computed": False,
        "items": [],
    }
    if len(eligible_docs) < 2:
        report["error"] = "need at least two sealed documents long enough for an item"
        return report
    packed = _e1_data().PackedCorpus(eligible_ids)
    items = packed.eval_items(len(eligible_docs), prefix=prefix, target=target, seed=seed)
    rows = []
    targets_ok = True
    disjoint_ok = True
    seen_docs: set[int] = set()
    for item in items:
        target_ok = item.true_ids[item.target_start :] == item.shuffled_ids[item.target_start :]
        targets_ok = targets_ok and target_ok and len(item.true_ids) == len(item.shuffled_ids)
        donor = _find_donor(
            list(item.shuffled_ids[: item.target_start]),
            eligible_ids,
            item.doc_index,
        )
        if donor is None:
            disjoint_ok = False
        seen_docs.add(item.doc_index)
        doc = eligible_docs[item.doc_index]
        rows.append(
            {
                "doc_id": doc.doc_id,
                "doc_index": item.doc_index,
                "target_start": item.target_start,
                "n_tokens": len(item.true_ids),
                "donor_doc_id": None if donor is None else eligible_docs[donor].doc_id,
                "target_identical": target_ok,
            }
        )
    report["items"] = rows
    report["n_items"] = len(rows)
    report["one_item_per_eligible_document"] = seen_docs == set(range(len(eligible_docs))) and len(rows) == len(eligible_docs)
    report["targets_identical"] = targets_ok and len(rows) > 0
    report["shuffled_prefix_document_disjoint"] = disjoint_ok and len(rows) > 0
    return report


# --------------------------------------------------------------------------
# Distributional contrast
# --------------------------------------------------------------------------


def _words(text: str, *, drop_stops: bool) -> list[str]:
    tokens = _WORD.findall(text.lower())
    if drop_stops:
        return [token for token in tokens if token not in _STOPWORDS]
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return float("nan")
    union = len(left | right)
    if union == 0:
        return float("nan")
    return len(left & right) / union


def _mean_pairwise(sets: list[set[str]], rng: random.Random, n_pairs: int) -> float:
    n = len(sets)
    if n < 2:
        return float("nan")
    vals: list[float] = []
    for _ in range(n_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        vals.append(_jaccard(sets[i], sets[j]))
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _mean_cross(left: list[set[str]], right: list[set[str]], rng: random.Random, n_pairs: int) -> float:
    if not left or not right:
        return float("nan")
    vals = [
        _jaccard(left[rng.randrange(len(left))], right[rng.randrange(len(right))])
        for _ in range(n_pairs)
    ]
    return sum(vals) / len(vals)


def _marker_rate(texts: list[str], pattern: re.Pattern[str]) -> float:
    if not texts:
        return float("nan")
    return sum(1 for text in texts if pattern.search(text)) / len(texts)


def lexical_rule(within_fda: float, within_primary: float, cross: float) -> str:
    if not all(math.isfinite(value) for value in (within_fda, within_primary, cross)):
        return "insufficient"
    gap = abs(within_fda - within_primary)
    margin = min(within_fda, within_primary) - cross
    if margin > gap and margin > CONTENT_JACCARD_MARGIN:
        return "different"
    return "not-different"


def marker_rule(dosage_gap: float, scholarly_gap: float) -> str:
    if not (math.isfinite(dosage_gap) and math.isfinite(scholarly_gap)):
        return "insufficient"
    if dosage_gap > MARKER_GAP and scholarly_gap > MARKER_GAP:
        return "different"
    if dosage_gap < 0.1 and scholarly_gap < 0.1:
        return "not-different"
    return "mixed"


def contrast_verdict(lexical: str, markers: str) -> str:
    if lexical == "insufficient" or markers == "insufficient":
        return "insufficient"
    if lexical == "different" and markers == "different":
        return "different"
    if lexical == "not-different" and markers == "not-different":
        return "not-different"
    return "mixed"


def _length_pair(texts: list[str]) -> dict:
    if not texts:
        return {"n": 0, "mean_chars": None, "mean_words": None, "median_words": None}
    chars = [len(text) for text in texts]
    word_counts = [len(_words(text, drop_stops=False)) for text in texts]
    ordered = sorted(word_counts)
    mid = ordered[len(ordered) // 2]
    return {
        "n": len(texts),
        "mean_chars": sum(chars) / len(chars),
        "mean_words": sum(word_counts) / len(word_counts),
        "median_words": mid,
    }


def _iter_primary(path: Path):
    """Full-text primary documents. A torn final line is skipped."""
    with path.open(encoding="utf-8") as fh:
        previous: str | None = None
        for line in fh:
            if previous is not None:
                yield from _primary_line(previous, final=False)
            previous = line
        if previous is not None:
            yield from _primary_line(previous, final=True)


def _primary_line(line: str, *, final: bool):
    stripped = line.strip()
    if not stripped:
        return
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        if final:
            return
        raise
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if meta.get("fulltext") != "ok":
        return
    text = payload.get("text") or ""
    if text:
        yield text


def register_contrast(
    fda_texts: list[str],
    primary_jsonl: str | Path | None,
    *,
    seed: int = 0,
    pair_samples: int = 200,
    n_pairs: int = 2000,
) -> dict:
    """Lexical contrast. ``fda_texts`` must already exclude the sealed split.

    MiniLM in ``experiments/pivot/corpus_stats.py`` is a model, so it is not
    used. The same question — is within-register similarity higher than
    cross-register similarity — is asked with word sets.
    """
    fda_sets = [set(_words(text, drop_stops=True)) for text in fda_texts]
    fda_vocab: set[str] = set()
    for tokens in fda_sets:
        fda_vocab |= tokens
    report: dict = {
        "fda_documents_compared": len(fda_texts),
        "fda_lengths": _length_pair(fda_texts),
        "sealed_documents_excluded": True,
        "primary_available": False,
    }
    if primary_jsonl is None or not Path(primary_jsonl).exists():
        report["verdict"] = "insufficient"
        report["reason"] = "primary corpus jsonl is not available"
        return report

    rng = random.Random(seed)
    primary_vocab: set[str] = set()
    primary_lengths_words: list[int] = []
    primary_lengths_chars: list[int] = []
    reservoir: list[set[str]] = []
    reservoir_texts: list[str] = []
    seen = 0
    for text in _iter_primary(Path(primary_jsonl)):
        tokens = _words(text, drop_stops=True)
        primary_vocab |= set(tokens)
        primary_lengths_words.append(len(_words(text, drop_stops=False)))
        primary_lengths_chars.append(len(text))
        seen += 1
        if len(reservoir) < pair_samples:
            reservoir.append(set(tokens))
            reservoir_texts.append(text)
        else:
            slot = rng.randrange(seen)
            if slot < pair_samples:
                reservoir[slot] = set(tokens)
                reservoir_texts[slot] = text
    if seen == 0:
        report["verdict"] = "insufficient"
        report["reason"] = "primary corpus has no full-text documents"
        return report

    fda_sample = fda_sets
    if len(fda_sample) > pair_samples:
        fda_sample = [fda_sets[i] for i in rng.sample(range(len(fda_sets)), pair_samples)]
    within_fda = _mean_pairwise(fda_sample, rng, n_pairs)
    within_primary = _mean_pairwise(reservoir, rng, n_pairs)
    cross = _mean_cross(fda_sample, reservoir, rng, n_pairs)
    lexical = lexical_rule(within_fda, within_primary, cross)
    dosage_fda = _marker_rate(fda_texts, _DOSAGE)
    dosage_primary = _marker_rate(reservoir_texts, _DOSAGE)
    scholarly_fda = _marker_rate(fda_texts, _SCHOLARLY)
    scholarly_primary = _marker_rate(reservoir_texts, _SCHOLARLY)
    markers = marker_rule(dosage_fda - dosage_primary, scholarly_primary - scholarly_fda)
    intersection = len(fda_vocab & primary_vocab)
    report.update(
        {
            "primary_available": True,
            "primary_fulltext_documents": seen,
            "primary_lengths": {
                "n": seen,
                "mean_chars": sum(primary_lengths_chars) / seen,
                "mean_words": sum(primary_lengths_words) / seen,
                "median_words": sorted(primary_lengths_words)[seen // 2],
            },
            "vocabulary": {
                "fda_types": len(fda_vocab),
                "primary_types": len(primary_vocab),
                "intersection": intersection,
                "jaccard": _jaccard(fda_vocab, primary_vocab),
                "fda_types_in_primary": intersection / len(fda_vocab) if fda_vocab else None,
                "primary_types_in_fda": intersection / len(primary_vocab) if primary_vocab else None,
            },
            "document_jaccard_content_words": {
                "within_fda": within_fda,
                "within_primary": within_primary,
                "cross": cross,
                "n_pairs": n_pairs,
                "pair_sample": pair_samples,
            },
            "markers": {
                "dosage_rate_fda": dosage_fda,
                "dosage_rate_primary": dosage_primary,
                "scholarly_rate_fda": scholarly_fda,
                "scholarly_rate_primary": scholarly_primary,
            },
            "lexical_rule": lexical,
            "marker_rule": markers,
            "verdict": contrast_verdict(lexical, markers),
            "rule": (
                "Content-word document-pair Jaccard (stoplist removed) is "
                "'different' when the cross-register mean is below both "
                "within-register means by more than the within-register gap "
                f"and by more than {CONTENT_JACCARD_MARGIN}. Marker rates are "
                "'different' when the FDA dosage rate exceeds the primary rate "
                f"by more than {MARKER_GAP} and the primary scholarly rate "
                "exceeds the FDA rate by the same margin. The verdict is "
                "'different' only when both rules agree."
            ),
        }
    )
    return report


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def paired_difference_report(n_items: int) -> dict:
    """Whether ``n_items`` can carry a paired difference, via ``stats.required_n``.

    The protocol's planning SD is a seed-level SD: each seed already averages
    ``EVAL_PREFIX`` / the 64-item evaluation. ``protocol_required_n`` is that
    seed count. The item-level rows treat each sealed item as one paired
    observation and do not apply the 10-seed inflation or the seed cap.
    Outcomes were not measured, so this is a sample-size statement, not a CI
    on a context-gain estimate.
    """
    protocol_n = required_n(
        PLANNING_SD,
        MPID,
        alpha=PROTOCOL_ALPHA,
        power=PROTOCOL_POWER,
        pilot_inflation=PROTOCOL_INFLATION,
        floor=24,
        cap=64,
    )
    item_rows = []
    for sd in (0.02, 0.05, 0.10, 0.20, 0.50):
        need = required_n(
            sd,
            MPID,
            alpha=PROTOCOL_ALPHA,
            power=PROTOCOL_POWER,
            pilot_inflation=1.0,
            floor=2,
        )
        item_rows.append(
            {
                "sd": sd,
                "required_n": need,
                "mde_at_n": None if n_items < 2 else mde(sd, n_items, alpha=PROTOCOL_ALPHA, power=PROTOCOL_POWER),
                "n_meets_required": n_items >= need,
            }
        )
    return {
        "n_items": n_items,
        "mpid": MPID,
        "alpha": PROTOCOL_ALPHA,
        "power": PROTOCOL_POWER,
        "protocol_planning_sd": PLANNING_SD,
        "protocol_pilot_inflation": PROTOCOL_INFLATION,
        "protocol_required_n_at_planning_sd": protocol_n,
        "e1_eval_items": 64,
        "sealed_items_cover_one_e1_eval": n_items >= 64,
        "item_level": item_rows,
        "outcomes_measured": False,
    }


def _quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarise_lengths(lengths: list[int], minimum: int) -> dict:
    if not lengths:
        return {"n": 0, "min": None, "median": None, "p10": None, "p90": None, "max": None, "n_at_least_item": 0}
    return {
        "n": len(lengths),
        "min": min(lengths),
        "median": _quantile(lengths, 0.5),
        "p10": _quantile(lengths, 0.10),
        "p90": _quantile(lengths, 0.90),
        "max": max(lengths),
        "n_at_least_item": sum(1 for value in lengths if value >= minimum),
    }


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------


def _select(docs: list[Document], id_by_doc: dict[str, list[int]], minimum: int) -> dict:
    licence_raw = Counter(doc.license for doc in docs)
    licence_canonical = Counter(normalise_license(doc.license) for doc in docs)
    dated = filter_by_date(docs, PROGRAMME_CUTOFF)
    licensed, licence_dropped = filter_by_license(dated, LICENSE_ALLOWLIST)
    complete, incomplete = filter_complete_sections(licensed)
    deduped, duplicates = dedupe_labels(complete)
    usable = [doc for doc in deduped if len(id_by_doc.get(doc.doc_id, ())) >= minimum]
    short = [doc for doc in deduped if doc.doc_id not in {item.doc_id for item in usable}]
    return {
        "licence_raw": dict(licence_raw),
        "licence_canonical": dict(licence_canonical),
        "n_date_dropped": len(docs) - len(dated),
        "n_licence_dropped": len(licence_dropped),
        "n_incomplete": len(incomplete),
        "n_duplicates": len(duplicates),
        "duplicates": duplicates,
        "usable": usable,
        "n_short": len(short),
    }


def assemble_register(
    jsonl: str | Path,
    *,
    card_path: str | Path,
    seal_path: str | Path,
    items_path: str | Path,
    primary_jsonl: str | Path | None = None,
    tokenize=None,
    tokenize_many=None,
    prefix: int = EVAL_PREFIX,
    target: int = EVAL_TARGET,
    seed: int = SPLIT_SEED,
    item_seed: int = ITEM_SEED,
    created: str = LICENSE_RETRIEVED,
) -> dict:
    """Seal the retained register, build items once, and write the card.

    Contrast uses the train split only. The sealed texts are read to count
    tokens and to build the item index; the index stores ids and offsets, and
    ``nll_computed`` stays false.
    """
    docs = load_register(jsonl)
    texts = [doc.text for doc in docs]
    if tokenize_many is not None:
        id_lists = tokenize_many(texts)
    elif tokenize is not None:
        id_lists = [list(tokenize(text)) for text in texts]
    else:
        id_lists = qwen_tokenize_many(texts)
    if len(id_lists) != len(docs):
        raise RuntimeError("tokenizer returned a different number of documents")
    id_by_doc = {doc.doc_id: ids for doc, ids in zip(docs, id_lists)}
    minimum = min_tokens_for_item(prefix, target)
    selected = _select(docs, id_by_doc, minimum)
    usable: list[Document] = selected["usable"]
    duplicates: list[Document] = selected["duplicates"]
    seal_path = Path(seal_path)
    items_path = Path(items_path)
    seal = None
    seal_created = False
    if seal_path.exists():
        seal = load_seal(seal_path)
        seal_created = False
    elif len(usable) >= 2:
        seal, seal_created = write_seal_once(
            usable, duplicates, seal_path, seed=seed, created=created
        )
    items_report: dict
    if seal is None:
        items_report = {"n_items": 0, "error": "no seal", "nll_computed": False}
    elif items_path.exists():
        items_report = json.loads(items_path.read_text(encoding="utf-8"))
        items_report["reused"] = True
    else:
        sealed_docs = _docs_in(usable, seal["test_id_sha256"])
        items_report = build_cross_register_items(
            sealed_docs, id_by_doc, prefix=prefix, target=target, seed=item_seed
        )
        _atomic_json(items_path, items_report)
        items_report["reused"] = False
    train_docs: list[Document] = []
    conflicts: list[str] = []
    if seal is not None:
        train_docs = _docs_in(usable, seal["train_id_sha256"])
        conflicts = training_conflict([doc.doc_id for doc in train_docs], seal)
    contrast = register_contrast(
        [doc.text for doc in train_docs],
        primary_jsonl,
        seed=seed,
    )
    harvest_meta = _load_json(harvest_sidecar(Path(jsonl)))
    n_items = int(items_report.get("n_items") or 0)
    power = paired_difference_report(n_items)
    distinct_ids = len({doc.doc_id for doc in docs})
    redistributable = sum(1 for doc in docs if doc.meta.get("redistributable") is True)
    card = {
        "gate": "G-1f",
        "source": "openFDA drug/label",
        "endpoint": OpenFdaLabelClient.BASE,
        "label_docs": LABEL_DOCS_URL,
        "search": harvest_meta.get("search"),
        "retrieved": created,
        "programme_cutoff": PROGRAMME_CUTOFF,
        "licence": {
            "exact": LICENSE_EXACT,
            "machine_field": LICENSE_MACHINE,
            "canonical": normalise_license(LICENSE_MACHINE),
            "url": LICENSE_URL,
            "terms_url": TERMS_URL,
            "retrieved": LICENSE_RETRIEVED,
            "quote": LICENSE_QUOTE,
            "exemption_quote": LICENSE_EXEMPTION_QUOTE,
            "caveat": LICENSE_CAVEAT,
            "api_meta_license_url": harvest_meta.get("license_url"),
            "api_meta_terms_url": harvest_meta.get("terms_url"),
            "api_last_updated": harvest_meta.get("api_last_updated"),
            "api_disclaimer": harvest_meta.get("disclaimer"),
            "redistributable_when_unmarked": True,
            "per_item_field": False,
            "redistributable_documents": redistributable,
            "not_redistributable_documents": len(docs) - redistributable,
        },
        "documents_fetched": len(docs),
        "distinct_ids": distinct_ids,
        "duplicate_id_rows": len(docs) - distinct_ids,
        "licence_raw": selected["licence_raw"],
        "licence_canonical": selected["licence_canonical"],
        "drops": {
            "date": selected["n_date_dropped"],
            "licence": selected["n_licence_dropped"],
            "incomplete": selected["n_incomplete"],
            "near_duplicate_set_id": selected["n_duplicates"],
            "shorter_than_item": selected["n_short"],
        },
        "documents_usable": len(usable),
        "token_length": summarise_lengths([len(ids) for ids in id_lists], minimum),
        "token_length_unit": "qwen3_tokens_plus_eos" if tokenize is None and tokenize_many is None else "caller_tokenizer",
        "network": {
            "aborted": harvest_meta.get("aborted"),
            "abort_detail": harvest_meta.get("abort_detail"),
            "pages": harvest_meta.get("pages"),
            "pages_skipped": harvest_meta.get("pages_skipped"),
            "page_failures": harvest_meta.get("page_failures") or [],
            "end": harvest_meta.get("end"),
            "fetched_rows_this_process": harvest_meta.get("fetched"),
            "written_this_process": harvest_meta.get("written"),
        },
        "seal": None
        if seal is None
        else {
            "path": str(seal_path),
            "created_this_call": seal_created,
            "seal_sha256": seal["seal_sha256"],
            "created": seal["created"],
            "seed": seal["seed"],
            "fractions": seal["fractions"],
            "n_held_out": len(seal["test_id_sha256"]),
            "n_dev": len(seal["dev_id_sha256"]),
            "n_train": len(seal["train_id_sha256"]),
            "n_blocked_duplicates": len(seal["blocked_duplicate_sha256"]),
            "training_conflicts": conflicts,
        },
        "items": {
            "n_items": n_items,
            "n_eligible": items_report.get("n_eligible"),
            "n_sealed_documents": items_report.get("n_sealed_documents"),
            "one_item_per_eligible_document": items_report.get("one_item_per_eligible_document"),
            "targets_identical": items_report.get("targets_identical"),
            "shuffled_prefix_document_disjoint": items_report.get("shuffled_prefix_document_disjoint"),
            "nll_computed": False,
            "reused": items_report.get("reused", False),
            "error": items_report.get("error"),
            "prefix": items_report.get("prefix", prefix),
            "target": items_report.get("target", target),
            "seed": items_report.get("seed", item_seed),
        },
        "paired_difference": power,
        "contrast": contrast,
        "data_path": str(jsonl),
        "published_in_git": False,
        "gitignore": "artifacts/",
    }
    mechanics = (
        seal is not None
        and not conflicts
        and card["items"]["targets_identical"] is True
        and card["items"]["shuffled_prefix_document_disjoint"] is True
        and card["items"]["one_item_per_eligible_document"] is True
    )
    card["g1f_mechanics_ok"] = mechanics
    card["g1f_endpoint_items"] = n_items >= 64
    _atomic_json(Path(card_path), card)
    return card


# --------------------------------------------------------------------------
# Tokenizer. Local files only. Not a model load.
# --------------------------------------------------------------------------


_TOKENIZER = None
_TOKENIZER_DIR_CANDIDATES = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    Path("/mnt/keepstar-extra/hf-cache/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
)


def qwen_tokenizer_dir() -> Path:
    for path in _TOKENIZER_DIR_CANDIDATES:
        if (path / "tokenizer.json").exists():
            return path
    raise FileNotFoundError("local Qwen3-1.7B-Base tokenizer snapshot was not found")


def _qwen():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir(), local_files_only=True)
        # The snapshot's model_max_length is 131072. Past that, a warning fires
        # and a default truncation setting would shorten the longest labels.
        tokenizer.model_max_length = 10**9
        _TOKENIZER = tokenizer
    return _TOKENIZER


def qwen_tokenize_many(texts: list[str]) -> list[list[int]]:
    """Qwen token ids plus the EOS id E1 appends. No model weights are loaded."""
    tokenizer = _qwen()
    eos = tokenizer.eos_token_id
    out: list[list[int]] = []
    step = 8
    for start in range(0, len(texts), step):
        encoded = tokenizer(
            texts[start : start + step],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        for row in encoded["input_ids"]:
            ids = [int(token) for token in row]
            if eos is not None:
                ids.append(int(eos))
            out.append(ids)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_build(
    *,
    out: Path,
    primary_jsonl: Path | None,
    max_docs: int,
    page_size: int,
    delay: float,
    retries: int,
    since: date,
    until: date,
    resume: bool,
    client=None,
    tokenize=None,
    tokenize_many=None,
    prefix: int = EVAL_PREFIX,
    target: int = EVAL_TARGET,
    usable_target: int = USABLE_TARGET,
    fetch_cap: int = FETCH_CAP,
    seed: int = SPLIT_SEED,
    item_seed: int = ITEM_SEED,
) -> dict:
    """Harvest until the usable target or the cap, then seal once."""
    out = Path(out)
    search = resolve_search(out, since, until, resume)
    target_docs = max_docs
    while True:
        counts = harvest_fda(
            out=out,
            search=search,
            max_docs=target_docs,
            page_size=page_size,
            delay=delay,
            retries=retries,
            resume=resume,
            client=client,
        )
        docs = load_register(out) if out.exists() else []
        # Cheap pre-check so we do not tokenize twice on the way to the target.
        # The authoritative usable count is computed in assemble. Here, a
        # character floor stands in only for the "fetch more?" decision when
        # the caller did not supply a tokenizer. With a tokenizer, assemble
        # is the one count that matters, so we fetch the requested max first.
        if counts.get("aborted") or counts.get("end") == "empty":
            break
        if len(docs) >= usable_target or target_docs >= fetch_cap:
            break
        # The harvest stopped because it hit max_docs. Raise the cap and resume.
        nxt = min(fetch_cap, target_docs + max(page_size * 4, usable_target - len(docs)))
        if nxt <= target_docs:
            break
        target_docs = nxt
    card_path = out.parent / "card.json"
    return assemble_register(
        out,
        card_path=card_path,
        seal_path=out.parent / "seal.json",
        items_path=out.parent / "items_sealed.json",
        primary_jsonl=primary_jsonl,
        tokenize=tokenize,
        tokenize_many=tokenize_many,
        prefix=prefix,
        target=target,
        seed=seed,
        item_seed=item_seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sealed FDA second register (G-1f)")
    parser.add_argument("--out", default="artifacts/corpus_fda/fda_2026.jsonl")
    parser.add_argument("--primary", default="artifacts/corpus/biorxiv_2026.jsonl")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--since", default="2026-01-01")
    parser.add_argument("--until", default=date.today().isoformat())
    parser.add_argument("--usable-target", type=int, default=USABLE_TARGET)
    parser.add_argument("--fetch-cap", type=int, default=FETCH_CAP)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="do not fetch; seal and report the JSONL already on disk",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    primary = Path(args.primary)
    if args.assemble_only:
        card = assemble_register(
            out,
            card_path=out.parent / "card.json",
            seal_path=out.parent / "seal.json",
            items_path=out.parent / "items_sealed.json",
            primary_jsonl=primary,
        )
    else:
        card = run_build(
            out=out,
            primary_jsonl=primary,
            max_docs=args.max_docs,
            page_size=args.page_size,
            delay=args.delay,
            retries=args.retries,
            since=date.fromisoformat(args.since),
            until=date.fromisoformat(args.until),
            resume=not args.no_resume,
            usable_target=args.usable_target,
            fetch_cap=args.fetch_cap,
        )
    summary = {
        "documents_fetched": card["documents_fetched"],
        "distinct_ids": card["distinct_ids"],
        "licence_canonical": card["licence_canonical"],
        "documents_usable": card["documents_usable"],
        "token_length": card["token_length"],
        "seal_sha256": None if card["seal"] is None else card["seal"]["seal_sha256"],
        "n_items": card["items"]["n_items"],
        "contrast_verdict": card["contrast"].get("verdict"),
        "aborted": card["network"].get("aborted"),
        "g1f_mechanics_ok": card["g1f_mechanics_ok"],
        "g1f_endpoint_items": card["g1f_endpoint_items"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not card["network"].get("aborted") else 2


if __name__ == "__main__":
    sys.exit(main())
