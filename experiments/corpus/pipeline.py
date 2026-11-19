"""The construction pipeline (docs/CORPUS.md §4).

Each function is pure and independently testable. The build script only sequences
them; it contains no scientific decisions of its own. That matters because every
step here is a *gate*: a document that fails is dropped and counted, and the
counts become part of the citable corpus card.

The forbidden things (§8) are enforced structurally where possible: splitting
takes documents and returns document IDs, so a paragraph-level split cannot be
expressed through this API.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from .sources import Document, normalise_license

# --------------------------------------------------------------------------
# 1-2. date
# --------------------------------------------------------------------------


def filter_by_date(
    docs: list[Document], cutoff: str, grace_days: int = 30
) -> list[Document]:
    """Keep documents dated strictly after ``cutoff + grace_days``.

    The boundary is excluded on purpose: a document posted on the cutoff+grace day
    had time to enter a crawl, so it is not safely post-cutoff.
    """
    first = date.fromisoformat(cutoff[:10]) + timedelta(days=grace_days)
    kept: list[Document] = []
    for doc in docs:
        try:
            d = doc.parsed_date()
        except ValueError:
            continue
        if d > first:
            kept.append(doc)
    return kept


# --------------------------------------------------------------------------
# 3. licence
# --------------------------------------------------------------------------


def filter_by_license(
    docs: list[Document], allowlist: set[str] | frozenset[str]
) -> tuple[list[Document], list[tuple[str, str]]]:
    """Keep items whose *own* licence is classifiable and allowed.

    Returns ``(kept, dropped)`` where dropped is ``(doc_id, reason)``. The licence
    is read per item; nothing is inferred from the source as a whole.
    """
    kept: list[Document] = []
    dropped: list[tuple[str, str]] = []
    for doc in docs:
        norm = normalise_license(doc.license)
        doc.license = norm
        if norm == "unknown":
            dropped.append((doc.doc_id, "licence:unclassifiable"))
        elif norm not in allowlist:
            dropped.append((doc.doc_id, f"licence:{norm}"))
        else:
            kept.append(doc)
    return kept, dropped


# --------------------------------------------------------------------------
# 4. body text
# --------------------------------------------------------------------------

_END_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited|"
    r"acknowledg(?:e)?ments?|funding|author contributions|"
    r"conflicts? of interest|competing interests)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CAPTION = re.compile(
    r"^\s*(?:figure|fig\.?|table|scheme|box)\s*\d+[.:)]", re.IGNORECASE
)
_WS = re.compile(r"[ \t]+")
_MANY_NL = re.compile(r"\n{3,}")


def _clean_block(text: str) -> str:
    # Cut trailing back-matter wholesale: it is never running prose.
    positions = [m.start() for m in _END_HEADING.finditer(text)]
    if positions:
        text = text[: min(positions)]
    lines = [ln for ln in text.splitlines() if not _CAPTION.match(ln)]
    text = "\n".join(lines)
    text = _WS.sub(" ", text)
    return _MANY_NL.sub("\n\n", text).strip()


def clean_text(doc: Document) -> Document:
    """Strip tables/captions/references/acknowledgements; keep running prose.

    The heuristics are line-based and deterministic. They are deliberately blunt:
    a false positive costs one document (counted), whereas a reference list left
    in place silently teaches the model citation boilerplate.
    """
    sections = {k: _clean_block(v) for k, v in doc.sections.items()}
    sections = {k: v for k, v in sections.items() if v}
    text = _clean_block(doc.text)
    if sections:
        # Rebuild the body from cleaned strata so no dirty paragraph survives via
        # the concatenated text.
        text = "\n\n".join(sections.values())
    return Document(
        doc_id=doc.doc_id,
        source=doc.source,
        title=doc.title,
        text=text,
        sections=sections,
        license=doc.license,
        date=doc.date,
        meta=dict(doc.meta),
    )


# --------------------------------------------------------------------------
# 5. strata helper
# --------------------------------------------------------------------------


def document_stratum(doc: Document, key: str = "stratum") -> str:
    """The document's stratum for splitting: meta override, else first section."""
    if key in doc.meta and doc.meta[key]:
        return str(doc.meta[key])
    if doc.sections:
        return sorted(doc.sections)[0]
    return "unknown"


# --------------------------------------------------------------------------
# 6. deduplicate
# --------------------------------------------------------------------------


def _shingles(text: str, k: int = 5) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        return frozenset()
    if len(words) < k:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(docs: list[Document], threshold: float = 0.95) -> list[Document]:
    """Drop near-duplicates, keeping the earliest version of each.

    Shingle Jaccard at the document level. Preprint→journal pairs are the common
    case, and keeping the earliest is what makes the choice a rule rather than an
    accident of iteration order.
    """
    ordered = sorted(docs, key=lambda d: (d.date, d.doc_id))
    shingles = [_shingles(d.text) for d in ordered]
    kept_idx: list[int] = []
    for i, si in enumerate(shingles):
        duplicate = False
        for j in kept_idx:
            sj = shingles[j]
            # Length prefilter: Jaccard <= min/max size, so skip impossible pairs.
            lo, hi = sorted((len(si), len(sj)))
            if hi and lo / hi < threshold:
                continue
            if _jaccard(si, sj) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept_idx.append(i)
    return [ordered[i] for i in kept_idx]


# --------------------------------------------------------------------------
# 7. memorisation screen
# --------------------------------------------------------------------------


def drop_low_nll(
    docs: list[Document], nll_fn, quantile: float = 0.10
) -> tuple[list[Document], list[Document]]:
    """Drop the lowest-NLL fraction of documents (docs/CORPUS.md §4 step 7).

    Low NLL means the base model already finds the text easy — leaked or
    boilerplate. Exactly ``round(quantile * n)`` documents are dropped, with a
    deterministic tie-break, so the count does not drift between runs.
    """
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must be in [0, 1)")
    scored = [(float(nll_fn(d)), d.doc_id, d) for d in docs]
    scored.sort(key=lambda t: (t[0], t[1]))
    n_drop = int(round(quantile * len(scored)))
    dropped = [t[2] for t in scored[:n_drop]]
    kept = [t[2] for t in scored[n_drop:]]
    return kept, dropped


# --------------------------------------------------------------------------
# 8. split by document
# --------------------------------------------------------------------------


def _stratum_rng(seed: int, stratum: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:corpus:{stratum}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def split_documents(
    docs: list[Document],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    strata_key: str = "stratum",
) -> dict[str, list[str]]:
    """Stratified, document-level split returning document IDs.

    Round-robin interleaving across strata makes every prefix of the order
    stratum-balanced, and the final slice hits the target proportions exactly.
    Document-level by construction: the only unit in the input is a document.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        groups[document_stratum(doc, strata_key)].append(doc.doc_id)

    per_stratum: list[list[str]] = []
    for stratum in sorted(groups):
        ids = sorted(groups[stratum])
        _stratum_rng(seed, stratum).shuffle(ids)
        per_stratum.append(ids)

    order: list[str] = []
    depth = 0
    while True:
        added = False
        for ids in per_stratum:
            if depth < len(ids):
                order.append(ids[depth])
                added = True
        if not added:
            break
        depth += 1

    n = len(order)
    n_train = min(int(round(fractions[0] * n)), n)
    n_dev = min(int(round(fractions[1] * n)), n - n_train)

    splits = {
        "train": order[:n_train],
        "dev": order[n_train : n_train + n_dev],
        "test": order[n_train + n_dev :],
    }
    all_ids = [i for v in splits.values() for i in v]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("a document appeared in more than one split")
    return splits


# --------------------------------------------------------------------------
# 9. freeze
# --------------------------------------------------------------------------


def token_stream_sha256(docs: list[Document], tokeniser) -> str:
    """Hash the exact token stream in a fixed document order.

    ``tokeniser`` maps text to a sequence of token ids. Sorting by ``doc_id``
    means the hash does not depend on harvest order; including the id means a
    document swapped for another changes the hash even if the tokens coincide.
    """
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        tokens = tokeniser(doc.text)
        h.update(doc.doc_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(" ".join(str(t) for t in tokens).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------
# 10. card
# --------------------------------------------------------------------------

CARD_KEYS: tuple[str, ...] = (
    "base_model",
    "recorded_cutoff",
    "sources",
    "documents_harvested",
    "drops_by_reason",
    "documents_retained",
    "tokens_per_stratum",
    "nll_decile",
    "contamination_gap_nats",
    "token_stream_sha256",
    "split_counts",
    "sealed_test_key_holder",
)


def build_card(
    docs: list[Document],
    splits: dict[str, list[str]],
    drops: dict[str, int],
    cutoff: str,
    extra: dict | None = None,
) -> dict:
    """Assemble the corpus card (docs/CORPUS.md §9). Pending values stay 'pending'."""
    extra = dict(extra or {})

    tokens_per_stratum: Counter[str] = Counter()
    for doc in docs:
        for stratum, text in doc.sections.items():
            tokens_per_stratum[stratum] += len(text.split())

    sources: Counter[str] = Counter()
    for doc in docs:
        sources[f"{doc.source}:{normalise_license(doc.license)}"] += 1

    card = {
        "base_model": extra.get("base_model", "pending"),
        "recorded_cutoff": cutoff,
        "sources": dict(sorted(sources.items())),
        "documents_harvested": extra.get("documents_harvested", len(docs)),
        "drops_by_reason": dict(sorted(drops.items())),
        "documents_retained": len(docs),
        "tokens_per_stratum": dict(sorted(tokens_per_stratum.items())),
        "nll_decile": extra.get("nll_decile", "pending"),
        "contamination_gap_nats": extra.get("contamination_gap_nats", "pending"),
        "token_stream_sha256": extra.get("token_stream_sha256", "pending"),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "sealed_test_key_holder": extra.get("sealed_test_key_holder", "pending"),
    }
    missing = [k for k in CARD_KEYS if k not in card]
    if missing:
        raise AssertionError(f"corpus card missing fields: {missing}")
    return card
