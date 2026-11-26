"""Document-disjoint G0 items from the post-cutoff bioRxiv JSONL.

One item per document. The prompt is the opening of a section. The target is
the held-out continuation of that same section, so the latent is not the
encoding of text the model was already shown. The rehearsal encoded the whole
paragraph, including the prompt, which hands the zero-prefix arm the answer
(FINDINGS F42).

A truncated final line is skipped. A malformed line anywhere earlier is
corruption and raises. The harvest is append-only and can be mid-write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The in-progress file was opened at the operative harvest cutoff 2025-06-01.
# Rows start the day after the 30-day grace, 2025-07-01. See docs/CORPUS.md.
MIN_DATE = "2025-07-01"
ALLOWED_LICENSES = frozenset({"cc-by", "cc0"})
SECTION_ORDER = ("results", "discussion", "introduction", "methods", "other", "abstract")
PROMPT_CHARS = 400
MIN_TARGET_CHARS = 800
TARGET_CHARS = 2000


class CorpusTooSmall(RuntimeError):
    """The corpus cannot support the null-width criterion. Do not start the GPU."""

    def __init__(self, have: int, need: int, detail: str) -> None:
        self.have = have
        self.need = need
        super().__init__(
            f"corpus has {have} document-disjoint full-text items; "
            f"G0's null half-width needs {need}. {detail}"
        )


@dataclass(frozen=True)
class Item:
    doc_id: str
    date: str
    license: str
    section: str
    prompt: str
    target: str
    partner_index: int
    null_split: str  # "a" or "b"; "" if this item is the odd one out

    @property
    def partner_is_other_document(self) -> bool:
        return self.partner_index >= 0


def iter_jsonl(path: Path):
    """Yield ``(line, is_final)``. The caller decides what a bad final line means."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        previous: str | None = None
        for line in fh:
            if previous is not None:
                yield previous, False
            previous = line
        if previous is not None:
            yield previous, True


def load_records(path: str | Path) -> tuple[list[dict], int]:
    """Parse a harvest JSONL. Returns rows and the number of skipped tail lines."""
    rows: list[dict] = []
    skipped_tail = 0
    for line, final in iter_jsonl(Path(path)):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            if final:
                skipped_tail += 1
                continue
            raise
        if not isinstance(payload, dict):
            if final:
                skipped_tail += 1
                continue
            raise ValueError("JSONL row is not an object")
        rows.append(payload)
    return rows, skipped_tail


def split_section(text: str) -> tuple[str, str] | None:
    """Split one section into a prompt and a non-overlapping held-out target."""
    body = " ".join((text or "").split())
    if len(body) < PROMPT_CHARS + MIN_TARGET_CHARS:
        return None
    cut = body.rfind(" ", 0, PROMPT_CHARS)
    if cut < 80:
        cut = PROMPT_CHARS
    prompt = body[:cut].strip()
    target = body[cut : cut + TARGET_CHARS].strip()
    if len(prompt) < 80 or len(target) < MIN_TARGET_CHARS:
        return None
    if prompt in target or target in prompt:
        return None
    return prompt, target


def _choose_section(record: dict) -> tuple[str, str, str] | None:
    sections = record.get("sections") or {}
    if isinstance(sections, dict):
        for name in SECTION_ORDER:
            found = split_section(str(sections.get(name) or ""))
            if found is not None:
                return name, found[0], found[1]
    found = split_section(str(record.get("text") or ""))
    if found is None:
        return None
    return "text", found[0], found[1]


def _drop_reason(record: dict) -> str | None:
    meta = record.get("meta") or {}
    if not isinstance(meta, dict) or meta.get("fulltext") != "ok":
        return "not_fulltext"
    license_name = str(record.get("license") or "")
    if license_name not in ALLOWED_LICENSES:
        return "licence"
    date = str(record.get("date") or "")[:10]
    if date < MIN_DATE:
        return "date"
    if not str(record.get("doc_id") or ""):
        return "no_id"
    return None


def items_from_records(
    records: list[dict], *, null_seed: int = 0
) -> tuple[list[Item], dict]:
    """One item per eligible document, partners rotated, null split disjoint."""
    drops = {"not_fulltext": 0, "licence": 0, "date": 0, "no_id": 0, "duplicate": 0, "short": 0}
    chosen: list[tuple[str, str, str, str, str, str]] = []
    seen: set[str] = set()
    for record in records:
        reason = _drop_reason(record)
        if reason is not None:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        doc_id = str(record["doc_id"])
        if doc_id in seen:
            drops["duplicate"] += 1
            continue
        picked = _choose_section(record)
        if picked is None:
            drops["short"] += 1
            continue
        section, prompt, target = picked
        seen.add(doc_id)
        chosen.append(
            (
                doc_id,
                str(record.get("date") or "")[:10],
                str(record.get("license") or ""),
                section,
                prompt,
                target,
            )
        )
    chosen.sort(key=lambda row: row[0])
    n = len(chosen)
    if n == 0:
        census = {"n_items": 0, "n_null_pairs": 0, "drops": drops, "null_seed": null_seed}
        return [], census

    rng = np.random.default_rng(null_seed)
    order = rng.permutation(n)
    n_pairs = n // 2
    split = {int(order[i]): "a" for i in range(n_pairs)}
    split.update({int(order[n_pairs + i]): "b" for i in range(n_pairs)})

    items: list[Item] = []
    for i, (doc_id, date, license_name, section, prompt, target) in enumerate(chosen):
        items.append(
            Item(
                doc_id=doc_id,
                date=date,
                license=license_name,
                section=section,
                prompt=prompt,
                target=target,
                partner_index=(i + 1) % n,
                null_split=split.get(i, ""),
            )
        )
    census = {
        "n_records": len(records),
        "n_items": n,
        "n_null_pairs": n_pairs,
        "n_null_held_out": n - 2 * n_pairs,
        "drops": drops,
        "null_seed": null_seed,
        "min_date": MIN_DATE,
        "licenses": sorted(ALLOWED_LICENSES),
        "document_disjoint": True,
    }
    return items, census


def require_items(n_items: int, need: int, *, measure_anyway: bool) -> None:
    """Refuse a GPU launch that cannot meet the null-width plan.

    ``measure_anyway`` is the explicit override for an SD-estimation run.
    The default is the loud failure: a corpus the size of the rehearsal
    (n = 14) must not proceed.
    """
    if measure_anyway:
        return
    if n_items < need:
        raise CorpusTooSmall(
            n_items,
            need,
            "Refusing to start the GPU. Re-run after the harvest grows, "
            "or pass --measure-anyway to estimate the SD on the documents "
            "that exist. A rehearsal-sized corpus is not a gate.",
        )


def census_path(path: str | Path, *, null_seed: int = 0) -> dict:
    """Stream a corpus file and return the item census. Does not encode."""
    records, skipped = load_records(path)
    items, census = items_from_records(records, null_seed=null_seed)
    census["path"] = str(path)
    census["skipped_truncated_tail"] = skipped
    census["n_items"] = len(items)
    return census
