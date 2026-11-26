"""E4-S1 items: one per document, four draft conditions.

A confirmatory item carries the document it came from, a target span whose
keywords are absent from the prompt, the true draft (the JEPA centroid of that
span — an oracle plan), a cross-document shuffled draft, a zero draft, and the
50% slerp between true and shuffled. The same seed rebuilds the same items.
Fit documents and the two null arms do not reuse an eval item's target span.

See ``docs/PROTOCOL.md`` §7.1 and ``docs/EXPERIMENTS.md`` §2 (spawn-keyed
``data`` stream; no ``seed + k``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from experiments.e4.metrics import normalise_entity
from experiments.protocol.seeds import spawn_key

# Content words. Short function words and the boilerplate of a paper are not
# a recall target: a generation that says "results" has not used the draft.
_STOP = frozenset(
    """
    a an the of and or to in on for with from by at as is are was were be been
    being it its this that these those their his her our your they them we you
    not no nor but if then than so such into over under about between among
    during after before within without across per via using used use based
    using study studies result results method methods data analysis figure
    table supplementary however therefore thus also can may might could would
    should which who whom whose what when where while than into onto from
    have has had having does did done do been being been other another same
    new both each few more most some any all one two three four five
    significant respectively including include shown show shows observed
    observed found find using associated associated compared compare
    """.split()
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
_WS = re.compile(r"\s+")

EncodeFn = Callable[[Sequence[str]], np.ndarray]

SECTION_ORDER = ("introduction", "methods", "results", "discussion", "other")
TARGET_PREFERENCE = ("results", "discussion", "methods", "other")

MIN_KEYWORDS = 4
MAX_KEYWORDS = 12
MIN_CHUNK = 280


@dataclass(frozen=True)
class E4Item:
    """One eval item. Vectors are L2 unit except ``z_none``, which is the zero draft."""

    item_index: int
    doc_id: str
    prompt: str
    target_span: str
    gold_entities: tuple[str, ...]
    plan_text: str
    context_text: str
    shuffled_doc_id: str
    shuffled_plan_text: str
    text_prefix_true: str
    text_prefix_random: str
    z_true: np.ndarray
    z_shuffled: np.ndarray
    z_dose: np.ndarray
    z_none: np.ndarray

    @property
    def run_stem(self) -> str:
        return f"e4s1/i{self.item_index:04d}"


@dataclass(frozen=True)
class FitExample:
    """A document the adapter may train on. Never an eval document."""

    doc_id: str
    prompt: str
    target: str
    z: np.ndarray


@dataclass(frozen=True)
class NullItem:
    """One noise-floor item. Disjoint from its pair, and not an eval target."""

    item_index: int
    doc_id: str
    prompt: str
    target_span: str
    gold_entities: tuple[str, ...]
    context_text: str
    arm: str

    @property
    def run_stem(self) -> str:
        return f"e4s1/null/{self.arm}/i{self.item_index:04d}"


@dataclass(frozen=True)
class E4Split:
    items: tuple[E4Item, ...]
    fit_examples: tuple[FitExample, ...]
    null_a: tuple[NullItem, ...]
    null_b: tuple[NullItem, ...]
    seed: int
    n_text: int
    allocation: dict[str, Any]

    @property
    def text_items(self) -> tuple[E4Item, ...]:
        """The positive-control subset. A prefix of the eval items, so the
        document set is nested and the seed still determines it."""
        return self.items[: self.n_text]


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation. ``t=0`` is ``a``, ``t=1`` is ``b``."""
    a = _unit(a)
    b = _unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-6:
        return a.copy()
    s = math.sin(omega)
    out = (math.sin((1.0 - t) * omega) / s) * a + (math.sin(t * omega) / s) * b
    return _unit(out)


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return (x / n).astype(np.float32)


def data_rng(seed: int) -> np.random.Generator:
    """Numpy generator for the ``data`` stream. Independent of every other stream."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(spawn_key(seed, "data"))


def extract_target_keywords(
    target: str, prompt: str, *, k: int = MAX_KEYWORDS
) -> list[str]:
    """Content words that occur in the target and not in the prompt.

    This is the gold list ``entity_recall`` scores. It is target-only on
    purpose: a hit cannot be explained by copying the prompt.
    """
    prompt_words = {normalise_entity(w) for w in _TOKEN.findall(prompt)}
    seen: list[str] = []
    for raw in _TOKEN.findall(target):
        word = normalise_entity(raw)
        if word in _STOP or word in prompt_words or word in seen:
            continue
        has_digit = any(ch.isdigit() for ch in word)
        if len(word) < 4 and not has_digit:
            continue
        if len(word) < 3:
            continue
        seen.append(word)
    seen.sort(key=lambda w: (-len(w), w))
    return seen[:k]


def _sentences(text: str) -> list[str]:
    """A local splitter so item building does not import the training stack.

    Abbreviations are not the point here. Chunks are recombined, and a bad
    boundary moves a few words between adjacent chunks of the same document.
    """
    text = _WS.sub(" ", text.replace("\n", " ")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_section(text: str, *, target_chars: int = 700) -> list[str]:
    """Split a section into paragraph-sized chunks of running prose."""
    if not text or not text.strip():
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) == 1:
        blocks = _sentences(text)
    chunks: list[str] = []
    buf: list[str] = []
    n = 0
    for block in blocks:
        piece = _WS.sub(" ", block).strip()
        if not piece:
            continue
        buf.append(piece)
        n += len(piece) + 1
        if n >= target_chars:
            chunk = " ".join(buf).strip()
            if len(chunk) >= MIN_CHUNK:
                chunks.append(chunk)
            buf, n = [], 0
    if buf:
        chunk = " ".join(buf).strip()
        if len(chunk) >= MIN_CHUNK:
            chunks.append(chunk)
    return chunks


def chunk_document(doc: dict[str, Any]) -> list[tuple[str, str]]:
    """``(section, chunk)`` in reading order. Empty sections contribute nothing."""
    sections = doc.get("sections") or {}
    out: list[tuple[str, str]] = []
    if isinstance(sections, dict) and any(sections.get(name) for name in SECTION_ORDER):
        for name in SECTION_ORDER:
            for chunk in chunk_section(str(sections.get(name) or "")):
                out.append((name, chunk))
    if len(out) < 3:
        out = [("body", c) for c in chunk_section(str(doc.get("text") or ""))]
    return out


def pool_texts(encode_sentences: EncodeFn, text: str) -> np.ndarray:
    """Mean-pool sentence vectors and L2-normalise. This is the draft."""
    sents = _sentences(text)
    if not sents:
        sents = [text.strip()[:1000] or " "]
    # Cap so one long paragraph cannot dominate a fit step or an audit sample.
    vecs = np.asarray(encode_sentences(sents[:16]), dtype=np.float64)
    if vecs.ndim != 2 or vecs.shape[0] != len(sents[:16]):
        raise ValueError(
            f"encoder returned {getattr(vecs, 'shape', None)} for {len(sents[:16])} sentences"
        )
    return _unit(vecs.mean(axis=0))


def hash_encode(texts: Sequence[str], dim: int = 512) -> np.ndarray:
    """Deterministic stand-in for the JEPA encoder. Tests only, never a result."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        v = rng.normal(size=dim)
        rows.append(v / np.linalg.norm(v))
    return np.stack(rows).astype(np.float32)


def _cut(text: str, n_chars: int) -> str:
    text = text.strip()
    if len(text) <= n_chars:
        return text
    clipped = text[:n_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped


def _candidate_targets(
    chunks: list[tuple[str, str]],
) -> list[int]:
    """Indices of chunks that may be the held-out span. Prefer results."""
    by_section: dict[str, list[int]] = {name: [] for name in TARGET_PREFERENCE}
    other: list[int] = []
    for i, (name, _) in enumerate(chunks):
        if i == 0:
            continue  # the opening chunk is the prompt, never the target
        if name in by_section:
            by_section[name].append(i)
        else:
            other.append(i)
    ordered: list[int] = []
    for name in TARGET_PREFERENCE:
        ordered.extend(by_section[name])
    ordered.extend(other)
    return ordered


def _prepare_one(
    doc: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any] | None:
    chunks = chunk_document(doc)
    if len(chunks) < 3:
        return None
    prompt = _cut(chunks[0][1], 700)
    candidates = _candidate_targets(chunks)
    if not candidates:
        return None
    # Draw among the preferred section first, falling through if keywords fail.
    order = candidates[:]
    rng.shuffle(order)
    # Stable preference: re-sort so results stay ahead of a random methods chunk,
    # but the draw *within* a section is the seed's.
    rank = {name: i for i, name in enumerate(TARGET_PREFERENCE)}
    order.sort(key=lambda i: rank.get(chunks[i][0], len(rank)))
    chosen = None
    keywords: list[str] = []
    for i in order:
        keywords = extract_target_keywords(chunks[i][1], prompt)
        if len(keywords) >= MIN_KEYWORDS:
            chosen = i
            break
    if chosen is None:
        return None
    target = chunks[chosen][1]
    context_parts = [text for j, (_, text) in enumerate(chunks) if j != chosen]
    context = _cut(" ".join(context_parts), 1500)
    # A longer on-topic prefix for the positive control. Still excludes the target.
    true_prefix = _cut(" ".join(context_parts[:2]), 1000)
    return {
        "doc_id": str(doc["doc_id"]),
        "prompt": prompt,
        "target_span": target,
        "gold_entities": tuple(keywords),
        "plan_text": target,
        "context_text": context,
        "text_prefix_true": true_prefix,
        "chunks": chunks,
        "target_index": chosen,
    }


def _null_from_prepared(
    prepared: dict[str, Any], *, arm: str, item_index: int, which: int
) -> NullItem | None:
    """A span that is not the eval target. ``which`` selects among the leftovers."""
    chunks = prepared["chunks"]
    leftovers = [i for i in range(len(chunks)) if i != prepared["target_index"] and i != 0]
    if not leftovers:
        return None
    idx = leftovers[which % len(leftovers)]
    # Two arms must not land on the same span when a document is reused.
    if which == 1 and len(leftovers) > 1:
        idx = leftovers[1]
    elif which == 0:
        idx = leftovers[0]
    target = chunks[idx][1]
    prompt = prepared["prompt"]
    keywords = extract_target_keywords(target, prompt)
    if len(keywords) < MIN_KEYWORDS:
        return None
    context = _cut(
        " ".join(text for j, (_, text) in enumerate(chunks) if j != idx),
        1500,
    )
    return NullItem(
        item_index=item_index,
        doc_id=prepared["doc_id"],
        prompt=prompt,
        target_span=target,
        gold_entities=tuple(keywords),
        context_text=context,
        arm=arm,
    )


def load_documents(
    path: str | Path,
    *,
    min_chars: int = 4000,
    require_fulltext_ok: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Stream the harvested JSONL. A torn final line is skipped, not fatal."""
    docs: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or ""
            if len(text) < min_chars:
                continue
            if require_fulltext_ok:
                status = (row.get("meta") or {}).get("fulltext")
                if status != "ok":
                    continue
            docs.append(row)
            if limit is not None and len(docs) >= limit:
                break
    return docs


def _encode_plan(encode_sentences: EncodeFn, text: str, dim: int) -> np.ndarray:
    z = pool_texts(encode_sentences, text)
    if z.shape != (dim,):
        raise ValueError(f"encoder dim {z.shape[-1]} != expected {dim}")
    return z


def build_split(
    docs: Sequence[dict[str, Any]],
    encode_sentences: EncodeFn,
    *,
    n_items: int = 192,
    n_fit: int = 64,
    n_null: int = 192,
    n_text: int = 96,
    seed: int = 0,
    dim: int = 512,
) -> E4Split:
    """Draw a document-disjoint eval set and the fit / null companions.

    Eval items are one per document. The shuffled draft is a derangement of
    those documents (a rotation of the selected order), so no item is paired
    with itself. When the corpus cannot supply two further document-disjoint
    null sets of size ``n_null``, the null pair is two leftover spans of the
    eval documents: the items are still disjoint, and the contrast is paired
    by document. Fit documents never intersect the eval set.
    """
    if n_text > n_items:
        raise ValueError("n_text cannot exceed n_items")
    if n_items < 2 or n_null < 2:
        raise ValueError("need at least 2 eval items and 2 null items")

    rng = data_rng(seed)
    prepared: list[dict[str, Any]] = []
    # Prepare in a shuffled document order so the seed, not file order, picks.
    order = rng.permutation(len(docs))
    for i in order:
        one = _prepare_one(docs[int(i)], rng)
        if one is not None:
            prepared.append(one)

    if len(prepared) < n_items:
        raise ValueError(
            f"need {n_items} eligible documents, found {len(prepared)}"
        )

    eval_rows = prepared[:n_items]
    rest = prepared[n_items:]
    if len(rest) >= 2 * n_null:
        null_mode = "disjoint_documents"
        null_src_a = rest[:n_null]
        null_src_b = rest[n_null : 2 * n_null]
        fit_src = rest[2 * n_null : 2 * n_null + n_fit]
    else:
        null_mode = "paired_spans"
        # Two spans inside each eval document. Fit stays on the leftover docs.
        if len(eval_rows) < n_null:
            raise ValueError(
                f"paired nulls need {n_null} eval documents, have {len(eval_rows)}"
            )
        null_src_a = eval_rows[:n_null]
        null_src_b = eval_rows[:n_null]
        fit_src = rest[:n_fit]

    if len(fit_src) < 1:
        raise ValueError(
            "no document left for the adapter fit; the eval set consumed the corpus"
        )

    # Donors: rotate the eval order. Item i receives item i-1's plan.
    donors = list(range(n_items))
    donor_of = {i: donors[(i - 1) % n_items] for i in donors}

    z_cache: dict[str, np.ndarray] = {}

    def z_of(text: str) -> np.ndarray:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in z_cache:
            z_cache[key] = _encode_plan(encode_sentences, text, dim)
        return z_cache[key]

    items: list[E4Item] = []
    for i, row in enumerate(eval_rows):
        donor = eval_rows[donor_of[i]]
        if donor["doc_id"] == row["doc_id"]:
            raise RuntimeError("derangement paired a document with itself")
        z_true = z_of(row["plan_text"])
        z_shuf = z_of(donor["plan_text"])
        items.append(
            E4Item(
                item_index=i,
                doc_id=row["doc_id"],
                prompt=row["prompt"],
                target_span=row["target_span"],
                gold_entities=row["gold_entities"],
                plan_text=row["plan_text"],
                context_text=row["context_text"],
                shuffled_doc_id=donor["doc_id"],
                shuffled_plan_text=donor["plan_text"],
                text_prefix_true=row["text_prefix_true"],
                text_prefix_random=_cut(donor["text_prefix_true"], len(row["text_prefix_true"])),
                z_true=z_true,
                z_shuffled=z_shuf,
                z_dose=slerp(z_true, z_shuf, 0.5),
                z_none=np.zeros(dim, dtype=np.float32),
            )
        )

    fit_examples = tuple(
        FitExample(
            doc_id=row["doc_id"],
            prompt=row["prompt"],
            target=row["target_span"],
            z=z_of(row["plan_text"]),
        )
        for row in fit_src
    )

    null_a: list[NullItem] = []
    null_b: list[NullItem] = []
    if null_mode == "disjoint_documents":
        for i, row in enumerate(null_src_a):
            item = _null_from_prepared(row, arm="a", item_index=i, which=0)
            if item is None:
                # The eval target was the only keyword-bearing span. Use it:
                # this document is not in the eval set, so the span is free.
                item = NullItem(
                    item_index=i,
                    doc_id=row["doc_id"],
                    prompt=row["prompt"],
                    target_span=row["target_span"],
                    gold_entities=row["gold_entities"],
                    context_text=row["context_text"],
                    arm="a",
                )
            null_a.append(item)
        for i, row in enumerate(null_src_b):
            item = _null_from_prepared(row, arm="b", item_index=i, which=0)
            if item is None:
                item = NullItem(
                    item_index=i,
                    doc_id=row["doc_id"],
                    prompt=row["prompt"],
                    target_span=row["target_span"],
                    gold_entities=row["gold_entities"],
                    context_text=row["context_text"],
                    arm="b",
                )
            null_b.append(item)
    else:
        for i, row in enumerate(null_src_a):
            a = _null_from_prepared(row, arm="a", item_index=i, which=0)
            b = _null_from_prepared(row, arm="b", item_index=i, which=1)
            if a is None or b is None or a.target_span == b.target_span:
                raise ValueError(
                    f"document {row['doc_id']} cannot supply two disjoint null spans"
                )
            null_a.append(a)
            null_b.append(b)

    if len(null_a) != n_null or len(null_b) != n_null:
        raise ValueError(
            f"null pair is {len(null_a)}/{len(null_b)}, expected {n_null}"
        )

    eval_ids = {item.doc_id for item in items}
    fit_ids = {ex.doc_id for ex in fit_examples}
    if eval_ids & fit_ids:
        raise RuntimeError("fit documents overlap the eval set")
    if len(eval_ids) != len(items):
        raise RuntimeError("eval items are not one per document")

    allocation = {
        "null_mode": null_mode,
        "n_eligible": len(prepared),
        "n_items": n_items,
        "n_fit": len(fit_examples),
        "n_null": n_null,
        "n_text": n_text,
        "dim": dim,
        "eval_doc_ids": [item.doc_id for item in items],
        "fit_doc_ids": [ex.doc_id for ex in fit_examples],
    }
    return E4Split(
        items=tuple(items),
        fit_examples=fit_examples,
        null_a=tuple(null_a),
        null_b=tuple(null_b),
        seed=seed,
        n_text=n_text,
        allocation=allocation,
    )


def corpus_signature(doc_ids: Iterable[str]) -> str:
    """SHA-256 of the selected document ids, in item order."""
    payload = "\n".join(doc_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
