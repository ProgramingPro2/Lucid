"""E4 endpoints: entity recall, leave-one-out centroid cosine, retrieval wins.

The point of this module is N1 in docs/PROTOCOL.md §1: the confirmatory endpoint
must not share a mechanism with the treatment.

- ``entity_recall`` reads the *target* entities out of the generated text. It
  uses no embedding and no model. This is the primary endpoint.
- ``loo_centroid_cosine`` uses a *different* encoder (MiniLM) and a leave-one-out
  centroid, so the item cannot score by matching itself.
- ``retrieval_win_rate`` is a paired, deterministic contrast against another
  document — a tournament rather than an absolute threshold.

The deleted error (FINDINGS F38) was an absolute cosine gate of 0.40 against an
unsteered null of 0.445. ``necessity.absolute_gate_is_valid`` exists so that
mistake cannot be made silently again.
"""

from __future__ import annotations

import re
import string

import numpy as np

_PUNCT = string.punctuation + "“”‘’—–…"


def normalise_entity(s: str) -> str:
    s = s.strip().strip(_PUNCT).lower()
    return re.sub(r"\s+", " ", s)


def _entity_pattern(entity: str, allow_plural: bool) -> re.Pattern[str]:
    norm = normalise_entity(entity)
    body = re.escape(norm)
    if allow_plural:
        body += r"(?:s|es)?"
    # Word boundaries that also work for hyphenated and multi-word entities.
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def entity_recall(
    generated_text: str, gold_entities: list[str], *, allow_plural: bool = False
) -> float:
    """Fraction of gold entities present in the text (target-only recall)."""
    if not gold_entities:
        raise ValueError("gold_entities must be non-empty")
    if not generated_text or not generated_text.strip():
        return 0.0
    hits = sum(
        1
        for e in gold_entities
        if _entity_pattern(e, allow_plural).search(generated_text)
    )
    return hits / len(gold_entities)


def _l2_rows(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("embeddings must be (N, D)")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("zero-norm embedding row")
    return x / norms


def loo_centroid_cosine(embeddings: np.ndarray, doc_ids) -> np.ndarray:
    """Cosine between each row and its own document's leave-one-out centroid.

    Rows that are the only member of their document get NaN; callers should
    exclude them. Keeping one item per document makes this equal to the document
    centroid, which is why E4-S1 draws one item per document.
    """
    x = _l2_rows(np.asarray(embeddings, dtype=np.float64))
    ids = np.asarray(doc_ids)
    if ids.shape[0] != x.shape[0]:
        raise ValueError("embeddings and doc_ids must align")

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    uniq, starts = np.unique(sorted_ids, return_index=True)
    bounds = np.append(starts, len(sorted_ids))

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    for k in range(len(uniq)):
        idx = order[bounds[k] : bounds[k + 1]]
        if idx.size < 2:
            continue
        block = x[idx]
        total = block.sum(axis=0)
        for j, row_idx in enumerate(idx):
            centroid = total - block[j]
            norm = np.linalg.norm(centroid)
            if norm < 1e-12:
                continue
            out[row_idx] = float(x[row_idx] @ (centroid / norm))
    return out


def retrieval_win_rate(
    gen_emb: np.ndarray, doc_ids
) -> float:
    """Paired retrieval tournament against the next document.

    For item *i*, compare cosine to its own document's LOO centroid against the
    centroid of the next document in sorted order (wrapping). Deterministic on
    purpose: a random draw would add variance to the control arm, which is the
    N2 failure mode.
    """
    x = _l2_rows(np.asarray(gen_emb, dtype=np.float64))
    ids = np.asarray(doc_ids)
    if ids.shape[0] != x.shape[0]:
        raise ValueError("gen_emb and doc_ids must align")
    uniq = np.unique(ids)
    if uniq.size < 2:
        raise ValueError("need at least 2 documents")

    own = loo_centroid_cosine(x, ids)
    centroids = {d: _l2_rows(x[ids == d].mean(axis=0, keepdims=True))[0] for d in uniq}

    wins = 0
    valid = 0
    for i, d in enumerate(ids):
        if np.isnan(own[i]):
            continue
        k = int(np.where(uniq == d)[0][0])
        other = uniq[(k + 1) % uniq.size]
        if other == d:
            continue
        wins += int(own[i] > float(x[i] @ centroids[other]))
        valid += 1
    if valid == 0:
        raise ValueError("no valid retrieval pairs")
    return wins / valid
