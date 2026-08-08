"""MiniLM proxy stats for CORPUS V2 narrative / topic structure.

Encodes held-out documents with sentence-transformers MiniLM (CPU) and reports:
  - consecutive-paragraph cos vs cross-doc-paragraph cos (delta)
  - document-centroid split-half cos vs cross-doc cos (delta)
  - consecutive-sentence cos vs within-doc-shuffled cos (delta)

Writes ``artifacts/corpus_v2/stats.json`` and prints a readable PASS table.
Does NOT load JEPA / heavy models — MiniLM proxy only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from experiments.pivot.corpus_v2 import load_corpus_v2  # noqa: E402

DEFAULT_CORPUS = _ROOT / "data" / "corpus_v2.txt"
DEFAULT_OUT = _ROOT / "artifacts" / "corpus_v2" / "stats.json"


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / n).astype(np.float32)


def _mean_pool_sents(sent_emb: np.ndarray) -> np.ndarray:
    if sent_emb.size == 0:
        return np.zeros(384, dtype=np.float32)
    return _l2(sent_emb.mean(axis=0))


def _encode_minilm(texts: list[str], batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 1000,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def _pairwise_consec(vecs: list[np.ndarray]) -> float:
    """Mean cosine of consecutive pairs within each sequence."""
    vals: list[float] = []
    for v in vecs:
        if len(v) < 2:
            continue
        # v is (n, d)
        dots = np.sum(v[:-1] * v[1:], axis=-1)
        vals.extend(float(x) for x in dots)
    return float(np.mean(vals)) if vals else float("nan")


def _cross_doc_sample(
    flat: np.ndarray,
    doc_ids: np.ndarray,
    rng: np.random.RandomState,
    n_pairs: int = 5000,
) -> float:
    """Mean cosine of random pairs from different documents."""
    n = flat.shape[0]
    if n < 2:
        return float("nan")
    vals: list[float] = []
    attempts = 0
    max_attempts = n_pairs * 20
    while len(vals) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = int(rng.randint(0, n)), int(rng.randint(0, n))
        if i == j or doc_ids[i] == doc_ids[j]:
            continue
        vals.append(float(np.dot(flat[i], flat[j])))
    return float(np.mean(vals)) if vals else float("nan")


def _split_half_cos(para_vecs: np.ndarray) -> float:
    """Cosine of mean of first half vs second half of paragraph vectors."""
    n = para_vecs.shape[0]
    if n < 2:
        return float("nan")
    mid = n // 2
    a = _l2(para_vecs[:mid].mean(axis=0))
    b = _l2(para_vecs[mid:].mean(axis=0))
    return float(np.dot(a, b))


def compute_stats(
    docs: list[list[list[str]]],
    held_frac: float = 0.10,
    seed: int = 42,
    max_held_docs: int | None = 2000,
    max_sents_per_doc: int | None = 200,
) -> dict[str, Any]:
    """Compute MiniLM topic/order stats on the last held_frac of docs."""
    n_docs = len(docs)
    if n_docs < 10:
        raise ValueError(f"Need >=10 docs for held-out stats; got {n_docs}")

    # Held-out = last 10% (deterministic, as requested)
    n_held = max(1, int(round(n_docs * held_frac)))
    held = docs[-n_held:]
    if max_held_docs is not None and len(held) > max_held_docs:
        held = held[:max_held_docs]

    rng = np.random.RandomState(seed)

    # Flatten sentences + paragraph texts for batch encode
    para_texts: list[str] = []
    para_doc_ids: list[int] = []
    sent_texts: list[str] = []
    sent_doc_ids: list[int] = []
    # Per-doc sentence index ranges for shuffle baseline
    doc_sent_ranges: list[tuple[int, int]] = []

    for di, doc in enumerate(held):
        s0 = len(sent_texts)
        for para in doc:
            para_texts.append(" ".join(para))
            para_doc_ids.append(di)
            for s in para:
                sent_texts.append(s)
                sent_doc_ids.append(di)
        s1 = len(sent_texts)
        if max_sents_per_doc is not None and (s1 - s0) > max_sents_per_doc:
            # truncate extras from this doc for sentence-order probe only
            excess = (s1 - s0) - max_sents_per_doc
            del sent_texts[-excess:]
            del sent_doc_ids[-excess:]
            s1 = len(sent_texts)
        doc_sent_ranges.append((s0, s1))

    print(
        f"[stats] held docs={len(held)} paras={len(para_texts)} "
        f"sents={len(sent_texts)}; encoding MiniLM...",
        flush=True,
    )
    # Encode all unique texts in one go (paras + sents) for speed
    all_texts = para_texts + sent_texts
    all_emb = _encode_minilm(all_texts)
    para_emb = all_emb[: len(para_texts)]
    sent_emb = all_emb[len(para_texts) :]

    # --- paragraph consecutive vs cross-doc ---
    para_by_doc: list[np.ndarray] = []
    offset = 0
    for di, doc in enumerate(held):
        n_p = len(doc)
        para_by_doc.append(para_emb[offset : offset + n_p])
        offset += n_p

    consec_para = _pairwise_consec(para_by_doc)
    cross_para = _cross_doc_sample(
        para_emb, np.asarray(para_doc_ids, dtype=np.int64), rng, n_pairs=5000
    )
    para_delta = consec_para - cross_para

    # --- document centroid split-half vs cross-doc ---
    doc_centroids = np.stack([_mean_pool_sents(p) for p in para_by_doc], axis=0)
    split_half_vals = [
        _split_half_cos(p) for p in para_by_doc if p.shape[0] >= 4
    ]
    split_half = float(np.mean(split_half_vals)) if split_half_vals else float("nan")
    # cross-doc centroid cos
    cross_cent_vals: list[float] = []
    nd = doc_centroids.shape[0]
    for _ in range(min(5000, nd * 10)):
        i, j = int(rng.randint(0, nd)), int(rng.randint(0, nd))
        if i == j:
            continue
        cross_cent_vals.append(float(np.dot(doc_centroids[i], doc_centroids[j])))
    cross_cent = float(np.mean(cross_cent_vals)) if cross_cent_vals else float("nan")
    split_delta = split_half - cross_cent

    # --- sentence order: consecutive vs within-doc shuffle ---
    consec_sent_vals: list[float] = []
    shuffle_sent_vals: list[float] = []
    for s0, s1 in doc_sent_ranges:
        if s1 - s0 < 2:
            continue
        v = sent_emb[s0:s1]
        consec_sent_vals.extend(float(x) for x in np.sum(v[:-1] * v[1:], axis=-1))
        # shuffle order within doc
        perm = rng.permutation(v.shape[0])
        vs = v[perm]
        shuffle_sent_vals.extend(float(x) for x in np.sum(vs[:-1] * vs[1:], axis=-1))

    consec_sent = float(np.mean(consec_sent_vals)) if consec_sent_vals else float("nan")
    shuffle_sent = float(np.mean(shuffle_sent_vals)) if shuffle_sent_vals else float("nan")
    sent_delta = consec_sent - shuffle_sent

    # Expectations / PASS notes
    # Positive sentence-order delta is the key question for the pivot.
    sent_pass = bool(sent_delta > 0.01)
    para_pass = bool(para_delta > 0.05)
    split_pass = bool(split_delta > 0.05)

    return {
        "n_docs_total": n_docs,
        "n_held_docs": len(held),
        "n_held_paragraphs": len(para_texts),
        "n_held_sentences": len(sent_texts),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "paragraph": {
            "consecutive_cos": consec_para,
            "cross_doc_cos": cross_para,
            "delta": para_delta,
            "pass": para_pass,
            "expectation": "delta >> 0 (topic persistence across paragraphs)",
        },
        "doc_centroid_split_half": {
            "split_half_cos": split_half,
            "cross_doc_cos": cross_cent,
            "delta": split_delta,
            "pass": split_pass,
            "expectation": "delta >> 0 (within-doc thematic coherence)",
        },
        "sentence_order": {
            "consecutive_cos": consec_sent,
            "within_doc_shuffled_cos": shuffle_sent,
            "delta": sent_delta,
            "pass": sent_pass,
            "expectation": (
                "delta > 0 (real narrative order signal; old shuffled corpus lacked this)"
            ),
        },
        "summary": {
            "sentence_order_signal_present": sent_pass,
            "paragraph_topic_persistence": para_pass,
            "doc_split_half_coherence": split_pass,
        },
    }


def print_table(stats: dict[str, Any]) -> None:
    def row(name: str, a: float, b: float, delta: float, passed: bool, note: str) -> None:
        flag = "PASS" if passed else "FAIL"
        print(
            f"{name:<28} {a:>8.4f}  {b:>8.4f}  {delta:>+8.4f}  {flag:<4}  {note}"
        )

    print()
    print("=" * 96)
    print("CORPUS V2 MiniLM proxy stats (held-out docs)")
    print("=" * 96)
    print(
        f"{'metric':<28} {'signal':>8}  {'baseline':>8}  {'delta':>8}  {'ok':<4}  note"
    )
    print("-" * 96)
    p = stats["paragraph"]
    row(
        "para consec vs cross-doc",
        p["consecutive_cos"],
        p["cross_doc_cos"],
        p["delta"],
        p["pass"],
        "expect +topic persist",
    )
    d = stats["doc_centroid_split_half"]
    row(
        "doc split-half vs cross",
        d["split_half_cos"],
        d["cross_doc_cos"],
        d["delta"],
        d["pass"],
        "expect +doc coherence",
    )
    s = stats["sentence_order"]
    row(
        "sent order vs doc-shuffle",
        s["consecutive_cos"],
        s["within_doc_shuffled_cos"],
        s["delta"],
        s["pass"],
        "KEY: narrative order?",
    )
    print("-" * 96)
    present = stats["summary"]["sentence_order_signal_present"]
    print(
        "Sentence-order signal present:"
        f" {'YES (positive delta)' if present else 'NO (absent / weak)'}"
    )
    print("=" * 96)
    print()


def main(argv: Sequence[str] | None = None) -> int:
    # Allow `python experiments/pivot/corpus_stats.py` from repo root
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    p = argparse.ArgumentParser(description="CORPUS V2 MiniLM stats")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--held-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-held-docs", type=int, default=2000)
    args = p.parse_args(argv)

    docs = load_corpus_v2(args.corpus)
    print(f"[stats] loaded {len(docs)} docs from {args.corpus}", flush=True)
    stats = compute_stats(
        docs,
        held_frac=args.held_frac,
        seed=args.seed,
        max_held_docs=args.max_held_docs,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print_table(stats)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
