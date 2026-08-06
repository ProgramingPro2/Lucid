#!/usr/bin/env python3
"""
Stage 0 diagnostics: adjacent-cosine wall investigation.

Compares four encoders on ~2000 adjacent sentence pairs from the Lucid corpus:
  1) JEPA encoder (project code)
  2) JEPA + ZCA whitening
  3) all-MiniLM-L6-v2 (SentenceTransformer; prefer project codec if available)
  4) MiniLM + ZCA whitening

CPU-only, robust to encoder load failures (continues).
Run:
  source .venv/bin/activate && PYTHONPATH=src python scripts/stage0_diagnostics.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


def _force_cpu_only() -> None:
    # Best-effort: make CUDA invisible to libraries that honor this.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repo_import_bootstrap() -> None:
    # Repo runs with PYTHONPATH=src; keep it explicit for robustness.
    if "src" not in sys.path:
        sys.path.insert(0, "src")


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


def _pairwise_cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    a_n = _normalize_rows(a)
    b_n = _normalize_rows(b)
    return float(np.sum(a_n * b_n, axis=1).mean())


@dataclass(frozen=True)
class Pair:
    doc_id: int
    i: int  # source sentence index within doc


@dataclass
class EvalData:
    docs_sentences: list[list[str]]  # only docs used for eval (subset), indexed by eval_doc_idx
    eval_doc_ids: list[int]  # eval_doc_idx -> original doc_id in corpus list
    pairs: list[Pair]  # pairs refer to eval_doc_idx in docs_sentences, via doc_id in Pair


def _sample_eval_pairs(
    all_docs_sentences: list[list[str]],
    n_pairs: int = 2000,
    seed: int = 0,
) -> EvalData:
    """
    Build adjacent in-order pairs within documents, then sample ~n_pairs.
    Keeps document boundaries so we can do within-doc and cross-doc controls.
    """
    rng = random.Random(seed)

    # Candidate pairs are within-doc adjacent indices (i -> i+1).
    candidates: list[tuple[int, int]] = []
    for doc_id, sents in enumerate(all_docs_sentences):
        if len(sents) < 2:
            continue
        for i in range(len(sents) - 1):
            candidates.append((doc_id, i))

    if not candidates:
        raise ValueError("No adjacent pairs found (corpus may be empty or too short).")

    if len(candidates) <= n_pairs:
        chosen = candidates
    else:
        chosen = rng.sample(candidates, n_pairs)

    chosen_doc_ids = sorted({doc_id for doc_id, _ in chosen})
    doc_id_to_eval_idx = {doc_id: k for k, doc_id in enumerate(chosen_doc_ids)}
    eval_docs = [all_docs_sentences[doc_id] for doc_id in chosen_doc_ids]

    pairs = [Pair(doc_id=doc_id_to_eval_idx[doc_id], i=i) for doc_id, i in chosen]
    return EvalData(docs_sentences=eval_docs, eval_doc_ids=chosen_doc_ids, pairs=pairs)


def _iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        batch_size = 1
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class Encoder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


def _encode_batched(enc: Encoder, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, enc.dim), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for batch in _iter_batches(texts, batch_size):
        vecs = enc.encode(batch)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f"{enc.name}.encode returned shape {vecs.shape}, expected 2D.")
        chunks.append(vecs)
    out = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    if out.shape[0] != len(texts):
        raise ValueError(
            f"{enc.name}.encode returned {out.shape[0]} rows for {len(texts)} texts."
        )
    return np.ascontiguousarray(out, dtype=np.float32)


@dataclass(frozen=True)
class ZCAWhitening:
    mean: np.ndarray  # (d,)
    W: np.ndarray  # (d, d)

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) @ self.W


def _fit_zca(X: np.ndarray, eps: float = 1e-5) -> ZCAWhitening:
    X64 = np.asarray(X, dtype=np.float64)
    if X64.ndim != 2:
        raise ValueError(f"ZCA fit expects 2D array, got shape {X64.shape}")
    if X64.shape[0] < 2:
        # Degenerate: identity transform.
        d = int(X64.shape[1])
        return ZCAWhitening(mean=np.zeros((d,), dtype=np.float32), W=np.eye(d, dtype=np.float32))

    mean = X64.mean(axis=0, keepdims=False)
    Xc = X64 - mean
    cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    # Symmetric PSD; use eigh.
    s, U = np.linalg.eigh(cov)
    s = np.maximum(s, 0.0)
    inv_sqrt = 1.0 / np.sqrt(s + eps)
    W = (U * inv_sqrt[None, :]) @ U.T
    return ZCAWhitening(mean=mean.astype(np.float32), W=W.astype(np.float32))


def _build_whiten_sample(
    all_docs_sentences: list[list[str]],
    exclude_doc_ids: set[int],
    n_sentences: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    available_doc_ids = [i for i in range(len(all_docs_sentences)) if i not in exclude_doc_ids]
    rng.shuffle(available_doc_ids)

    sample: list[str] = []
    for doc_id in available_doc_ids:
        sents = all_docs_sentences[doc_id]
        for s in sents:
            if not s.strip():
                continue
            sample.append(s)
            if len(sample) >= n_sentences:
                return sample
    # Fallback: if corpus is small, allow sampling from excluded docs too.
    if len(sample) < n_sentences:
        all_flat = [s for doc in all_docs_sentences for s in doc if s.strip()]
        rng.shuffle(all_flat)
        sample = (sample + all_flat)[:n_sentences]
    return sample


@dataclass
class Metrics:
    dim: int
    n_pairs: int
    adjacent_cos: float
    within_doc_shuf_cos: float
    cross_doc_shuf_cos: float
    random_floor: float
    m3_top1: float  # fraction
    m3_mrr: float


def _compute_controls(
    rng: random.Random,
    pairs: list[Pair],
    doc_vectors: list[np.ndarray],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Returns (adjacent_pairs, within_doc_shuf_pairs, cross_doc_shuf_pairs) where
    each element is a list of (a_vecs, b_vecs) arrays ready for cosine mean.
    """
    # Adjacent in-order.
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for p in pairs:
        dv = doc_vectors[p.doc_id]
        a_list.append(dv[p.i])
        b_list.append(dv[p.i + 1])
    A = np.stack(a_list, axis=0) if a_list else np.zeros((0, doc_vectors[0].shape[1]), dtype=np.float32)
    B = np.stack(b_list, axis=0) if b_list else np.zeros((0, doc_vectors[0].shape[1]), dtype=np.float32)

    # Within-doc shuffled: permute targets within each doc among the doc's pairs.
    within_A: list[np.ndarray] = []
    within_B: list[np.ndarray] = []
    by_doc: dict[int, list[int]] = {}
    for idx, p in enumerate(pairs):
        by_doc.setdefault(p.doc_id, []).append(idx)
    for doc_id, idxs in by_doc.items():
        # targets are the p.i+1 vectors for these idxs; permute them.
        targets = [pairs[j].i + 1 for j in idxs]
        perm = targets[:]
        rng.shuffle(perm)
        dv = doc_vectors[doc_id]
        for j, tgt_i in zip(idxs, perm, strict=True):
            src_i = pairs[j].i
            within_A.append(dv[src_i])
            within_B.append(dv[tgt_i])
    WA = np.stack(within_A, axis=0) if within_A else A[:0]
    WB = np.stack(within_B, axis=0) if within_B else B[:0]

    # Cross-doc shuffled: for each source, pick a random target from a different doc.
    # We'll sample from the pool of true adjacent targets (p.i+1) but require doc != source doc.
    all_targets: list[tuple[int, int]] = [(p.doc_id, p.i + 1) for p in pairs]
    cross_A: list[np.ndarray] = []
    cross_B: list[np.ndarray] = []
    for p in pairs:
        dv_src = doc_vectors[p.doc_id]
        cross_A.append(dv_src[p.i])
        chosen: tuple[int, int] | None = None
        # Try a few random picks; if unlucky, fall back to scan.
        for _ in range(30):
            cand = rng.choice(all_targets)
            if cand[0] != p.doc_id:
                chosen = cand
                break
        if chosen is None:
            for cand in all_targets:
                if cand[0] != p.doc_id:
                    chosen = cand
                    break
        if chosen is None:
            # Degenerate: only one doc in data; use within-doc but non-adjacent if possible.
            doc_id = p.doc_id
            dv = doc_vectors[doc_id]
            pool = [k for k in range(len(dv)) if k != p.i + 1]
            tgt_i = rng.choice(pool) if pool else (p.i + 1)
            cross_B.append(dv[tgt_i])
        else:
            dv_tgt = doc_vectors[chosen[0]]
            cross_B.append(dv_tgt[chosen[1]])
    CA = np.stack(cross_A, axis=0) if cross_A else A[:0]
    CB = np.stack(cross_B, axis=0) if cross_B else B[:0]

    return [(A, B)], [(WA, WB)], [(CA, CB)]


def _m3_metrics(
    rng: random.Random,
    pairs: list[Pair],
    doc_vectors: list[np.ndarray],
    n_candidates: int = 100,
) -> tuple[float, float]:
    """
    Given z_t and candidates (true z_{t+1} + distractors from same doc),
    compute top-1 accuracy and MRR under cosine similarity.
    """
    if not pairs:
        return float("nan"), float("nan")

    top1 = 0
    rr_sum = 0.0

    for p in pairs:
        dv = doc_vectors[p.doc_id]
        if p.i + 1 >= len(dv):
            continue
        q = dv[p.i]
        true = dv[p.i + 1]

        # Distractors from same doc (excluding the true next).
        pool = [k for k in range(len(dv)) if k != (p.i + 1)]
        # Optionally exclude the query itself too, though it's not specified; avoid trivial self-match.
        if p.i in pool:
            pool.remove(p.i)

        need = max(1, n_candidates) - 1
        if not pool:
            distractors_idx = [p.i + 1] * need
        elif len(pool) >= need:
            distractors_idx = rng.sample(pool, need)
        else:
            # Sample with replacement if doc is short.
            distractors_idx = [rng.choice(pool) for _ in range(need)]

        cands = np.vstack([true, dv[distractors_idx]])
        qn = _normalize_rows(q)[0]
        cn = _normalize_rows(cands)
        sims = cn @ qn
        # Rank of true candidate (index 0) among all candidates, descending.
        order = np.argsort(-sims)
        rank0 = int(np.where(order == 0)[0][0]) + 1  # 1-based
        if rank0 == 1:
            top1 += 1
        rr_sum += 1.0 / rank0

    n = len(pairs)
    return top1 / n, rr_sum / n


def _print_table(title: str, m: Metrics) -> None:
    def f(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return f"{x:.4f}"

    rows = [
        ("dim", str(m.dim)),
        ("pairs", str(m.n_pairs)),
        ("adjacent cosine (mean)", f(m.adjacent_cos)),
        ("within-doc shuffled cosine (mean)", f(m.within_doc_shuf_cos)),
        ("cross-doc shuffled cosine (mean)", f(m.cross_doc_shuf_cos)),
        ("random-vector floor (1/sqrt(dim))", f(m.random_floor)),
        ("M3 top-1 (%)", f(100.0 * m.m3_top1)),
        ("M3 MRR", f(m.m3_mrr)),
    ]

    w = max(len(k) for k, _ in rows)
    print(f"\n== {title} ==")
    for k, v in rows:
        print(f"{k:<{w}}  {v}")


def _load_jepa_encoder() -> Encoder:
    from conceptpoc.persistence import load_joint
    from conceptpoc.codec.jepa_codec import JepaCodec

    model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt", device="cpu")
    codec = JepaCodec(model, tok, 128, device="cpu")

    class _JEPA(Encoder):
        def __init__(self, c: Any) -> None:
            self.name = "JEPA"
            self.dim = int(c.dim)
            self._codec = c

        def encode(self, texts: list[str]) -> np.ndarray:
            return self._codec.encode(texts)

    return _JEPA(codec)


def _load_minilm_encoder() -> Encoder:
    # Prefer the project codec if available; enforce CPU.
    try:
        from conceptpoc.codec.sentence_transformer_codec import SentenceTransformerCodec

        codec = SentenceTransformerCodec(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        class _MiniLM(Encoder):
            def __init__(self, c: Any) -> None:
                self.name = "MiniLM"
                self.dim = int(c.dim)
                self._codec = c

            def encode(self, texts: list[str]) -> np.ndarray:
                return self._codec.encode(texts)

        return _MiniLM(codec)
    except Exception:
        # Fallback: load directly from sentence_transformers.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        model.eval()
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dim_ = int(get_dim())

        class _MiniLM(Encoder):
            def __init__(self, m: Any, d: int) -> None:
                self.name = "MiniLM"
                self.dim = int(d)
                self._model = m

            def encode(self, texts: list[str]) -> np.ndarray:
                if not texts:
                    return np.zeros((0, self.dim), dtype=np.float32)
                vecs = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return np.ascontiguousarray(vecs, dtype=np.float32)

        return _MiniLM(model, dim_)


def _evaluate_encoder(
    enc: Encoder,
    eval_data: EvalData,
    whiten: ZCAWhitening | None,
    batch_size: int,
    seed: int,
) -> Metrics:
    rng = random.Random(seed)

    # Encode all eval docs in one pass, then split back into docs.
    flat: list[str] = []
    offsets: list[int] = [0]
    for sents in eval_data.docs_sentences:
        flat.extend(sents)
        offsets.append(len(flat))

    vecs = _encode_batched(enc, flat, batch_size=batch_size)
    if whiten is not None:
        vecs = whiten.apply(vecs)

    # Split per doc.
    doc_vecs: list[np.ndarray] = []
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        doc_vecs.append(np.ascontiguousarray(vecs[a:b], dtype=np.float32))

    # Controls.
    (adj,), (within,), (cross,) = _compute_controls(rng, eval_data.pairs, doc_vecs)
    adjacent_cos = _pairwise_cos_mean(*adj)
    within_cos = _pairwise_cos_mean(*within)
    cross_cos = _pairwise_cos_mean(*cross)

    top1, mrr = _m3_metrics(rng, eval_data.pairs, doc_vecs, n_candidates=100)

    return Metrics(
        dim=int(doc_vecs[0].shape[1]) if doc_vecs else int(enc.dim),
        n_pairs=len(eval_data.pairs),
        adjacent_cos=adjacent_cos,
        within_doc_shuf_cos=within_cos,
        cross_doc_shuf_cos=cross_cos,
        random_floor=1.0 / math.sqrt(max(1, int(enc.dim))),
        m3_top1=top1,
        m3_mrr=mrr,
    )


def main() -> None:
    _force_cpu_only()
    _repo_import_bootstrap()

    # Lazy imports after env setup.
    from conceptpoc.data.corpus import load_documents

    seed = 0
    rng = random.Random(seed)
    corpus_path = "data/corpus_1m.txt"

    t0 = time.time()
    docs = load_documents(corpus_path)
    # Strip blanks; keep per-doc boundaries.
    all_docs_sentences = [[s for s in d.sentences if s.strip()] for d in docs]

    eval_data = _sample_eval_pairs(all_docs_sentences, n_pairs=2000, seed=seed)
    print(
        f"Loaded docs={len(all_docs_sentences)} | eval_docs={len(eval_data.docs_sentences)} "
        f"| eval_pairs={len(eval_data.pairs)} | corpus={corpus_path}"
    )

    # Whitening fit sample size.
    whiten_n = 5000
    exclude = set(eval_data.eval_doc_ids)

    base_encoders: list[tuple[str, Callable[[], Encoder], int]] = [
        ("JEPA", _load_jepa_encoder, 64),
        ("MiniLM", _load_minilm_encoder, 64),
    ]

    for base_name, factory, batch_size in base_encoders:
        try:
            enc = factory()
        except Exception as e:
            print(f"\n== {base_name} (FAILED TO LOAD) ==\n{type(e).__name__}: {e}")
            continue

        # Fit whitening on held-out sentences (best-effort).
        zca: ZCAWhitening | None = None
        try:
            sample_texts = _build_whiten_sample(
                all_docs_sentences,
                exclude_doc_ids=exclude,
                n_sentences=whiten_n,
                seed=seed + 13,
            )
            sample_vecs = _encode_batched(enc, sample_texts, batch_size=batch_size)
            zca = _fit_zca(sample_vecs, eps=1e-5)
        except Exception as e:
            print(
                f"\n[{base_name}] Whitening fit failed; continuing without ZCA. "
                f"{type(e).__name__}: {e}"
            )
            zca = None

        # Evaluate raw.
        try:
            m_raw = _evaluate_encoder(
                enc, eval_data, whiten=None, batch_size=batch_size, seed=seed + 1
            )
            _print_table(f"{base_name} (raw)", m_raw)
        except Exception as e:
            print(f"\n== {base_name} (raw) FAILED ==\n{type(e).__name__}: {e}")

        # Evaluate whitened.
        if zca is not None:
            try:
                m_w = _evaluate_encoder(
                    enc, eval_data, whiten=zca, batch_size=batch_size, seed=seed + 2
                )
                _print_table(f"{base_name} (ZCA whitened)", m_w)
            except Exception as e:
                print(
                    f"\n== {base_name} (ZCA whitened) FAILED ==\n{type(e).__name__}: {e}"
                )

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s (CPU-only).")


if __name__ == "__main__":
    main()

