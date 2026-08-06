#!/usr/bin/env python3
"""G1 granularity probe: is forward topic predictability stronger at paragraph scale?

Groups corpus sentences into paragraph-ish blocks of size G in {1,2,3,5,10}
(rolling mean of sentence vectors), then for each G reports:
  - adjacent(group) cosine
  - within-doc shuffled(group) cosine
  - cross-doc shuffled(group) cosine
  - M3 top-1% (retrieve z_{t+1} from z_t among true-next + 99 in-doc distractors)

Primary encoder: frozen all-MiniLM-L6-v2. JEPA also attempted if available.

Interpretation: if M3 top-1 climbs substantially as G increases, topic
predictability lives at coarser time-scales -> paragraph-level concepts worth
building. Flat M3 => coarsening does not help.

CPU-only. Run from repo root:
  source .venv/bin/activate && PYTHONPATH=src python scripts/probe_granularity.py
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def _force_cpu_only() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repo_bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


def _pairwise_cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sum(_normalize_rows(a) * _normalize_rows(b), axis=1).mean())


def _iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    bs = max(1, batch_size)
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


@dataclass
class Encoder:
    name: str
    dim: int
    encode_fn: Callable[[list[str]], np.ndarray]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self.encode_fn(texts), dtype=np.float32)


def _encode_batched(enc: Encoder, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, enc.dim), dtype=np.float32)
    chunks = [enc.encode(b) for b in _iter_batches(texts, batch_size)]
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] != len(texts):
        raise ValueError(f"{enc.name}: got {out.shape[0]} vecs for {len(texts)} texts")
    return np.ascontiguousarray(out, dtype=np.float32)


def _load_minilm() -> Encoder:
    try:
        from conceptpoc.codec.sentence_transformer_codec import SentenceTransformerCodec

        codec = SentenceTransformerCodec(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        return Encoder("MiniLM", int(codec.dim), codec.encode)
    except Exception:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        model.eval()
        dim = int(model.get_sentence_embedding_dimension())

        def _enc(texts: list[str]) -> np.ndarray:
            return np.ascontiguousarray(
                model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )

        return Encoder("MiniLM", dim, _enc)


def _load_jepa() -> Encoder:
    from conceptpoc.persistence import load_joint
    from conceptpoc.codec.jepa_codec import JepaCodec

    model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt", device="cpu")
    codec = JepaCodec(model, tok, 128, device="cpu")
    return Encoder("JEPA", int(codec.dim), codec.encode)


def _group_vectors(sent_vecs: np.ndarray, g: int) -> np.ndarray:
    """Non-overlapping mean-pool of consecutive sentence vectors into groups of size g."""
    n, d = sent_vecs.shape
    if g <= 1:
        return sent_vecs
    n_groups = n // g
    if n_groups == 0:
        return np.zeros((0, d), dtype=np.float32)
    trimmed = sent_vecs[: n_groups * g].reshape(n_groups, g, d)
    return np.ascontiguousarray(trimmed.mean(axis=1), dtype=np.float32)


@dataclass(frozen=True)
class Pair:
    doc_id: int
    i: int  # group index within doc


def _build_group_docs(
    docs_sent_vecs: list[np.ndarray],
    g: int,
) -> list[np.ndarray]:
    return [_group_vectors(dv, g) for dv in docs_sent_vecs]


def _sample_pairs(
    group_docs: list[np.ndarray],
    n_pairs: int,
    seed: int,
) -> list[Pair]:
    rng = random.Random(seed)
    candidates: list[Pair] = []
    for doc_id, gv in enumerate(group_docs):
        if len(gv) < 2:
            continue
        for i in range(len(gv) - 1):
            candidates.append(Pair(doc_id=doc_id, i=i))
    if not candidates:
        return []
    if len(candidates) <= n_pairs:
        return candidates
    return rng.sample(candidates, n_pairs)


def _controls(
    rng: random.Random,
    pairs: list[Pair],
    group_docs: list[np.ndarray],
) -> tuple[float, float, float]:
    if not pairs:
        return float("nan"), float("nan"), float("nan")

    a_adj, b_adj = [], []
    for p in pairs:
        dv = group_docs[p.doc_id]
        a_adj.append(dv[p.i])
        b_adj.append(dv[p.i + 1])
    adj = _pairwise_cos_mean(np.stack(a_adj), np.stack(b_adj))

    # Within-doc shuffle of targets among this doc's sampled pairs.
    by_doc: dict[int, list[int]] = {}
    for idx, p in enumerate(pairs):
        by_doc.setdefault(p.doc_id, []).append(idx)
    wa, wb = [], []
    for doc_id, idxs in by_doc.items():
        targets = [pairs[j].i + 1 for j in idxs]
        perm = targets[:]
        rng.shuffle(perm)
        dv = group_docs[doc_id]
        for j, tgt in zip(idxs, perm, strict=True):
            wa.append(dv[pairs[j].i])
            wb.append(dv[tgt])
    within = _pairwise_cos_mean(np.stack(wa), np.stack(wb))

    # Cross-doc: random target from a different doc's adjacent targets.
    all_tgts = [(p.doc_id, p.i + 1) for p in pairs]
    ca, cb = [], []
    for p in pairs:
        ca.append(group_docs[p.doc_id][p.i])
        chosen = None
        for _ in range(40):
            cand = rng.choice(all_tgts)
            if cand[0] != p.doc_id:
                chosen = cand
                break
        if chosen is None:
            for cand in all_tgts:
                if cand[0] != p.doc_id:
                    chosen = cand
                    break
        if chosen is None:
            dv = group_docs[p.doc_id]
            pool = [k for k in range(len(dv)) if k != p.i + 1]
            tgt = rng.choice(pool) if pool else p.i + 1
            cb.append(dv[tgt])
        else:
            cb.append(group_docs[chosen[0]][chosen[1]])
    cross = _pairwise_cos_mean(np.stack(ca), np.stack(cb))
    return adj, within, cross


def _m3_top1(
    rng: random.Random,
    pairs: list[Pair],
    group_docs: list[np.ndarray],
    n_candidates: int = 100,
) -> float:
    if not pairs:
        return float("nan")
    hits = 0
    n = 0
    for p in pairs:
        dv = group_docs[p.doc_id]
        if p.i + 1 >= len(dv):
            continue
        q = dv[p.i]
        true = dv[p.i + 1]
        pool = [k for k in range(len(dv)) if k not in (p.i, p.i + 1)]
        need = max(1, n_candidates) - 1
        if not pool:
            # Degenerate short doc: pad with true (rank stays 1) — skip instead.
            continue
        if len(pool) >= need:
            dist_idx = rng.sample(pool, need)
        else:
            dist_idx = [rng.choice(pool) for _ in range(need)]
        cands = np.vstack([true, dv[dist_idx]])
        sims = _normalize_rows(cands) @ _normalize_rows(q)[0]
        rank0 = int(np.argsort(-sims).tolist().index(0)) + 1
        if rank0 == 1:
            hits += 1
        n += 1
    return hits / n if n else float("nan")


@dataclass
class Row:
    g: int
    n_pairs: int
    adjacent: float
    within: float
    cross: float
    m3_top1: float  # fraction


def _evaluate_g(
    docs_sent_vecs: list[np.ndarray],
    g: int,
    n_pairs: int,
    seed: int,
) -> Row:
    group_docs = _build_group_docs(docs_sent_vecs, g)
    # Drop empty / single-group docs for pairing.
    usable = [gv for gv in group_docs if len(gv) >= 2]
    if not usable:
        return Row(g, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    # Remap: only keep docs with >=2 groups; rebuild index map.
    # Simpler: keep original indexing but sample only from docs with len>=2.
    pairs = _sample_pairs(group_docs, n_pairs=n_pairs, seed=seed + g)
    rng = random.Random(seed + 1000 + g)
    adj, within, cross = _controls(rng, pairs, group_docs)
    m3 = _m3_top1(random.Random(seed + 2000 + g), pairs, group_docs, n_candidates=100)
    return Row(g, len(pairs), adj, within, cross, m3)


def _print_table_clean(title: str, rows: list[Row]) -> None:
    def f(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "nan"
        return f"{x:.4f}"

    print(f"\n== {title} ==")
    hdr = (
        f"{'G':>4}  {'n_pairs':>8}  {'adj cos':>10}  {'within-shuf':>12}  "
        f"{'cross-shuf':>11}  {'M3 top-1%':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        m3pct = "nan" if math.isnan(r.m3_top1) else f"{100.0 * r.m3_top1:9.2f}"
        print(
            f"{r.g:4d}  {r.n_pairs:8d}  {f(r.adjacent):>10}  {f(r.within):>12}  "
            f"{f(r.cross):>11}  {m3pct:>10}"
        )


def _select_docs_for_encoding(
    all_docs: list[list[str]],
    max_docs: int,
    min_sents: int,
    seed: int,
) -> list[list[str]]:
    """Pick docs long enough to support G=10 grouping, subsample for CPU budget."""
    rng = random.Random(seed)
    eligible = [d for d in all_docs if len(d) >= min_sents]
    if not eligible:
        # Fall back to any doc with >=2 sentences.
        eligible = [d for d in all_docs if len(d) >= 2]
    if len(eligible) > max_docs:
        eligible = rng.sample(eligible, max_docs)
    return eligible


def _run_encoder(
    enc: Encoder,
    docs: list[list[str]],
    n_pairs: int,
    batch_size: int,
    seed: int,
) -> list[Row]:
    flat: list[str] = []
    offsets = [0]
    for sents in docs:
        flat.extend(sents)
        offsets.append(len(flat))
    print(f"  encoding {len(flat)} sentences across {len(docs)} docs ({enc.name})...")
    vecs = _encode_batched(enc, flat, batch_size=batch_size)
    doc_vecs = [
        np.ascontiguousarray(vecs[a:b], dtype=np.float32)
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    rows: list[Row] = []
    for g in (1, 2, 3, 5, 10):
        row = _evaluate_g(doc_vecs, g=g, n_pairs=n_pairs, seed=seed)
        rows.append(row)
        if math.isnan(row.m3_top1):
            print(f"  G={g}: pairs={row.n_pairs} (insufficient data)")
        else:
            print(
                f"  G={g}: pairs={row.n_pairs} adj={row.adjacent:.4f} "
                f"within={row.within:.4f} cross={row.cross:.4f} "
                f"M3={100 * row.m3_top1:.2f}%"
            )
    return rows


def main() -> int:
    _force_cpu_only()
    root = _repo_bootstrap()
    t0 = time.time()
    print("== G1 granularity probe (CPU-only) ==")
    print(f"cwd={root}")

    try:
        from conceptpoc.data.corpus import load_documents
    except Exception as e:
        print(f"IMPORT FAILED: {type(e).__name__}: {e}")
        return 1

    corpus_path = "data/corpus_1m.txt"
    n_pairs = 1800
    seed = 0

    try:
        docs = load_documents(corpus_path)
        all_docs = [[s for s in d.sentences if s.strip()] for d in docs]
        all_docs = [d for d in all_docs if d]
    except Exception as e:
        print(f"CORPUS LOAD FAILED: {type(e).__name__}: {e}")
        return 1

    # Need docs with >= 20 sentences so G=10 yields >=2 groups.
    selected = _select_docs_for_encoding(
        all_docs, max_docs=400, min_sents=20, seed=seed
    )
    print(
        f"Loaded corpus docs={len(all_docs)} | selected={len(selected)} "
        f"| target_pairs≈{n_pairs} | corpus={corpus_path}"
    )

    loaders: list[tuple[str, Callable[[], Encoder], int]] = [
        ("MiniLM", _load_minilm, 64),
        ("JEPA", _load_jepa, 32),
    ]

    any_ok = False
    for name, factory, batch_size in loaders:
        try:
            enc = factory()
        except Exception as e:
            print(f"\n== {name} (FAILED TO LOAD) ==\n{type(e).__name__}: {e}")
            continue
        try:
            rows = _run_encoder(
                enc, selected, n_pairs=n_pairs, batch_size=batch_size, seed=seed
            )
            _print_table_clean(f"{name} (grouped mean-pool)", rows)
            any_ok = True
            # Quick interpretation for MiniLM (required arm).
            if name == "MiniLM" and len(rows) >= 2:
                m3_g1 = rows[0].m3_top1
                m3_g10 = rows[-1].m3_top1
                if not (math.isnan(m3_g1) or math.isnan(m3_g10)):
                    delta = 100.0 * (m3_g10 - m3_g1)
                    print(
                        f"\nINTERPRETATION ({name}): M3 top-1% G=1->{rows[-1].g}: "
                        f"{100*m3_g1:.2f}% -> {100*m3_g10:.2f}% (Δ={delta:+.2f} pp). "
                        + (
                            "Substantial climb => topic predictability at coarser scale; "
                            "paragraph-level concepts worth building."
                            if delta >= 5.0
                            else "Flat/small change => coarsening does not clearly help."
                        )
                    )
        except Exception as e:
            print(f"\n== {name} EVAL FAILED ==\n{type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    if not any_ok:
        print("No encoder completed successfully.")
        return 1

    print(f"\nDone in {time.time() - t0:.1f}s (CPU-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
