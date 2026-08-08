#!/usr/bin/env python3
"""Cheap CPU probes for coarse-steer (Fable §7).

P0 — paragraph/topic centroid stability on the repaired-codec sentence space.
Numpy only at runtime (corpus loader may import torch). No retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.sentence_pred.data import SentenceCorpus, load_sentence_corpus  # noqa: E402

DEFAULT_CORPUS = "artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1"
DEFAULT_OUT = "artifacts/sentence_pred/coarse_steer/probes_P0.json"


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x)) + eps
        return (x / n).astype(np.float32)
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / denom).astype(np.float32)


def _mean_pool_normalize(vecs: np.ndarray) -> np.ndarray:
    return _l2_normalize(vecs.mean(axis=0))


def _paragraph_centroids(doc_vecs: np.ndarray, g: int) -> np.ndarray:
    """Non-overlapping G-sentence mean-pooled, L2-normalized centroids."""
    n = int(doc_vecs.shape[0])
    n_blocks = n // g
    if n_blocks == 0:
        return np.zeros((0, doc_vecs.shape[1]), dtype=np.float32)
    blocks = doc_vecs[: n_blocks * g].reshape(n_blocks, g, -1).mean(axis=1)
    return _l2_normalize(blocks)


def _pairwise_mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine between aligned rows of unit vectors a, b."""
    if a.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.sum(a * b, axis=-1)))


def _cross_pair_mean_cos(
    groups: list[np.ndarray],
    rng: np.random.RandomState,
    n_pairs: int,
) -> float:
    """Mean cosine of random pairs drawn from different groups."""
    eligible = [g for g in groups if g.shape[0] > 0]
    if len(eligible) < 2:
        return float("nan")
    # Flatten with group ids for sampling
    parts = []
    gids = []
    for i, g in enumerate(eligible):
        parts.append(g)
        gids.append(np.full(g.shape[0], i, dtype=np.int64))
    bank = np.concatenate(parts, axis=0)
    group_of = np.concatenate(gids, axis=0)
    n = bank.shape[0]
    if n < 2:
        return float("nan")

    cos_vals: list[float] = []
    attempts = 0
    max_attempts = n_pairs * 20
    while len(cos_vals) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.randint(0, n, size=2)
        if i == j or group_of[i] == group_of[j]:
            continue
        cos_vals.append(float(np.dot(bank[i], bank[j])))
    if not cos_vals:
        return float("nan")
    return float(np.mean(cos_vals))


def _metric_a_paragraph(
    corpus: SentenceCorpus,
    held_ids: np.ndarray,
    g: int,
    rng: np.random.RandomState,
    n_cross_pairs: int = 20_000,
) -> dict:
    para_groups: list[np.ndarray] = []
    consec_cos: list[float] = []

    for doc_id in held_ids:
        s, e = corpus.doc_slice(int(doc_id))
        doc_vecs = corpus.sent_vecs[s:e]
        paras = _paragraph_centroids(doc_vecs, g)
        if paras.shape[0] == 0:
            continue
        para_groups.append(paras)
        if paras.shape[0] >= 2:
            consec_cos.extend(
                float(np.dot(paras[i], paras[i + 1])) for i in range(paras.shape[0] - 1)
            )

    consec_para_cos = float(np.mean(consec_cos)) if consec_cos else float("nan")
    cross_para_cos = _cross_pair_mean_cos(para_groups, rng, n_cross_pairs)
    delta = consec_para_cos - cross_para_cos
    return {
        "G": g,
        "n_consec_pairs": len(consec_cos),
        "n_para_groups": len(para_groups),
        "consec_para_cos": consec_para_cos,
        "cross_para_cos": cross_para_cos,
        "delta": float(delta),
        "pass": bool(delta >= 0.10),
        "kill_band": bool(delta < 0.05),
    }


def _metric_b_doc_half(
    corpus: SentenceCorpus,
    held_ids: np.ndarray,
    rng: np.random.RandomState,
    n_cross_pairs: int = 20_000,
    min_sents: int = 10,
) -> dict:
    half_cos: list[float] = []
    first_halves: list[np.ndarray] = []

    for doc_id in held_ids:
        s, e = corpus.doc_slice(int(doc_id))
        doc_vecs = corpus.sent_vecs[s:e]
        n = int(doc_vecs.shape[0])
        if n < min_sents:
            continue
        mid = n // 2
        c1 = _mean_pool_normalize(doc_vecs[:mid])
        c2 = _mean_pool_normalize(doc_vecs[mid:])
        half_cos.append(float(np.dot(c1, c2)))
        first_halves.append(c1.reshape(1, -1))

    doc_half_cos = float(np.mean(half_cos)) if half_cos else float("nan")
    cross_doc_centroid_cos = _cross_pair_mean_cos(first_halves, rng, n_cross_pairs)
    delta = doc_half_cos - cross_doc_centroid_cos
    return {
        "n_docs": len(half_cos),
        "doc_half_cos": doc_half_cos,
        "cross_doc_centroid_cos": cross_doc_centroid_cos,
        "delta": float(delta),
        "pass": bool(delta >= 0.15),
        "kill_band": bool(delta < 0.10),
    }


def _metric_c_delta_cos(
    corpus: SentenceCorpus,
    held_ids: np.ndarray,
) -> dict:
    """Mean cos(delta_t, delta_{t+1}) where delta_t = z_{t+1} - z_t (raw diffs)."""
    vals: list[float] = []
    for doc_id in held_ids:
        s, e = corpus.doc_slice(int(doc_id))
        z = corpus.sent_vecs[s:e]
        if z.shape[0] < 3:
            continue
        d0 = z[1:-1] - z[:-2]
        d1 = z[2:] - z[1:-1]
        # Normalize deltas for cosine (direction of change)
        d0n = _l2_normalize(d0)
        d1n = _l2_normalize(d1)
        vals.extend(np.sum(d0n * d1n, axis=-1).tolist())
    mean_cos = float(np.mean(vals)) if vals else float("nan")
    return {
        "n_triples": len(vals),
        "delta_cos": mean_cos,
        "expect_lt": -0.2,
        "confirms_mean_reversion": bool(mean_cos < -0.2),
    }


def run_p0(
    corpus: SentenceCorpus,
    max_held_docs: int = 500,
    seed: int = 1234,
) -> dict:
    held = np.asarray(corpus.held_doc_ids, dtype=np.int64)
    if held.shape[0] > max_held_docs:
        rng_pick = np.random.RandomState(seed)
        held = np.sort(rng_pick.choice(held, size=max_held_docs, replace=False))
    rng = np.random.RandomState(seed)

    a5 = _metric_a_paragraph(corpus, held, g=5, rng=rng)
    a10 = _metric_a_paragraph(corpus, held, g=10, rng=np.random.RandomState(seed + 1))
    b = _metric_b_doc_half(corpus, held, rng=np.random.RandomState(seed + 2))
    c = _metric_c_delta_cos(corpus, held)

    delta_a = float(a5["delta"])
    delta_b = float(b["delta"])
    go = bool(delta_a >= 0.10 or delta_b >= 0.15)
    kill = bool(delta_a < 0.05 and delta_b < 0.10)
    if kill:
        overall = "KILL"
    elif go:
        overall = "GO"
    else:
        overall = "MARGINAL"

    return {
        "probe": "P0",
        "corpus": None,  # filled by caller
        "n_held_docs_used": int(held.shape[0]),
        "seed": seed,
        "a_G5": a5,
        "a_G10": a10,
        "b_doc_half": b,
        "c_delta": c,
        "verdicts": {
            "a_pass": bool(a5["pass"]),
            "b_pass": bool(b["pass"]),
            "a_delta": delta_a,
            "b_delta": delta_b,
            "overall": overall,
            "rules": {
                "a_pass_if": "consec_para_cos - cross_para_cos >= +0.10",
                "b_pass_if": "doc_half_cos - cross_doc_centroid_cos >= +0.15",
                "kill_if": "(a) < +0.05 AND (b) < +0.10",
                "go_if": "(a) >= +0.10 OR (b) >= +0.15",
            },
        },
    }


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def print_p0_table(results: dict) -> None:
    a5 = results["a_G5"]
    a10 = results["a_G10"]
    b = results["b_doc_half"]
    c = results["c_delta"]
    v = results["verdicts"]

    print("=" * 72)
    print("P0 — Coarse stability on repaired codec (held docs)")
    print(f"held docs used: {results['n_held_docs_used']}  seed={results['seed']}")
    print("=" * 72)
    print()
    print(f"{'metric':<42} {'value':>10}  {'note'}")
    print("-" * 72)
    print(
        f"{'(a) consec_para_cos G=5':<42} {_fmt(a5['consec_para_cos']):>10}  "
        f"n_pairs={a5['n_consec_pairs']}"
    )
    print(
        f"{'(a) cross_para_cos G=5':<42} {_fmt(a5['cross_para_cos']):>10}  "
        f"control"
    )
    print(
        f"{'(a) delta G=5 (consec-cross)':<42} {_fmt(a5['delta']):>10}  "
        f"{'PASS' if a5['pass'] else 'KILL'}  (need >= +0.10)"
    )
    print(
        f"{'(a) consec_para_cos G=10':<42} {_fmt(a10['consec_para_cos']):>10}  "
        f"n_pairs={a10['n_consec_pairs']}  (context)"
    )
    print(
        f"{'(a) cross_para_cos G=10':<42} {_fmt(a10['cross_para_cos']):>10}  "
        f"control (context)"
    )
    print(
        f"{'(a) delta G=10 (consec-cross)':<42} {_fmt(a10['delta']):>10}  "
        f"context only"
    )
    print(
        f"{'(b) doc_half_cos':<42} {_fmt(b['doc_half_cos']):>10}  "
        f"n_docs={b['n_docs']}"
    )
    print(
        f"{'(b) cross_doc_centroid_cos':<42} {_fmt(b['cross_doc_centroid_cos']):>10}  "
        f"control"
    )
    print(
        f"{'(b) delta (half-cross)':<42} {_fmt(b['delta']):>10}  "
        f"{'PASS' if b['pass'] else 'KILL'}  (need >= +0.15)"
    )
    print(
        f"{'(c) consec sentence-delta cos':<42} {_fmt(c['delta_cos']):>10}  "
        f"expect < -0.2; n={c['n_triples']}"
    )
    print("-" * 72)
    print(
        f"Overall verdict: {v['overall']}  "
        f"(GO if a>=+0.10 or b>=+0.15; KILL if a<+0.05 and b<+0.10)"
    )
    print(
        f"  a_pass={v['a_pass']} (delta={_fmt(v['a_delta'])})  "
        f"b_pass={v['b_pass']} (delta={_fmt(v['b_delta'])})"
    )
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Coarse-steer probes (Fable §7)")
    p.add_argument("--probe", required=True, choices=["P0"], help="Probe id")
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="SentenceCorpus cache dir")
    p.add_argument("--out", default=DEFAULT_OUT, help="JSON output path")
    p.add_argument("--max-held-docs", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    if args.probe != "P0":
        raise SystemExit(f"Unsupported probe: {args.probe}")

    corpus_path = Path(args.corpus)
    print(f"Loading corpus from {corpus_path} ...", flush=True)
    corpus = load_sentence_corpus(corpus_path)
    print(
        f"  n_docs={corpus.n_docs} n_sent={corpus.n_sentences} "
        f"held={len(corpus.held_doc_ids)}",
        flush=True,
    )

    results = run_p0(corpus, max_held_docs=args.max_held_docs, seed=args.seed)
    results["corpus"] = str(corpus_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print_p0_table(results)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
