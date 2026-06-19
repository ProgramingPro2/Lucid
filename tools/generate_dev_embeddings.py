"""Generate dev sample embeddings when no ML cache exists.

Reads sentences from data/books_corpus.txt, assigns random 512-d vectors,
PCA-projects to 3D, clusters with KMeans, and writes embeddings.json.

This is a dev-only fallback — for real semantic layout run tools/export_embeddings.py
after `conceptpoc prepare`.

Usage:
    python tools/generate_dev_embeddings.py [--limit 20000]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "books_corpus.txt"
OUT = ROOT / "rating-app" / "viz" / "public" / "embeddings.json"

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _segment_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for chunk in _SENT_RE.split(para):
            chunk = chunk.strip()
            if len(chunk) >= 12:
                sentences.append(chunk)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dev embeddings JSON")
    parser.add_argument("--limit", type=int, default=20_000, help="Max sentences to export")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS}")

    print(f"Reading {CORPUS}")
    sentences = _segment_sentences(CORPUS.read_text(encoding="utf-8"))[: args.limit]
    if not sentences:
        raise RuntimeError("No sentences extracted from corpus")

    rng = np.random.default_rng(args.seed)
    n = len(sentences)
    dim = 512
    print(f"Generating random {n:,} x {dim} vectors...")
    vecs = rng.standard_normal((n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8

    print("PCA -> 3D, KMeans -> 12 clusters...")
    xyz = PCA(n_components=3, random_state=args.seed).fit_transform(vecs)
    clusters = KMeans(n_clusters=12, n_init=10, random_state=args.seed).fit_predict(vecs)

    out = [
        {
            "id": i,
            "text": sentences[i],
            "x": float(xyz[i, 0]),
            "y": float(xyz[i, 1]),
            "z": float(xyz[i, 2]),
            "cluster": int(clusters[i]),
        }
        for i in range(n)
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"Wrote {len(out):,} points to {OUT}")


if __name__ == "__main__":
    main()
