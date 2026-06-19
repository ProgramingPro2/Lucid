"""Export PCA 3D embeddings for the concept visualizer.

Reads a corpus cache (.npz) and writes rating-app/viz/public/embeddings.json.

Usage:
    python tools/export_embeddings.py
    python tools/export_embeddings.py --config configs/jepa_austen.json
    python tools/export_embeddings.py --cache artifacts/austen_jepa/cache/corpus_*.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rating-app" / "viz" / "public" / "embeddings.json"


def _cache_from_config(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts_dir = ROOT / cfg["artifacts_dir"] / "cache"
    caches = sorted(artifacts_dir.glob("corpus_*.npz"))
    if not caches:
        raise FileNotFoundError(
            f"No cache in {artifacts_dir}. Run eval or prepare with --config {config_path}"
        )
    return caches[-1]


def _latest_cache() -> Path:
    caches = sorted(
        ROOT.glob("artifacts/**/cache/corpus_*.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    if caches:
        return caches[-1]
    raise FileNotFoundError(
        "No corpus cache found. Run eval/prepare on a config, e.g.:\n"
        "  python -m conceptpoc.cli --config configs/jepa_austen.json eval\n"
        "  python tools/export_embeddings.py --config configs/jepa_austen.json"
    )


def resolve_cache(config: Path | None, cache: Path | None) -> Path:
    if cache is not None:
        if cache.exists():
            return cache
        matches = sorted(ROOT.glob(str(cache)))
        if matches:
            return matches[-1]
        raise FileNotFoundError(f"Cache not found: {cache}")
    if config is not None:
        return _cache_from_config(config)
    return _latest_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3D embeddings for the visualizer")
    parser.add_argument(
        "--config",
        type=Path,
        help="Config JSON; use its artifacts_dir/cache/ (e.g. configs/jepa_austen.json)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Explicit .npz path (glob ok), overrides --config",
    )
    args = parser.parse_args()

    cache_path = resolve_cache(args.config, args.cache)
    print(f"Loading {cache_path}")
    d = np.load(cache_path, allow_pickle=True)
    vecs = d["vectors"].astype("float32")
    sents = list(d["sentences"])

    print(f"Projecting {len(sents):,} vectors with PCA...")
    xyz = PCA(n_components=3).fit_transform(vecs)
    cl = KMeans(n_clusters=12, n_init=10, random_state=0).fit_predict(vecs)

    out = [
        {
            "id": i,
            "text": sents[i],
            "x": float(xyz[i, 0]),
            "y": float(xyz[i, 1]),
            "z": float(xyz[i, 2]),
            "cluster": int(cl[i]),
        }
        for i in range(len(sents))
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"Wrote {len(out):,} points to {OUT}")


if __name__ == "__main__":
    main()
