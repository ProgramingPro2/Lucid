"""Phase 0: encode corpus-v2 paragraphs with the frozen repaired JEPA codec.

Produces ``artifacts/corpus_v2/zbank.npz``:
  paragraph_z (N, 512) float16, doc_id, para_idx, token_len
plus consecutive-pair cosine drift stats and a doc-disjoint 98/1/1 split.

Cache is keyed by input file mtimes (corpus + JEPA checkpoint). Re-run is a
no-op when the cache is fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.persistence import load_joint  # noqa: E402
from experiments.pivot.corpus_v2 import load_corpus_v2  # noqa: E402

DEFAULT_CORPUS = _ROOT / "data" / "corpus_v2.txt"
DEFAULT_JEPA = _ROOT / "artifacts" / "jepa_repair_1m" / "jepa" / "joint.pt"
DEFAULT_OUT = _ROOT / "artifacts" / "corpus_v2" / "zbank.npz"
DEFAULT_META = _ROOT / "artifacts" / "corpus_v2" / "zbank_meta.json"
MAX_TOKENS = 128
BATCH = 256
SPLIT = (0.98, 0.01, 0.01)
SEED = 42


def _mtime(p: Path) -> float:
    return float(p.stat().st_mtime) if p.exists() else -1.0


def cache_key(corpus: Path, jepa: Path, max_tokens: int) -> str:
    # v2 = pooled sentence centroids (matches 0.705 drift protocol)
    raw = (
        f"v2-pooled|{corpus.resolve()}|{_mtime(corpus)}|"
        f"{jepa.resolve()}|{_mtime(jepa)}|{max_tokens}"
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def flatten_paragraphs(
    docs: list[list[list[str]]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (para_texts, doc_id, para_idx)."""
    texts: list[str] = []
    doc_ids: list[int] = []
    para_idxs: list[int] = []
    for di, doc in enumerate(docs):
        for pi, para in enumerate(doc):
            texts.append(" ".join(para).strip())
            doc_ids.append(di)
            para_idxs.append(pi)
    return (
        texts,
        np.asarray(doc_ids, dtype=np.int32),
        np.asarray(para_idxs, dtype=np.int32),
    )


def _token_lens(codec: JepaCodec, texts: list[str], batch: int) -> np.ndarray:
    """Approximate token lengths with the JEPA tokenizer (capped at max_tokens)."""
    out = np.zeros(len(texts), dtype=np.int16)
    tok = codec._tok
    mt = codec._max_tokens
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids, pad = tok.encode_batch(chunk, mt)
        # pad is True where padded; count non-pad
        lengths = (~pad).sum(dim=1).cpu().numpy().astype(np.int16)
        out[i : i + len(chunk)] = lengths
    return out


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return (x / (float(np.linalg.norm(x)) + eps)).astype(np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / n).astype(np.float32)


def pool_paragraph_centroid(sent_vecs: np.ndarray) -> np.ndarray:
    """L2-normalize(mean of sentence JEPA vectors) — the pooled fiber centroid."""
    if sent_vecs.size == 0:
        return np.zeros(512, dtype=np.float32)
    return _l2_normalize(np.asarray(sent_vecs, dtype=np.float32).mean(axis=0))


@torch.no_grad()
def encode_paragraphs_pooled(
    codec: JepaCodec,
    docs: list[list[list[str]]],
    batch: int = BATCH,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode every sentence, mean-pool within each paragraph → centroids.

    This matches the measured consecutive-para band (~0.705): G-pooled sentence
    JEPA vectors, not a single truncated paragraph encode (which collapses to
    ~0.38 on v2 because max_tokens=128 truncates long Wiki paragraphs).
    """
    # Flatten sentences with paragraph ownership.
    sent_texts: list[str] = []
    para_of_sent: list[int] = []
    doc_ids: list[int] = []
    para_idxs: list[int] = []
    para_id = 0
    for di, doc in enumerate(docs):
        for pi, para in enumerate(doc):
            for s in para:
                t = s.strip()
                if not t:
                    continue
                sent_texts.append(t)
                para_of_sent.append(para_id)
            doc_ids.append(di)
            para_idxs.append(pi)
            para_id += 1
    n_paras = para_id
    dim = int(codec.dim)
    print(
        f"[encode_bank] encoding {len(sent_texts)} sentences → {n_paras} para centroids",
        flush=True,
    )

    sent_z = np.zeros((len(sent_texts), dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(sent_texts), batch):
        chunk = sent_texts[i : i + batch]
        z = codec.encode(chunk)
        sent_z[i : i + len(chunk)] = z
        if progress and ((i // batch) % 40 == 0 or i + batch >= len(sent_texts)):
            done = min(i + batch, len(sent_texts))
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"[encode_bank] {done}/{len(sent_texts)} sents "
                f"({100.0 * done / max(len(sent_texts), 1):.1f}%) "
                f"{rate:.1f} sent/s",
                flush=True,
            )

    # Pool per paragraph
    zs = np.zeros((n_paras, dim), dtype=np.float32)
    counts = np.zeros(n_paras, dtype=np.int32)
    for si, pid in enumerate(para_of_sent):
        zs[pid] += sent_z[si]
        counts[pid] += 1
    for pid in range(n_paras):
        if counts[pid] > 0:
            zs[pid] = _l2_normalize(zs[pid] / float(counts[pid]))
        else:
            zs[pid] = _l2_normalize(np.random.randn(dim).astype(np.float32))

    # token_len ≈ sentence count * rough proxy; real token lens filled by caller
    token_len = counts.astype(np.int16)  # placeholder overwritten
    return (
        zs,
        np.asarray(doc_ids, dtype=np.int32),
        np.asarray(para_idxs, dtype=np.int32),
        token_len,
    )


def consecutive_pair_cos(
    z: np.ndarray, doc_id: np.ndarray, para_idx: np.ndarray
) -> np.ndarray:
    """Cosine of consecutive paragraphs within the same document."""
    # Rows are already ordered by (doc_id, para_idx) from flatten_paragraphs.
    cos_vals: list[float] = []
    for i in range(len(z) - 1):
        if doc_id[i] == doc_id[i + 1] and para_idx[i] + 1 == para_idx[i + 1]:
            cos_vals.append(float(np.dot(z[i], z[i + 1])))
    return np.asarray(cos_vals, dtype=np.float32)


def doc_disjoint_split(
    n_docs: int, seed: int = SEED, ratios: tuple[float, float, float] = SPLIT
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    ids = np.arange(n_docs, dtype=np.int32)
    rng.shuffle(ids)
    n_train = int(round(ratios[0] * n_docs))
    n_val = int(round(ratios[1] * n_docs))
    # remainder -> held/test
    train = np.sort(ids[:n_train])
    val = np.sort(ids[n_train : n_train + n_val])
    held = np.sort(ids[n_train + n_val :])
    return {"train_doc_ids": train, "val_doc_ids": val, "held_doc_ids": held}


def build_bank(
    corpus_path: Path = DEFAULT_CORPUS,
    jepa_path: Path = DEFAULT_JEPA,
    out_path: Path = DEFAULT_OUT,
    meta_path: Path = DEFAULT_META,
    max_tokens: int = MAX_TOKENS,
    batch: int = BATCH,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    jepa_path = Path(jepa_path)
    out_path = Path(out_path)
    meta_path = Path(meta_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    key = cache_key(corpus_path, jepa_path, max_tokens)
    if out_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_key") == key:
            print(f"[encode_bank] cache hit ({key}); skipping encode", flush=True)
            return meta

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[encode_bank] loading corpus {corpus_path}", flush=True)
    docs = load_corpus_v2(corpus_path)
    texts, _doc_flat, _para_flat = flatten_paragraphs(docs)
    print(
        f"[encode_bank] {len(docs)} docs, {len(texts)} paragraphs",
        flush=True,
    )

    print(f"[encode_bank] loading JEPA codec from {jepa_path} on {device}", flush=True)
    model, tok = load_joint(str(jepa_path), device=device)
    codec = JepaCodec(model, tok, max_tokens=max_tokens, device=device)

    t0 = time.time()
    z, doc_id, para_idx, _sent_counts = encode_paragraphs_pooled(
        codec, docs, batch=batch, progress=True
    )
    assert len(texts) == len(z), f"para count mismatch {len(texts)} vs {len(z)}"
    token_len = _token_lens(codec, texts, batch=min(batch, 64))
    encode_s = time.time() - t0

    # free encoder VRAM before writing
    del codec, model, tok
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    z16 = z.astype(np.float16)
    pair_cos = consecutive_pair_cos(z, doc_id, para_idx)
    drift_mean = float(pair_cos.mean()) if pair_cos.size else float("nan")
    drift_std = float(pair_cos.std()) if pair_cos.size else float("nan")
    split = doc_disjoint_split(len(docs), seed=SEED, ratios=SPLIT)

    np.savez_compressed(
        out_path,
        paragraph_z=z16,
        doc_id=doc_id,
        para_idx=para_idx,
        token_len=token_len,
        consec_pair_cos=pair_cos,
        **split,
    )

    meta: dict[str, Any] = {
        "cache_key": key,
        "corpus": str(corpus_path),
        "jepa": str(jepa_path),
        "out": str(out_path),
        "n_docs": int(len(docs)),
        "n_paragraphs": int(len(texts)),
        "n_consec_pairs": int(pair_cos.size),
        "drift_mean": drift_mean,
        "drift_std": drift_std,
        "drift_band_expected": [0.655, 0.755],
        "drift_in_band": bool(0.655 <= drift_mean <= 0.755) if pair_cos.size else False,
        "max_tokens": int(max_tokens),
        "batch": int(batch),
        "device": device,
        "encode_s": float(encode_s),
        "split": {
            "train_docs": int(split["train_doc_ids"].size),
            "val_docs": int(split["val_doc_ids"].size),
            "held_docs": int(split["held_doc_ids"].size),
            "ratios": list(SPLIT),
            "seed": SEED,
        },
        "encoding": "pooled_sentence_centroids",
        "dtype": "float16",
        "dim": int(z.shape[1]),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[encode_bank] wrote {out_path}  n={meta['n_paragraphs']}  "
        f"drift={drift_mean:.4f}±{drift_std:.4f}  "
        f"pairs={meta['n_consec_pairs']}  ({encode_s:.1f}s)",
        flush=True,
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Encode corpus-v2 paragraph JEPA bank")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--jepa", type=Path, default=DEFAULT_JEPA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    build_bank(
        corpus_path=args.corpus,
        jepa_path=args.jepa,
        out_path=args.out,
        meta_path=args.meta,
        max_tokens=args.max_tokens,
        batch=args.batch,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
