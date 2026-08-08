#!/usr/bin/env python3
"""Phase-3 pooled-topic renderability A/B probe.

For each held G=5 paragraph block:
  1. paragraph_concept = L2-normalize(mean of its 5 sentence vectors)
  2. render G sentences under that frozen concept (prefix-chained)
  3. MiniLM cos(rendered paragraph text, source paragraph text)

Compare phase-3 writer vs old two-channel writer on the same items.

Gate: phase-3 mean cos >= 0.25; delta vs old >= +0.08 is a clear win.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from conceptpoc.jepa.tokenizer import Tokenizer  # noqa: E402
from conceptpoc.persistence import load_writer  # noqa: E402
from experiments.sentence_pred.data import load_sentence_corpus  # noqa: E402
from experiments.sentence_pred.forward_gen import _load_minilm, _l2  # noqa: E402
from experiments.sentence_pred.paragraph_writer import (  # noqa: E402
    G_DEFAULT,
    join_paragraph,
    render_paragraph,
)
from conceptpoc.writer.train_phase3 import pool_paragraph_concept  # noqa: E402

DEFAULT_CORPUS = "artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1"
DEFAULT_P3 = "artifacts/writer_p3/writer_paragraph.pt"
DEFAULT_P25 = "artifacts/writer_p25/writer_two_channel.pt"
DEFAULT_OUT = "artifacts/writer_p3/p3_probe.json"

PASS_GATE = 0.25
WIN_DELTA = 0.08


def _collect_held_paragraphs(corpus, g: int, n_items: int, seed: int) -> list[dict]:
    rng = np.random.RandomState(seed)
    items: list[dict] = []
    held = np.asarray(corpus.held_doc_ids, dtype=np.int64)
    for doc_id in held.tolist():
        a, b = corpus.doc_slice(int(doc_id))
        n = b - a
        if n < g:
            continue
        # Prefer a mid-doc block so first_prefix exists (matches inference chaining).
        if n >= g + 1:
            block_start = 1
            first_prefix = corpus.sentences[a + block_start - 1]
        else:
            block_start = 0
            first_prefix = None
        lo = a + block_start
        hi = lo + g
        texts = [corpus.sentences[i] for i in range(lo, hi)]
        vecs = corpus.sent_vecs[lo:hi]
        items.append(
            {
                "doc_id": int(doc_id),
                "block_start": int(block_start),
                "para_text": " ".join(texts),
                "first_prefix": first_prefix,
                "centroid": pool_paragraph_concept(vecs),
            }
        )
    rng.shuffle(items)
    return items[: int(n_items)]


@torch.no_grad()
def _score_writer(
    writer,
    tokenizer,
    items: list[dict],
    minilm,
    device: str,
    g: int,
    label: str,
) -> dict:
    scores: list[float] = []
    writer.to(device)
    writer.eval()
    for i, it in enumerate(items):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[{label}] item {i + 1}/{len(items)}", flush=True)
        concept = torch.from_numpy(it["centroid"][None, :]).to(device)
        rendered = render_paragraph(
            writer,
            tokenizer,
            concept,
            first_prefix=it["first_prefix"],
            n_sentences=g,
            temperature=0.8,
            top_k=50,
            max_tokens=96,
        )
        rend_text = join_paragraph(rendered)
        if not rend_text.strip():
            scores.append(0.0)
            continue
        emb = _l2(
            np.asarray(
                minilm.encode([rend_text, it["para_text"]]),
                dtype=np.float32,
            )
        )
        scores.append(float(np.dot(emb[0], emb[1])))
    arr = np.asarray(scores, dtype=np.float32)
    return {
        "n_items": len(scores),
        "mean_cos": float(arr.mean()) if len(arr) else float("nan"),
        "median_cos": float(np.median(arr)) if len(arr) else float("nan"),
        "std_cos": float(arr.std()) if len(arr) else float("nan"),
        "scores": [float(x) for x in scores],
    }


def run_probe(
    corpus_dir: str = DEFAULT_CORPUS,
    phase3_path: str = DEFAULT_P3,
    phase25_path: str = DEFAULT_P25,
    n_items: int = 300,
    g: int = G_DEFAULT,
    seed: int = 1234,
    device: str = "auto",
    out_path: str = DEFAULT_OUT,
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = load_sentence_corpus(corpus_dir)
    items = _collect_held_paragraphs(corpus, g=g, n_items=n_items, seed=seed)
    if not items:
        raise RuntimeError("no held paragraph blocks found")
    print(f"[p3_probe] scoring {len(items)} held G={g} paragraphs on {device}", flush=True)

    tokenizer = Tokenizer("gpt2")
    minilm = _load_minilm()

    # Phase-3 writer
    p3 = load_writer(phase3_path, device=device)
    p3_stats = _score_writer(p3, tokenizer, items, minilm, device, g, "phase3")
    del p3
    if device == "cuda":
        torch.cuda.empty_cache()

    # Old two-channel control (same items)
    old = load_writer(phase25_path, device=device)
    old_stats = _score_writer(old, tokenizer, items, minilm, device, g, "two_channel")
    del old
    if device == "cuda":
        torch.cuda.empty_cache()

    delta = float(p3_stats["mean_cos"] - old_stats["mean_cos"])
    pass_gate = bool(p3_stats["mean_cos"] >= PASS_GATE)
    clear_win = bool(delta >= WIN_DELTA)
    if pass_gate and clear_win:
        verdict = "PASS_CLEAR_WIN"
    elif pass_gate:
        verdict = "PASS"
    elif p3_stats["mean_cos"] <= 0.10:
        verdict = "KILL"
    else:
        verdict = "FAIL"

    results = {
        "probe": "P3_pooled_renderability",
        "n_items": len(items),
        "G": g,
        "device": device,
        "phase3_path": phase3_path,
        "phase25_path": phase25_path,
        "phase3": {
            "mean_cos": p3_stats["mean_cos"],
            "median_cos": p3_stats["median_cos"],
            "std_cos": p3_stats["std_cos"],
        },
        "two_channel": {
            "mean_cos": old_stats["mean_cos"],
            "median_cos": old_stats["median_cos"],
            "std_cos": old_stats["std_cos"],
        },
        "delta_phase3_minus_old": delta,
        "gates": {
            "pass_if": f"phase3 mean MiniLM cos >= {PASS_GATE}",
            "clear_win_if": f"delta >= +{WIN_DELTA}",
            "kill_if": "phase3 mean <= 0.10",
            "pass": pass_gate,
            "clear_win": clear_win,
        },
        "verdict": verdict,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print(f"[p3_probe] wrote {out}", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-3 pooled renderability A/B probe")
    p.add_argument("--corpus-dir", type=str, default=DEFAULT_CORPUS)
    p.add_argument("--phase3", type=str, default=DEFAULT_P3)
    p.add_argument("--phase25", type=str, default=DEFAULT_P25)
    p.add_argument("--n-items", type=int, default=300)
    p.add_argument("--g", type=int, default=G_DEFAULT)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = p.parse_args()
    run_probe(
        corpus_dir=args.corpus_dir,
        phase3_path=args.phase3,
        phase25_path=args.phase25,
        n_items=args.n_items,
        g=args.g,
        seed=args.seed,
        device=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
