#!/usr/bin/env python3
"""G1-pilot table for the R1 encoder-repair run (Fable §1.3).

Builds a held-out sentence corpus from the trained repair checkpoint, runs
``evalsuite.m1_table``, writes ``artifacts/jepa_repair_100k/m1_pilot.json``,
and prints the G1-pilot pass/fail table. Reconstruction is skipped for pilot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.sentence_pred.data import build_sentence_corpus  # noqa: E402
from experiments.sentence_pred.evalsuite import m1_table  # noqa: E402


def _gate_rows(table: dict) -> list[dict]:
    adj = float(table["adjacent_cos"])
    within = float(table["within_doc_shuf_cos"])
    cross = float(table["cross_doc_shuf_cos"])
    floor = float(table["floor_1_sqrt_d"])
    m3 = float(table["m3_identity_indoc_top1"])
    mean_cands = float(table["mean_distinct_candidates"])
    chance = 1.0 / mean_cands if mean_cands > 0 else float("nan")
    er = float(table["effective_rank"]["effective_rank"])
    dim = int(table.get("dim", table["effective_rank"].get("dim", 512)))

    rows = [
        {
            "metric": "adjacent_cos",
            "value": adj,
            "threshold": ">= 0.15",
            "pass": adj >= 0.15,
        },
        {
            "metric": "adjacent - within_doc_shuf",
            "value": adj - within,
            "threshold": ">= +0.05",
            "pass": (adj - within) >= 0.05,
        },
        {
            "metric": "adjacent_cos (upper)",
            "value": adj,
            "threshold": "<= 0.60",
            "pass": adj <= 0.60,
        },
        {
            "metric": "cross_doc_shuf_cos",
            "value": cross,
            "threshold": "<= 0.10",
            "pass": cross <= 0.10,
        },
        {
            "metric": "M3-identity indoc top1",
            "value": m3,
            "threshold": f">= chance+5pts (chance={chance:.4f})",
            "pass": m3 >= (chance + 0.05) if chance == chance else False,
            "chance": chance,
            "mean_distinct_candidates": mean_cands,
        },
        {
            "metric": "effective_rank",
            "value": er,
            "threshold": f">= 100/{dim}",
            "pass": er >= 100.0,
            "dim": dim,
        },
        {
            "metric": "floor_1_sqrt_d",
            "value": floor,
            "threshold": "(report only)",
            "pass": True,
        },
        {
            "metric": "within_doc_shuf_cos",
            "value": within,
            "threshold": "(report only)",
            "pass": True,
        },
    ]
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="G1-pilot m1 table for jepa_repair_100k")
    p.add_argument(
        "--ckpt",
        default="artifacts/jepa_repair_100k/jepa/joint.pt",
        help="trained joint JEPA checkpoint",
    )
    p.add_argument("--corpus", default="data/corpus_100k.txt")
    p.add_argument("--out", default="artifacts/jepa_repair_100k/m1_pilot.json")
    p.add_argument("--cache-dir", default="artifacts/jepa_repair_100k/m1_corpus")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--n-pairs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if _cuda() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    print(f"building sentence corpus from {args.corpus} with {ckpt} on {args.device}...")
    corpus = build_sentence_corpus(
        corpus_path=args.corpus,
        jepa_ckpt=str(ckpt),
        out_dir=args.cache_dir,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        min_sents=4,
        held_out_frac=0.10,
        seed=args.seed,
        detok_wikitext=True,
        device=args.device,
    )
    print(
        f"corpus: {corpus.n_docs} docs, {corpus.n_sentences} sents, "
        f"held_out={len(corpus.held_doc_ids)}"
    )

    table = m1_table(None, corpus, n_pairs=args.n_pairs, seed=args.seed)
    if "error" in table:
        raise SystemExit(f"m1_table failed: {table}")

    rows = _gate_rows(table)
    gate_pass = all(r["pass"] for r in rows if r["threshold"] != "(report only)")
    payload = {
        "ckpt": str(ckpt),
        "corpus": args.corpus,
        "n_pairs": table.get("n_pairs"),
        "m1": table,
        "g1_pilot": rows,
        "g1_pilot_pass": gate_pass,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"wrote {out}")

    print("\n=== G1-pilot table ===")
    print(f"{'metric':<32} {'value':>10}  {'threshold':<36}  result")
    print("-" * 90)
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        if r["threshold"] == "(report only)":
            mark = "—"
        print(
            f"{r['metric']:<32} {r['value']:>10.4f}  {r['threshold']:<36}  {mark}"
        )
    print("-" * 90)
    print(f"G1-pilot overall: {'PASS' if gate_pass else 'FAIL'}")
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(type(o))


if __name__ == "__main__":
    raise SystemExit(main())
