"""Score every harvested document with the base model's own NLL.

The memorisation screen (docs/CORPUS.md §4 step 7) drops the lowest-NLL decile:
low NLL means the model already finds the text easy, which is what leakage and
boilerplate look like. This script produces that table. It is deliberately
separate from ``build`` so a long GPU pass is not repeated when a non-NLL gate
changes.

Scoring uses up to four evenly spaced windows per document, so a 20 k-token paper
is not represented by its abstract alone and a short abstract is not padded into
a different distribution. The reported value is the token-weighted mean NLL in
nats/token.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.corpus.nll \
        --docs-jsonl artifacts/corpus/biorxiv_2026.jsonl \
        --out artifacts/corpus/nll_qwen3_1.7b.json \
        --max-tokens 2048 --limit 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .harvest import load_documents


def token_windows(ids: list[int], length: int, n_windows: int = 4) -> list[list[int]]:
    """Up to ``n_windows`` non-overlapping-as-possible windows of ``length``.

    Short documents are one window. Long documents are sampled evenly from start
    to end, so methods and discussion are both represented. Deterministic: no
    RNG, so the table can be regenerated exactly.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if len(ids) <= length:
        return [ids]
    n = max(1, n_windows)
    if n == 1:
        return [ids[:length]]
    starts = np.linspace(0, len(ids) - length, num=n, dtype=int)
    return [ids[int(s) : int(s) + length] for s in starts]


def score_document(ids: list[int], loss_fn, *, length: int, n_windows: int = 4) -> float:
    """Token-weighted mean NLL over the selected windows.

    ``loss_fn`` maps a non-empty id list to a mean NLL. Injecting it keeps the
    windowing testable without a GPU.
    """
    windows = [w for w in token_windows(ids, length, n_windows) if len(w) > 1]
    if not windows:
        return float("nan")
    total_tokens = sum(len(w) - 1 for w in windows)
    if total_tokens <= 0:
        return float("nan")
    return float(
        sum(loss_fn(w) * (len(w) - 1) for w in windows) / total_tokens
    )


def _model_loss_fn(lm, device: str):
    import torch

    def loss_fn(ids: list[int]) -> float:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = lm.model(input_ids=x, labels=x)
        return float(out.loss.item())

    return loss_fn


def score_jsonl(
    docs_path: str | Path,
    out_path: str | Path,
    *,
    model_id: str = "Qwen/Qwen3-1.7B-Base",
    max_tokens: int = 2048,
    n_windows: int = 4,
    limit: int | None = None,
) -> dict:
    from experiments.runner.model import LM

    docs = load_documents(docs_path)
    if limit is not None:
        docs = docs[:limit]
    out_path = Path(out_path)
    table: dict[str, float] = {}
    if out_path.exists():
        table = json.loads(out_path.read_text()).get("nll", {})

    todo = [d for d in docs if d.doc_id not in table]
    print(f"[nll] {len(docs)} documents, {len(todo)} to score", flush=True)
    if not todo:
        return table

    lm = LM(model_id)
    device = str(lm.model.device)
    loss_fn = _model_loss_fn(lm, device)
    started = time.time()
    scored = 0
    for doc in todo:
        ids = lm.tokenizer(doc.text, add_special_tokens=False)["input_ids"]
        value = score_document(
            ids, loss_fn, length=max_tokens, n_windows=n_windows
        )
        if math.isfinite(value):
            table[doc.doc_id] = value
        scored += 1
        if scored % 10 == 0:
            rate = scored / max(1e-9, time.time() - started)
            print(f"[nll] {scored}/{len(todo)}  {rate * 60:.1f} docs/min", flush=True)
            payload = {
                "model": model_id,
                "max_tokens": max_tokens,
                "n_windows": n_windows,
                "nll": table,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": model_id,
                "max_tokens": max_tokens,
                "n_windows": n_windows,
                "nll": table,
            },
            indent=2,
        )
    )
    print(f"[nll] wrote {len(table)} scores to {out_path}", flush=True)
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--docs-jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    score_jsonl(
        args.docs_jsonl,
        args.out,
        model_id=args.model,
        max_tokens=args.max_tokens,
        n_windows=args.n_windows,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
