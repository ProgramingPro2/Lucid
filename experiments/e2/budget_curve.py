"""CoT budget dose-response on the E2 task families (calibration only).

The capability screen fixes one CoT budget per family. This script asks whether a
family's apparent weakness is a capability limit or a truncation artifact, by
holding the items and the direct arm fixed and varying only the number of new
tokens available for explicit working.

It is calibration, not a confirmatory result: it decides how to match FLOPs in
the trained arm, and it is never pooled with E2's n.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.e2.budget_curve \
        --model Qwen/Qwen3-1.7B --family all --n 20 \
        --budgets 128 256 512 1024 2048 --out artifacts/e2/budget_1.7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.protocol.stats import paired_summary
from experiments.runner.model import LM

from .evaluate import _items, run_mode
from .tasks import chance


def summarise_budgets(
    rows: list[dict], budgets: list[int]
) -> dict:
    """Per-family accuracy and CoT−direct contrast at each budget.

    ``rows`` must carry ``budget`` for CoT rows; direct rows are shared across
    budgets and must have ``mode == "direct"``.
    """
    out: dict = {}
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family]
        direct = [r for r in fam if r["mode"] == "direct"]
        if not direct:
            continue
        d_acc = [r["correct"] for r in direct]
        entry: dict = {
            "n": len(direct),
            "chance": chance(family),
            "direct": {
                "accuracy": round(float(np.mean(d_acc)), 4),
                "marker_rate": round(
                    float(np.mean([r["has_marker"] for r in direct])), 4
                ),
                "mean_chars": int(np.mean([r["chars"] for r in direct])),
            },
            "cot": {},
        }
        for budget in budgets:
            cot = [
                r
                for r in fam
                if r["mode"] == "cot" and r.get("budget") == budget
            ]
            if not cot:
                continue
            diff = paired_summary([r["correct"] for r in cot], d_acc)
            entry["cot"][str(budget)] = {
                "accuracy": round(float(np.mean([r["correct"] for r in cot])), 4),
                "marker_rate": round(
                    float(np.mean([r["has_marker"] for r in cot])), 4
                ),
                "mean_chars": int(np.mean([r["chars"] for r in cot])),
                "cot_minus_direct": {
                    "mean": round(diff.mean, 4),
                    "ci_low": round(diff.ci_low, 4),
                    "ci_high": round(diff.ci_high, 4),
                    "n": diff.n,
                },
            }
        out[family] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--family", default="all", choices=["f1", "f2", "f3", "all"])
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--max-new-tokens-direct", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    families = ["f1", "f2", "f3"] if args.family == "all" else [args.family]
    lm = LM(args.model)
    rows: list[dict] = []

    for family in families:
        items = _items(family, "test", args.n, args.start)
        print(f"[budget] {family}: {len(items)} items", flush=True)
        direct = run_mode(
            lm,
            items,
            "direct",
            max_new_tokens=args.max_new_tokens_direct,
            batch_size=args.batch_size,
        )
        rows.extend(direct)
        for budget in args.budgets:
            print(f"[budget] {family} cot @ {budget}", flush=True)
            cot = run_mode(
                lm,
                items,
                "cot",
                max_new_tokens=budget,
                batch_size=args.batch_size,
            )
            for row in cot:
                row["budget"] = budget
            rows.extend(cot)

    report = {
        "note": (
            "CoT budget dose-response. Calibration for matched-FLOP design; not "
            "the trained E2 arm and never pooled with it."
        ),
        "model": args.model,
        "n_per_family": args.n,
        "budgets": args.budgets,
        "thinking": False,
        "summary": summarise_budgets(rows, args.budgets),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
