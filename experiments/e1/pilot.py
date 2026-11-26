"""Run an E1 pilot grid and summarise it.

This is a *pilot*: it estimates the per-seed paired SD so the confirmatory
sample size can be fixed, and it produces the first real context-gain numbers.
It never declares a PASS — that requires the pre-registered *n* from
``docs/PROTOCOL.md`` §5.4 on the sealed post-cutoff corpus.

Each run is a separate process, so a crash in one arm cannot corrupt another
and GPU memory is returned to the driver between runs. Existing output files
are skipped, which makes the driver resumable.

Usage::

    PYTHONPATH=src .venv/bin/python -m experiments.e1.pilot \
        --max-docs 4000 --out-dir artifacts/e1/pilot

Each arm's token count defaults to its arm-table budget. ``--tokens``
overrides every arm and warns when that override disagrees with the table.
The voided 2026-11-20 pilot was launched with one shared ``--tokens``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.e1.arms import resolve_token_budget  # noqa: E402
from experiments.protocol.stats import pilot_sd_inflation, required_n  # noqa: E402

# The pilot plan. A and B get more seeds because their paired difference is the
# quantity that sizes the confirmatory run; the other arms are descriptive.
DEFAULT_PLAN: dict[str, list[int]] = {
    "A": [0, 1, 2, 3],
    "B": [0, 1, 2, 3],
    "A_prime": [0, 1],
    "B_prime": [0, 1],
    "E": [0, 1],
    "D": [0, 1],
    "N": [0],
    "PC_hi": [0, 1],
    "PC_lo": [0, 1],
}


def train_command(
    arm: str,
    seed: int,
    args: argparse.Namespace,
    out_path: Path,
) -> list[str]:
    """The ``experiments.e1.train`` invocation for one pilot cell.

    ``--tokens`` is omitted when the pilot was not given an override, so
    the trainer uses that arm's own budget. A shared override is forwarded
    and the trainer prints the mismatch warning.
    """
    cmd = [
        sys.executable,
        "-m",
        "experiments.e1.train",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--context",
        str(args.context),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--jepa-weight",
        str(args.jepa_weight),
        "--delta",
        str(args.delta),
        "--eval-items",
        str(args.eval_items),
        "--model",
        args.model,
        "--corpus",
        args.corpus,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--log-every",
        str(args.log_every),
        "--out",
        str(out_path),
    ]
    if args.tokens is not None:
        cmd.extend(["--tokens", str(args.tokens)])
    if args.max_docs is not None:
        cmd += ["--max-docs", str(args.max_docs)]
    if args.max_tokens is not None:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if args.layer is not None:
        cmd += ["--layer", str(args.layer)]
    return cmd


def _run_one(
    arm: str,
    seed: int,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    _, warning = resolve_token_budget(arm, args.tokens)
    if warning:
        print(warning, file=sys.stderr, flush=True)
    print(f"[pilot] {arm} seed={seed} -> {out_path}", flush=True)
    subprocess.run(train_command(arm, seed, args, out_path), check=True)


def _load(out_dir: Path, arm: str, seed: int) -> dict:
    return json.loads((out_dir / f"{arm}_seed{seed}.json").read_text())


def summarise(
    out_dir: Path, plan: dict[str, list[int]], mpid: float
) -> dict:
    results: dict[str, dict] = {}
    for arm, seeds in plan.items():
        rows = [
            _load(out_dir, arm, s)
            for s in seeds
            if (out_dir / f"{arm}_seed{s}.json").exists()
        ]
        if not rows:
            continue
        cg = np.asarray([r["eval"]["context_gain_mean"] for r in rows])
        entry: dict = {
            "n": len(rows),
            "context_gain_mean": float(cg.mean()),
            "context_gain_sd_across_seeds": (
                float(cg.std(ddof=1)) if len(cg) > 1 else None
            ),
            "nll_true": float(np.mean([r["eval"]["nll_true"] for r in rows])),
            "seeds": [r["seed"] for r in rows],
        }
        if "necessity_mean" in rows[0]["eval"]:
            nec = np.asarray([r["eval"]["necessity_mean"] for r in rows])
            entry["necessity_mean"] = float(nec.mean())
            entry["necessity_sd_across_seeds"] = (
                float(nec.std(ddof=1)) if len(nec) > 1 else None
            )
        results[arm] = entry

    # Paired A - B across seeds that both arms share.
    if "A" in results and "B" in results:
        common = sorted(set(plan["A"]) & set(plan["B"]))
        diffs = []
        for s in common:
            if not (
                (out_dir / f"A_seed{s}.json").exists()
                and (out_dir / f"B_seed{s}.json").exists()
            ):
                continue
            a = _load(out_dir, "A", s)["eval"]["context_gain_mean"]
            b = _load(out_dir, "B", s)["eval"]["context_gain_mean"]
            diffs.append(a - b)
        if diffs:
            d = np.asarray(diffs)
            sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
            # ``sd`` is the SD of the within-seed paired difference. It is not
            # a run-level variance, and the sample-size formula has to be told
            # how many seeds it was estimated from. Floor 24 and cap 64 are
            # the protocol §5.4 bounds; the old call used the defaults
            # (inflation 1, floor 1) and printed "required n = 1".
            n_seeds = len(d)
            inflation = pilot_sd_inflation(n_seeds) if n_seeds >= 2 else None
            estimate = (
                required_n(
                    sd,
                    mpid,
                    alpha=0.025,
                    pilot_inflation=inflation,
                    floor=24,
                    cap=64,
                )
                if inflation is not None and math.isfinite(sd) and sd > 0
                else None
            )
            results["A_minus_B"] = {
                "n_pairs": len(d),
                "mean": float(d.mean()),
                "sd": sd,
                "per_seed": dict(zip(common, d.tolist())),
                "pilot_sd_inflation": inflation,
                "estimated_n_within_design": estimate,
                "variance_note": (
                    "within-design paired SD across seeds, not a run-level variance"
                ),
            }
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="override every arm's token budget (default: each arm's own budget)",
    )
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--jepa-weight", type=float, default=0.1)
    p.add_argument("--delta", type=int, default=8)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--eval-items", type=int, default=64)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--corpus", default="data/corpus_v2.txt")
    p.add_argument("--max-docs", type=int, default=4000)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--mpid", type=float, default=0.0112)
    p.add_argument("--arms", default=None, help="comma-separated subset")
    p.add_argument("--seeds", default=None, help="comma-separated, overrides plan")
    p.add_argument("--out-dir", default="artifacts/e1/pilot")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    plan = {k: list(v) for k, v in DEFAULT_PLAN.items()}
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        plan = {k: v for k, v in plan.items() if k in keep}
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        plan = {k: list(seeds) for k in plan}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm, seeds in plan.items():
        for seed in seeds:
            out_path = out_dir / f"{arm}_seed{seed}.json"
            if out_path.exists():
                print(f"[pilot] skip existing {out_path}", flush=True)
                continue
            _run_one(arm, seed, args, out_path)

    summary = summarise(out_dir, plan, args.mpid)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 68)
    print("E1 PILOT SUMMARY (pilot, not a confirmatory result)")
    print("=" * 68)
    for arm, row in summary.items():
        if arm == "A_minus_B":
            continue
        line = (
            f"{arm:8s} n={row['n']}  P1={row['context_gain_mean']:+.5f}"
        )
        if "necessity_mean" in row:
            line += f"  P3={row['necessity_mean']:+.5f}"
        print(line)
    if "A_minus_B" in summary:
        d = summary["A_minus_B"]
        print("-" * 68)
        print(
            f"A - B    n={d['n_pairs']}  mean={d['mean']:+.5f}  "
            f"sd={d['sd']:.5f}"
        )
        if d["estimated_n_within_design"] is not None:
            print(
                "estimated n from a within-design paired SD "
                "(not a run-level variance), "
                f"{d['n_pairs']} seeds, inflation {d['pilot_sd_inflation']:.2f}, "
                f"floor 24, cap 64, MPID={args.mpid}: "
                f"{d['estimated_n_within_design']}"
            )
    print("=" * 68)
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
