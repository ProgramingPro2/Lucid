"""Materialise E2 task files and print a generation audit.

Usage (from the repository root)::

    PYTHONPATH=. .venv/bin/python -m experiments.e2.build \
        --family all --split test --n 2000 --out artifacts/e2

The report checks properties of the *generated data*, not of any model: prompt
collisions, oracle recoverability, answer length and file hash. "Random-answer
accuracy" is a sanity check that the chance level is what the protocol says it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Support both `python -m experiments.e2.build` and a direct script invocation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.e2 import solver  # noqa: E402
from experiments.e2.tasks import (  # noqa: E402
    FAMILIES,
    Item,
    chance,
    generate_many,
    verify,
)



def _random_answer(item: Item, rng: np.random.Generator) -> str:
    """Draw one answer uniformly from that family's answer space."""
    if item.family == "f1":
        return str(int(rng.integers(10_000, 1_000_000)))
    if item.family == "f2":
        entities = item.meta["entities"]
        return str(entities[int(rng.integers(0, len(entities)))])
    tokens = item.meta["tokens"]
    return str(tokens[int(rng.integers(0, len(tokens)))])


def _collisions(items: list[Item], attribute: str) -> int:
    """Count prompts that map to more than one distinct gold answer."""
    golds: dict[str, set[str]] = defaultdict(set)
    for item in items:
        golds[getattr(item, attribute)].add(item.gold)
    return sum(1 for values in golds.values() if len(values) > 1)


def _median_int(values: list[int]) -> int:
    return int(round(statistics.median(values)))


def _report(
    family: str,
    split: str,
    items: list[Item],
    path: Path,
    digest: str,
) -> dict[str, Any]:
    oracle_correct = 0
    for item in items:
        answer = solver.solve(item)
        oracle_correct += int(verify(item.family, f"ANSWER: {answer}", item.gold))

    seed_bytes = hashlib.sha256(f"random:{family}:{split}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
    random_correct = sum(
        int(verify(item.family, f"ANSWER: {_random_answer(item, rng)}", item.gold))
        for item in items
    )

    lengths = [len(item.gold) for item in items]
    direct_collisions = _collisions(items, "prompt_direct")
    cot_collisions = _collisions(items, "prompt_cot")
    n = len(items)

    report = {
        "family": family,
        "split": split,
        "file": str(path),
        "n": n,
        "prompt_direct_collisions": direct_collisions,
        "prompt_cot_collisions": cot_collisions,
        "oracle_accuracy": oracle_correct / n if n else 0.0,
        "random_accuracy": random_correct / n if n else 0.0,
        "chance": chance(family),
        "gold_length_min": min(lengths) if lengths else 0,
        "gold_length_median": _median_int(lengths) if lengths else 0,
        "gold_length_max": max(lengths) if lengths else 0,
        "sha256": digest,
    }

    print(f"{path.name}")
    print(f"  n                            {n}")
    print(
        "  prompt->gold collisions       "
        f"{direct_collisions} (direct), {cot_collisions} (cot)"
    )
    print(f"  oracle accuracy               {report['oracle_accuracy']:.3f}")
    print(
        "  random-answer accuracy        "
        f"{report['random_accuracy']:.3f} (chance {report['chance']:.6f})"
    )
    print(
        "  gold length min/median/max    "
        f"{report['gold_length_min']}/"
        f"{report['gold_length_median']}/"
        f"{report['gold_length_max']}"
    )
    print(f"  sha256                        {digest}")
    return report


def _build_one(family: str, split: str, n: int, out: Path) -> dict[str, Any]:
    items = generate_many(family, split, n)
    payload = "".join(item.to_json() + "\n" for item in items)
    data = payload.encode("utf-8")
    path = out / f"{family}_{split}.jsonl"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return _report(family, split, items, path, digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate E2 JSONL tasks and audit the generated data."
    )
    parser.add_argument(
        "--family",
        default="all",
        choices=["all", *FAMILIES],
        help="task family, or 'all' (default: all)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="split to materialise (default: test)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="items per family (maximum 2000 per split)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/e2",
        help="output directory (default: artifacts/e2)",
    )
    args = parser.parse_args(argv)

    if args.n < 1:
        parser.error("--n must be positive")

    families = list(FAMILIES) if args.family == "all" else [args.family]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"E2 generation audit: split={args.split} n={args.n} out={out}")
    reports = [_build_one(family, args.split, args.n, out) for family in families]

    valid = all(
        report["prompt_direct_collisions"] == 0
        and report["prompt_cot_collisions"] == 0
        and abs(report["oracle_accuracy"] - 1.0) < 1e-12
        for report in reports
    )
    print(f"VALIDATION {'PASS' if valid else 'FAIL'}")
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
