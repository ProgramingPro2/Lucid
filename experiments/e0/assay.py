"""G0 assay driver.

Default mode is a CPU dry run. ``--execute`` is the GPU launch and refuses,
before loading a model, when the corpus is smaller than the null-width plan.

    python -m experiments.e0.assay --print-n
    python -m experiments.e0.assay --dry-run --out artifacts/e0/g0_dry_run.json
    python -m experiments.e0.assay --execute --corpus artifacts/corpus/biorxiv_2026.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.e0.collect import collect_scores
from experiments.e0.criteria import diagnose_rehearsal, evaluate, sample_size_plan
from experiments.e0.items import (
    CorpusTooSmall,
    census_path,
    items_from_records,
    load_records,
    require_items,
)
from experiments.e0.synthetic import bow_embed, echoing_generate

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = _ROOT / "artifacts" / "corpus" / "biorxiv_2026.jsonl"
DEFAULT_DRY_OUT = _ROOT / "artifacts" / "e0" / "g0_dry_run.json"


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            raise ValueError("non-finite number in the G0 report")
        return value
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(report)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def print_sample_size(plan: dict, census: dict | None) -> None:
    """Print the null-width n and, when a corpus is present, the n on disk."""
    print("G0 null-width sample size (mean_ci half-width <= 0.015)")
    print(f"  rehearsal shuffled-margin SD: {plan['arm_sd']:.4f}")
    print(f"  null-pair SD (sd * sqrt(2)):  {plan['pair_sd']:.4f}")
    print(f"  required null pairs:          {plan['required_pairs']}")
    print(f"  required documents:           {plan['required_documents']}")
    print(f"  single-arm reading (items):   {plan['single_arm_items']}")
    print(
        "  half-width at the protocol's n=256, pair SD: "
        f"{plan['half_width_at_protocol_n_256_pair']:.4f}"
    )
    print(
        "  power n for a 0.03 dose contrast at the rehearsal SD: "
        f"{plan['required_n_power_mpid_0.03']}"
    )
    print(
        "  n to detect the 0.0036 rehearsal dip as an effect: "
        f"{plan['required_n_to_detect_rehearsal_dip']}"
    )
    if census is None:
        print("  corpus: not provided")
        return
    have = int(census["n_items"])
    pairs = int(census["n_null_pairs"])
    print(f"  corpus: {census.get('path', '')}")
    print(f"  achievable full-text items:   {have}")
    print(f"  achievable null pairs:        {pairs}")
    print(f"  skipped truncated tail lines: {census.get('skipped_truncated_tail', 0)}")
    print(f"  enough for the pair criterion: {have >= plan['required_documents']}")
    print(f"  enough for the single-arm reading: {have >= plan['single_arm_items']}")


def synthetic_records(n: int = 48) -> list[dict]:
    """In-memory harvest rows. Long enough to split, short enough to hash."""
    rows = []
    for i in range(n):
        words = [f"d{i}w{k}" for k in range(700)]
        body = " ".join(words)
        rows.append(
            {
                "doc_id": f"biorxiv:10.example/{i:04d}",
                "source": "biorxiv",
                "title": f"Synthetic paper {i}",
                "text": body,
                "sections": {"results": body},
                "license": "cc-by",
                "date": "2026-01-15",
                "meta": {"fulltext": "ok", "doi": f"10.example/{i:04d}"},
            }
        )
    return rows


def build_dry_report(*, n_items: int = 48, dim: int = 32, null_seed: int = 2) -> dict:
    """End-to-end CPU path. The generator ignores the latent on purpose."""
    records = synthetic_records(n_items)
    items, item_census = items_from_records(records, null_seed=null_seed)

    def embed(text: str) -> np.ndarray:
        return bow_embed(text, dim=dim)

    scores = collect_scores(
        items,
        echoing_generate,
        embed,
        dose_seeds=(11, 12, 13),
        zero_seed=21,
        text_seed=31,
        null_seed_a=101,
        null_seed_b=202,
    )
    report = evaluate(scores)
    plan = sample_size_plan()
    report.update(
        {
            "mode": "cpu-dry-run",
            "not_g0": True,
            "note": (
                "CPU dry run on synthetic documents. The generator echoes the "
                "prompt and ignores the latent. Verdicts here are plumbing, "
                "not a measurement of the adapter."
            ),
            "synthetic_channel": "decorative",
            "item_census": item_census,
            "sample_size": plan,
            "rehearsal_diagnosis": diagnose_rehearsal(),
            "planning_n_met": item_census["n_items"] >= plan["required_documents"],
        }
    )
    return report


def _live_census(path: Path) -> dict | None:
    if not path.exists():
        return None
    return census_path(path)


def execute_gpu(args: argparse.Namespace, plan: dict) -> dict:
    """Load the Phase-A adapter and run the assay. Not called by the dry run."""
    records, skipped = load_records(args.corpus)
    items, census = items_from_records(records, null_seed=args.seed)
    census["skipped_truncated_tail"] = skipped
    census["path"] = str(args.corpus)
    require_items(
        census["n_items"],
        plan["required_documents"],
        measure_anyway=args.measure_anyway,
    )

    import torch

    from experiments.pivot.gate_g1 import (
        _load_cfg,
        encode_paragraph_pooled,
        generate_paragraph,
        load_stage_a_for_infer,
    )
    from experiments.protocol.seeds import spawn_key, stream

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("G0 --execute needs CUDA. Refusing to occupy a CPU for this.")
    cfg = _load_cfg(_ROOT / args.cfg)
    model, tokenizer, meta = load_stage_a_for_infer(cfg, Path(args.ckpt), device)

    from conceptpoc.codec.jepa_codec import JepaCodec
    from conceptpoc.persistence import load_joint

    jepa_model, jepa_tok = load_joint(str(_ROOT / args.jepa), device="cpu")
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")

    def embed(text: str) -> np.ndarray:
        return encode_paragraph_pooled(codec, text)

    def generate(prompt: str, latent: np.ndarray | None, seed: int) -> str:
        torch.manual_seed(int(seed) % (2**31 - 1))
        return generate_paragraph(
            model,
            tokenizer,
            None if latent is None else np.asarray(latent, dtype=np.float32),
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            device=device,
        )

    gen = stream(args.seed, "decode")
    dose_seeds = [
        int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()) for _ in range(3)
    ]
    null_seed_a = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
    # docs/experiments protocol seeds.py: null arms use seed + 10000.
    null_seed_b = int(spawn_key(args.seed + 10000, "decode") % (2**31 - 1))
    zero_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
    text_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())

    scores = collect_scores(
        items,
        generate,
        embed,
        dose_seeds=dose_seeds,
        zero_seed=zero_seed,
        text_seed=text_seed,
        null_seed_a=null_seed_a,
        null_seed_b=null_seed_b,
    )
    report = evaluate(scores)
    report.update(
        {
            "mode": "gpu",
            "not_g0": False,
            "note": (
                "G0 on the post-cutoff corpus. Codec-space margin calibrates "
                "the instrument. It is not evidence for C1-C3."
            ),
            "backbone": meta.get("backbone"),
            "ckpt": meta.get("ckpt"),
            "item_census": census,
            "sample_size": plan,
            "rehearsal_diagnosis": diagnose_rehearsal(),
            "planning_n_met": census["n_items"] >= plan["required_documents"],
            "measure_anyway": bool(args.measure_anyway),
            "max_new_tokens": args.max_new_tokens,
            "seeds": {
                "run": args.seed,
                "dose": dose_seeds,
                "null_a": null_seed_a,
                "null_b": null_seed_b,
            },
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G0 assay validation")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--cfg", default="configs/pivot_step1.json")
    parser.add_argument("--ckpt", default="artifacts/pivot_stageA/checkpoint_latest.pt")
    parser.add_argument("--jepa", default="artifacts/jepa_repair_1m/jepa/joint.pt")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--print-n", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--measure-anyway", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)

    if args.execute and args.dry_run:
        print("pass only one of --execute and --dry-run", file=sys.stderr)
        return 2

    plan = sample_size_plan()
    census = _live_census(Path(args.corpus))
    print_sample_size(plan, census)

    if args.print_n and not args.execute and not args.dry_run:
        return 0

    if args.execute:
        try:
            report = execute_gpu(args, plan)
        except CorpusTooSmall as exc:
            print(str(exc), file=sys.stderr)
            return 2
        out = Path(args.out) if args.out else _ROOT / "artifacts" / "e0" / "g0.json"
    else:
        report = build_dry_report()
        if census is not None:
            report["live_corpus"] = {
                "path": census["path"],
                "n_items": census["n_items"],
                "n_null_pairs": census["n_null_pairs"],
                "skipped_truncated_tail": census["skipped_truncated_tail"],
                "drops": census["drops"],
                "enough_for_pair_criterion": census["n_items"] >= plan["required_documents"],
                "enough_for_single_arm_reading": census["n_items"] >= plan["single_arm_items"],
            }
        out = Path(args.out) if args.out else DEFAULT_DRY_OUT

    write_report(report, out)
    print(f"verdict: {report['verdict']}")
    for criterion in report["criteria"]:
        print(f"  {criterion['name']}: {criterion['verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
