#!/usr/bin/env python3
"""Post-training validation: compare regression baseline vs diffusion predictor.

Usage (after ``train-diffusion`` finishes):
  python scripts/validate_diffusion.py --config configs/jepa_1m_diffusion.json

Writes ``artifacts/jepa_1m_diffusion/validation_results.json`` and runs
``eval-harness`` for both models (heuristic judge offline; set
``JUDGE_BACKEND=lemonade`` for the real judge).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conceptpoc.config import Config
from conceptpoc.loader import artifact_paths


def _run_harness(config_path: str, out_dir: str, label: str) -> dict:
    env = os.environ.copy()
    judge = env.get("JUDGE_BACKEND", "heuristic")
    cmd = [
        sys.executable,
        "-m",
        "conceptpoc.cli",
        "eval-harness",
        "--config",
        config_path,
        "--out",
        out_dir,
        "-q",
    ]
    print(f"Running eval-harness ({label}) judge={judge} -> {out_dir}")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    report = json.loads((ROOT / out_dir / "report.json").read_text())
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/jepa_1m_diffusion.json")
    p.add_argument(
        "--skip-baseline", action="store_true", help="only eval the diffusion config"
    )
    args = p.parse_args()

    cfg = Config.load(args.config)
    diff_ckpt = artifact_paths(cfg)["predictor_diffusion"]
    if not diff_ckpt.exists():
        raise SystemExit(
            f"Diffusion checkpoint missing at {diff_ckpt}; run train-diffusion first."
        )

    results: dict = {"diffusion_config": args.config, "judge_backend": os.environ.get("JUDGE_BACKEND", "heuristic")}
    if not args.skip_baseline:
        baseline_cfg = ROOT / "configs/jepa_1m_unif.json"
        results["baseline"] = _run_harness(
            str(baseline_cfg),
            "artifacts/jepa_1m_unif/eval_diffusion_baseline",
            "regression baseline",
        )
    results["diffusion"] = _run_harness(
        args.config,
        str(cfg.artifacts_dir / "eval_diffusion"),
        "diffusion",
    )

    out_path = cfg.artifacts_dir / "validation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    def _latent(report: dict) -> tuple[float, float]:
        overall = report.get("overall", {})
        return (
            overall.get("mean_max_cos_to_bank", float("nan")),
            overall.get("mean_consecutive_cos", float("nan")),
        )

    if "baseline" in results:
        b_bank, b_seq = _latent(results["baseline"])
        d_bank, d_seq = _latent(results["diffusion"])
        print(f"Baseline  latent: bank={b_bank:.3f}  consecutive={b_seq:.3f}")
        print(f"Diffusion latent: bank={d_bank:.3f}  consecutive={d_seq:.3f}")
        print(f"Diffusion judge overall: {results['diffusion']['overall']['mean_score']:.3f}")


if __name__ == "__main__":
    main()
