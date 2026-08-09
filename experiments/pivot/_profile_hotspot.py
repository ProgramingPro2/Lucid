"""CPU-only profile helpers + optional full-loop timing (no src edits)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def profile_dataloader(n: int = 200) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from experiments.pivot.train_stage_a import ParaPairDataset, _load_cfg, _make_collate

    cfg = _load_cfg(_ROOT / "configs" / "pivot_step1.json")
    t0 = time.perf_counter()
    ds = ParaPairDataset(_ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="train")
    build_s = time.perf_counter() - t0
    tok = AutoTokenizer.from_pretrained(cfg["backbone"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = _make_collate(tok, cfg)
    t0 = time.perf_counter()
    for i in range(n):
        collate([ds[i]])
    ms = (time.perf_counter() - t0) * 1000 / n
    return {"n": n, "dataset_build_s": build_s, "collate_ms": ms, "pairs": len(ds)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    out = profile_dataloader(args.n)
    print(json.dumps(out, indent=2))
