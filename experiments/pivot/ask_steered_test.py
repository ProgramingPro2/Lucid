"""Quick steered-generation qualitative test (inference only).

Encodes a QUESTION into a JEPA topic vector z (segment + mean-pool + L2),
loads Phase-A checkpoint, generates 3 sampled paragraphs with prompt as prev.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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
from conceptpoc.data.segment import segment_sentences  # noqa: E402
from conceptpoc.persistence import load_joint  # noqa: E402
from experiments.pivot.encode_bank import pool_paragraph_centroid  # noqa: E402
from experiments.pivot.gate_g1 import (  # noqa: E402
    _cos,
    _l2,
    encode_paragraph_pooled,
    generate_paragraph,
    load_stage_a_for_infer,
)
from experiments.pivot.train_stage_a import _load_cfg  # noqa: E402

PROMPT = "which is closer, the sun or the moon, to earth?"
CKPT = _ROOT / "artifacts" / "pivot_stageA" / "checkpoint_epoch01.pt"
JEPA = _ROOT / "artifacts" / "jepa_repair_1m" / "jepa" / "joint.pt"
CFG = _ROOT / "configs" / "pivot_step1.json"
SEEDS = [7, 42, 123]
MAX_NEW = 144
TEMP = 0.8


def encode_command_z(codec: JepaCodec, text: str) -> np.ndarray:
    """Match pipeline command vector: segment_sentences → encode → mean-pool → L2."""
    sents = segment_sentences(text) or [text.strip()]
    sents = [s.strip() for s in sents if s and s.strip()]
    if not sents:
        sents = [text.strip() or " "]
    vecs = codec.encode(sents)
    return pool_paragraph_centroid(np.asarray(vecs, dtype=np.float32))


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ask] device={device} free-ish start", flush=True)

    print(f"[ask] loading frozen JEPA from {JEPA}", flush=True)
    jepa_model, jepa_tok = load_joint(str(JEPA), device="cpu")
    jepa_model.eval()
    for p in jepa_model.parameters():
        p.requires_grad_(False)
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")

    z = encode_command_z(codec, PROMPT)
    print(
        f"[ask] prompt-derived z: dim={z.shape[0]} ||z||={float(np.linalg.norm(z)):.4f}",
        flush=True,
    )

    cfg = _load_cfg(CFG)
    model, tokenizer, meta = load_stage_a_for_infer(cfg, CKPT, device)
    print(f"[ask] stage-A ready meta={meta}", flush=True)

    samples = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        text = generate_paragraph(
            model,
            tokenizer,
            z,
            PROMPT,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=TEMP,
            device=device,
        )
        fed = None
        if text.strip():
            y = encode_paragraph_pooled(codec, text)
            fed = _cos(y, z)
        row = {
            "seed": seed,
            "z_steer": "prompt-derived",
            "generated": text,
            "fed_vs_concept": fed,
        }
        samples.append(row)
        print(f"\n===== SAMPLE seed={seed} fed_vs_concept={fed} =====", flush=True)
        print(text, flush=True)

    out = {
        "prompt": PROMPT,
        "z_steer": "prompt-derived",
        "z_norm": float(np.linalg.norm(z)),
        "ckpt": str(CKPT),
        "jepa": str(JEPA),
        "max_new_tokens": MAX_NEW,
        "temperature": TEMP,
        "samples": samples,
        "elapsed_s": time.time() - t0,
    }
    out_path = _ROOT / "artifacts" / "pivot_stageA" / "ask_steered_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[ask] wrote {out_path} elapsed={out['elapsed_s']:.1f}s", flush=True)
    # Also dump a plain text report for the deliverable.
    report = _ROOT / "artifacts" / "pivot_stageA" / "ask_steered_test_report.txt"
    lines = [
        "Lucid Phase-A steered generation qualitative test",
        f"prompt: {PROMPT}",
        f"z_steer: prompt-derived (segment_sentences → JEPA encode → mean-pool → L2)",
        f"ckpt: {CKPT}",
        f"jepa: {JEPA}",
        f"max_new_tokens={MAX_NEW} temperature={TEMP}",
        "",
    ]
    for s in samples:
        lines.append(f"--- sample seed={s['seed']} ---")
        lines.append(f"z_steer: {s['z_steer']}")
        lines.append(f"fed_vs_concept: {s['fed_vs_concept']}")
        lines.append("generated (verbatim):")
        lines.append(s["generated"])
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ask] wrote {report}", flush=True)


if __name__ == "__main__":
    main()
