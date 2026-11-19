"""E0 rehearsal — does the assay's dose ladder actually work?

This is **not** G0. G0 runs on the new post-cutoff corpus specified in
docs/CORPUS.md. This runs on the legacy corpus with the existing Phase-A
adapter, purely to answer one question before any GPU-week is committed:

    can the instrument detect a known effect (a dose of the true latent),
    and how wide is its noise floor at n items?

If the dose ladder is flat here, G0 will be flat too, and the endpoint is wrong —
which is exactly what G0 exists to find out, at half a day's cost.

Readout, in order of independence:
  1. `cos(gen, z_true)` from the frozen JEPA codec — the *codec-space* margin.
     Declared circular by construction (docs/PROTOCOL.md §4) and used only to
     calibrate, never as evidence for C1-C3.
  2. The paired margin `cos(gen, z_true) - cos(gen, z_shuf)`, which is the
     quantity E4-S1 uses.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.e0.rehearsal --items 16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.pivot.gate_g1 import (  # noqa: E402
    _l2,
    _load_cfg,
    encode_paragraph_pooled,
    generate_paragraph,
    load_stage_a_for_infer,
)
from experiments.protocol.stats import mean_ci, paired_summary  # noqa: E402


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two unit vectors."""
    a = _l2(np.asarray(a, dtype=np.float32))
    b = _l2(np.asarray(b, dtype=np.float32))
    dot = float(np.clip(a @ b, -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-6:
        return a
    s = math.sin(omega)
    return _l2(
        (math.sin((1 - t) * omega) / s) * a + (math.sin(t * omega) / s) * b
    )


def load_paragraphs(path: Path, *, min_chars: int = 500, limit: int = 400) -> list[str]:
    """Stream a corpus file and keep the first `limit` clean paragraphs."""
    out: list[str] = []
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                buf.append(line.strip())
                continue
            text = " ".join(buf).strip()
            buf = []
            if len(text) >= min_chars and text.count(".") >= 3:
                out.append(text)
                if len(out) >= limit:
                    break
    return out


def prose_prefix(text: str, fraction: float = 0.35) -> str:
    """A plausible generation prompt: the opening of the paragraph."""
    cut = max(80, int(len(text) * fraction))
    return text[:cut].rsplit(" ", 1)[0] + " "


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--corpus", default="data/corpus_v2.txt")
    ap.add_argument("--cfg", default="configs/pivot_step1.json")
    ap.add_argument("--ckpt", default="artifacts/pivot_stageA/checkpoint_latest.pt")
    ap.add_argument("--jepa", default="artifacts/jepa_repair_1m/jepa/joint.pt")
    ap.add_argument("--out", default="artifacts/e0/rehearsal.json")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(_ROOT / args.cfg)

    paras = load_paragraphs(_ROOT / args.corpus, limit=max(4 * args.items, 40))
    if len(paras) < args.items + 2:
        print(f"only {len(paras)} paragraphs available", file=sys.stderr)
        return 2

    model, tokenizer, meta = load_stage_a_for_infer(
        cfg, Path(args.ckpt), device
    )

    from conceptpoc.codec.jepa_codec import JepaCodec
    from conceptpoc.persistence import load_joint

    print("[E0] loading codec", flush=True)
    jepa_model, jepa_tok = load_joint(str(_ROOT / args.jepa), device="cpu")
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")

    items = paras[: args.items]
    others = paras[args.items : args.items + args.items]

    print(f"[E0] encoding {len(items)} targets + {len(others)} shuffled", flush=True)
    z_true = [encode_paragraph_pooled(codec, t) for t in items]
    z_shuf = [encode_paragraph_pooled(codec, t) for t in others]

    conditions = [
        "true",
        "slerp50",
        "shuffled",
        "zero",
        "textprefix",
        "textprefix_rand",
    ]
    rows: list[dict] = []

    for i, text in enumerate(items):
        prev = prose_prefix(text)
        for cond in conditions:
            if cond == "true":
                z = z_true[i]
                prompt = prev
            elif cond == "slerp50":
                z = slerp(z_true[i], z_shuf[i], 0.5)
                prompt = prev
            elif cond == "shuffled":
                z = z_shuf[i]
                prompt = prev
            elif cond == "zero":
                z = None
                prompt = prev
            elif cond == "textprefix":
                z = None
                prompt = prose_prefix(text, 0.7)
            elif cond == "textprefix_rand":
                # The positive control that matters: the *same* amount of
                # content-bearing text, from a different document.
                z = None
                prompt = prose_prefix(others[i], 0.7)
            else:
                raise ValueError(cond)

            gen = generate_paragraph(
                model,
                tokenizer,
                z,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                device=device,
            )
            gz = encode_paragraph_pooled(codec, gen) if gen.strip() else np.zeros(512, np.float32)
            c_true = float(gz @ z_true[i])
            c_shuf = float(gz @ z_shuf[i])
            rows.append(
                {
                    "item": i,
                    "cond": cond,
                    "cos_true": c_true,
                    "cos_shuf": c_shuf,
                    "margin": c_true - c_shuf,
                    "gen_chars": len(gen),
                    "gen_head": gen[:120],
                }
            )
        print(f"[E0] item {i + 1}/{len(items)} done", flush=True)

    # Aggregate per condition.
    summary: dict[str, dict] = {}
    for cond in conditions:
        vals = [r["margin"] for r in rows if r["cond"] == cond]
        s = mean_ci(vals)
        summary[cond] = {
            "n": s.n,
            "mean_margin": round(s.mean, 4),
            "ci_low": round(s.ci_low, 4),
            "ci_high": round(s.ci_high, 4),
            "mean_cos_true": round(
                float(np.mean([r["cos_true"] for r in rows if r["cond"] == cond])), 4
            ),
            "mean_gen_chars": int(
                np.mean([r["gen_chars"] for r in rows if r["cond"] == cond])
            ),
        }

    # Necessity: true vs shuffled, paired by item.
    true_m = [r["margin"] for r in rows if r["cond"] == "true"]
    shuf_m = [r["margin"] for r in rows if r["cond"] == "shuffled"]
    nec = paired_summary(true_m, shuf_m)

    # Dose ladder: 0% (= shuffled), 50%, 100% (= true).
    ladder = [
        summary["shuffled"]["mean_cos_true"],
        summary["slerp50"]["mean_cos_true"],
        summary["true"]["mean_cos_true"],
    ]
    monotone = ladder[0] <= ladder[1] <= ladder[2]

    # Positive control: does the readout see real content when it is given one?
    tp = [r["cos_true"] for r in rows if r["cond"] == "textprefix"]
    tpr = [r["cos_true"] for r in rows if r["cond"] == "textprefix_rand"]
    pos = paired_summary(tp, tpr)
    textprefix_vs_zero = paired_summary(
        tp, [r["cos_true"] for r in rows if r["cond"] == "zero"]
    )

    report = {
        "note": "E0 REHEARSAL on the legacy corpus — not G0, not evidence for C1-C3",
        "backbone": meta.get("backbone"),
        "ckpt": meta.get("ckpt"),
        "n_items": args.items,
        "max_new_tokens": args.max_new_tokens,
        "conditions": summary,
        "necessity_true_minus_shuffled": {
            "mean": round(nec.mean, 4),
            "ci_low": round(nec.ci_low, 4),
            "ci_high": round(nec.ci_high, 4),
            "n": nec.n,
        },
        "positive_control_textprefix_minus_random": {
            "mean": round(pos.mean, 4),
            "ci_low": round(pos.ci_low, 4),
            "ci_high": round(pos.ci_high, 4),
            "n": pos.n,
            "passes_0.03_gate": bool(pos.ci_low > 0.03),
        },
        "textprefix_minus_zero": {
            "mean": round(textprefix_vs_zero.mean, 4),
            "ci_low": round(textprefix_vs_zero.ci_low, 4),
            "ci_high": round(textprefix_vs_zero.ci_high, 4),
        },
        "dose_ladder_cos_true": [round(x, 4) for x in ladder],
        "dose_monotone": bool(monotone),
        "samples": {c: next(r["gen_head"] for r in rows if r["cond"] == c) for c in conditions},
    }

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
