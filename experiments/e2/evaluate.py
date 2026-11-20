"""Run an E2 capability screen on a backbone.

This is **not** the full E2 treatment. E2's arm A trains a discrete latent
emission gate; that requires a training run. What this script measures is the
pre-registered *VOID gate* in docs/PROTOCOL.md §6.4:

    B_text − E < 0.10  ⇒  VOID

That is: before training a latent arm, can the backbone show a text-CoT advantage
on these tasks at all? If the answer is no, the assay cannot see the effect it is
supposed to measure, and no amount of training will make the contrast
interpretable.

It reports exact-match accuracy from the same verifier used everywhere else, the
fraction of replies that even contain the ``ANSWER:`` marker, and the paired
CoT − direct difference with a t interval.

The backbone must be instruction-tuned (docs/PROTOCOL.md §6.3). Qwen3-1.7B-Base
fails the screen because explicit CoT degenerates, not because the tasks are too
hard; the default is therefore the instruct checkpoint.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.e2.evaluate \
        --model Qwen/Qwen3-1.7B --family f1 --n 20 --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.protocol.stats import paired_summary
from experiments.runner.model import GenConfig, LM

from .tasks import chance, generate_many, verify

MODES = ("direct", "cot")


def _items(family: str, split: str, n: int, start: int):
    return generate_many(family, split, n, start=start)


def _wrap(
    lm: LM, prompt: str, use_chat_template: bool, think: bool = False
) -> str:
    """Optionally present the task as a user turn.

    A base model given the instruction as raw text emitted EOS immediately: it
    has never been trained to answer a directive, so the prompt looks like the
    end of a document. The chat template is a *formatting* change (same weights)
    and makes the screen measure task ability rather than format compliance. It
    is reported, because it is a deviation from a raw-completion baseline.

    ``think`` controls Qwen3's hidden reasoning mode. It defaults off: the B_text
    arm is *explicit* text CoT, and hidden thinking consumed the entire output
    budget without reaching an answer in the pilot. Turning it off makes the two
    arms differ in the way the protocol specifies — available working, not a
    different inference mode.
    """
    if use_chat_template and getattr(lm.tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return lm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=think,
            )
        except (TypeError, ValueError):
            return lm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return prompt


def run_mode(
    lm: LM,
    items,
    mode: str,
    *,
    max_new_tokens: int,
    batch_size: int,
    use_chat_template: bool = True,
    think: bool = False,
) -> list[dict]:
    """Generate one reply per item for one prompt mode and verify it."""
    prompts = [
        _wrap(
            lm,
            it.prompt_direct if mode == "direct" else it.prompt_cot,
            use_chat_template,
            think,
        )
        for it in items
    ]
    cfg = GenConfig(max_new_tokens=max_new_tokens, batch_size=batch_size)

    # Chunk so that a progress line appears without holding every generation in
    # one call; the batch size still controls the actual decode batching.
    rows: list[dict] = []
    chunk = max(batch_size, 8)
    for lo in range(0, len(items), chunk):
        hi = min(lo + chunk, len(items))
        texts, _ = lm.generate(prompts[lo:hi], cfg)
        for item, text in zip(items[lo:hi], texts):
            rows.append(
                {
                    "id": item.id,
                    "family": item.family,
                    "mode": mode,
                    "gold": item.gold,
                    "correct": bool(verify(item.family, text, item.gold)),
                    "has_marker": "answer:" in text.lower(),
                    "chars": len(text),
                    "text": text[:600],
                }
            )
        print(f"    {mode}: {hi}/{len(items)}", flush=True)
    return rows


def _accuracy(rows: list[dict]) -> float:
    return float(np.mean([r["correct"] for r in rows])) if rows else float("nan")


def summarise(rows: list[dict]) -> dict:
    """Per-family and overall accuracy, marker rate, and the paired contrast."""
    out: dict = {"by_family": {}, "overall": {}}
    families = sorted({r["family"] for r in rows})

    for family in families:
        fam = [r for r in rows if r["family"] == family]
        by_mode: dict[str, list[dict]] = {
            m: [r for r in fam if r["mode"] == m] for m in MODES
        }
        entry: dict = {
            "n": max((len(v) for v in by_mode.values()), default=0),
            "chance": chance(family),
        }
        for mode in MODES:
            rs = by_mode[mode]
            if not rs:
                continue
            entry[mode] = {
                "accuracy": round(_accuracy(rs), 4),
                "marker_rate": round(float(np.mean([r["has_marker"] for r in rs])), 4),
                "mean_chars": int(np.mean([r["chars"] for r in rs])),
            }

        if all(by_mode[m] for m in MODES):
            direct = [r["correct"] for r in by_mode["direct"]]
            cot = [r["correct"] for r in by_mode["cot"]]
            diff = paired_summary(cot, direct)
            entry["cot_minus_direct"] = {
                "mean": round(diff.mean, 4),
                "ci_low": round(diff.ci_low, 4),
                "ci_high": round(diff.ci_high, 4),
                "n": diff.n,
            }
        out["by_family"][family] = entry

    if all(any(r["mode"] == m for r in rows) for m in MODES):
        direct = [r["correct"] for r in rows if r["mode"] == "direct"]
        cot = [r["correct"] for r in rows if r["mode"] == "cot"]
        diff = paired_summary(cot, direct)
        mean_acc = float(np.mean(direct + cot))
        out["overall"] = {
            "n_items": len(direct),
            "accuracy_mean_modes": round(mean_acc, 4),
            "direct_accuracy": round(float(np.mean(direct)), 4),
            "cot_accuracy": round(float(np.mean(cot)), 4),
            "cot_minus_direct": {
                "mean": round(diff.mean, 4),
                "ci_low": round(diff.ci_low, 4),
                "ci_high": round(diff.ci_high, 4),
                "n": diff.n,
            },
            # The §6.4 VOID gate, applied to this prompt-only proxy for B_text.
            "capability_gate": "PASS" if diff.mean >= 0.10 else "VOID",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--family", default="f1", choices=["f1", "f2", "f3", "all"])
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--mode", default="both", choices=["direct", "cot", "both"])
    p.add_argument("--max-new-tokens-direct", type=int, default=24)
    p.add_argument(
        "--max-new-tokens-cot",
        type=int,
        default=1024,
        help="1024 is the protocol budget; below it F3 truncates (F48)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="send the task as raw completion text (the model often emits EOS)",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="enable Qwen3 hidden reasoning for the CoT arm (off by default)",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    families = ["f1", "f2", "f3"] if args.family == "all" else [args.family]
    modes = MODES if args.mode == "both" else (args.mode,)

    lm = LM(args.model)
    print(f"[E2] {args.model}: {lm.n_params() / 1e6:.0f}M params", flush=True)

    all_rows: list[dict] = []
    for family in families:
        items = _items(family, args.split, args.n, args.start)
        print(f"[E2] {family}: {len(items)} items, chance={chance(family):.3f}", flush=True)
        for mode in modes:
            budget = (
                args.max_new_tokens_direct
                if mode == "direct"
                else args.max_new_tokens_cot
            )
            if family == "f3" and mode == "cot" and budget < 1024:
                print(
                    "    warning: F3 traces need >=1024 tokens; a smaller budget "
                    "truncates them and scores 0 (artifact, not capability)",
                    flush=True,
                )
            all_rows.extend(
                run_mode(
                    lm,
                    items,
                    mode,
                    max_new_tokens=budget,
                    batch_size=args.batch_size,
                    use_chat_template=not args.no_chat_template,
                    think=args.thinking and mode == "cot",
                )
            )

    report = {
        "note": (
            "E2 CAPABILITY SCREEN on an untrained backbone. This is the §6.4 VOID "
            "gate as a prompt-only proxy for B_text, not the trained arm. It says "
            "whether the assay can see a text-CoT effect, not whether latents work."
        ),
        "model": args.model,
        "chat_template": bool(
            not args.no_chat_template
            and getattr(lm.tokenizer, "chat_template", None)
        ),
        "thinking": bool(args.thinking),
        "split": args.split,
        "start": args.start,
        "n_per_family": args.n,
        "max_new_tokens": {
            "direct": args.max_new_tokens_direct,
            "cot": args.max_new_tokens_cot,
        },
        "summary": summarise(all_rows),
        "rows": all_rows,
    }

    print(json.dumps(report["summary"], indent=2))
    out = Path(args.out) if args.out else Path(f"artifacts/e2/capability_{args.family}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
