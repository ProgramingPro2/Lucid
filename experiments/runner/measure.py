"""Measure decode throughput and VRAM for a backbone.

This is the measurement behind the ">= 2,000 tok/s" qualification gate in
docs/PROTOCOL.md §10. It reports generation throughput (tokens written), not
prefill throughput, because the arms are generation-bound.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.runner.measure \
        --model Qwen/Qwen3-1.7B-Base --batch-size 4 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import json
import sys

from .model import GenConfig, LM


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    prompt = (
        "Summarise the following paragraph in one sentence.\n\n"
        "The study recruited 412 participants across three sites and measured "
        "change in systolic blood pressure over twelve weeks. Adherence was "
        "assessed by pill count at every visit, and the primary analysis used "
        "a mixed-effects model with site as a random intercept.\n\nSummary:"
    )

    lm = LM(args.model, load_in_4bit=args.load_in_4bit)
    cfg = GenConfig(max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    prompts = [prompt] * args.n_prompts

    _, warm = lm.generate(prompts[: args.batch_size], cfg)  # warm-up, excluded
    texts, stats = lm.generate(prompts, cfg)

    report = {
        "model": args.model,
        "params_m": round(lm.n_params() / 1e6, 1),
        "load_in_4bit": args.load_in_4bit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        **stats.as_dict(),
        "sample": texts[0][:200] if texts else "",
        "warmup_seconds": round(warm.seconds, 3),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
