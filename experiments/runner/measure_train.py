"""Measure training throughput and VRAM — the real content of the G0 gate.

``docs/PROTOCOL.md`` §10 requires >= 2,000 tokens/s at ctx 512-1024 with
micro-batch >= 4 before any experiment starts. That is a *training* number
(forward + backward), not a decode number; ``measure.py`` covers decode.

The batching requirement is a power requirement, not a speed one: n scales as
sigma^2, and micro-batch-1 gradient variance inflates sigma.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def build(model_id: str, *, load_in_4bit: bool, lora_r: int = 16):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16
    kwargs: dict = {"torch_dtype": dtype}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, cfg)
    model.train()
    return model


@torch.no_grad()
def _guard_finite(model) -> int:
    bad = 0
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad += 1
    return bad


def bench(
    model_id: str,
    *,
    load_in_4bit: bool,
    micro_batch: int,
    ctx: int,
    steps: int = 12,
    warmup: int = 3,
    lr: float = 1e-4,
) -> dict:
    model = build(model_id, load_in_4bit=load_in_4bit)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95)
    )
    g = torch.Generator(device="cuda").manual_seed(0)
    tokens_per_step = micro_batch * ctx

    def one_step() -> float:
        ids = torch.randint(0, 1000, (micro_batch, ctx), device="cuda", generator=g)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    losses = []
    for _ in range(steps):
        losses.append(one_step())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tok_s = (steps * tokens_per_step) / elapsed
    return {
        "model": model_id,
        "load_in_4bit": load_in_4bit,
        "micro_batch": micro_batch,
        "ctx": ctx,
        "steps_timed": steps,
        "trainable_params_m": round(trainable / 1e6, 2),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(tok_s, 0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "nonfinite_grad_params": _guard_finite(model),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rep = bench(
        args.model,
        load_in_4bit=args.load_in_4bit,
        micro_batch=args.micro_batch,
        ctx=args.ctx,
        steps=args.steps,
    )
    print(json.dumps(rep, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
