"""Train one E2 cell and report the pre-registered endpoint machinery.

This is the harness, not a result. A short run on train items does not read the
sealed test split and cannot pass G2b. It exists to show that the latent arm
trains, that the gate can emit, and that the matched-FLOP verdict is computed
from measurements.

Usage (proof run, stays under an 8 GB share of a 24 GB card)::

    python -m experiments.e2.train --arm A --family f1 --steps 200 --capacity 4

The backbone is ``Qwen/Qwen3-1.7B`` (instruct). Thinking is disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.protocol.seeds import spawn_key

from .analysis import decorative_report, necessity_fraction
from .arms import ANSWER_BUDGET, COT_BUDGET, ArmSpec, arm_spec, family_code, run_id
from .chat import wrap_user
from .codebook import VectorQuantizer, is_degenerate, usage_stats
from .flops import judge_flop_ratio, measure_calls, plan_depth_repeats, project_gpu_hours
from .gate import EmissionGate
from .latent import (
    TinyBackbone,
    emit,
    emit_depth,
    span_loss,
    text_target_nll,
)
from .tasks import generate_many, verify

VRAM_BUDGET_GB = 8.0


@dataclass
class TrainConfig:
    arm: str = "A"
    family: str = "f1"
    seed: int = 0
    steps: int = 200
    lr: float = 1e-4
    capacity: int = 4
    depth_repeats: int = 1
    answer_tokens: int = ANSWER_BUDGET
    cot_tokens: int = COT_BUDGET
    lora_r: int = 8
    model_id: str = "Qwen/Qwen3-1.7B"
    max_prompt_tokens: int = 384
    max_trace_tokens: int = 128
    eval_items: int = 8
    log_every: int = 20
    measure_flops: bool = True
    tiny: bool = False
    device: str = "cpu"
    out: str = "artifacts/e2/pilot_a_f1.json"


def _seed_int(seed: int, stream: str) -> int:
    return spawn_key(seed, stream) % (2**31 - 1)


def _hash_ids(text: str, vocab: int, limit: int) -> list[int]:
    ids: list[int] = []
    words = text.split() or ["x"]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(1 + int.from_bytes(digest[:4], "big") % (vocab - 2))
        if len(ids) >= limit:
            break
    return ids


def _gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return float(free) / 1024**3


def _refuse_if_no_room() -> None:
    free = _gpu_free_gb()
    if free < VRAM_BUDGET_GB:
        raise SystemExit(
            f"refusing to start: {free:.2f} GB free, need {VRAM_BUDGET_GB:.0f} GB "
            "so this process stays inside its share of the card"
        )


class QwenBackbone:
    """The instruct model, with LoRA modules already injected into ``causal``."""

    def __init__(self, causal: torch.nn.Module, tokenizer) -> None:
        self.causal = causal
        self.peft = None
        self.tokenizer = tokenizer
        self.d_model = int(causal.config.hidden_size)
        self.n_layers = int(causal.config.num_hidden_layers)

    def train(self, mode: bool = True):
        if self.peft is not None:
            self.peft.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self) -> bool:
        if self.peft is None:
            return True
        return bool(self.peft.training)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.causal.get_input_embeddings()(ids)

    def forward_hidden(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.causal.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state

    def reapply(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """One more full pass of the decoder stack. Counted as extra depth."""
        return self.forward_hidden(hidden, attention_mask)

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.causal.lm_head(hidden)


def _load_qwen(cfg: TrainConfig) -> tuple[QwenBackbone, torch.nn.Module, object]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _refuse_if_no_room()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    causal = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    causal.config.use_cache = False
    causal.gradient_checkpointing_enable()
    torch.manual_seed(_seed_int(cfg.seed, "lora_init"))
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    peft_model = get_peft_model(causal, lora)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    peft_model.train()
    base = peft_model.get_base_model()
    wrapped = QwenBackbone(base, tokenizer)
    wrapped.peft = peft_model
    return wrapped, peft_model, tokenizer


def _build_tiny(cfg: TrainConfig) -> tuple[TinyBackbone, None, None]:
    torch.manual_seed(_seed_int(cfg.seed, "block_init"))
    backbone = TinyBackbone()
    backbone.train()
    return backbone, None, None


def _tokenize(cfg: TrainConfig, tokenizer, text: str, limit: int) -> list[int]:
    if tokenizer is None:
        vocab = 48
        return _hash_ids(text, vocab, limit)
    wrapped = wrap_user(tokenizer, text, think=False)
    ids = tokenizer(wrapped, add_special_tokens=False, truncation=True, max_length=limit).input_ids
    if not ids:
        ids = tokenizer(" ", add_special_tokens=False).input_ids or [tokenizer.eos_token_id]
    return list(ids)


def _answer_text(gold: str) -> str:
    return f"ANSWER: {gold}"


def _ids_tensor(ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=device)


def _step(
    backbone,
    spec: ArmSpec,
    gate: EmissionGate | None,
    codebook: VectorQuantizer | None,
    depth_token: torch.nn.Parameter | None,
    prompt_embeds: torch.Tensor,
    answer_ids: torch.Tensor,
    target_ids: torch.Tensor | None,
    baseline: torch.Tensor | None,
) -> tuple[object, torch.Tensor]:
    if spec.kind == "vehicle":
        from .latent import Span

        span = Span(
            embeds=prompt_embeds,
            mask=torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device),
            n_latent=0,
            n_text=0,
            n_forwards=0,
            commitment=prompt_embeds.new_zeros(()),
            prompt_len=int(prompt_embeds.shape[1]),
            capacity=spec.capacity,
        )
        return span_loss(
            backbone, span, answer_ids,
            lambda_token=0.0, commitment_beta=0.0, baseline=None, policy_coef=0.0,
        )
    if spec.kind == "text":
        assert target_ids is not None
        ce = text_target_nll(backbone, prompt_embeds, target_ids)
        from .latent import LossParts

        parts = LossParts(
            loss=ce, ce=ce, charge=ce.new_zeros(()), commitment=ce.new_zeros(()),
            policy=ce.new_zeros(()), reward=-(ce.detach()),
            n_latent=0, n_text=int(target_ids.numel()), code_indices=[],
        )
        return parts, ce.detach()
    if spec.kind == "depth":
        assert depth_token is not None
        span = emit_depth(
            backbone, depth_token, prompt_embeds,
            capacity=spec.capacity, depth_repeats=spec.depth_repeats,
        )
        return span_loss(
            backbone, span, answer_ids,
            lambda_token=0.0, commitment_beta=0.0, baseline=None, policy_coef=0.0,
        )
    assert gate is not None and codebook is not None
    span = emit(
        backbone, gate, codebook, prompt_embeds,
        capacity=spec.capacity,
        depth_repeats=spec.depth_repeats,
        allow_text=spec.allow_text,
    )
    return span_loss(
        backbone, span, answer_ids,
        lambda_token=spec.lambda_token,
        commitment_beta=spec.commitment_beta,
        baseline=baseline,
    )


def _trainable(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in modules:
        if module is None:
            continue
        params.extend(p for p in module.parameters() if p.requires_grad)
    return params


def _greedy_answer(backbone, tokenizer, embeds, mask, n_tokens: int) -> str:
    seq = embeds
    attn = mask
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    for _ in range(n_tokens):
        hidden = backbone.forward_hidden(seq, attn)
        logits = backbone.logits_from_hidden(hidden[:, -1, :])
        tok = int(logits.argmax(dim=-1).item())
        ids.append(tok)
        if eos is not None and tok == eos:
            break
        nxt = torch.tensor([[tok]], dtype=torch.long, device=seq.device)
        seq = torch.cat([seq, backbone.embed_tokens(nxt)], dim=1)
        attn = torch.cat(
            [attn, torch.ones(1, 1, dtype=attn.dtype, device=attn.device)], dim=1
        )
    if tokenizer is None:
        return " ".join(str(i) for i in ids)
    return tokenizer.decode(ids, skip_special_tokens=True)


def _eval_latent(backbone, gate, codebook, tokenizer, cfg: TrainConfig, spec: ArmSpec, items) -> dict:
    gate.eval()
    codebook.eval()
    backbone_was_training = backbone.training if hasattr(backbone, "training") else False
    if hasattr(backbone, "eval"):
        backbone.eval()
    rows = []
    with torch.no_grad():
        for item in items:
            prompt_ids = _tokenize(cfg, tokenizer, item.prompt_direct, cfg.max_prompt_tokens)
            device = next(gate.parameters()).device
            embeds = backbone.embed_tokens(_ids_tensor(prompt_ids, device))
            full = emit(
                backbone, gate, codebook, embeds,
                capacity=spec.capacity, depth_repeats=spec.depth_repeats, allow_text=spec.allow_text,
            )
            ablated = emit(
                backbone, gate, codebook, embeds,
                capacity=spec.capacity, depth_repeats=spec.depth_repeats,
                allow_text=spec.allow_text, ablate=True,
            )
            halted = emit(
                backbone, gate, codebook, embeds,
                capacity=spec.capacity, depth_repeats=spec.depth_repeats,
                allow_text=False, force_halt=True,
            )
            text = _greedy_answer(backbone, tokenizer, full.embeds, full.mask, spec.answer_tokens)
            text_ab = _greedy_answer(backbone, tokenizer, ablated.embeds, ablated.mask, spec.answer_tokens)
            text_halt = _greedy_answer(backbone, tokenizer, halted.embeds, halted.mask, spec.answer_tokens)
            rows.append({
                "id": item.id,
                "correct": bool(verify(item.family, text, item.gold)),
                "correct_ablated": bool(verify(item.family, text_ab, item.gold)),
                "correct_halt": bool(verify(item.family, text_halt, item.gold)),
                "n_latent": full.n_latent,
                "emission_rate": full.emission_rate,
                "text": text[:300],
            })
    if hasattr(backbone, "train") and backbone_was_training:
        backbone.train()
    gate.train()
    codebook.train()
    acc = float(sum(r["correct"] for r in rows) / len(rows))
    acc_ab = float(sum(r["correct_ablated"] for r in rows) / len(rows))
    acc_halt = float(sum(r["correct_halt"] for r in rows) / len(rows))
    rate = float(sum(r["emission_rate"] for r in rows) / len(rows))
    return {
        "n": len(rows),
        "split": "train",
        "sealed_test_read": False,
        "accuracy": acc,
        "ablated_accuracy": acc_ab,
        "halt_accuracy": acc_halt,
        "emission_rate": rate,
        "ablation_drop": acc - acc_ab,
        # Same weights with the gate forced off. Not a separately trained E arm.
        "necessity_vs_halt_proxy": necessity_fraction(acc, acc_ab, acc_halt),
        "decorative": decorative_report(rate, acc, acc_ab),
        "rows": rows,
    }


def _measure(cfg: TrainConfig, backbone, gate, codebook, tokenizer, spec: ArmSpec, prompt_embeds) -> dict:
    """Wall-clock and FLOP-counter measurements. Neither is treated as the other."""
    if not cfg.measure_flops:
        return {"measured": False}

    def _latent():
        with torch.no_grad():
            emit(
                backbone, gate, codebook, prompt_embeds,
                capacity=spec.capacity, depth_repeats=spec.depth_repeats, allow_text=False,
            )

    latent = measure_calls(_latent, repeats=3, warmup=1)

    text = None
    generate_note = "no tokenizer generate on the tiny backbone"
    if tokenizer is not None and not cfg.tiny:
        # A short generate is what the counter can finish. 1024 is timed separately
        # without the counter, because the counter multiplies decode cost.
        item_prompt = "What is 2 multiplied by 2?"

        def _short():
            ids = tokenizer(wrap_user(tokenizer, item_prompt, think=False), return_tensors="pt")
            ids = {k: v.to(backbone.causal.device) for k, v in ids.items()}
            with torch.no_grad():
                backbone.causal.generate(
                    **ids, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        text = measure_calls(_short, repeats=2, warmup=1)
        # Direct wall-clock of the protocol budget. One prompt, greedy, no counter.
        ids = tokenizer(wrap_user(tokenizer, item_prompt, think=False), return_tensors="pt")
        ids = {k: v.to(backbone.causal.device) for k, v in ids.items()}
        walls = []
        for new in (128, 1024):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                backbone.causal.generate(
                    **ids, max_new_tokens=new, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            walls.append((new, time.perf_counter() - start))
        generate_note = "1024-token generate timed once; flop counter used only on 8 new tokens"
        wall_1024 = walls[-1][1]
        wall_128 = walls[0][1]
    else:
        wall_1024 = None
        wall_128 = None

    per_text = None
    extrapolated = None
    if text is not None and text.flops_mean > 0:
        per_text = text.flops_mean / 8.0
        # One proof-config emit versus a 1024-token decode extrapolated from the
        # 8-token measurement. The extrapolation is labelled. The 1024 wall-clock
        # below is not extrapolated.
        extrapolated = judge_flop_ratio(
            latent.flops_mean, per_text * cfg.cot_tokens
        ).as_dict()
    direct = None
    if text is not None:
        direct = judge_flop_ratio(latent.flops_mean, text.flops_mean).as_dict()

    plan = None
    if per_text is not None and latent.flops_mean > 0:
        # Per latent step ≈ the measured emit divided by the forwards it did.
        # At depth_repeats=1 a full-cap emit does `capacity` forwards plus the
        # halt check's extra forward. Use the mean per forward as the step cost
        # only as a planning input; the verdict above uses the raw totals.
        plan = plan_depth_repeats(
            latent.flops_mean / max(spec.capacity, 1),
            per_text,
            capacity=16,
            cot_tokens=cfg.cot_tokens,
        )

    wall_ratio = None
    if wall_1024 is not None and latent.wall_mean_s > 0:
        wall_ratio = judge_flop_ratio(latent.wall_mean_s, wall_1024).as_dict()

    return {
        "measured": True,
        "latent_emit": latent.as_dict(),
        "text_generate_8": None if text is None else text.as_dict(),
        "generate_wall_s": {"128": wall_128, "1024": wall_1024},
        "direct_emit_vs_8_token_generate": direct,
        "extrapolated_emit_vs_1024": extrapolated,
        "wall_emit_vs_1024_generate": wall_ratio,
        "plan_depth_repeats_capacity_16": plan,
        "note": generate_note,
    }


def train_cell(cfg: TrainConfig) -> dict:
    """Train one cell. Returns a report. Does not read the test split."""
    spec = arm_spec(
        cfg.arm,
        depth_repeats=cfg.depth_repeats,
        capacity=cfg.capacity,
        cot_tokens=cfg.cot_tokens,
        answer_tokens=cfg.answer_tokens,
    )
    if cfg.tiny:
        backbone, peft_model, tokenizer = _build_tiny(cfg)
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        if cfg.device != "cuda":
            raise SystemExit("the instruct backbone is loaded on cuda; pass --tiny for cpu")
        backbone, peft_model, tokenizer = _load_qwen(cfg)
        device = torch.device("cuda")
        dtype = torch.bfloat16

    torch.manual_seed(_seed_int(cfg.seed, "block_init"))
    gate = None
    codebook = None
    depth_token = None
    modules: list[torch.nn.Module] = []
    if peft_model is not None:
        modules.append(peft_model)
    else:
        modules.append(backbone)
    if spec.kind == "latent":
        codebook = VectorQuantizer(spec.n_codes, 64 if not cfg.tiny else 8, commitment_beta=spec.commitment_beta)
        gate = EmissionGate(backbone.d_model, codebook.dim)
        gate.to(device=device, dtype=dtype)
        codebook.to(device=device)
        modules.extend([gate, codebook])
    if spec.kind == "depth":
        depth_token = torch.nn.Parameter(torch.zeros(backbone.d_model, device=device, dtype=dtype))

    params = _trainable(modules)
    if depth_token is not None:
        params.append(depth_token)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    items = generate_many(family_code(cfg.family), "train", min(cfg.steps, 2000), start=0)
    if not items:
        raise RuntimeError("no training items")

    baseline: torch.Tensor | None = None
    losses: list[float] = []
    rates: list[float] = []
    indices: list[int] = []
    saw_action_grad = False
    tokens = 0
    if torch.cuda.is_available() and not cfg.tiny:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(cfg.steps):
        item = items[step % len(items)]
        prompt_text = item.prompt_cot if spec.kind == "text" else item.prompt_direct
        prompt_ids = _tokenize(cfg, tokenizer, prompt_text, cfg.max_prompt_tokens)
        answer_ids = _ids_tensor(
            _tokenize(cfg, tokenizer, _answer_text(item.gold), cfg.answer_tokens),
            device,
        )
        prompt_embeds = backbone.embed_tokens(_ids_tensor(prompt_ids, device))
        target_ids = None
        if spec.kind == "text":
            trace = item.solution + "\n" + _answer_text(item.gold)
            target_ids = _ids_tensor(_tokenize(cfg, tokenizer, trace, cfg.max_trace_tokens), device)
        opt.zero_grad(set_to_none=True)
        parts, baseline = _step(
            backbone, spec, gate, codebook, depth_token,
            prompt_embeds, answer_ids, target_ids, baseline,
        )
        if not torch.isfinite(parts.loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        parts.loss.backward()
        if gate is not None and gate.action.weight.grad is not None:
            if float(gate.action.weight.grad.abs().sum()) > 0:
                saw_action_grad = True
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(parts.loss.detach()))
        rates.append(parts.n_latent / float(spec.capacity))
        indices.extend(parts.code_indices)
        tokens += int(prompt_embeds.shape[1] + answer_ids.numel() + parts.n_latent + parts.n_text)
        if cfg.log_every and step % cfg.log_every == 0:
            print(
                f"step {step} loss {losses[-1]:.4f} ce {float(parts.ce.detach()):.4f} "
                f"latent {parts.n_latent}/{spec.capacity} text {parts.n_text}",
                flush=True,
            )
    seconds = time.perf_counter() - start
    peak = 0.0
    if torch.cuda.is_available() and not cfg.tiny:
        peak = float(torch.cuda.max_memory_allocated()) / 1024**3

    n_codes = codebook.n_codes if codebook is not None else spec.n_codes
    if indices:
        usage = usage_stats(torch.tensor(indices), n_codes)
        degenerate = is_degenerate(torch.tensor(indices), n_codes)
    else:
        usage = {"n_used": 0, "perplexity": 0.0, "dead_fraction": 1.0, "max_fraction": 0.0}
        degenerate = spec.kind == "latent"

    eval_report = None
    if cfg.eval_items > 0 and spec.kind == "latent" and gate is not None and codebook is not None:
        # Train items only. The sealed test split is not generated.
        eval_items = items[: cfg.eval_items]
        eval_report = _eval_latent(backbone, gate, codebook, tokenizer, cfg, spec, eval_items)

    measure = {"measured": False}
    if spec.kind == "latent" and gate is not None and codebook is not None and cfg.measure_flops:
        probe_item = items[0]
        probe_ids = _tokenize(cfg, tokenizer, probe_item.prompt_direct, cfg.max_prompt_tokens)
        probe_embeds = backbone.embed_tokens(_ids_tensor(probe_ids, device))
        gate.eval()
        codebook.eval()
        measure = _measure(cfg, backbone, gate, codebook, tokenizer, spec, probe_embeds.detach())
        gate.train()
        codebook.train()

    # Planning projections. steps_per_run is an assumption, stated as one.
    steps_per_run = 2000  # one pass over the 2,000 train items
    n_runs = 189
    proof_hours = project_gpu_hours(seconds / max(cfg.steps, 1), steps_per_run=steps_per_run, n_runs=n_runs, scale=1.0)
    # Matched-depth scale: proof uses capacity `c` and repeats `r`. The manifest
    # cap is 16. Repeats come from the planner when it fitted; otherwise the
    # scale is reported as unknown rather than invented.
    plan = measure.get("plan_depth_repeats_capacity_16") if isinstance(measure, dict) else None
    if plan and plan.get("repeats"):
        scale = (16 / spec.capacity) * (float(plan["repeats"]) / spec.depth_repeats)
        matched = project_gpu_hours(
            seconds / max(cfg.steps, 1),
            steps_per_run=steps_per_run,
            n_runs=n_runs,
            scale=scale,
        )
        matched["assumption"] = (
            "step time scales linearly with capacity and depth_repeats; "
            "2000 steps is one pass over the train split, not a tuned budget"
        )
    else:
        matched = None

    chat_on = bool(tokenizer is not None and getattr(tokenizer, "chat_template", None))
    report = {
        "claim": False,
        "note": (
            "Pipeline proof. Train split only. Not a G2b verdict. "
            "Thinking disabled. Backbone is the instruct checkpoint when tiny is false."
        ),
        "run_id": run_id(cfg.family, cfg.arm, cfg.seed),
        "arm": spec.as_dict(),
        "model_id": None if cfg.tiny else cfg.model_id,
        "tiny": cfg.tiny,
        "thinking": False,
        "chat_template": chat_on,
        "steps": cfg.steps,
        "seconds": seconds,
        "steps_per_second": cfg.steps / seconds if seconds else None,
        "tokens": tokens,
        "tokens_per_second": tokens / seconds if seconds else None,
        "peak_vram_gb": peak,
        "over_8gb": bool(peak > VRAM_BUDGET_GB + 0.25),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_finite": all(map(lambda x: x == x, losses)),  # NaN != NaN
        "mean_emission_rate": float(sum(rates) / len(rates)) if rates else 0.0,
        "steps_with_latent": int(sum(r > 0 for r in rates)),
        "saw_action_grad": saw_action_grad,
        "codes_used": int(usage["n_used"]),
        "code_perplexity": float(usage["perplexity"]),
        "degenerate_codes": bool(degenerate) if indices else spec.kind == "latent",
        "eval": eval_report,
        "flops": measure,
        "projection_proof_config": proof_hours,
        "projection_matched_depth": matched,
    }
    out = Path(cfg.out)
    if not cfg.tiny:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {out}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train one E2 cell")
    p.add_argument("--arm", default="A")
    p.add_argument("--family", default="f1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--capacity", type=int, default=4)
    p.add_argument("--depth-repeats", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-items", type=int, default=8)
    p.add_argument("--measure-flops", action="store_true", default=True)
    p.add_argument("--no-measure-flops", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out", default="artifacts/e2/pilot_a_f1.json")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = p.parse_args(argv)
    cfg = TrainConfig(
        arm=args.arm,
        family=args.family,
        seed=args.seed,
        steps=args.steps,
        capacity=args.capacity,
        depth_repeats=args.depth_repeats,
        lr=args.lr,
        eval_items=0 if args.tiny else args.eval_items,
        measure_flops=not args.no_measure_flops,
        tiny=args.tiny,
        device="cpu" if args.tiny else "cuda",
        out=args.out,
        model_id=args.model,
    )
    if not cfg.tiny and "Base" in cfg.model_id:
        raise SystemExit("refusing Qwen3-1.7B-Base: F44, the base model is VOID for E2")
    report = train_cell(cfg)
    summary = {
        "run_id": report["run_id"],
        "loss_first": report["loss_first"],
        "loss_last": report["loss_last"],
        "mean_emission_rate": report["mean_emission_rate"],
        "codes_used": report["codes_used"],
        "steps_per_second": report["steps_per_second"],
        "tokens_per_second": report["tokens_per_second"],
        "peak_vram_gb": report["peak_vram_gb"],
        "over_8gb": report["over_8gb"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
