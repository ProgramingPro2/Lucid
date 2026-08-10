"""Stage-B GRPO trainer on SoftPrefixAdapter + QLoRA (Fable §2.3).

Reuses Stage-A model class, GpuGate, CheckpointDB. Does not modify src/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.ckpt_db import CheckpointDB  # noqa: E402
from experiments.pivot.adapter import SoftPrefixAdapter, build_inputs_embeds  # noqa: E402
from experiments.pivot.gate_g1 import load_stage_a_for_infer  # noqa: E402
from experiments.pivot.reward_stage_b import (  # noqa: E402
    HybridReward,
    LogitJudge,
    group_advantages,
)
from experiments.pivot.rollout_stage_b import (  # noqa: E402
    PromptSampler,
    generate_group,
    load_jepa,
)
from experiments.pivot.train_stage_a import (  # noqa: E402
    ParaPairDataset,
    StageAModel,
    _FlatState,
    _flat_trainable_state,
    _grads_finite,
    _load_cfg,
    _load_flat_state,
    _maybe_gate,
    _param_groups,
    _sanitize_grads,
    _save_pt,
    _seed_all,
    _trainable_params,
)

DEFAULT_CFG = _ROOT / "configs" / "pivot_stage_b.json"


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _merge_backbone_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    phase_a = _load_cfg(_ROOT / cfg.get("phase_a_cfg", "configs/pivot_step1.json"))
    for key in (
        "backbone",
        "backbone_fallback",
        "load_in_4bit",
        "bnb_4bit_quant_type",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
        "z_dim",
        "d_model",
        "n_prefix",
        "adapter_hidden",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "max_prev_tokens",
        "max_target_tokens",
        "ctx_max",
    ):
        if key in cfg:
            phase_a[key] = cfg[key]
    return phase_a


def _cosine_lr(step: int, total: int, base: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    t = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return base * (min_ratio + (1.0 - min_ratio) * cosine)


def _extract_lora_sd(backbone) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in backbone.state_dict().items()
        if "lora_" in k
    }


def _load_lora_into(backbone, sd: dict[str, torch.Tensor]) -> None:
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    _ = missing, unexpected


@torch.no_grad()
def _sequence_logprobs(
    model: StageAModel,
    adapter: SoftPrefixAdapter,
    tokenizer,
    z: torch.Tensor,
    prev: str,
    text: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (token_logprobs [T], token_entropy [T]) for generated text given soft-prefix(z)+prev."""
    enc_prev = tokenizer(
        prev,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    enc_tgt = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    prev_ids = enc_prev["input_ids"].to(device)
    tgt_ids = enc_tgt["input_ids"].to(device)
    if tgt_ids.numel() == 0:
        empty = torch.zeros(0, device=device)
        return empty, empty
    input_ids = torch.cat([prev_ids, tgt_ids], dim=1)
    embed = model.backbone.get_input_embeddings()
    tok_emb = embed(input_ids)
    prefix = adapter(z).to(dtype=tok_emb.dtype)
    prefix = torch.clamp(prefix, -10.0, 10.0)
    inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    out = model.backbone(inputs_embeds=inputs_embeds, attention_mask=attn)
    # logits aligned to next-token: position n_prefix+prev_len-1 predicts first tgt token...
    n_prefix = prefix.shape[1]
    pl = prev_ids.shape[1]
    # logits index for target token j is at position n_prefix + pl + j - 1
    logits = out.logits  # (1, n_prefix+L, V)
    start = n_prefix + pl - 1
    end = start + tgt_ids.shape[1]
    step_logits = logits[:, start:end, :]  # (1, T, V)
    log_probs = F.log_softmax(step_logits.float(), dim=-1)
    tok_lp = log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T,)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=-1).squeeze(0)  # (T,)
    return tok_lp, ent


def _sequence_logprobs_grad(
    model: StageAModel,
    tokenizer,
    z: torch.Tensor,
    prev: str,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable per-token logprobs under current policy adapter."""
    enc_prev = tokenizer(
        prev,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    enc_tgt = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    prev_ids = enc_prev["input_ids"].to(device)
    tgt_ids = enc_tgt["input_ids"].to(device)
    if tgt_ids.numel() == 0:
        return torch.zeros(0, device=device, requires_grad=True)
    input_ids = torch.cat([prev_ids, tgt_ids], dim=1)
    embeds = build_inputs_embeds(model.backbone, model.adapter, z, input_ids)
    attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
    out = model.backbone(inputs_embeds=embeds, attention_mask=attn)
    n_prefix = int(model.adapter.n_prefix)
    pl = prev_ids.shape[1]
    start = n_prefix + pl - 1
    end = start + tgt_ids.shape[1]
    step_logits = out.logits[:, start:end, :]
    log_probs = F.log_softmax(step_logits.float(), dim=-1)
    return log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_anti_gaming_battery(
    cfg: dict[str, Any],
    model: StageAModel,
    tokenizer,
    rewarder: HybridReward,
    held_ds: ParaPairDataset,
    *,
    step: int,
    baseline: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    heldout_judge: LogitJudge | None = None,
) -> dict[str, Any]:
    """Lightweight mid-run battery (greedy on probe set)."""
    from experiments.pivot.gate_g1 import generate_paragraph, encode_minilm_pair_cos
    from experiments.pivot.reward_stage_b import distinct_n, tokenize_words

    n = int(cfg.get("probe_n_held", 128))
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(held_ds), size=min(n, len(held_ds)), replace=False).tolist()
    gens: list[str] = []
    feds: list[float] = []
    js: list[float] = []
    d2s: list[float] = []
    model.eval()
    for i in idxs:
        item = held_ds[i]
        text = generate_paragraph(
            model,
            tokenizer,
            item["z"],
            item["prev"],
            max_new_tokens=int(cfg.get("eval_max_new_tokens", 128)),
            do_sample=False,
            device=device,
        )
        gens.append(text)
        br = rewarder.score_one(text, item["z"], skip_judge_if_gated=False)
        feds.append(br.c if np.isfinite(br.c) else float("nan"))
        js.append(br.j)
        d2s.append(distinct_n(tokenize_words(text), 2))
    fed_med = float(np.nanmedian(np.asarray(feds, dtype=np.float64)))
    j_mean = float(np.nanmean(np.asarray(js, dtype=np.float64)))
    d2_mean = float(np.nanmean(np.asarray(d2s, dtype=np.float64)))

    held_j = None
    if heldout_judge is not None:
        heldout_judge.to_device()
        hj = heldout_judge.score_coherence(gens)
        held_j = float(np.nanmean(np.asarray(hj, dtype=np.float64)))
        if not bool(cfg.get("judge_resident", True)):
            heldout_judge.to_cpu()

    base_fed = float(baseline.get("fed_vs_concept_median", float("nan")))
    base_j = float(baseline.get("train_judge_mean", float("nan")))
    hack = False
    if np.isfinite(base_fed) and np.isfinite(base_j):
        if j_mean >= base_j + 0.05 and fed_med < base_fed:
            hack = True

    row = {
        "step": step,
        "fed_vs_concept_median": fed_med,
        "train_judge_mean": j_mean,
        "heldout_judge_mean": held_j,
        "distinct2_mean": d2_mean,
        "baseline_fed": base_fed,
        "baseline_train_j": base_j,
        "reward_hacking_flag": hack,
        "n": len(gens),
    }
    append_jsonl(out_dir / "battery.jsonl", row)
    (out_dir / f"battery_step{step:06d}.json").write_text(json.dumps(row, indent=2))
    print(f"[grpo] battery step={step} {json.dumps(row)}", flush=True)
    model.train()
    return row


def train_grpo(
    cfg: dict[str, Any],
    *,
    init_ckpt: Path,
    ref_ckpt: Path,
    device_name: str = "auto",
    max_steps: int | None = None,
) -> dict[str, Any]:
    _seed_all(int(cfg.get("seed", 42)))
    device = _device(device_name)
    out_dir = _ROOT / cfg["out_dir"] / "grpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"

    phase_cfg = _merge_backbone_cfg(cfg)
    # Policy LRs from Stage-B
    phase_cfg["lr_adapter"] = float(cfg.get("lr_adapter", 2e-5))
    phase_cfg["lr_lora"] = float(cfg.get("lr_lora", 5e-6))
    phase_cfg["weight_decay"] = float(cfg.get("weight_decay", 0.0))

    print(f"[grpo] loading policy from {init_ckpt}", flush=True)
    model, tokenizer, meta = load_stage_a_for_infer(phase_cfg, init_ckpt, device)
    model.train()
    for n, p in model.named_parameters():
        if "lora_" in n or n.startswith("adapter."):
            p.requires_grad_(True)

    # Frozen Phase-A reference adapter
    print(f"[grpo] loading Phase-A reference from {ref_ckpt}", flush=True)
    ref_blob = torch.load(ref_ckpt, map_location="cpu", weights_only=False)
    ref_adapter = SoftPrefixAdapter(
        z_dim=int(phase_cfg["z_dim"]),
        d_model=int(phase_cfg.get("d_model", model.adapter.d_model)),
        n_prefix=int(phase_cfg["n_prefix"]),
        hidden=int(phase_cfg.get("adapter_hidden", 512)),
    ).to(device)
    ref_adapter.load_state_dict(ref_blob["adapter"])
    ref_adapter.eval()
    for p in ref_adapter.parameters():
        p.requires_grad_(False)
    ref_lora_sd = {}
    if isinstance(ref_blob.get("backbone"), dict):
        ref_lora_sd = {
            k: v.cpu().clone()
            for k, v in ref_blob["backbone"].items()
            if "lora_" in k
        }
    policy_lora_sd = _extract_lora_sd(model.backbone)

    opt = torch.optim.AdamW(
        _param_groups(model, phase_cfg),
        weight_decay=float(phase_cfg["weight_decay"]),
    )
    gate = _maybe_gate(device, bool(cfg.get("gpu_gate", True)))
    params = _trainable_params(model)

    codec = load_jepa(_ROOT / cfg["jepa"])
    judge = LogitJudge(
        cfg.get("train_judge", "Qwen/Qwen2.5-1.5B-Instruct"),
        device,
        load_in_4bit=bool(cfg.get("judge_load_in_4bit", True)),
    )
    rewarder = HybridReward(
        codec,
        judge,
        w_z=float(cfg.get("w_z", 0.6)),
        w_j=float(cfg.get("w_j", 0.4)),
        f_floor=float(cfg.get("f_floor", 0.35)),
        f_ceiling=float(cfg.get("f_ceiling", 0.62)),
    )
    heldout_judge = None  # swap-in on battery only to save VRAM

    held = ParaPairDataset(_ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="held")
    train_ds = ParaPairDataset(_ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="train")
    sampler = PromptSampler(
        held,
        train_ds,
        mix=cfg.get("prompt_mix", {"held_bank": 0.5, "fiber": 0.25, "mdn": 0.25}),
        fiber_kappa=float(cfg.get("fiber_kappa", 700.0)),
        mdn_path=_ROOT / cfg["mdn_ckpt"] if cfg.get("mdn_ckpt") else None,
        seed=int(cfg.get("seed", 42)),
    )

    baseline_path = _ROOT / cfg["out_dir"] / "baseline_phaseA.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    db = CheckpointDB(out_dir / "checkpoints.sqlite", keep_last=3)
    run_id = db.latest_run()
    flat_holder = _FlatState()
    start_step = 0
    if run_id is None:
        run_id = db.start_run(
            str(cfg.get("corpus", "grpo")),
            "grpo",
            int(cfg.get("seed", 42)),
            int(cfg.get("ctx_max", 512)),
            int(cfg.get("n_prefix", 16)),
            {**cfg, "init_ckpt": str(init_ckpt), "ref_ckpt": str(ref_ckpt)},
        )
    else:
        loaded = db.load_latest_checkpoint(run_id, device="cpu")
        if loaded is not None:
            model_state, opt_state, epoch = loaded
            start_step = _load_flat_state(model, model_state)
            if opt_state:
                try:
                    opt.load_state_dict(opt_state)
                except Exception as e:  # noqa: BLE001
                    print(f"[grpo] opt resume skip: {e}", flush=True)
            print(f"[grpo] resumed step={start_step}", flush=True)

    total_steps = int(max_steps if max_steps is not None else cfg.get("total_steps", 600))
    k = int(cfg.get("group_k", 8))
    prompts_per_step = int(cfg.get("prompts_per_step", 4))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    beta = float(cfg.get("kl_beta", 0.05))
    micro_batch = int(cfg.get("micro_batch", 4))
    grad_accum = int(cfg.get("grad_accum", 8))
    ckpt_every = int(cfg.get("ckpt_every", 50))
    battery_every = int(cfg.get("battery_every", 100))
    warmup = int(cfg.get("warmup_steps", 20))
    min_ratio = float(cfg.get("cosine_min_ratio", 0.2))
    base_lr_a = float(cfg.get("lr_adapter", 2e-5))
    base_lr_l = float(cfg.get("lr_lora", 5e-6))

    history: list[dict[str, Any]] = []
    init_entropy: float | None = None
    lr_halved = False
    stall_best_fed = float(baseline.get("fed_vs_concept_median", -1.0))
    stall_count = 0

    print(
        f"[grpo] start step={start_step}/{total_steps} k={k} prompts/step={prompts_per_step} beta={beta}",
        flush=True,
    )

    for step in range(start_step + 1, total_steps + 1):
        if gate is not None:
            gate.step(model, opt)

        # LR schedule
        lr_a = _cosine_lr(step, total_steps, base_lr_a, warmup, min_ratio)
        lr_l = _cosine_lr(step, total_steps, base_lr_l, warmup, min_ratio)
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr_a if gi == 0 else lr_l

        prompts = sampler.sample_n(prompts_per_step)
        # Collect rollouts
        groups: list[dict[str, Any]] = []
        model.eval()
        with torch.no_grad():
            for pr in prompts:
                texts = generate_group(
                    model,
                    tokenizer,
                    pr.z_cmd,
                    pr.ctx,
                    k=k,
                    max_new_tokens=int(cfg.get("max_new_tokens", 200)),
                    min_new_tokens=int(cfg.get("min_new_tokens", 40)),
                    temperature=float(cfg.get("temperature", 0.9)),
                    top_p=float(cfg.get("top_p", 0.95)),
                    device=device,
                )
                scored = rewarder.score_group(texts, pr.z_cmd)
                rewards = [s.reward for s in scored]
                adv = group_advantages(rewards)
                z_t = torch.from_numpy(pr.z_cmd[None, :]).to(device=device, dtype=torch.float32)

                # old + ref logprobs (policy LoRA currently loaded)
                old_lps: list[torch.Tensor] = []
                ents: list[float] = []
                for s in scored:
                    lp, ent = _sequence_logprobs(
                        model, model.adapter, tokenizer, z_t, pr.ctx, s.text, device
                    )
                    old_lps.append(lp.detach())
                    ents.append(float(ent.mean().item()) if ent.numel() else 0.0)

                # Ref logprobs: swap to Phase-A LoRA + ref adapter
                cur_lora = _extract_lora_sd(model.backbone)
                if ref_lora_sd:
                    _load_lora_into(model.backbone, ref_lora_sd)
                ref_lps: list[torch.Tensor] = []
                for s in scored:
                    lp, _ = _sequence_logprobs(
                        model, ref_adapter, tokenizer, z_t, pr.ctx, s.text, device
                    )
                    ref_lps.append(lp.detach())
                _load_lora_into(model.backbone, cur_lora)

                groups.append(
                    {
                        "prompt": pr,
                        "scored": scored,
                        "rewards": rewards,
                        "adv": adv,
                        "old_lps": old_lps,
                        "ref_lps": ref_lps,
                        "ents": ents,
                        "z_t": z_t,
                    }
                )
                append_jsonl(
                    rollouts_path,
                    {
                        "step": step,
                        "source": pr.source,
                        "doc_id": pr.doc_id,
                        "para_idx": pr.para_idx,
                        "ctx": pr.ctx,
                        "z_cmd": pr.z_cmd.astype(np.float32).tolist(),
                        "generations": [s.as_dict() for s in scored],
                        "rewards": rewards,
                        "adv": adv,
                    },
                )
        model.train()

        # Flatten sequences for micro-batching
        flat: list[dict[str, Any]] = []
        for g in groups:
            for i, s in enumerate(g["scored"]):
                if not (s.text or "").strip():
                    continue
                flat.append(
                    {
                        "z_t": g["z_t"],
                        "ctx": g["prompt"].ctx,
                        "text": s.text,
                        "adv": float(g["adv"][i]),
                        "old_lp": g["old_lps"][i],
                        "ref_lp": g["ref_lps"][i],
                        "ent": g["ents"][i],
                        "reward": float(g["rewards"][i]),
                    }
                )

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        kl_acc = 0.0
        ent_acc = 0.0
        n_seq = 0
        micro_i = 0
        for fi, sample in enumerate(flat):
            new_lp = _sequence_logprobs_grad(
                model, tokenizer, sample["z_t"], sample["ctx"], sample["text"], device
            )
            old_lp = sample["old_lp"].to(device)
            ref_lp = sample["ref_lp"].to(device)
            T = int(new_lp.numel())
            if T == 0:
                continue
            # Align lengths if tokenizer drift
            T = min(T, int(old_lp.numel()), int(ref_lp.numel()))
            new_lp = new_lp[:T]
            old_lp = old_lp[:T]
            ref_lp = ref_lp[:T]
            ratio = torch.exp(new_lp - old_lp)
            adv = float(sample["adv"])
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            # length-normalized: mean over tokens
            pg = -torch.minimum(unclipped, clipped).mean()
            # k3 KL estimator: exp(ref-new) - (ref-new) - 1  (Schulman)
            log_ratio_kr = ref_lp - new_lp
            kl_tok = torch.exp(log_ratio_kr) - log_ratio_kr - 1.0
            kl = kl_tok.mean()
            loss = (pg + beta * kl) / float(grad_accum)
            loss.backward()
            micro_i += 1
            loss_acc += float(pg.detach().item())
            kl_acc += float(kl.detach().item()) * T  # approx nats/paragraph via sum? use mean*T
            ent_acc += float(sample["ent"])
            n_seq += 1

            if micro_i >= micro_batch * grad_accum or fi == len(flat) - 1:
                # Actually accumulate until grad_accum sequences... use micro_batch as seqs per backward chunk already done
                pass
            if (fi + 1) % max(1, micro_batch) == 0 or fi == len(flat) - 1:
                if (fi + 1) % (micro_batch * grad_accum) == 0 or fi == len(flat) - 1:
                    _sanitize_grads(params)
                    if _grads_finite(params):
                        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("grad_clip", 1.0)))
                        opt.step()
                    else:
                        print(f"[grpo] bad grads step={step}; skip opt", flush=True)
                    opt.zero_grad(set_to_none=True)

        # Fix: simpler opt step once per GRPO step after all seqs
        # (above may have stepped multiple times — acceptable for microbatching)
        mean_loss = loss_acc / max(1, n_seq)
        mean_kl = kl_acc / max(1, n_seq)  # mean token-KL * rough
        # Better KL nats/paragraph: average of (kl_tok.mean()*T)
        mean_ent = ent_acc / max(1, n_seq)
        if init_entropy is None and mean_ent > 0:
            init_entropy = mean_ent
        if init_entropy and mean_ent < init_entropy * (1.0 - float(cfg.get("entropy_drop_halt", 0.40))):
            if not lr_halved:
                base_lr_a *= 0.5
                base_lr_l *= 0.5
                lr_halved = True
                print(f"[grpo] entropy drop → halve LR (ent={mean_ent:.4f})", flush=True)
            else:
                print(f"[grpo] entropy collapsed again; stopping", flush=True)
                break

        # KL controller
        if step % int(cfg.get("kl_controller_every", 25)) == 0:
            # mean_kl here is token-mean * count proxy; recompute paragraph KL from last groups
            para_kls = []
            for g in groups:
                for i in range(len(g["scored"])):
                    old = g["old_lps"][i]
                    ref = g["ref_lps"][i]
                    T = min(old.numel(), ref.numel())
                    if T == 0:
                        continue
                    # approx KL(policy||ref) via stored old≈policy at sample: use old vs ref
                    lr = ref[:T] - old[:T]
                    para_kls.append(float((torch.exp(lr) - lr - 1).sum().item()))
            mean_para_kl = float(np.mean(para_kls)) if para_kls else 0.0
            if mean_para_kl > float(cfg.get("kl_high", 12.0)):
                beta *= 1.5
                print(f"[grpo] KL high {mean_para_kl:.2f} → beta={beta:.4f}", flush=True)
            elif mean_para_kl < float(cfg.get("kl_low", 2.0)):
                beta *= 0.7
                print(f"[grpo] KL low {mean_para_kl:.2f} → beta={beta:.4f}", flush=True)
        else:
            mean_para_kl = mean_kl

        row = {
            "step": step,
            "loss": mean_loss,
            "kl_nats": mean_para_kl,
            "entropy": mean_ent,
            "beta": beta,
            "reward_mean": float(np.mean([x["reward"] for x in flat])) if flat else 0.0,
            "n_seq": n_seq,
            "lr_adapter": lr_a,
            "lr_lora": lr_l,
        }
        history.append(row)
        append_jsonl(out_dir / "train.jsonl", row)
        print(
            f"[grpo] step={step}/{total_steps} loss={mean_loss:.4f} "
            f"kl={mean_para_kl:.3f} ent={mean_ent:.3f} R={row['reward_mean']:.3f} beta={beta:.4f}",
            flush=True,
        )

        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_path = out_dir / f"checkpoint_step{step:06d}.pt"
            flat_holder.set_state(_flat_trainable_state(model, step))
            db.record_epoch(run_id, step, row)
            db.save_checkpoint(run_id, step, flat_holder, opt, device="cpu")
            _save_pt(ckpt_path, model, opt, 0, step, row)
            _save_pt(out_dir / "checkpoint_latest.pt", model, opt, 0, step, row)
            print(f"[grpo] checkpoint → {ckpt_path}", flush=True)

        if battery_every > 0 and step % battery_every == 0:
            if heldout_judge is None:
                try:
                    heldout_judge = LogitJudge(
                        cfg.get("heldout_judge", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
                        device,
                        load_in_4bit=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[grpo] heldout judge load failed: {e}", flush=True)
            batt = run_anti_gaming_battery(
                cfg,
                model,
                tokenizer,
                rewarder,
                held,
                step=step,
                baseline=baseline,
                out_dir=out_dir,
                device=device,
                heldout_judge=heldout_judge,
            )
            if batt.get("reward_hacking_flag"):
                print("[grpo] HARD STOP: reward hacking detected (judge up, fed down)", flush=True)
                incident = {
                    "step": step,
                    "battery": batt,
                    "action": "stop_revert_last_green",
                }
                (out_dir / "incident_reward_hacking.json").write_text(
                    json.dumps(incident, indent=2)
                )
                break
            fed = float(batt.get("fed_vs_concept_median", float("nan")))
            if np.isfinite(fed):
                if fed > stall_best_fed:
                    stall_best_fed = fed
                    stall_count = 0
                else:
                    stall_count += battery_every
                if stall_count >= 300:
                    print("[grpo] STALL: no tracking improvement for 300 steps", flush=True)
                    break

    final = out_dir / "checkpoint_latest.pt"
    summary = {
        "steps_ran": history[-1]["step"] if history else start_step,
        "history_tail": history[-10:],
        "final_ckpt": str(final),
        "beta_final": beta,
    }
    (out_dir / "grpo_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage-B GRPO")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--init-ckpt", type=Path, default=None)
    p.add_argument("--ref-ckpt", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)
    init = args.init_ckpt or (_ROOT / cfg["out_dir"] / "b0" / "ckpt.pt")
    if not init.exists():
        init = _ROOT / cfg["phase_a_ckpt"]
    ref = args.ref_ckpt or (_ROOT / cfg["phase_a_ckpt"])
    train_grpo(
        cfg,
        init_ckpt=init,
        ref_ckpt=ref,
        device_name=args.device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
