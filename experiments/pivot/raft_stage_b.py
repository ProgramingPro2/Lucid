"""Stage-B0 RAFT: best-of-k distill on hybrid-reward winners.

1) Score rollouts JSONL → winners JSONL (gate=1 AND c>=0.50 AND J>=0.5, argmax-R)
2) One CE epoch via Stage-A trainer loop retargeted at winners (LR halved).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.ckpt_db import CheckpointDB  # noqa: E402
from conceptpoc.config import GpuGateConfig  # noqa: E402
from conceptpoc.gpu_gate import GpuGate  # noqa: E402
from experiments.pivot.adapter import SoftPrefixAdapter, collate_prefix_batch  # noqa: E402
from experiments.pivot.gate_g1 import load_stage_a_for_infer  # noqa: E402
from experiments.pivot.rollout_stage_b import run_rollouts  # noqa: E402
from experiments.pivot.train_stage_a import (  # noqa: E402
    StageAModel,
    _FlatState,
    _flat_trainable_state,
    _grads_finite,
    _load_cfg,
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


def select_winners(
    rollouts_path: Path,
    winners_path: Path,
    *,
    min_cos: float = 0.50,
    min_j: float = 0.5,
) -> dict[str, Any]:
    """Keep argmax-R sample per prompt if acceptance bar clears."""
    winners_path.parent.mkdir(parents=True, exist_ok=True)
    n_prompts = 0
    n_accept = 0
    rewards_all: list[float] = []
    best_rewards: list[float] = []
    with rollouts_path.open("r", encoding="utf-8") as fin, winners_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            n_prompts += 1
            gens = row["generations"]
            rewards = [float(g.get("reward", 0.0)) for g in gens]
            rewards_all.extend(rewards)
            best_i = int(np.argmax(rewards)) if rewards else 0
            best = gens[best_i]
            best_rewards.append(float(best.get("reward", 0.0)))
            ok = (
                int(best.get("gate", 0)) == 1
                and float(best.get("c", float("nan"))) >= min_cos
                and float(best.get("j", 0.0)) >= min_j
            )
            if not ok:
                continue
            n_accept += 1
            out = {
                "prompt_idx": row.get("prompt_idx"),
                "source": row.get("source"),
                "doc_id": row.get("doc_id"),
                "para_idx": row.get("para_idx"),
                "ctx": row.get("ctx"),
                "z_cmd": row.get("z_cmd"),
                "target": best.get("text", ""),
                "c": best.get("c"),
                "j": best.get("j"),
                "f": best.get("f"),
                "reward": best.get("reward"),
                "gate": best.get("gate"),
                "best_idx": best_i,
                "k": row.get("k"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    stats = {
        "n_prompts": n_prompts,
        "n_winners": n_accept,
        "winner_rate": float(n_accept) / max(1, n_prompts),
        "reward_mean_all": float(np.mean(rewards_all)) if rewards_all else 0.0,
        "reward_std_all": float(np.std(rewards_all)) if rewards_all else 0.0,
        "reward_mean_best": float(np.mean(best_rewards)) if best_rewards else 0.0,
        "min_cos": min_cos,
        "min_j": min_j,
        "rollouts": str(rollouts_path),
        "winners": str(winners_path),
    }
    meta_path = winners_path.with_suffix(".stats.json")
    meta_path.write_text(json.dumps(stats, indent=2))
    print(f"[raft] winners {n_accept}/{n_prompts} ({stats['winner_rate']:.3f}) → {winners_path}", flush=True)
    return stats


class WinnersDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.rows[i]
        z = np.asarray(r["z_cmd"], dtype=np.float32)
        z = z / (np.linalg.norm(z) + 1e-12)
        return {
            "prev": r.get("ctx") or "",
            "target": r.get("target") or "",
            "z": z,
            "doc_id": int(r.get("doc_id") or 0),
            "para_idx": int(r.get("para_idx") or 0),
        }


def _make_winners_collate(tokenizer, cfg: dict[str, Any]):
    n_prefix = int(cfg["n_prefix"])

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        z = torch.from_numpy(np.stack([b["z"] for b in batch], axis=0))
        prev = [b["prev"] for b in batch]
        tgt = [b["target"] for b in batch]
        packed = collate_prefix_batch(
            tokenizer,
            prev,
            tgt,
            max_prev=int(cfg.get("max_prev_tokens", 256)),
            max_target=int(cfg.get("max_target_tokens", 256)),
            ctx_max=int(cfg.get("ctx_max", 512)),
            n_prefix=n_prefix,
        )
        packed["z"] = z
        packed["prev_text"] = prev
        packed["target_text"] = tgt
        return packed

    return _collate


def distill_winners(
    cfg: dict[str, Any],
    winners_path: Path,
    *,
    init_ckpt: Path,
    out_dir: Path,
    device_name: str = "auto",
) -> Path:
    """One CE epoch on winners; LR = Stage-A LR * b0_lr_scale (halved)."""
    _seed_all(int(cfg.get("seed", 42)))
    device = _device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_a_cfg = _load_cfg(_ROOT / cfg.get("phase_a_cfg", "configs/pivot_step1.json"))
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
        "grad_clip",
    ):
        if key in cfg:
            phase_a_cfg[key] = cfg[key]
        elif key in phase_a_cfg:
            pass

    # Halved LRs relative to Phase-A
    scale = float(cfg.get("b0_lr_scale", 0.5))
    phase_a_cfg["lr_adapter"] = float(phase_a_cfg.get("lr_adapter", 5e-4)) * scale
    phase_a_cfg["lr_lora"] = float(phase_a_cfg.get("lr_lora", 1e-4)) * scale
    phase_a_cfg["weight_decay"] = float(phase_a_cfg.get("weight_decay", 0.01))
    micro = int(cfg.get("b0_micro_batch", 1))
    accum = int(cfg.get("b0_grad_accum", 64))

    model, tokenizer, meta = load_stage_a_for_infer(phase_a_cfg, init_ckpt, device)
    model.train()
    # Ensure LoRA params require grad
    for n, p in model.named_parameters():
        if "lora_" in n or n.startswith("adapter."):
            p.requires_grad_(True)

    opt = torch.optim.AdamW(
        _param_groups(model, phase_a_cfg),
        weight_decay=float(phase_a_cfg["weight_decay"]),
    )
    gate = _maybe_gate(device, bool(cfg.get("gpu_gate", True)))

    ds = WinnersDataset(winners_path)
    if len(ds) == 0:
        raise RuntimeError(f"no winners in {winners_path}")
    loader = DataLoader(
        ds,
        batch_size=micro,
        shuffle=True,
        num_workers=0,
        collate_fn=_make_winners_collate(tokenizer, phase_a_cfg),
        drop_last=False,
    )

    db_path = out_dir / "checkpoints.sqlite"
    db = CheckpointDB(db_path, keep_last=3)
    run_id = db.start_run(
        str(cfg.get("corpus", "b0")),
        "raft_b0",
        int(cfg.get("seed", 42)),
        int(cfg.get("ctx_max", 512)),
        int(cfg.get("n_prefix", 16)),
        {**cfg, "init_ckpt": str(init_ckpt), "n_winners": len(ds)},
    )
    flat_holder = _FlatState()

    step = 0
    epoch = 0
    t0 = time.time()
    loss_ema = None
    n_epochs = int(cfg.get("b0_epochs", 1))
    params = _trainable_params(model)

    print(
        f"[raft] distill n={len(ds)} epochs={n_epochs} "
        f"lr_adapter={phase_a_cfg['lr_adapter']} lr_lora={phase_a_cfg['lr_lora']} "
        f"micro={micro} accum={accum}",
        flush=True,
    )

    for ep in range(n_epochs):
        opt.zero_grad(set_to_none=True)
        micro_i = 0
        for batch in loader:
            if gate is not None:
                gate.step(model, opt)
            z = batch["z"].to(device=device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(z, input_ids, attention_mask, labels)
            loss = out.loss / float(accum)
            if not torch.isfinite(loss):
                print(f"[raft] non-finite loss at step {step}; skip", flush=True)
                opt.zero_grad(set_to_none=True)
                micro_i = 0
                continue
            loss.backward()
            micro_i += 1
            if micro_i < accum:
                continue
            _sanitize_grads(params)
            if not _grads_finite(params):
                print(f"[raft] bad grads at step {step}; skip", flush=True)
                opt.zero_grad(set_to_none=True)
                micro_i = 0
                continue
            torch.nn.utils.clip_grad_norm_(params, float(phase_a_cfg.get("grad_clip", 1.0)))
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro_i = 0
            step += 1
            lv = float(loss.detach().float().item()) * accum
            loss_ema = lv if loss_ema is None else 0.9 * loss_ema + 0.1 * lv
            if step % 10 == 0:
                print(
                    f"[raft] ep={ep} step={step} loss={lv:.4f} ema={loss_ema:.4f} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
            if step % 50 == 0:
                ckpt_path = out_dir / f"checkpoint_step{step:06d}.pt"
                _save_pt(
                    ckpt_path,
                    model,
                    opt,
                    ep,
                    step,
                    {"loss": loss_ema, "phase": "b0"},
                )
                flat_holder.set_state(_flat_trainable_state(model, step))
                db.record_epoch(run_id, ep, {"step": step, "loss": loss_ema, "phase": "b0"})
                db.save_checkpoint(run_id, ep, flat_holder, opt, device="cpu")
        # leftover micro-batches
        if micro_i > 0:
            _sanitize_grads(params)
            if _grads_finite(params):
                torch.nn.utils.clip_grad_norm_(params, float(phase_a_cfg.get("grad_clip", 1.0)))
                opt.step()
                step += 1
            opt.zero_grad(set_to_none=True)
        epoch = ep

    final = out_dir / "ckpt.pt"
    _save_pt(
        final,
        model,
        opt,
        epoch,
        step,
        {"loss": loss_ema, "phase": "b0", "n_winners": len(ds)},
    )
    # also alias
    latest = out_dir / "checkpoint_latest.pt"
    _save_pt(latest, model, opt, epoch, step, {"loss": loss_ema, "phase": "b0"})
    flat_holder.set_state(_flat_trainable_state(model, step))
    db.record_epoch(run_id, epoch, {"step": step, "loss": loss_ema, "phase": "b0_final"})
    db.save_checkpoint(run_id, epoch, flat_holder, opt, device="cpu")
    print(f"[raft] wrote {final} steps={step}", flush=True)
    return final


def run_b0(
    cfg: dict[str, Any],
    *,
    skip_rollouts: bool = False,
    skip_distill: bool = False,
    device_name: str = "auto",
) -> dict[str, Any]:
    out_dir = _ROOT / cfg["out_dir"] / "b0"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = out_dir / "rollouts.jsonl"
    winners = out_dir / "winners.jsonl"
    init_ckpt = _ROOT / cfg["phase_a_ckpt"]

    if not skip_rollouts:
        run_rollouts(
            cfg,
            n_prompts=int(cfg.get("b0_n_prompts", 4000)),
            out_jsonl=rollouts,
            ckpt_path=init_ckpt,
            resume=True,
            device_name=device_name,
        )
    else:
        if not rollouts.exists():
            raise FileNotFoundError(rollouts)

    stats = select_winners(
        rollouts,
        winners,
        min_cos=float(cfg.get("raft_min_cos", 0.50)),
        min_j=float(cfg.get("raft_min_j", 0.5)),
    )
    result: dict[str, Any] = {"winners_stats": stats}

    if not skip_distill:
        # Free rollout models: re-enter distill in a fresh process ideally;
        # here we rely on caller launching phases separately if VRAM tight.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ckpt = distill_winners(
            cfg,
            winners,
            init_ckpt=init_ckpt,
            out_dir=out_dir,
            device_name=device_name,
        )
        result["ckpt"] = str(ckpt)
    (out_dir / "b0_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage-B0 RAFT distill")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--skip-rollouts", action="store_true")
    p.add_argument("--skip-distill", action="store_true")
    p.add_argument("--select-only", action="store_true", help="only select winners from existing rollouts")
    p.add_argument("--rollouts", type=Path, default=None)
    p.add_argument("--winners", type=Path, default=None)
    p.add_argument("--distill-only", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)

    if args.select_only:
        rollouts = args.rollouts or (_ROOT / cfg["out_dir"] / "b0" / "rollouts.jsonl")
        winners = args.winners or (_ROOT / cfg["out_dir"] / "b0" / "winners.jsonl")
        select_winners(
            rollouts,
            winners,
            min_cos=float(cfg.get("raft_min_cos", 0.50)),
            min_j=float(cfg.get("raft_min_j", 0.5)),
        )
        return

    if args.distill_only:
        winners = args.winners or (_ROOT / cfg["out_dir"] / "b0" / "winners.jsonl")
        distill_winners(
            cfg,
            winners,
            init_ckpt=_ROOT / cfg["phase_a_ckpt"],
            out_dir=_ROOT / cfg["out_dir"] / "b0",
            device_name=args.device,
        )
        return

    run_b0(
        cfg,
        skip_rollouts=args.skip_rollouts,
        skip_distill=args.skip_distill,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
