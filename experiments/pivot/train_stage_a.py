"""Phase A Stage-A trainer: QLoRA + z→soft-prefix RC objective.

CE of target paragraph given [soft-prefix(z̃); previous paragraph],
z̃ ~ vMF(E(x_target), κ≈700), 10% prefix-drop. GpuGate + SQLite resume.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.config import GpuGateConfig  # noqa: E402
from conceptpoc.gpu_gate import GpuGate  # noqa: E402
from conceptpoc.persistence import load_joint  # noqa: E402
from experiments.pivot.adapter import (  # noqa: E402
    SoftPrefixAdapter,
    build_inputs_embeds,
    collate_prefix_batch,
    fiber_corrupt,
    human_param_count,
    lora_target_modules_for,
)
from experiments.pivot.corpus_v2 import load_corpus_v2  # noqa: E402

DEFAULT_CFG = _ROOT / "configs" / "pivot_step1.json"


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cfg(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ParaPairDataset(Dataset):
    """Consecutive paragraph pairs (prev, target, z_target) from zbank + corpus."""

    def __init__(
        self,
        corpus_path: Path,
        zbank_path: Path,
        split: str = "train",
        min_target_chars: int = 40,
    ) -> None:
        docs = load_corpus_v2(corpus_path)
        bank = np.load(zbank_path, allow_pickle=False)
        z = np.asarray(bank["paragraph_z"], dtype=np.float32)
        # Re-normalize float16→float32
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        doc_id = np.asarray(bank["doc_id"], dtype=np.int32)
        para_idx = np.asarray(bank["para_idx"], dtype=np.int32)

        split_key = {
            "train": "train_doc_ids",
            "val": "val_doc_ids",
            "held": "held_doc_ids",
        }[split]
        allowed = set(int(x) for x in bank[split_key].tolist())

        # Map (doc, para) → flat index
        index: dict[tuple[int, int], int] = {}
        for i in range(len(doc_id)):
            index[(int(doc_id[i]), int(para_idx[i]))] = i

        items: list[tuple[str, str, np.ndarray, int, int]] = []
        for di in sorted(allowed):
            if di < 0 or di >= len(docs):
                continue
            doc = docs[di]
            for pi in range(1, len(doc)):
                prev = " ".join(doc[pi - 1]).strip()
                tgt = " ".join(doc[pi]).strip()
                if len(tgt) < min_target_chars:
                    continue
                key = (di, pi)
                if key not in index:
                    continue
                zi = z[index[key]]
                items.append((prev, tgt, zi.astype(np.float32), di, pi))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        prev, tgt, z, di, pi = self.items[i]
        return {
            "prev": prev,
            "target": tgt,
            "z": z,
            "doc_id": di,
            "para_idx": pi,
        }


def _make_collate(tokenizer, cfg: dict[str, Any]):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        prev = [b["prev"] for b in batch]
        tgt = [b["target"] for b in batch]
        z = torch.from_numpy(np.stack([b["z"] for b in batch], axis=0))
        packed = collate_prefix_batch(
            tokenizer,
            prev,
            tgt,
            max_prev=int(cfg["max_prev_tokens"]),
            max_target=int(cfg["max_target_tokens"]),
            ctx_max=int(cfg["ctx_max"]),
            n_prefix=int(cfg["n_prefix"]),
        )
        packed["z"] = z
        packed["prev_text"] = prev
        packed["target_text"] = tgt
        packed["doc_id"] = torch.tensor([b["doc_id"] for b in batch], dtype=torch.long)
        packed["para_idx"] = torch.tensor([b["para_idx"] for b in batch], dtype=torch.long)
        return packed

    return _collate


def _resolve_backbone(cfg: dict[str, Any], device: torch.device) -> tuple[Any, Any, str]:
    """Load 4-bit backbone; fall back to OLMo if Qwen fails / OOM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    primary = cfg["backbone"]
    fallback = cfg.get("backbone_fallback", "allenai/OLMo-2-0425-1B")
    compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16"))

    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
    )
    if device.type == "cuda":
        map_dev = device.index if device.index is not None else 0
    else:
        map_dev = "cpu"

    last_err: Exception | None = None
    for name in (primary, fallback):
        try:
            print(f"[stageA] loading backbone {name} (4-bit NF4)...", flush=True)
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb,
                device_map={"": map_dev},
                trust_remote_code=True,
                dtype=compute_dtype,
            )
            # Confirm d_model
            d_model = int(model.config.hidden_size)
            cfg["d_model"] = d_model
            print(
                f"[stageA] backbone ok: {name} hidden_size={d_model} "
                f"vram={_vram_gb():.2f}GB",
                flush=True,
            )
            return model, tok, name
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[stageA] failed to load {name}: {e}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"could not load any backbone; last error: {last_err}")


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _attach_lora(model, cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False
    )
    targets = lora_target_modules_for(getattr(model.config, "model_type", "") or "")
    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    # Soft-prefix path: make sure grads flow into inputs_embeds.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # Gradient checkpointing + custom inputs_embeds is a known NaN source.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.print_trainable_parameters()
    return model


def _maybe_gate(dev: torch.device, enabled: bool) -> GpuGate | None:
    if not enabled or dev.type != "cuda":
        return None
    # release_memory=False: bitsandbytes 4-bit modules cannot safely .to(cpu)/.to(cuda)
    gate = GpuGate(
        GpuGateConfig(enabled=True, release_memory=False),
        dev,
        progress=True,
    )
    return gate if gate.active else None


class StageAModel(nn.Module):
    """Backbone (PEFT) + soft-prefix adapter."""

    def __init__(self, backbone, adapter: SoftPrefixAdapter) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(
        self,
        z: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        drop_mask: torch.Tensor | None = None,
    ):
        embeds = build_inputs_embeds(
            self.backbone, self.adapter, z, input_ids, drop_mask=drop_mask
        )
        return self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


def _param_groups(model: StageAModel, cfg: dict[str, Any]):
    adapter_params = list(model.adapter.parameters())
    adapter_ids = {id(p) for p in adapter_params}
    lora_params = [
        p for p in model.backbone.parameters() if p.requires_grad and id(p) not in adapter_ids
    ]
    return [
        {"params": adapter_params, "lr": float(cfg["lr_adapter"])},
        {"params": lora_params, "lr": float(cfg["lr_lora"])},
    ]


def _save_pt(
    path: Path,
    model: StageAModel,
    opt: torch.optim.Optimizer,
    epoch: int,
    step: int,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "adapter": model.adapter.state_dict(),
        "backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items() if "lora_" in k or "modules_to_save" in k},
        "optimizer": opt.state_dict(),
        "meta": meta,
    }
    # Prefer peft's save for LoRA weights alongside our adapter blob.
    lora_dir = path.parent / "lora_adapter"
    try:
        model.backbone.save_pretrained(str(lora_dir))
        payload["lora_dir"] = str(lora_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[stageA] peft save_pretrained warn: {e}", flush=True)
    torch.save(payload, path)
    torch.save(model.adapter.state_dict(), path.parent / "adapter.pt")


def _flat_trainable_state(model: StageAModel, step: int) -> dict[str, torch.Tensor]:
    """Flatten adapter + LoRA (+ step) into a tensor-only state_dict for CheckpointDB."""
    out: dict[str, torch.Tensor] = {}
    for k, v in model.adapter.state_dict().items():
        out[f"adapter.{k}"] = v.detach().cpu()
    for k, v in model.backbone.state_dict().items():
        if "lora_" in k:
            out[f"lora.{k}"] = v.detach().cpu()
    out["__step"] = torch.tensor([int(step)], dtype=torch.long)
    return out


def _load_flat_state(model: StageAModel, flat: dict[str, Any]) -> int:
    adapter_sd = {
        k[len("adapter.") :]: v for k, v in flat.items() if k.startswith("adapter.")
    }
    lora_sd = {k[len("lora.") :]: v for k, v in flat.items() if k.startswith("lora.")}
    if adapter_sd:
        model.adapter.load_state_dict(adapter_sd)
    if lora_sd:
        missing, _unexpected = model.backbone.load_state_dict(lora_sd, strict=False)
        if missing:
            print(f"[stageA] resume missing lora keys: {len(missing)}", flush=True)
    step_t = flat.get("__step")
    if step_t is None:
        return 0
    if isinstance(step_t, torch.Tensor):
        return int(step_t.reshape(-1)[0].item())
    return int(step_t)


class _FlatState(nn.Module):
    """Holds a flat tensor state_dict so CheckpointDB can serialize it."""

    def __init__(self) -> None:
        super().__init__()
        self._sd: dict[str, torch.Tensor] = {}

    def set_state(self, sd: dict[str, torch.Tensor]) -> None:
        self._sd = sd

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return dict(self._sd)

    def load_state_dict(self, state_dict, *args, **kwargs):  # type: ignore[override]
        self._sd = dict(state_dict)
        return torch.nn.modules.module._IncompatibleKeys([], [])


def _load_resume(
    db: CheckpointDB,
    run_id: int,
    model: StageAModel,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    loaded = db.load_latest_checkpoint(run_id, device="cpu")
    if loaded is None:
        # Also try peft adapter dir + adapter.pt sidecar
        return 0, 0
    model_state, opt_state, epoch = loaded
    step = _load_flat_state(model, model_state)
    if opt_state:
        try:
            opt.load_state_dict(opt_state)
        except Exception as e:  # noqa: BLE001
            print(f"[stageA] optimizer resume skipped: {e}", flush=True)
    print(f"[stageA] resumed from epoch {epoch} step={step}", flush=True)
    return int(epoch) + 1, step

@torch.no_grad()
def eval_sample_log(
    model: StageAModel,
    tokenizer,
    codec: JepaCodec | None,
    held_ds: ParaPairDataset,
    out_path: Path,
    n: int = 64,
    max_new_tokens: int = 128,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Generate n paragraphs from held-out bank z; optionally re-encode."""
    model.eval()
    device = device or next(model.adapter.parameters()).device
    n = min(n, len(held_ds))
    if n == 0:
        return []
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(held_ds), size=n, replace=False).tolist()
    rows: list[dict[str, Any]] = []
    for i in idxs:
        item = held_ds[i]
        z = torch.from_numpy(item["z"][None, :]).to(device=device, dtype=torch.float32)
        prev = item["prev"]
        # tokenize prev only; generation continues after
        enc = tokenizer(
            prev,
            add_special_tokens=False,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        prefix = model.adapter(z)
        embed = model.backbone.get_input_embeddings()
        inputs_embeds = torch.cat([prefix, embed(input_ids)], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        gen = model.backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # generate with inputs_embeds returns only new tokens for some versions;
        # handle both.
        if gen.shape[1] > input_ids.shape[1]:
            new_ids = gen[:, input_ids.shape[1] :]
        else:
            new_ids = gen
        text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        y = None
        if codec is not None and text:
            y = codec.encode([text])[0].tolist()
        rows.append(
            {
                "z_cmd": item["z"].tolist(),
                "ctx": prev,
                "y_text": text,
                "y": y,
                "doc_id": int(item["doc_id"]),
                "para_idx": int(item["para_idx"]),
                "target_ref": item["target"],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[stageA] wrote eval sample log n={len(rows)} → {out_path}", flush=True)
    model.train()
    return rows


def train(cfg: dict[str, Any], *, resume: bool = True, max_steps: int | None = None) -> dict[str, Any]:
    _seed_all(int(cfg.get("seed", 42)))
    out_dir = _ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    device = _device("auto")

    # Ensure zbank exists
    zbank = _ROOT / cfg["zbank"]
    if not zbank.exists():
        from experiments.pivot.encode_bank import build_bank

        print("[stageA] zbank missing; building...", flush=True)
        build_bank(out_path=zbank)

    print(f"[stageA] loading train pairs from {zbank}...", flush=True)
    train_ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="train")
    print(f"[stageA] loading held pairs...", flush=True)
    held_ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="held")
    n_paras = len(train_ds)
    print(
        f"[stageA] train pairs={n_paras} held={len(held_ds)} "
        f"(need ≥{cfg.get('min_paragraphs', 150000)})",
        flush=True,
    )
    if n_paras < int(cfg.get("min_paragraphs", 150000)):
        print(
            f"[stageA] WARNING: train pairs {n_paras} < min_paragraphs; "
            "continuing anyway (anti-starvation still applies via ≥2 epochs).",
            flush=True,
        )

    backbone, tokenizer, backbone_name = _resolve_backbone(cfg, device)
    backbone = _attach_lora(backbone, cfg)
    adapter = SoftPrefixAdapter(
        z_dim=int(cfg["z_dim"]),
        d_model=int(cfg["d_model"]),
        n_prefix=int(cfg["n_prefix"]),
        hidden=int(cfg.get("adapter_hidden", 512)),
    ).to(device)
    print(
        f"[stageA] adapter params={human_param_count(adapter.n_trainable())}",
        flush=True,
    )
    model = StageAModel(backbone, adapter)
    model.train()

    opt = torch.optim.AdamW(
        _param_groups(model, cfg),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    db_path = _ROOT / cfg.get("db_path", str(out_dir / "checkpoints.sqlite"))
    db = CheckpointDB(db_path, keep_last=3)
    run_id = db.latest_run() if resume else None
    start_epoch = 0
    global_step = 0
    flat_holder = _FlatState()
    if run_id is None:
        run_id = db.start_run(
            str(cfg["corpus"]),
            backbone_name,
            int(cfg.get("seed", 42)),
            int(cfg["ctx_max"]),
            int(cfg["n_prefix"]),
            {**cfg, "backbone_resolved": backbone_name},
        )
        print(f"[stageA] new run_id={run_id}", flush=True)
    else:
        start_epoch, global_step = _load_resume(db, run_id, model, opt, device)

    gate = _maybe_gate(device, bool(cfg.get("gpu_gate", True)))
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("micro_batch", 1)),
        shuffle=True,
        num_workers=0,
        collate_fn=_make_collate(tokenizer, cfg),
        drop_last=True,
    )
    grad_accum = int(cfg.get("grad_accum", 64))
    epochs = int(cfg.get("epochs", 2))
    kappa = float(cfg.get("kappa_aug", 700.0))
    drop_p = float(cfg.get("prefix_drop_p", 0.10))
    log_every = int(cfg.get("log_every", 20))
    eval_every = int(cfg.get("eval_every_steps", 500))
    clip = float(cfg.get("grad_clip", 1.0))

    # Cheap no-condition CE baseline (prefix always dropped) before training.
    baseline_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 32:
                break
            z = batch["z"].to(device)
            drop = torch.ones(z.size(0), dtype=torch.bool, device=device)
            out = model(
                z,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
                drop_mask=drop,
            )
            baseline_losses.append(float(out.loss.item()))
    baseline_ce = float(np.mean(baseline_losses)) if baseline_losses else float("nan")
    model.train()
    (out_dir / "baseline_uncond.json").write_text(
        json.dumps({"n": len(baseline_losses), "ce": baseline_ce, "backbone": backbone_name}, indent=2)
        + "\n"
    )

    # Optional frozen JEPA for eval re-encode (CPU to save VRAM)
    codec: JepaCodec | None = None
    try:
        jepa_path = _ROOT / cfg.get("jepa", "artifacts/jepa_repair_1m/jepa/joint.pt")
        jm, jt = load_joint(str(jepa_path), device="cpu")
        codec = JepaCodec(jm, jt, max_tokens=128, device="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"[stageA] JEPA codec for eval skipped: {e}", flush=True)

    history: list[dict[str, Any]] = []
    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
        print("[stageA] interrupt flagged; will checkpoint after step", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    t_run = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        def log(msg: str) -> None:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        log(
            f"start run={run_id} backbone={backbone_name} "
            f"train_pairs={n_paras} epochs={epochs} "
            f"eff_batch={grad_accum * int(cfg.get('micro_batch', 1))} "
            f"resume_epoch={start_epoch} step={global_step} vram={_vram_gb():.2f}GB "
            f"baseline_uncond_ce={baseline_ce:.4f}"
        )

        for epoch in range(start_epoch, epochs):
            model.train()
            running = 0.0
            n_tok = 0
            opt.zero_grad(set_to_none=True)
            micro = 0
            for bi, batch in enumerate(loader):
                if stop["flag"] or (max_steps is not None and global_step >= max_steps):
                    break
                if gate is not None:
                    gate.step(model, opt)

                z = batch["z"].to(device)
                z_tilde = fiber_corrupt(z, kappa=kappa)
                if not torch.isfinite(z_tilde).all():
                    z_tilde = F.normalize(z.float(), dim=-1)
                drop = torch.rand(z.size(0), device=device) < drop_p
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                if int((labels != -100).sum()) == 0:
                    continue

                out = model(z_tilde, input_ids, attn, labels, drop_mask=drop)
                raw_loss = out.loss
                if raw_loss is None or not torch.isfinite(raw_loss):
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    continue
                loss = raw_loss / grad_accum
                loss.backward()
                micro += 1

                running += float(raw_loss.detach().item())
                n_tok += 1

                if micro >= grad_accum:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], clip
                    )
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    global_step += 1

                    if global_step % log_every == 0 or global_step <= 5:
                        avg = running / max(n_tok, 1)
                        log(
                            f"epoch={epoch} step={global_step} loss={avg:.4f} "
                            f"vram={_vram_gb():.2f}GB"
                        )
                        history.append(
                            {"epoch": epoch, "step": global_step, "loss": avg}
                        )
                        running = 0.0
                        n_tok = 0

                    if global_step % eval_every == 0:
                        eval_path = out_dir / f"eval_samples_step{global_step:06d}.jsonl"
                        try:
                            eval_sample_log(
                                model,
                                tokenizer,
                                codec,
                                held_ds,
                                eval_path,
                                n=int(cfg.get("eval_n", 64)),
                                device=device,
                            )
                        except Exception as e:  # noqa: BLE001
                            log(f"eval_sample_log failed: {e}")

            # end epoch: checkpoint
            meta = {
                "backbone": backbone_name,
                "epoch": epoch,
                "step": global_step,
                "loss_last": history[-1]["loss"] if history else None,
            }
            ckpt_path = out_dir / f"checkpoint_epoch{epoch:02d}.pt"
            flat_holder.set_state(_flat_trainable_state(model, global_step))
            db.record_epoch(run_id, epoch, meta)
            db.save_checkpoint(run_id, epoch, flat_holder, opt, device="cpu")
            _save_pt(ckpt_path, model, opt, epoch, global_step, meta)
            log(f"checkpoint epoch={epoch} → {ckpt_path}")

            # Epoch-end eval sample log
            eval_path = out_dir / f"eval_samples_epoch{epoch:02d}.jsonl"
            try:
                eval_sample_log(
                    model,
                    tokenizer,
                    codec,
                    held_ds,
                    eval_path,
                    n=int(cfg.get("eval_n", 64)),
                    device=device,
                )
            except Exception as e:  # noqa: BLE001
                log(f"epoch eval_sample_log failed: {e}")

            if stop["flag"] or (max_steps is not None and global_step >= max_steps):
                break

        # Final pointer
        latest = out_dir / "checkpoint_latest.pt"
        if (out_dir / f"checkpoint_epoch{max(start_epoch, epochs - 1):02d}.pt").exists():
            import shutil

            src = out_dir / f"checkpoint_epoch{epochs - 1:02d}.pt"
            # Prefer the highest existing epoch file
            existing = sorted(out_dir.glob("checkpoint_epoch*.pt"))
            if existing:
                shutil.copy2(existing[-1], latest)
        log(f"done elapsed_h={(time.time() - t_run) / 3600:.2f} steps={global_step}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "run_id": run_id,
        "backbone": backbone_name,
        "steps": global_step,
        "history": history,
        "out_dir": str(out_dir),
        "vram_gb": _vram_gb(),
    }


def baseline_unconditioned_loss(
    cfg: dict[str, Any], n_batches: int = 32
) -> float:
    """Cheap no-condition CE baseline (prefix always dropped) for first-epoch compare."""
    device = _device("auto")
    zbank = _ROOT / cfg["zbank"]
    ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="train")
    backbone, tokenizer, name = _resolve_backbone(cfg, device)
    # No LoRA — frozen 4-bit CE with zero prefix via a throwaway adapter
    adapter = SoftPrefixAdapter(
        z_dim=int(cfg["z_dim"]),
        d_model=int(getattr(backbone.config, "hidden_size", cfg["d_model"])),
        n_prefix=int(cfg["n_prefix"]),
        hidden=int(cfg.get("adapter_hidden", 512)),
    ).to(device)
    cfg = dict(cfg)
    cfg["d_model"] = int(backbone.config.hidden_size)
    model = StageAModel(backbone, adapter).eval()
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        collate_fn=_make_collate(tokenizer, cfg),
    )
    losses: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            z = batch["z"].to(device)
            drop = torch.ones(z.size(0), dtype=torch.bool, device=device)
            out = model(
                z,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
                drop_mask=drop,
            )
            losses.append(float(out.loss.item()))
    mean = float(np.mean(losses)) if losses else float("nan")
    print(f"[stageA] no-condition baseline CE ({n_batches} batches, {name}): {mean:.4f}", flush=True)
    del model, backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pivot Stage-A QLoRA + soft-prefix trainer")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--no-gpu-gate", action="store_true")
    args = p.parse_args(argv)
    print(f"[stageA] boot config={args.config} resume={not args.no_resume}", flush=True)
    cfg = _load_cfg(args.config)
    if args.no_gpu_gate:
        cfg["gpu_gate"] = False
    if args.baseline_only:
        baseline_unconditioned_loss(cfg)
        return
    train(cfg, resume=not args.no_resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
