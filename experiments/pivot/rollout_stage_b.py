"""Stage-B rollouts: prompt sampler + grouped generation + reward logging.

Writes full (z_cmd, ctx, k generations, rewards) JSONL for RAFT / DPO fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

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

from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.persistence import load_joint  # noqa: E402
from experiments.pivot.adapter import fiber_corrupt  # noqa: E402
from experiments.pivot.gate_g1 import generate_paragraph, load_stage_a_for_infer  # noqa: E402
from experiments.pivot.reward_stage_b import HybridReward, LogitJudge, _l2  # noqa: E402
from experiments.pivot.train_stage_a import (  # noqa: E402
    ParaPairDataset,
    _load_cfg,
    _seed_all,
)
from experiments.sentence_pred.forward_gen import load_mdn_predictor  # noqa: E402

DEFAULT_CFG = _ROOT / "configs" / "pivot_stage_b.json"


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_jepa(path: Path) -> JepaCodec:
    jepa_model, jepa_tok = load_joint(str(path), device="cpu")
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")
    return codec


from dataclasses import dataclass  # noqa: E402


@dataclass
class StageBPrompt:
    z_cmd: np.ndarray
    ctx: str
    source: str  # held_bank | fiber | mdn
    doc_id: int
    para_idx: int
    meta: dict[str, Any]


class PromptSampler:
    """50% held bank / 25% fiber / 25% MDN novel destinations (§5.1)."""

    def __init__(
        self,
        held_ds: ParaPairDataset,
        train_ds: ParaPairDataset | None,
        *,
        mix: dict[str, float],
        fiber_kappa: float = 700.0,
        mdn_path: Path | None = None,
        seed: int = 42,
    ) -> None:
        self.held = held_ds
        self.train = train_ds
        self.mix = mix
        self.fiber_kappa = float(fiber_kappa)
        self.rng = np.random.RandomState(seed)
        self._mdn = None
        self._bank_z: np.ndarray | None = None
        self._bank_ctx: list[str] = []
        if mdn_path is not None and Path(mdn_path).exists():
            self._mdn = load_mdn_predictor(mdn_path, device="cpu")
            # Nearest-bank context index from held+train items
            items = list(held_ds.items)
            if train_ds is not None:
                items = items + list(train_ds.items)
            zs = np.stack([it[2] for it in items], axis=0).astype(np.float32)
            self._bank_z = _l2(zs)
            self._bank_ctx = [it[0] for it in items]
            self._bank_meta = [(int(it[3]), int(it[4])) for it in items]

    def _pick_pool_item(self) -> tuple[str, str, np.ndarray, int, int]:
        # Prefer held; fall back to train tail
        use_held = self.rng.rand() < 0.7 or self.train is None or len(self.train) == 0
        ds = self.held if use_held else self.train
        i = int(self.rng.randint(0, len(ds)))
        prev, tgt, z, di, pi = ds.items[i]
        return prev, tgt, z.astype(np.float32), int(di), int(pi)

    def _nearest_ctx(self, z: np.ndarray) -> tuple[str, int, int]:
        assert self._bank_z is not None
        z = _l2(z)
        sims = self._bank_z @ z
        j = int(np.argmax(sims))
        di, pi = self._bank_meta[j]
        return self._bank_ctx[j], di, pi

    def sample_one(self) -> StageBPrompt:
        u = float(self.rng.rand())
        p_held = float(self.mix.get("held_bank", 0.5))
        p_fiber = float(self.mix.get("fiber", 0.25))
        if u < p_held:
            prev, _tgt, z, di, pi = self._pick_pool_item()
            return StageBPrompt(
                z_cmd=_l2(z),
                ctx=prev,
                source="held_bank",
                doc_id=di,
                para_idx=pi,
                meta={},
            )
        if u < p_held + p_fiber:
            prev, _tgt, z, di, pi = self._pick_pool_item()
            zt = torch.from_numpy(_l2(z)[None, :])
            z_tilde = fiber_corrupt(zt, kappa=self.fiber_kappa).numpy()[0]
            return StageBPrompt(
                z_cmd=_l2(z_tilde),
                ctx=prev,
                source="fiber",
                doc_id=di,
                para_idx=pi,
                meta={"kappa": self.fiber_kappa},
            )
        # MDN novel
        if self._mdn is None or self._bank_z is None:
            prev, _tgt, z, di, pi = self._pick_pool_item()
            zt = torch.from_numpy(_l2(z)[None, :])
            z_tilde = fiber_corrupt(zt, kappa=200.0).numpy()[0]
            return StageBPrompt(
                z_cmd=_l2(z_tilde),
                ctx=prev,
                source="mdn_fallback_fiber",
                doc_id=di,
                para_idx=pi,
                meta={},
            )
        # Context = random short bank trajectory (1–4 prev zs)
        n_ctx = int(self.rng.randint(1, 5))
        idxs = self.rng.choice(len(self._bank_z), size=n_ctx, replace=False)
        ctx_z = torch.from_numpy(self._bank_z[idxs][None, :, :]).float()  # (1,T,D)
        pad = torch.zeros((1, n_ctx), dtype=torch.bool)
        samp = self._mdn.sample(ctx_z, pad_mask=pad, n_samples=1)  # (1,1,D)
        z_new = _l2(samp[0, 0].cpu().numpy())
        ctx, di, pi = self._nearest_ctx(z_new)
        return StageBPrompt(
            z_cmd=z_new,
            ctx=ctx,
            source="mdn",
            doc_id=di,
            para_idx=pi,
            meta={"mdn_ctx_n": n_ctx},
        )

    def sample_n(self, n: int) -> list[StageBPrompt]:
        return [self.sample_one() for _ in range(n)]


@torch.no_grad()
def generate_group(
    model,
    tokenizer,
    z: np.ndarray,
    prev: str,
    *,
    k: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """Generate k samples for one (z, ctx). Batched when possible."""
    texts: list[str] = []
    # Batch as one forward with num_return_sequences when supported via inputs_embeds.
    z_t = torch.from_numpy(_l2(z)[None, :]).to(device=device, dtype=torch.float32)
    enc = tokenizer(
        prev,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    embed = model.backbone.get_input_embeddings()
    tok_emb = embed(input_ids)
    prefix = model.adapter(z_t).to(dtype=tok_emb.dtype)
    prefix = torch.clamp(prefix, -10.0, 10.0)
    inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    # Expand batch for k sequences
    inputs_embeds = inputs_embeds.expand(k, -1, -1).contiguous()
    attn = attn.expand(k, -1).contiguous()

    gen_kwargs: dict[str, Any] = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(min_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    try:
        gen = model.backbone.generate(**gen_kwargs)
    except Exception as e:  # noqa: BLE001
        # Fallback: sequential
        print(f"[rollout] batched generate failed ({e}); falling back sequential", flush=True)
        for _ in range(k):
            texts.append(
                generate_paragraph(
                    model,
                    tokenizer,
                    z,
                    prev,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    device=device,
                )
            )
        return texts

    # With inputs_embeds, HF usually returns only new tokens
    if gen.dim() == 2:
        for i in range(gen.shape[0]):
            texts.append(tokenizer.decode(gen[i], skip_special_tokens=True).strip())
    return texts


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def run_rollouts(
    cfg: dict[str, Any],
    *,
    n_prompts: int | None = None,
    out_jsonl: Path | None = None,
    ckpt_path: Path | None = None,
    resume: bool = True,
    device_name: str = "auto",
) -> Path:
    _seed_all(int(cfg.get("seed", 42)))
    device = _device(device_name)
    out_dir = _ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_jsonl or (out_dir / "b0" / "rollouts.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_prompts = int(n_prompts if n_prompts is not None else cfg.get("b0_n_prompts", 4000))
    k = int(cfg.get("b0_k", cfg.get("group_k", 8)))
    done = count_jsonl(out_jsonl) if resume else 0
    if done >= n_prompts:
        print(f"[rollout] already complete n={done} >= {n_prompts}", flush=True)
        return out_jsonl

    # Load Phase-A cfg for backbone dims / paths
    phase_a_cfg = _load_cfg(_ROOT / cfg.get("phase_a_cfg", "configs/pivot_step1.json"))
    # Merge Stage-B backbone knobs
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
    ):
        if key in cfg:
            phase_a_cfg[key] = cfg[key]

    ckpt = Path(ckpt_path or (_ROOT / cfg["phase_a_ckpt"]))
    model, tokenizer, meta = load_stage_a_for_infer(phase_a_cfg, ckpt, device)
    print(f"[rollout] loaded policy meta={meta}", flush=True)

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

    held = ParaPairDataset(
        _ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="held"
    )
    train_ds = ParaPairDataset(
        _ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="train"
    )
    sampler = PromptSampler(
        held,
        train_ds,
        mix=cfg.get("prompt_mix", {"held_bank": 0.5, "fiber": 0.25, "mdn": 0.25}),
        fiber_kappa=float(cfg.get("fiber_kappa", 700.0)),
        mdn_path=_ROOT / cfg["mdn_ckpt"] if cfg.get("mdn_ckpt") else None,
        seed=int(cfg.get("seed", 42)) + done,
    )

    t0 = time.time()
    for i in range(done, n_prompts):
        prompt = sampler.sample_one()
        texts = generate_group(
            model,
            tokenizer,
            prompt.z_cmd,
            prompt.ctx,
            k=k,
            max_new_tokens=int(cfg.get("max_new_tokens", 200)),
            min_new_tokens=int(cfg.get("min_new_tokens", 40)),
            temperature=float(cfg.get("temperature", 0.9)),
            top_p=float(cfg.get("top_p", 0.95)),
            device=device,
        )
        scored = rewarder.score_group(texts, prompt.z_cmd)
        row = {
            "prompt_idx": i,
            "source": prompt.source,
            "doc_id": prompt.doc_id,
            "para_idx": prompt.para_idx,
            "ctx": prompt.ctx,
            "z_cmd": prompt.z_cmd.astype(np.float32).tolist(),
            "meta": prompt.meta,
            "k": k,
            "generations": [s.as_dict() for s in scored],
            "rewards": [s.reward for s in scored],
            "best_idx": int(np.argmax([s.reward for s in scored])),
            "ckpt": str(ckpt),
            "ts": time.time(),
        }
        append_jsonl(out_jsonl, row)
        if (i + 1) % 10 == 0 or i == done:
            elapsed = time.time() - t0
            rate = (i + 1 - done) / max(1e-6, elapsed)
            rmean = float(np.mean(row["rewards"]))
            print(
                f"[rollout] {i+1}/{n_prompts} source={prompt.source} "
                f"rmean={rmean:.3f} rate={rate*3600:.1f} prompts/h "
                f"vram={torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0:.2f}GB",
                flush=True,
            )
            t0 = time.time()
            # reset timer baseline for rate on next window — keep cumulative via done offset
            done_at = i + 1
            # rewrite t0 relative: keep simple per-window
            _ = done_at
    print(f"[rollout] done → {out_jsonl}", flush=True)
    return out_jsonl


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage-B grouped rollouts")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--n-prompts", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)
    run_rollouts(
        cfg,
        n_prompts=args.n_prompts,
        out_jsonl=args.out,
        ckpt_path=args.ckpt,
        resume=not args.no_resume,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
