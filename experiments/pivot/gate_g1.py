"""G1 kill-gate for Phase-A concept-conditioned generation (inference only).

Pre-registered thresholds (novelty plan §5):
  - median MiniLM writer-fidelity >= 0.40
  - median JEPA fed_vs_concept >= 0.55
  - 2-AFC z-identification >= 0.85 (pairs at cross-doc cos ~0.57)
  - fluency guard >= 0.45 win-rate vs unconditioned

KILL (first / pre-corrective read, as tasked):
  MiniLM median < 0.30 OR 2-AFC < 0.70  -> kill concept-conditioned generation.

Does not modify src/ or tests/. Does not retrain.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

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
from conceptpoc.persistence import load_joint  # noqa: E402
from experiments.pivot.adapter import SoftPrefixAdapter  # noqa: E402
from experiments.pivot.train_stage_a import (  # noqa: E402
    ParaPairDataset,
    StageAModel,
    _attach_lora,
    _load_cfg,
    _resolve_backbone,
    _seed_all,
)

DEFAULT_CFG = _ROOT / "configs" / "pivot_step1.json"
DEFAULT_CKPT = _ROOT / "artifacts" / "pivot_stageA" / "checkpoint_epoch01.pt"
DEFAULT_OUT = _ROOT / "artifacts" / "pivot_stageA" / "gate_g1_epoch01.json"

# Pre-registered thresholds
THR_MINILM = 0.40
THR_FED = 0.55
THR_2AFC = 0.85
THR_FLUENCY = 0.45
KILL_MINILM = 0.30
KILL_2AFC = 0.70

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / n).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2(a)
    b = _l2(b)
    return float(np.dot(a, b))


def _split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts if parts else [t]


def load_stage_a_for_infer(
    cfg: dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
) -> tuple[StageAModel, Any, dict[str, Any]]:
    """Load QLoRA backbone + SoftPrefixAdapter from a Stage-A checkpoint."""
    backbone, tokenizer, name = _resolve_backbone(cfg, device)
    cfg["backbone_resolved"] = name
    backbone = _attach_lora(backbone, cfg)
    adapter = SoftPrefixAdapter(
        z_dim=int(cfg["z_dim"]),
        d_model=int(cfg.get("d_model", backbone.config.hidden_size)),
        n_prefix=int(cfg["n_prefix"]),
        hidden=int(cfg.get("adapter_hidden", 512)),
    ).to(device)
    model = StageAModel(backbone, adapter)

    print(f"[G1] loading checkpoint {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "adapter" not in ckpt:
        raise KeyError(f"checkpoint missing 'adapter' keys: {list(ckpt.keys())}")
    model.adapter.load_state_dict(ckpt["adapter"])

    # Prefer in-checkpoint LoRA tensors; fall back to peft dir.
    loaded_lora = False
    if isinstance(ckpt.get("backbone"), dict) and ckpt["backbone"]:
        missing, unexpected = model.backbone.load_state_dict(
            ckpt["backbone"], strict=False
        )
        n_lora = sum(1 for k in ckpt["backbone"] if "lora_" in k)
        print(
            f"[G1] loaded {n_lora} LoRA tensors from ckpt "
            f"(missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
        )
        loaded_lora = n_lora > 0
    if not loaded_lora:
        lora_dir = ckpt.get("lora_dir") or str(ckpt_path.parent / "lora_adapter")
        if Path(lora_dir).exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            st_path = Path(lora_dir) / "adapter_model.safetensors"
            bin_path = Path(lora_dir) / "adapter_model.bin"
            if st_path.exists():
                lora_sd = load_file(str(st_path))
            elif bin_path.exists():
                lora_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
            else:
                lora_sd = None
            if lora_sd is not None:
                print(f"[G1] loading LoRA from {lora_dir}", flush=True)
                set_peft_model_state_dict(model.backbone, lora_sd)
                loaded_lora = True
    if not loaded_lora:
        print("[G1] WARNING: no LoRA weights loaded — adapter-only inference", flush=True)

    model.eval()
    meta = {
        "epoch": int(ckpt.get("epoch", -1)),
        "step": int(ckpt.get("step", -1)),
        "meta": ckpt.get("meta", {}),
        "ckpt": str(ckpt_path),
        "backbone": name,
    }
    return model, tokenizer, meta


@torch.no_grad()
def generate_paragraph(
    model: StageAModel,
    tokenizer,
    z: np.ndarray | None,
    prev: str,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    device: torch.device,
) -> str:
    """z -> soft-prefix -> Qwen generate (matches train_stage_a.eval_sample_log).

    z=None ⇒ unconditioned (zero soft-prefix / CFG drop).
    """
    if z is None:
        z_t = torch.zeros((1, model.adapter.z_dim), device=device, dtype=torch.float32)
        drop = True
    else:
        z_t = torch.from_numpy(_l2(z)[None, :]).to(device=device, dtype=torch.float32)
        drop = False

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
    if drop:
        prefix = torch.zeros_like(prefix)
    inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    gen_kwargs: dict[str, Any] = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    gen = model.backbone.generate(**gen_kwargs)
    # With inputs_embeds, HF usually returns only new token ids; some versions
    # return a longer sequence — mirror eval_sample_log handling.
    if gen.shape[1] > input_ids.shape[1]:
        new_ids = gen[:, input_ids.shape[1] :]
    else:
        new_ids = gen
    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def length_norm_nll(
    model: StageAModel,
    tokenizer,
    prev: str,
    text: str,
    device: torch.device,
) -> float:
    """Length-normalized NLL of ``text`` under the backbone given ``prev`` (no soft-prefix).

    Fluency proxy: scores plain token continuation quality, independent of z.
    """
    if not text.strip():
        return float("inf")
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
        return float("inf")
    input_ids = torch.cat([prev_ids, tgt_ids], dim=1)
    # labels: ignore prev
    labels = input_ids.clone()
    labels[:, : prev_ids.shape[1]] = -100
    out = model.backbone(input_ids=input_ids, labels=labels)
    # out.loss is mean over non-ignored tokens
    return float(out.loss.detach().float().item())


def encode_paragraph_pooled(codec: JepaCodec, text: str) -> np.ndarray:
    """Sentence-split → JEPA encode → mean-pool → L2 (matches zbank protocol)."""
    sents = _split_sentences(text)
    if not sents:
        return np.zeros(codec.dim, dtype=np.float32)
    # Cap sentences to keep CPU encode bounded
    if len(sents) > 32:
        sents = sents[:32]
    vecs = codec.encode(sents)
    return _l2(np.asarray(vecs, dtype=np.float32).mean(axis=0))


def encode_minilm_pair_cos(
    gen_texts: list[str],
    src_texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    # Encode each paragraph as one unit (writer-fidelity protocol).
    all_t = list(gen_texts) + list(src_texts)
    emb = model.encode(
        all_t,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    n = len(gen_texts)
    g, s = emb[:n], emb[n:]
    return np.sum(g * s, axis=-1).astype(np.float32)


def sample_held_indices(
    held_ds: ParaPairDataset,
    n: int,
    seed: int = 0,
) -> list[int]:
    rng = np.random.RandomState(seed)
    n = min(n, len(held_ds))
    return rng.choice(len(held_ds), size=n, replace=False).tolist()


def sample_2afc_pairs(
    zs: np.ndarray,
    rng: np.random.RandomState,
    n_pairs: int,
    target_cos: float = 0.57,
    tol: float = 0.05,
    max_tries: int = 200_000,
) -> list[tuple[int, int]]:
    """Sample index pairs with cos(z_i, z_j) ≈ target (cross-doc band)."""
    z = _l2(zs)
    n = z.shape[0]
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    lo, hi = target_cos - tol, target_cos + tol
    tries = 0
    while len(pairs) < n_pairs and tries < max_tries:
        tries += 1
        i = int(rng.randint(0, n))
        j = int(rng.randint(0, n))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        c = float(np.dot(z[i], z[j]))
        if lo <= c <= hi:
            seen.add((a, b))
            pairs.append((i, j))
    if len(pairs) < n_pairs:
        # Widen tolerance once if bank is sparse
        print(
            f"[G1] 2-AFC: only {len(pairs)}/{n_pairs} pairs in [{lo:.2f},{hi:.2f}]; "
            f"widening tol to 0.10",
            flush=True,
        )
        lo, hi = target_cos - 0.10, target_cos + 0.10
        while len(pairs) < n_pairs and tries < max_tries * 2:
            tries += 1
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            c = float(np.dot(z[i], z[j]))
            if lo <= c <= hi:
                seen.add((a, b))
                pairs.append((i, j))
    return pairs


def identify_held_split(zbank_path: Path) -> dict[str, Any]:
    """Document how held z were identified."""
    bank = np.load(zbank_path, allow_pickle=False)
    held_ids = np.asarray(bank["held_doc_ids"], dtype=np.int32)
    doc_id = np.asarray(bank["doc_id"], dtype=np.int32)
    n_docs = int(doc_id.max()) + 1 if doc_id.size else 0
    n_held_paras = int(np.isin(doc_id, held_ids).sum())
    # Positional last-10% for comparison (NOT used — overlaps train)
    n_last10 = max(1, int(round(0.10 * n_docs)))
    last10 = set(range(n_docs - n_last10, n_docs))
    held_set = set(int(x) for x in held_ids.tolist())
    return {
        "method": "zbank.npz held_doc_ids (doc-disjoint 98/1/1, seed=42)",
        "note": (
            "Phase-A ParaPairDataset uses the same held_doc_ids. "
            "Positional last-10% of docs would overlap the train split "
            "(random 98%) and is NOT used for G1."
        ),
        "n_docs_total": n_docs,
        "n_held_docs": int(held_ids.size),
        "n_held_paragraphs": n_held_paras,
        "held_doc_ids_min": int(held_ids.min()) if held_ids.size else None,
        "held_doc_ids_max": int(held_ids.max()) if held_ids.size else None,
        "overlap_held_vs_positional_last10_docs": int(len(held_set & last10)),
    }


def run_gate(
    *,
    cfg_path: Path = DEFAULT_CFG,
    ckpt_path: Path = DEFAULT_CKPT,
    out_path: Path = DEFAULT_OUT,
    n: int = 200,
    n_2afc_pairs: int = 100,
    n_fluency: int = 64,
    max_new_tokens: int = 128,
    seed: int = 0,
    device_name: str = "auto",
) -> dict[str, Any]:
    t_wall0 = time.time()
    cfg = _load_cfg(cfg_path)
    _seed_all(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
        if device_name != "auto"
        else "cpu"
    )
    print(f"[G1] device={device} n={n} ckpt={ckpt_path}", flush=True)

    zbank = _ROOT / cfg["zbank"]
    held_info = identify_held_split(zbank)
    print(f"[G1] held split: {json.dumps(held_info, indent=2)}", flush=True)

    held_ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="held")
    print(f"[G1] held pairs available: {len(held_ds)}", flush=True)
    if len(held_ds) == 0:
        raise RuntimeError("no held pairs in ParaPairDataset")

    idxs = sample_held_indices(held_ds, n, seed=seed)
    items = [held_ds[i] for i in idxs]
    print(f"[G1] sampled n={len(items)} held commands", flush=True)

    # ---- load actuator ----
    model, tokenizer, ckpt_meta = load_stage_a_for_infer(cfg, ckpt_path, device)
    print(
        f"[G1] checkpoint epoch={ckpt_meta['epoch']} step={ckpt_meta['step']} "
        f"backbone={ckpt_meta['backbone']}",
        flush=True,
    )

    # ---- generate conditioned paragraphs ----
    gens: list[str] = []
    t_gen0 = time.time()
    for i, item in enumerate(items):
        text = generate_paragraph(
            model,
            tokenizer,
            item["z"],
            item["prev"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
        )
        gens.append(text)
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_gen0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(items) - i - 1) / max(rate, 1e-6)
            print(
                f"[G1] gen {i+1}/{len(items)} "
                f"({rate:.2f}/s eta={eta/60:.1f}m) "
                f"chars={len(text)} preview={text[:80]!r}",
                flush=True,
            )
    t_gen = time.time() - t_gen0
    print(f"[G1] conditioned generation done in {t_gen:.1f}s", flush=True)

    # Free GPU briefly before CPU encode / MiniLM (keep model for fluency + uncond)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- JEPA re-encode (CPU) ----
    jepa_path = _ROOT / cfg.get("jepa", "artifacts/jepa_repair_1m/jepa/joint.pt")
    print(f"[G1] loading JEPA codec (cpu) from {jepa_path}", flush=True)
    jepa_model, jepa_tok = load_joint(str(jepa_path), device="cpu")
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")

    ys: list[np.ndarray] = []
    feds: list[float] = []
    for i, (item, text) in enumerate(zip(items, gens)):
        y = encode_paragraph_pooled(codec, text) if text.strip() else np.zeros(512, np.float32)
        ys.append(y)
        feds.append(_cos(y, item["z"]) if text.strip() else float("nan"))
        if (i + 1) % 50 == 0:
            print(f"[G1] JEPA encode {i+1}/{len(items)}", flush=True)
    fed_arr = np.asarray(feds, dtype=np.float32)
    fed_median = float(np.nanmedian(fed_arr))
    fed_mean = float(np.nanmean(fed_arr))
    print(f"[G1] fed_vs_concept median={fed_median:.4f} mean={fed_mean:.4f}", flush=True)

    # ---- MiniLM writer-fidelity ----
    src_texts = [item["target"] for item in items]
    print("[G1] encoding MiniLM writer-fidelity...", flush=True)
    minilm_cos = encode_minilm_pair_cos(gens, src_texts)
    minilm_median = float(np.median(minilm_cos))
    minilm_mean = float(np.mean(minilm_cos))
    print(
        f"[G1] MiniLM fidelity median={minilm_median:.4f} mean={minilm_mean:.4f}",
        flush=True,
    )

    # ---- 2-AFC controllability ----
    zs = np.stack([_l2(it["z"]) for it in items], axis=0)
    y_mat = np.stack([_l2(y) for y in ys], axis=0)
    rng = np.random.RandomState(seed + 7)
    pairs = sample_2afc_pairs(zs, rng, n_pairs=min(n_2afc_pairs, len(items) // 2))
    pair_cos = [float(np.dot(zs[i], zs[j])) for i, j in pairs]
    correct = 0
    for i, j in pairs:
        # generated_i should be closer to z_i than to z_j
        # (and symmetrically we count each directed? Spec: accuracy that
        # generated_a re-encodes closer to z_a than z_b — one check per pair arm)
        # Use both arms for a fuller estimate:
        ok_a = float(np.dot(y_mat[i], zs[i])) > float(np.dot(y_mat[i], zs[j]))
        ok_b = float(np.dot(y_mat[j], zs[j])) > float(np.dot(y_mat[j], zs[i]))
        correct += int(ok_a) + int(ok_b)
    n_2afc_trials = 2 * len(pairs) if pairs else 0
    afc_acc = float(correct / n_2afc_trials) if n_2afc_trials else float("nan")
    print(
        f"[G1] 2-AFC acc={afc_acc:.4f} over {n_2afc_trials} directed trials "
        f"({len(pairs)} pairs, mean pair cos={float(np.mean(pair_cos)) if pair_cos else float('nan'):.3f})",
        flush=True,
    )

    # ---- Fluency proxy: length-norm NLL win-rate vs unconditioned ----
    flu_n = min(n_fluency, len(items))
    flu_idxs = list(range(flu_n))
    print(f"[G1] fluency proxy: generating {flu_n} unconditioned paras...", flush=True)
    uncond_gens: list[str] = []
    t_u0 = time.time()
    for k, i in enumerate(flu_idxs):
        ut = generate_paragraph(
            model,
            tokenizer,
            None,
            items[i]["prev"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
        )
        uncond_gens.append(ut)
        if (k + 1) % 16 == 0 or k == 0:
            print(f"[G1] uncond gen {k+1}/{flu_n}", flush=True)
    print(f"[G1] uncond gen done in {time.time()-t_u0:.1f}s", flush=True)

    wins = 0
    nll_cond: list[float] = []
    nll_uncond: list[float] = []
    for k, i in enumerate(flu_idxs):
        nc = length_norm_nll(model, tokenizer, items[i]["prev"], gens[i], device)
        nu = length_norm_nll(model, tokenizer, items[i]["prev"], uncond_gens[k], device)
        nll_cond.append(nc)
        nll_uncond.append(nu)
        # Lower NLL = more fluent under the backbone. Cond "wins" if <= uncond.
        if nc <= nu:
            wins += 1
    fluency_winrate = float(wins / flu_n) if flu_n else float("nan")
    mean_nll_c = float(np.mean(nll_cond)) if nll_cond else float("nan")
    mean_nll_u = float(np.mean(nll_uncond)) if nll_uncond else float("nan")
    # perplexity ratio: ppl_cond / ppl_uncond  (<1 means cond more fluent)
    ppl_ratio = float(np.exp(mean_nll_c - mean_nll_u)) if nll_cond else float("nan")
    print(
        f"[G1] fluency winrate={fluency_winrate:.4f} "
        f"(cond NLL lower/equal than uncond); "
        f"mean_nll_cond={mean_nll_c:.4f} mean_nll_uncond={mean_nll_u:.4f} "
        f"ppl_ratio={ppl_ratio:.4f}",
        flush=True,
    )

    # ---- PASS / FAIL / KILL ----
    pass_minilm = bool(minilm_median >= THR_MINILM)
    pass_fed = bool(fed_median >= THR_FED)
    pass_render = pass_minilm and pass_fed
    pass_2afc = bool(afc_acc >= THR_2AFC)
    pass_fluency = bool(fluency_winrate >= THR_FLUENCY)
    # First (pre-corrective) read: apply kill thresholds directly
    kill = bool(minilm_median < KILL_MINILM or afc_acc < KILL_2AFC)
    overall_pass = pass_render and pass_2afc and pass_fluency

    # Sample rows for qualitative inspection
    order = np.argsort(-fed_arr)  # best fed first, then sprinkle mid/low
    sample_ix = []
    if len(order):
        sample_ix.append(int(order[0]))
        sample_ix.append(int(order[len(order) // 4]))
        sample_ix.append(int(order[len(order) // 2]))
        sample_ix.append(int(order[(3 * len(order)) // 4]))
        sample_ix.append(int(order[-1]))
        # one random
        sample_ix.append(int(rng.choice(len(items))))
    # dedupe preserve order
    seen_s: set[int] = set()
    sample_rows: list[dict[str, Any]] = []
    for i in sample_ix:
        if i in seen_s:
            continue
        seen_s.add(i)
        z = items[i]["z"]
        sample_rows.append(
            {
                "doc_id": int(items[i]["doc_id"]),
                "para_idx": int(items[i]["para_idx"]),
                "z_cmd_norm": float(np.linalg.norm(z)),
                "z_cmd_preview": [float(x) for x in z[:8]],
                "source_paragraph": items[i]["target"][:500],
                "generated_paragraph": gens[i][:500],
                "fed_vs_concept": float(fed_arr[i]),
                "minilm_fidelity": float(minilm_cos[i]),
                "ctx_preview": items[i]["prev"][:200],
            }
        )

    # Optional: summarize train-time eval_samples if present
    eval_path = ckpt_path.parent / "eval_samples_epoch01.jsonl"
    eval_summary: dict[str, Any] | None = None
    if eval_path.exists():
        erows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
        efeds = []
        for r in erows:
            if r.get("y") and r.get("z_cmd"):
                efeds.append(_cos(np.asarray(r["y"], np.float32), np.asarray(r["z_cmd"], np.float32)))
        eval_summary = {
            "path": str(eval_path),
            "n": len(erows),
            "fed_vs_concept_median_precomputed_y": float(np.median(efeds)) if efeds else None,
            "empty_gens": sum(1 for r in erows if not (r.get("y_text") or "").strip()),
            "note": (
                "Train-time y may be single-shot encode (not sentence-pooled); "
                "G1 numbers below re-encode with pooled protocol."
            ),
        }

    report: dict[str, Any] = {
        "gate": "G1",
        "checkpoint": ckpt_meta,
        "n_held_sampled": len(items),
        "n_held_available": len(held_ds),
        "held_split": held_info,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "seconds": t_gen,
            "empty_gens": sum(1 for g in gens if not g.strip()),
            "mean_chars": float(np.mean([len(g) for g in gens])) if gens else 0.0,
        },
        "metrics": {
            "minilm_writer_fidelity": {
                "median": minilm_median,
                "mean": minilm_mean,
                "threshold": THR_MINILM,
                "pass": pass_minilm,
            },
            "jepa_fed_vs_concept": {
                "median": fed_median,
                "mean": fed_mean,
                "threshold": THR_FED,
                "pass": pass_fed,
            },
            "renderability_and": {
                "pass": pass_render,
                "rule": "MiniLM>=0.40 AND fed_vs_concept>=0.55",
            },
            "controllability_2afc": {
                "accuracy": afc_acc,
                "n_pairs": len(pairs),
                "n_directed_trials": n_2afc_trials,
                "mean_pair_cos": float(np.mean(pair_cos)) if pair_cos else None,
                "target_pair_cos": 0.57,
                "threshold": THR_2AFC,
                "pass": pass_2afc,
            },
            "fluency_guard": {
                "method": (
                    "length-normalized NLL under the loaded Stage-A backbone "
                    "(no soft-prefix) of conditioned vs unconditioned (zero-prefix) "
                    "generations from the same prev context; win = NLL_cond <= NLL_uncond"
                ),
                "win_rate": fluency_winrate,
                "n": flu_n,
                "mean_nll_cond": mean_nll_c,
                "mean_nll_uncond": mean_nll_u,
                "ppl_ratio_cond_over_uncond": ppl_ratio,
                "threshold": THR_FLUENCY,
                "pass": pass_fluency,
            },
        },
        "verdicts": {
            "g1_pass": overall_pass,
            "kill_rule": {
                "triggered": kill,
                "rule": "MiniLM median < 0.30 OR 2-AFC < 0.70 (pre-corrective first read)",
                "minilm_median": minilm_median,
                "kill_minilm_threshold": KILL_MINILM,
                "afc_accuracy": afc_acc,
                "kill_afc_threshold": KILL_2AFC,
            },
            "summary": (
                "KILL" if kill else ("PASS" if overall_pass else "FAIL_BUT_NOT_KILL")
            ),
        },
        "samples": sample_rows,
        "train_eval_samples_summary": eval_summary,
        "wall_seconds": time.time() - t_wall0,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    # also write a compact generations dump
    gens_path = out_path.with_suffix(".generations.jsonl")
    with gens_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            f.write(
                json.dumps(
                    {
                        "doc_id": int(item["doc_id"]),
                        "para_idx": int(item["para_idx"]),
                        "generated": gens[i],
                        "source": item["target"],
                        "fed_vs_concept": float(fed_arr[i]),
                        "minilm_fidelity": float(minilm_cos[i]),
                    }
                )
                + "\n"
            )
    print(f"[G1] wrote {out_path}", flush=True)
    print(f"[G1] wrote {gens_path}", flush=True)
    print(f"[G1] VERDICT: {report['verdicts']['summary']}", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="G1 kill-gate for Phase-A actuator")
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=200, help="held-out commands (default 200; up to 500)")
    p.add_argument("--n-2afc-pairs", type=int, default=100)
    p.add_argument("--n-fluency", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(argv)
    run_gate(
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        out_path=args.out,
        n=args.n,
        n_2afc_pairs=args.n_2afc_pairs,
        n_fluency=args.n_fluency,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
