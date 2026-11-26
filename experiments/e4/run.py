"""Run E4-S1: build items, fit adapters, decode the six arms, write a verdict.

The headline verdict is ``NOT_MEASURED`` unless the run is the pre-registered
192 × 3 design and both controls behave. A smoke run exists to prove the
pipeline, not to answer Q1.

GPU: this process refuses to load a model onto CUDA when another compute job
is already resident or when free memory cannot hold a footprint under 11 GB.
It also refuses a confirmatory run whose estimate would still be going at
23:30 America/New_York, which is when this machine's wifi drops.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.e4.arms import MANIFEST_FITS, MANIFEST_ITEMS, MANIFEST_NULL, MANIFEST_TEXT, arm_plan, plan_counts
from experiments.e4.data import build_split, corpus_signature, hash_encode, load_documents
from experiments.e4.drift import audit_zbank, write_audit
from experiments.e4.evaluate import score_and_decide
from experiments.e4.fit import DEFAULT_FIT_SEEDS, fit_adapter
from experiments.e4.generate import DecodeConfig, generate_arm, generate_null
from experiments.protocol.manifest import capture_git_state

_ROOT = Path(__file__).resolve().parents[2]
_TZ = ZoneInfo("America/New_York")
_WIFI_OFF = (23, 30)
# Leave a margin so a job is not still writing when the radio drops.
_WIFI_MARGIN_MIN = 20
_GPU_FOOTPRINT_MIB = 11_000


def minutes_until_wifi_off(now: datetime | None = None) -> float:
    now = now or datetime.now(_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_TZ)
    cutoff = now.replace(hour=_WIFI_OFF[0], minute=_WIFI_OFF[1], second=0, microsecond=0)
    return (cutoff - now).total_seconds() / 60.0


def gpu_snapshot() -> dict:
    """What is on the card right now. Empty if nvidia-smi is unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        mem = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": str(exc), "apps": [], "free_mib": 0}
    apps = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        apps.append(
            {"pid": int(parts[0]), "name": parts[1], "used_mib": int(parts[2])}
        )
    free = used = total = 0
    row = mem.stdout.strip().splitlines()
    if row:
        nums = [int(p.strip()) for p in row[0].split(",")]
        used, free, total = nums[0], nums[1], nums[2]
    return {
        "ok": True,
        "apps": apps,
        "used_mib": used,
        "free_mib": free,
        "total_mib": total,
    }


def gpu_available_for_e4(snapshot: dict | None = None) -> tuple[bool, str]:
    """True when this process can put a <11 GB footprint on the card.

    llama-server and desktop helpers are the background load. Another
    training process (anything python over 2 GB that is not us) means a
    second GPU job, which this machine does not have room for.
    """
    snap = snapshot if snapshot is not None else gpu_snapshot()
    if not snap.get("ok"):
        return False, snap.get("reason", "nvidia-smi failed")
    me = _own_pids()
    blockers = []
    for app in snap["apps"]:
        name = app["name"].lower()
        if app["pid"] in me:
            continue
        if "llama-server" in name or "sunshine" in name or "ead" in name:
            continue
        if app["used_mib"] >= 2000 and "python" in name:
            blockers.append(f"pid {app['pid']} {app['name']} {app['used_mib']} MiB")
    if blockers:
        return False, "another GPU job is resident: " + "; ".join(blockers)
    if snap["free_mib"] < 9000:
        return False, f"only {snap['free_mib']} MiB free; need 9000 MiB to stay under 11 GB"
    return True, "ok"


def _own_pids() -> set[int]:
    import os

    return {os.getpid()}


def _records_from_arm(items, arm: str, fit: int, texts: list[str]) -> list[dict]:
    if len(items) != len(texts):
        raise RuntimeError(f"{arm} produced {len(texts)} texts for {len(items)} items")
    rows = []
    for item, text in zip(items, texts):
        rows.append(
            {
                "arm": arm,
                "fit": int(fit),
                "item_index": int(item.item_index),
                "doc_id": str(item.doc_id),
                "text": text,
                "gold": tuple(item.gold_entities),
                "context": item.context_text,
            }
        )
    return rows


def _load_lm(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer


def _jepa_encode(jepa_path: Path, device: str):
    """Return ``(encode_fn, close)``. encode_fn maps sentences to rows."""
    import sys

    for p in (str(_ROOT / "src"), str(_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from conceptpoc.codec.jepa_codec import JepaCodec
    from conceptpoc.persistence import load_joint

    model, tok = load_joint(str(jepa_path), device=device)
    codec = JepaCodec(model, tok, max_tokens=128, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return codec.encode(list(texts))

    def close() -> None:
        nonlocal model, tok, codec
        del codec, model, tok
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return encode, close


def _minilm(device: str):
    from sentence_transformers import SentenceTransformer

    # MiniLM is a scorer, not the refiner. Keep it off the card the refiner
    # just filled; CPU is fast enough at this batch size.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    return embed


def run(
    *,
    corpus: Path,
    out: Path,
    n_items: int,
    n_fits: int,
    n_text: int,
    n_null: int,
    n_fit_docs: int,
    fit_steps: int,
    max_new_tokens: int,
    seed: int,
    device: str,
    model_id: str,
    jepa_path: Path,
    hash_encoder: bool,
    n_boot: int,
    batch_size: int,
) -> dict:
    t0 = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    drift = audit_zbank()
    write_audit(out.parent / "zbank_drift_audit.json", drift)
    if drift.get("drift_in_band"):
        raise RuntimeError("drift audit unexpectedly reports the legacy band; re-check the bank")
    # The bank is a calibration record. Drafts below are encoded fresh.
    if not drift.get("drafts_separable"):
        drift_note = "encoder cone may be too narrow; necessity will be biased toward zero"
    else:
        drift_note = drift["reason"]

    docs = load_documents(corpus, min_chars=4000)
    encode_device = "cpu" if hash_encoder else ("cuda" if device == "cuda" else "cpu")
    closer = None
    if hash_encoder:
        def encode(texts, _dim=512):
            return hash_encode(texts, dim=_dim)
    else:
        if encode_device == "cuda":
            ok, why = gpu_available_for_e4()
            if not ok:
                encode_device = "cpu"
                drift_note += f" JEPA encode fell back to CPU ({why})."
        encode, closer = _jepa_encode(jepa_path, encode_device)

    try:
        split = build_split(
            docs,
            encode,
            n_items=n_items,
            n_fit=n_fit_docs,
            n_null=n_null,
            n_text=n_text,
            seed=seed,
        )
    finally:
        if closer is not None:
            closer()

    if device == "cuda":
        ok, why = gpu_available_for_e4()
        if not ok:
            raise RuntimeError(f"refusing to load the refiner on CUDA: {why}")
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_lm(model_id, device)
    cfg = DecodeConfig(max_new_tokens=max_new_tokens, batch_size=batch_size)
    seeds = list(DEFAULT_FIT_SEEDS[:n_fits])
    if len(seeds) < n_fits:
        raise RuntimeError(f"only {len(DEFAULT_FIT_SEEDS)} pre-registered fit seeds")

    records: list[dict] = []
    fit_stats: list[dict] = []
    gen_started = time.perf_counter()
    n_gen = 0
    text_items = list(split.text_items)
    for fit_i, fit_seed in enumerate(seeds):
        adapter, stats = fit_adapter(
            model,
            tokenizer,
            split.fit_examples,
            steps=fit_steps,
            seed=fit_seed,
            device=device,
            max_prev=128,
            max_target=64,
        )
        ckpt = out.parent / f"adapter_f{fit_i}.pt"
        torch.save(
            {"adapter": adapter.state_dict(), "stats": stats, "seed": fit_seed},
            ckpt,
        )
        stats = dict(stats)
        stats["checkpoint"] = str(ckpt)
        stats["fit_index"] = fit_i
        fit_stats.append(stats)
        model.eval()
        for arm in ("true", "shuf", "none", "dose50"):
            texts = generate_arm(
                model, tokenizer, adapter, list(split.items), arm, cfg, seed=seed
            )
            records.extend(_records_from_arm(split.items, arm, fit_i, texts))
            n_gen += len(texts)
        # Text-prefix does not use the adapter. One fit is the analysis; the
        # manifest's ×3 is recorded as scheduled and not re-decoded, because a
        # frozen greedy backbone would copy the same strings (pseudoreplication).
        if fit_i == 0:
            for arm in ("textpc_true", "textpc_rand"):
                texts = generate_arm(
                    model, tokenizer, adapter, text_items, arm, cfg, seed=seed
                )
                records.extend(_records_from_arm(text_items, arm, 0, texts))
                n_gen += len(texts)
            for arm_name, group in (("null_a", split.null_a), ("null_b", split.null_b)):
                texts = generate_null(
                    model,
                    tokenizer,
                    adapter,
                    list(group),
                    cfg,
                    seed=seed,
                    arm=arm_name[-1],
                )
                records.extend(_records_from_arm(group, arm_name, 0, texts))
                n_gen += len(texts)
        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()

    gen_seconds = time.perf_counter() - gen_started
    embed = _minilm(device)
    confirmatory = (
        n_items == MANIFEST_ITEMS
        and n_fits == MANIFEST_FITS
        and n_text == MANIFEST_TEXT
        and n_null == MANIFEST_NULL
        and not hash_encoder
        and model_id.endswith("Qwen3-1.7B-Base")
    )
    result = score_and_decide(
        records, embed, confirmatory=confirmatory, n_boot=n_boot, seed=seed
    )
    scheduled = plan_counts(
        arm_plan(n_items=n_items, n_fits=n_fits, n_text=n_text, n_null=n_null)
    )
    # Executed text-prefix and null are one fit, not n_fits. Say so.
    executed = {
        "true": n_items * n_fits,
        "shuf": n_items * n_fits,
        "none": n_items * n_fits,
        "dose50": n_items * n_fits,
        "textpc_true": n_text,
        "textpc_rand": n_text,
        "null_a": n_null,
        "null_b": n_null,
        "adapter_fits": n_fits,
        "generations": n_gen,
    }
    git = capture_git_state(_ROOT)
    peak = 0.0
    if device == "cuda" and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
    result.update(
        {
            "model_id": model_id,
            "device": device,
            "hash_encoder": bool(hash_encoder),
            "seed": seed,
            "fit_steps": fit_steps,
            "max_new_tokens": max_new_tokens,
            "fit": fit_stats,
            "arm_counts_scheduled": scheduled,
            "arm_counts_executed": executed,
            "text_prefix_executed_once": True,
            "text_prefix_reason": (
                "The refiner is frozen and the text-prefix arms do not read "
                "the adapter, so greedy replicates across fits are the same "
                "strings. They are decoded once. The manifest still schedules "
                f"{n_text * n_fits} per condition; those extra copies are not "
                "pooled, because pooling identical rows would be a fake n."
            ),
            "allocation": {
                k: split.allocation[k]
                for k in ("null_mode", "n_eligible", "n_items", "n_fit", "n_null", "n_text")
            },
            "corpus_sha256": corpus_signature(item.doc_id for item in split.items),
            "zbank_drift": {
                "drift_mean": drift["drift_mean"],
                "drift_std": drift["drift_std"],
                "legacy_band": drift["legacy_band"],
                "drift_in_band": drift["drift_in_band"],
                "drift_above_band": drift["drift_above_band"],
                "cross_doc_mean": drift["cross_doc_mean"],
                "consecutive_minus_cross": drift["consecutive_minus_cross"],
                "drafts_separable": drift["drafts_separable"],
                "meta_matches_recompute": drift["meta_matches_recompute"],
                "used_as_e4_draft_source": False,
                "affects_paired_contrasts": False,
                "note": drift_note,
            },
            "git_commit": git.commit,
            "git_dirty": git.dirty,
            "seconds": time.perf_counter() - t0,
            "generation_seconds": gen_seconds,
            "generations_per_second": n_gen / gen_seconds if gen_seconds else None,
            "peak_vram_gb": peak,
            "sample_generations": [
                {"arm": r["arm"], "doc_id": r["doc_id"], "head": r["text"][:180]}
                for r in records[:12]
            ],
        }
    )
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_generations(out.with_suffix(".generations.jsonl"), records)
    return result


def _write_generations(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            payload = dict(row)
            payload["gold"] = list(payload["gold"])
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _add_scale(p: argparse.ArgumentParser, *, full: bool) -> None:
    if full:
        p.add_argument("--n-items", type=int, default=MANIFEST_ITEMS)
        p.add_argument("--n-fits", type=int, default=MANIFEST_FITS)
        p.add_argument("--n-text", type=int, default=MANIFEST_TEXT)
        p.add_argument("--n-null", type=int, default=MANIFEST_NULL)
        p.add_argument("--n-fit-docs", type=int, default=64)
        p.add_argument("--fit-steps", type=int, default=200)
        p.add_argument("--max-new-tokens", type=int, default=64)
        p.add_argument("--n-boot", type=int, default=10_000)
    else:
        p.add_argument("--n-items", type=int, default=8)
        p.add_argument("--n-fits", type=int, default=1)
        p.add_argument("--n-text", type=int, default=8)
        p.add_argument("--n-null", type=int, default=8)
        p.add_argument("--n-fit-docs", type=int, default=8)
        p.add_argument("--fit-steps", type=int, default=3)
        p.add_argument("--max-new-tokens", type=int, default=32)
        p.add_argument("--n-boot", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--corpus", type=Path, default=_ROOT / "artifacts" / "corpus" / "biorxiv_2026.jsonl")
    p.add_argument("--jepa", type=Path, default=_ROOT / "artifacts" / "jepa_repair_1m" / "jepa" / "joint.pt")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--hash-encoder", action="store_true")
    p.add_argument("--allow-past-window", action="store_true")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E4-S1 interface harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="recompute the zbank drift numbers")
    audit.add_argument("--out", type=Path, default=_ROOT / "artifacts" / "e4s1" / "zbank_drift_audit.json")
    audit.add_argument("--encode-sample", action="store_true")
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--corpus", type=Path, default=_ROOT / "artifacts" / "corpus" / "biorxiv_2026.jsonl")

    smoke = sub.add_parser("smoke", help="handful of items, proves the pipeline")
    smoke.add_argument("--out", type=Path, default=_ROOT / "artifacts" / "e4s1" / "smoke_result.json")
    _add_scale(smoke, full=False)

    full = sub.add_parser("full", help="pre-registered 192 × 3, if the window allows")
    full.add_argument("--out", type=Path, default=_ROOT / "artifacts" / "e4s1" / "result.json")
    _add_scale(full, full=True)

    pre = sub.add_parser("preflight", help="GPU and wifi-window check, no model load")
    pre.add_argument("--estimate-minutes", type=float, default=None)

    args = p.parse_args(argv)
    if args.cmd == "audit":
        report = audit_zbank()
        if args.encode_sample:
            from experiments.e4.drift import encode_sample_drift

            docs = load_documents(args.corpus, min_chars=4000)
            report["harvest_sample"] = encode_sample_drift(
                docs, device=args.device
            )
        write_audit(args.out, report)
        print(json.dumps({
            "drift_mean": report["drift_mean"],
            "drift_in_band": report["drift_in_band"],
            "cross_doc_mean": report["cross_doc_mean"],
            "consecutive_minus_cross": report["consecutive_minus_cross"],
            "affects_paired_contrasts": report["affects_paired_contrasts"],
            "out": str(args.out),
        }, indent=2))
        return 0

    if args.cmd == "preflight":
        ok, why = gpu_available_for_e4()
        left = minutes_until_wifi_off()
        payload = {"gpu_ok": ok, "gpu_reason": why, "minutes_until_wifi_off": left, "snapshot": gpu_snapshot()}
        if args.estimate_minutes is not None:
            payload["fits_window"] = args.estimate_minutes < left - _WIFI_MARGIN_MIN
        print(json.dumps(payload, indent=2))
        return 0 if ok else 2

    if args.cmd == "full" and not args.allow_past_window:
        left = minutes_until_wifi_off()
        # Conservative until a smoke has measured a rate: the revised protocol
        # figure is ~4 GPU-hours. Refuse rather than guess low.
        estimate = 4 * 60
        if estimate >= left - _WIFI_MARGIN_MIN:
            print(
                json.dumps(
                    {
                        "started": False,
                        "reason": (
                            f"refusing the 192-item run: a 4-hour estimate does not "
                            f"finish { _WIFI_MARGIN_MIN } min before 23:30 "
                            f"({left:.0f} min left)"
                        ),
                    }
                )
            )
            return 3

    if args.device == "cuda":
        ok, why = gpu_available_for_e4()
        if not ok:
            print(json.dumps({"started": False, "reason": why}))
            return 2

    result = run(
        corpus=args.corpus,
        out=args.out,
        n_items=args.n_items,
        n_fits=args.n_fits,
        n_text=args.n_text,
        n_null=args.n_null,
        n_fit_docs=args.n_fit_docs,
        fit_steps=args.fit_steps,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device=args.device,
        model_id=args.model,
        jepa_path=args.jepa,
        hash_encoder=args.hash_encoder,
        n_boot=args.n_boot,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "arithmetic_verdict": result.get("arithmetic_verdict"),
        "reason": result["reason"],
        "out": str(args.out),
        "seconds": result["seconds"],
        "peak_vram_gb": result["peak_vram_gb"],
        "generations_per_second": result["generations_per_second"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
