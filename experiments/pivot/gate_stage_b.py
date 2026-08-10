"""Stage-B final gate battery vs Phase-A baseline (Fable §6.1).

Pre-registered thresholds; HARD STOP if train-judge rose but fed_vs_concept fell.
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
from experiments.pivot.gate_g1 import (  # noqa: E402
    encode_minilm_pair_cos,
    encode_paragraph_pooled,
    generate_paragraph,
    identify_held_split,
    load_stage_a_for_infer,
    sample_2afc_pairs,
    sample_held_indices,
)
from experiments.pivot.reward_stage_b import (  # noqa: E402
    LogitJudge,
    _cos,
    _l2,
    distinct_n,
    tokenize_words,
)
from experiments.pivot.train_stage_a import ParaPairDataset, _load_cfg, _seed_all  # noqa: E402

DEFAULT_CFG = _ROOT / "configs" / "pivot_stage_b.json"


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _merge_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
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
        "corpus",
        "zbank",
        "jepa",
    ):
        if key in cfg:
            phase_a[key] = cfg[key]
    return phase_a


def self_bleu4(texts: list[str], n_pairs: int = 200, seed: int = 0) -> float:
    """Approximate corpus self-BLEU-4 via random pairs (n-gram precision)."""
    rng = np.random.RandomState(seed)
    toks = [tokenize_words(t) for t in texts]

    def _ngrams(tok: list[str], n: int = 4) -> list[tuple[str, ...]]:
        if len(tok) < n:
            return []
        return [tuple(tok[i : i + n]) for i in range(len(tok) - n + 1)]

    scores: list[float] = []
    n = len(texts)
    if n < 2:
        return 0.0
    for _ in range(n_pairs):
        i = int(rng.randint(0, n))
        j = int(rng.randint(0, n))
        if i == j:
            continue
        ref = _ngrams(toks[j], 4)
        hyp = _ngrams(toks[i], 4)
        if not hyp:
            continue
        from collections import Counter

        rc = Counter(ref)
        match = 0
        for g in hyp:
            if rc[g] > 0:
                match += 1
                rc[g] -= 1
        scores.append(match / len(hyp))
    return float(np.mean(scores)) if scores else 0.0


def freeze_phase_a_baseline(
    cfg: dict[str, Any],
    *,
    device_name: str = "auto",
    n: int | None = None,
) -> dict[str, Any]:
    """Measure Phase-A on probe set P; write baseline_phaseA.json."""
    phase_cfg = _merge_cfg(cfg)
    device = _device(device_name)
    out_dir = _ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _ROOT / cfg["phase_a_ckpt"]
    _seed_all(int(cfg.get("seed", 42)))

    held = ParaPairDataset(_ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="held")
    n_held = int(n if n is not None else cfg.get("probe_n_held", 128))
    idxs = sample_held_indices(held, n_held, seed=0)
    items = [held[i] for i in idxs]

    model, tokenizer, meta = load_stage_a_for_infer(phase_cfg, ckpt, device)
    gens: list[str] = []
    for item in items:
        gens.append(
            generate_paragraph(
                model,
                tokenizer,
                item["z"],
                item["prev"],
                max_new_tokens=int(cfg.get("eval_max_new_tokens", 128)),
                do_sample=False,
                device=device,
            )
        )

    jepa_model, jepa_tok = load_joint(str(_ROOT / cfg["jepa"]), device="cpu")
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")
    feds = []
    for item, text in zip(items, gens):
        if not text.strip():
            feds.append(float("nan"))
            continue
        y = encode_paragraph_pooled(codec, text)
        feds.append(_cos(y, item["z"]))
    fed_med = float(np.nanmedian(np.asarray(feds, dtype=np.float64)))

    minilm = encode_minilm_pair_cos(gens, [it["target"] for it in items])
    minilm_med = float(np.median(minilm))

    # Train judge
    judge = LogitJudge(
        cfg.get("train_judge", "Qwen/Qwen2.5-1.5B-Instruct"),
        device,
        load_in_4bit=True,
    )
    j_train = judge.score_coherence(gens)
    j_train_mean = float(np.nanmean(np.asarray(j_train, dtype=np.float64)))
    del judge
    if device.type == "cuda":
        torch.cuda.empty_cache()

    held_j_mean = None
    try:
        hj = LogitJudge(cfg.get("heldout_judge", "HuggingFaceTB/SmolLM2-1.7B-Instruct"), device, load_in_4bit=True)
        j_held = hj.score_coherence(gens)
        held_j_mean = float(np.nanmean(np.asarray(j_held, dtype=np.float64)))
        del hj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[gateB] heldout judge baseline failed: {e}", flush=True)

    d2 = float(np.mean([distinct_n(tokenize_words(t), 2) for t in gens]))
    sbleu = self_bleu4(gens)

    # 2-AFC on same gens
    zs = np.stack([_l2(it["z"]) for it in items], axis=0)
    ys = []
    for text in gens:
        if text.strip():
            ys.append(encode_paragraph_pooled(codec, text))
        else:
            ys.append(np.zeros(512, np.float32))
    ys_a = np.stack(ys, axis=0)
    rng = np.random.RandomState(1)
    pairs = sample_2afc_pairs(zs, rng, n_pairs=min(100, len(items) // 2))
    correct = 0
    trials = 0
    for i, j in pairs:
        # directed: is y_i closer to z_i than z_j?
        for a, b in ((i, j), (j, i)):
            ca = float(np.dot(_l2(ys_a[a]), zs[a]))
            cb = float(np.dot(_l2(ys_a[a]), zs[b]))
            correct += int(ca >= cb)
            trials += 1
    afc = float(correct) / max(1, trials)

    baseline = {
        "checkpoint": str(ckpt),
        "meta": meta,
        "n": len(items),
        "fed_vs_concept_median": fed_med,
        "minilm_median": minilm_med,
        "train_judge_mean": j_train_mean,
        "heldout_judge_mean": held_j_mean,
        "distinct2_mean": d2,
        "self_bleu4": sbleu,
        "afc_accuracy": afc,
        "held_split": identify_held_split(_ROOT / cfg["zbank"]),
        "generations_path": str(out_dir / "baseline_phaseA.generations.jsonl"),
    }
    with (out_dir / "baseline_phaseA.generations.jsonl").open("w", encoding="utf-8") as f:
        for item, text, fed in zip(items, gens, feds):
            f.write(
                json.dumps(
                    {
                        "doc_id": int(item["doc_id"]),
                        "para_idx": int(item["para_idx"]),
                        "ctx": item["prev"],
                        "target": item["target"],
                        "generated": text,
                        "fed": fed,
                        "z_cmd": item["z"].astype(np.float32).tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (out_dir / "baseline_phaseA.json").write_text(json.dumps(baseline, indent=2))
    print(f"[gateB] baseline written: {json.dumps(baseline, indent=2)}", flush=True)
    return baseline


def evaluate_checkpoint(
    cfg: dict[str, Any],
    ckpt_path: Path,
    baseline: dict[str, Any],
    *,
    device_name: str = "auto",
    n_seeds: int = 1,
    label: str = "stageB",
) -> dict[str, Any]:
    phase_cfg = _merge_cfg(cfg)
    device = _device(device_name)
    out_dir = _ROOT / cfg["out_dir"]
    held = ParaPairDataset(_ROOT / cfg["corpus"], _ROOT / cfg["zbank"], split="held")
    n_held = int(cfg.get("probe_n_held", 128))
    idxs = sample_held_indices(held, n_held, seed=0)
    items = [held[i] for i in idxs]

    model, tokenizer, meta = load_stage_a_for_infer(phase_cfg, ckpt_path, device)

    # Multi-seed: average metrics (greedy seed0 + sampled seeds)
    all_fed: list[float] = []
    all_d2: list[float] = []
    seed_gens: list[list[str]] = []
    for seed in range(max(1, n_seeds)):
        gens: list[str] = []
        do_sample = seed > 0
        for item in items:
            gens.append(
                generate_paragraph(
                    model,
                    tokenizer,
                    item["z"],
                    item["prev"],
                    max_new_tokens=int(cfg.get("eval_max_new_tokens", 128)),
                    do_sample=do_sample,
                    temperature=0.9 if do_sample else 1.0,
                    device=device,
                )
            )
        seed_gens.append(gens)

    # Primary metrics on greedy (seed 0)
    gens0 = seed_gens[0]
    jepa_model, jepa_tok = load_joint(str(_ROOT / cfg["jepa"]), device="cpu")
    codec = JepaCodec(jepa_model, jepa_tok, max_tokens=128, device="cpu")
    feds = []
    for item, text in zip(items, gens0):
        if not text.strip():
            feds.append(float("nan"))
            continue
        y = encode_paragraph_pooled(codec, text)
        feds.append(_cos(y, item["z"]))
    # Aggregate across seeds for fed
    for gens in seed_gens:
        for item, text in zip(items, gens):
            if not text.strip():
                continue
            y = encode_paragraph_pooled(codec, text)
            all_fed.append(_cos(y, item["z"]))
            all_d2.append(distinct_n(tokenize_words(text), 2))

    fed_med = float(np.nanmedian(np.asarray(feds, dtype=np.float64)))
    fed_med_seeds = float(np.nanmedian(np.asarray(all_fed, dtype=np.float64))) if all_fed else fed_med
    minilm = encode_minilm_pair_cos(gens0, [it["target"] for it in items])
    minilm_med = float(np.median(minilm))
    d2 = float(np.mean(all_d2)) if all_d2 else float(
        np.mean([distinct_n(tokenize_words(t), 2) for t in gens0])
    )
    sbleu = self_bleu4(gens0)

    judge = LogitJudge(cfg.get("train_judge", "Qwen/Qwen2.5-1.5B-Instruct"), device, load_in_4bit=True)
    j_train = float(np.nanmean(np.asarray(judge.score_coherence(gens0), dtype=np.float64)))
    del judge
    if device.type == "cuda":
        torch.cuda.empty_cache()

    j_held = None
    try:
        hj = LogitJudge(cfg.get("heldout_judge", "HuggingFaceTB/SmolLM2-1.7B-Instruct"), device, load_in_4bit=True)
        j_held = float(np.nanmean(np.asarray(hj.score_coherence(gens0), dtype=np.float64)))
        del hj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[gateB] heldout judge failed: {e}", flush=True)

    # 2-AFC
    zs = np.stack([_l2(it["z"]) for it in items], axis=0)
    ys = []
    for text in gens0:
        ys.append(
            encode_paragraph_pooled(codec, text)
            if text.strip()
            else np.zeros(512, np.float32)
        )
    ys_a = np.stack(ys, axis=0)
    rng = np.random.RandomState(1)
    pairs = sample_2afc_pairs(zs, rng, n_pairs=min(100, len(items) // 2))
    correct = 0
    trials = 0
    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            ca = float(np.dot(_l2(ys_a[a]), zs[a]))
            cb = float(np.dot(_l2(ys_a[a]), zs[b]))
            correct += int(ca >= cb)
            trials += 1
    afc = float(correct) / max(1, trials)

    # Pairwise fluency vs Phase-A baseline gens
    base_gens_path = _ROOT / cfg["out_dir"] / "baseline_phaseA.generations.jsonl"
    fluency_wr = None
    if base_gens_path.exists() and j_held is not None:
        base_gens = []
        with base_gens_path.open("r", encoding="utf-8") as f:
            for line in f:
                base_gens.append(json.loads(line)["generated"])
        # Use train judge for pairwise if heldout already freed — reload heldout briefly
        try:
            hj = LogitJudge(cfg.get("heldout_judge", "HuggingFaceTB/SmolLM2-1.7B-Instruct"), device, load_in_4bit=True)
            wins = 0
            n_pair = min(64, len(gens0), len(base_gens))
            for i in range(n_pair):
                a, b = gens0[i], base_gens[i]
                # position-swap: score both, win if stageB higher
                sa = hj.score_coherence([a])[0]
                sb = hj.score_coherence([b])[0]
                if np.isfinite(sa) and np.isfinite(sb) and sa >= sb:
                    wins += 1
            fluency_wr = float(wins) / max(1, n_pair)
            del hj
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[gateB] fluency pairwise failed: {e}", flush=True)

    base_fed = float(baseline.get("fed_vs_concept_median", 0.0))
    base_afc = float(baseline.get("afc_accuracy", 0.0))
    base_minilm = float(baseline.get("minilm_median", 0.0))
    base_held_j = baseline.get("heldout_judge_mean")
    base_train_j = float(baseline.get("train_judge_mean", 0.0))
    base_sbleu = float(baseline.get("self_bleu4", 0.0))

    gates = {
        "fed_vs_concept": {
            "value": fed_med_seeds,
            "greedy": fed_med,
            "need": f">= baseline+{cfg.get('gate_fed_delta', 0.03)} AND >= {cfg.get('gate_fed_floor', 0.58)}",
            "pass": (
                fed_med_seeds >= base_fed + float(cfg.get("gate_fed_delta", 0.03))
                and fed_med_seeds >= float(cfg.get("gate_fed_floor", 0.58))
            ),
        },
        "afc_2": {
            "value": afc,
            "need": f">= baseline AND >= {cfg.get('gate_2afc_floor', 0.88)}",
            "pass": afc >= base_afc and afc >= float(cfg.get("gate_2afc_floor", 0.88)),
        },
        "heldout_judge": {
            "value": j_held,
            "baseline": base_held_j,
            "need": f">= baseline+{cfg.get('gate_heldout_j_delta', 0.08)}",
            "pass": (
                j_held is not None
                and base_held_j is not None
                and j_held >= float(base_held_j) + float(cfg.get("gate_heldout_j_delta", 0.08))
            ),
        },
        "minilm": {
            "value": minilm_med,
            "need": f">= baseline-{cfg.get('gate_minilm_slack', 0.02)}",
            "pass": minilm_med >= base_minilm - float(cfg.get("gate_minilm_slack", 0.02)),
        },
        "fluency_winrate": {
            "value": fluency_wr,
            "need": f">= {cfg.get('gate_fluency_winrate', 0.55)}",
            "pass": fluency_wr is not None
            and fluency_wr >= float(cfg.get("gate_fluency_winrate", 0.55)),
        },
        "distinct2": {
            "value": d2,
            "need": f">= {cfg.get('gate_distinct2', 0.45)}",
            "pass": d2 >= float(cfg.get("gate_distinct2", 0.45)),
        },
        "self_bleu4": {
            "value": sbleu,
            "need": f"<= baseline+{cfg.get('gate_self_bleu_slack', 0.05)}",
            "pass": sbleu <= base_sbleu + float(cfg.get("gate_self_bleu_slack", 0.05)),
        },
    }

    hard_stop = bool(j_train > base_train_j and fed_med_seeds < base_fed)
    all_pass = all(g["pass"] for g in gates.values()) and not hard_stop

    result = {
        "label": label,
        "checkpoint": str(ckpt_path),
        "meta": meta,
        "n": len(items),
        "n_seeds": n_seeds,
        "metrics": {
            "fed_vs_concept_median": fed_med_seeds,
            "fed_vs_concept_greedy": fed_med,
            "minilm_median": minilm_med,
            "train_judge_mean": j_train,
            "heldout_judge_mean": j_held,
            "afc_accuracy": afc,
            "distinct2_mean": d2,
            "self_bleu4": sbleu,
            "fluency_winrate": fluency_wr,
        },
        "baseline": {
            "fed_vs_concept_median": base_fed,
            "afc_accuracy": base_afc,
            "minilm_median": base_minilm,
            "heldout_judge_mean": base_held_j,
            "train_judge_mean": base_train_j,
            "self_bleu4": base_sbleu,
        },
        "gates": gates,
        "hard_stop_reward_hacking": hard_stop,
        "overall_pass": all_pass,
        "samples": [
            {
                "doc_id": int(items[i]["doc_id"]),
                "para_idx": int(items[i]["para_idx"]),
                "generated": gens0[i][:400],
                "fed": feds[i],
            }
            for i in range(min(6, len(items)))
        ],
    }
    out_path = out_dir / f"gate_{label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[gateB] {label} overall_pass={all_pass} hard_stop={hard_stop}", flush=True)
    print(json.dumps({k: {"value": v["value"], "pass": v["pass"]} for k, v in gates.items()}, indent=2), flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage-B gates")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--freeze-baseline", action="store_true")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--label", type=str, default="stageB")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--probe-n", type=int, default=None)
    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)
    if args.freeze_baseline:
        freeze_phase_a_baseline(cfg, device_name=args.device, n=args.probe_n)
        return
    baseline_path = _ROOT / cfg["out_dir"] / "baseline_phaseA.json"
    if not baseline_path.exists():
        freeze_phase_a_baseline(cfg, device_name=args.device, n=args.probe_n)
    baseline = json.loads(baseline_path.read_text())
    ckpt = args.ckpt
    if ckpt is None:
        b0 = _ROOT / cfg["out_dir"] / "b0" / "ckpt.pt"
        grpo = _ROOT / cfg["out_dir"] / "grpo" / "checkpoint_latest.pt"
        ckpt = grpo if grpo.exists() else b0 if b0.exists() else _ROOT / cfg["phase_a_ckpt"]
    n_seeds = int(args.n_seeds if args.n_seeds is not None else cfg.get("final_n_seeds", 20))
    evaluate_checkpoint(
        cfg,
        ckpt,
        baseline,
        device_name=args.device,
        n_seeds=n_seeds,
        label=args.label,
    )


if __name__ == "__main__":
    main()
