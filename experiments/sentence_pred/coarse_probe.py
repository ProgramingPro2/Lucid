#!/usr/bin/env python3
"""Cheap probes for coarse-steer (Fable §7).

P0 — paragraph/topic centroid stability (CPU, numpy).
P2 — concept-space anchored rollout, no writer (GPU).
P1 — FiLM legibility of pooled/slerped concepts (GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.sentence_pred.data import SentenceCorpus, load_sentence_corpus  # noqa: E402

DEFAULT_CORPUS = "artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1"
DEFAULT_OUT = "artifacts/sentence_pred/coarse_steer/probes_P0.json"
DEFAULT_MDN = "artifacts/sentence_pred/mdn.pt"
DEFAULT_WRITER = "artifacts/writer_p25/writer_two_channel.pt"
DEFAULT_JEPA = "artifacts/jepa_repair_1m/jepa/joint.pt"
SEEDS_PATH = "artifacts/sentence_pred/forward_gen/seeds.json"


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x)) + eps
        return (x / n).astype(np.float32)
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / denom).astype(np.float32)


def _mean_pool_normalize(vecs: np.ndarray) -> np.ndarray:
    return _l2_normalize(vecs.mean(axis=0))


def _paragraph_centroids(doc_vecs: np.ndarray, g: int) -> np.ndarray:
    """Non-overlapping G-sentence mean-pooled, L2-normalized centroids."""
    n = int(doc_vecs.shape[0])
    n_blocks = n // g
    if n_blocks == 0:
        return np.zeros((0, doc_vecs.shape[1]), dtype=np.float32)
    blocks = doc_vecs[: n_blocks * g].reshape(n_blocks, g, -1).mean(axis=1)
    return _l2_normalize(blocks)


def _pairwise_mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine between aligned rows of unit vectors a, b."""
    if a.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.sum(a * b, axis=-1)))


def _cross_pair_mean_cos(
    groups: list[np.ndarray],
    rng: np.random.RandomState,
    n_pairs: int,
) -> float:
    """Mean cosine of random pairs drawn from different groups."""
    eligible = [g for g in groups if g.shape[0] > 0]
    if len(eligible) < 2:
        return float("nan")
    # Flatten with group ids for sampling
    parts = []
    gids = []
    for i, g in enumerate(eligible):
        parts.append(g)
        gids.append(np.full(g.shape[0], i, dtype=np.int64))
    bank = np.concatenate(parts, axis=0)
    group_of = np.concatenate(gids, axis=0)
    n = bank.shape[0]
    if n < 2:
        return float("nan")

    cos_vals: list[float] = []
    attempts = 0
    max_attempts = n_pairs * 20
    while len(cos_vals) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.randint(0, n, size=2)
        if i == j or group_of[i] == group_of[j]:
            continue
        cos_vals.append(float(np.dot(bank[i], bank[j])))
    if not cos_vals:
        return float("nan")
    return float(np.mean(cos_vals))


def _metric_a_paragraph(
    corpus: SentenceCorpus,
    held_ids: np.ndarray,
    g: int,
    rng: np.random.RandomState,
    n_cross_pairs: int = 20_000,
) -> dict:
    para_groups: list[np.ndarray] = []
    consec_cos: list[float] = []

    for doc_id in held_ids:
        s, e = corpus.doc_slice(int(doc_id))
        doc_vecs = corpus.sent_vecs[s:e]
        paras = _paragraph_centroids(doc_vecs, g)
        if paras.shape[0] == 0:
            continue
        para_groups.append(paras)
        if paras.shape[0] >= 2:
            consec_cos.extend(
                float(np.dot(paras[i], paras[i + 1])) for i in range(paras.shape[0] - 1)
            )

    consec_para_cos = float(np.mean(consec_cos)) if consec_cos else float("nan")
    cross_para_cos = _cross_pair_mean_cos(para_groups, rng, n_cross_pairs)
    delta = consec_para_cos - cross_para_cos
    return {
        "G": g,
        "n_consec_pairs": len(consec_cos),
        "n_para_groups": len(para_groups),
        "consec_para_cos": consec_para_cos,
        "cross_para_cos": cross_para_cos,
        "delta": float(delta),
        "pass": bool(delta >= 0.10),
        "kill_band": bool(delta < 0.05),
    }


def _metric_b_doc_half(
    corpus: SentenceCorpus,
    held_ids: np.ndarray,
    rng: np.random.RandomState,
    n_cross_pairs: int = 20_000,
    min_sents: int = 10,
) -> dict:
    half_cos: list[float] = []
    first_halves: list[np.ndarray] = []

    for doc_id in held_ids:
        s, e = corpus.doc_slice(int(doc_id))
        doc_vecs = corpus.sent_vecs[s:e]
        n = int(doc_vecs.shape[0])
        if n < min_sents:
            continue
        mid = n // 2
        c1 = _mean_pool_normalize(doc_vecs[:mid])
        c2 = _mean_pool_normalize(doc_vecs[mid:])
        half_cos.append(float(np.dot(c1, c2)))
        first_halves.append(c1.reshape(1, -1))

    doc_half_cos = float(np.mean(half_cos)) if half_cos else float("nan")
    cross_doc_centroid_cos = _cross_pair_mean_cos(first_halves, rng, n_cross_pairs)
    delta = doc_half_cos - cross_doc_centroid_cos
    return {
        "n_docs": len(half_cos),
        "doc_half_cos": doc_half_cos,
        "cross_doc_centroid_cos": cross_doc_centroid_cos,
        "delta": float(delta),
        "pass": bool(delta >= 0.15),
        "kill_band": bool(delta < 0.10),
    }


def _metric_c_delta_cos(
    corpus: SentenceCorpus,
    held_ids: np.ndarray,
) -> dict:
    """Mean cos(delta_t, delta_{t+1}) where delta_t = z_{t+1} - z_t (raw diffs)."""
    vals: list[float] = []
    for doc_id in held_ids:
        s, e = corpus.doc_slice(int(doc_id))
        z = corpus.sent_vecs[s:e]
        if z.shape[0] < 3:
            continue
        d0 = z[1:-1] - z[:-2]
        d1 = z[2:] - z[1:-1]
        # Normalize deltas for cosine (direction of change)
        d0n = _l2_normalize(d0)
        d1n = _l2_normalize(d1)
        vals.extend(np.sum(d0n * d1n, axis=-1).tolist())
    mean_cos = float(np.mean(vals)) if vals else float("nan")
    return {
        "n_triples": len(vals),
        "delta_cos": mean_cos,
        "expect_lt": -0.2,
        "confirms_mean_reversion": bool(mean_cos < -0.2),
    }


def run_p0(
    corpus: SentenceCorpus,
    max_held_docs: int = 500,
    seed: int = 1234,
) -> dict:
    held = np.asarray(corpus.held_doc_ids, dtype=np.int64)
    if held.shape[0] > max_held_docs:
        rng_pick = np.random.RandomState(seed)
        held = np.sort(rng_pick.choice(held, size=max_held_docs, replace=False))
    rng = np.random.RandomState(seed)

    a5 = _metric_a_paragraph(corpus, held, g=5, rng=rng)
    a10 = _metric_a_paragraph(corpus, held, g=10, rng=np.random.RandomState(seed + 1))
    b = _metric_b_doc_half(corpus, held, rng=np.random.RandomState(seed + 2))
    c = _metric_c_delta_cos(corpus, held)

    delta_a = float(a5["delta"])
    delta_b = float(b["delta"])
    go = bool(delta_a >= 0.10 or delta_b >= 0.15)
    kill = bool(delta_a < 0.05 and delta_b < 0.10)
    if kill:
        overall = "KILL"
    elif go:
        overall = "GO"
    else:
        overall = "MARGINAL"

    return {
        "probe": "P0",
        "corpus": None,  # filled by caller
        "n_held_docs_used": int(held.shape[0]),
        "seed": seed,
        "a_G5": a5,
        "a_G10": a10,
        "b_doc_half": b,
        "c_delta": c,
        "verdicts": {
            "a_pass": bool(a5["pass"]),
            "b_pass": bool(b["pass"]),
            "a_delta": delta_a,
            "b_delta": delta_b,
            "overall": overall,
            "rules": {
                "a_pass_if": "consec_para_cos - cross_para_cos >= +0.10",
                "b_pass_if": "doc_half_cos - cross_doc_centroid_cos >= +0.15",
                "kill_if": "(a) < +0.05 AND (b) < +0.10",
                "go_if": "(a) >= +0.10 OR (b) >= +0.15",
            },
        },
    }


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def print_p0_table(results: dict) -> None:
    a5 = results["a_G5"]
    a10 = results["a_G10"]
    b = results["b_doc_half"]
    c = results["c_delta"]
    v = results["verdicts"]

    print("=" * 72)
    print("P0 — Coarse stability on repaired codec (held docs)")
    print(f"held docs used: {results['n_held_docs_used']}  seed={results['seed']}")
    print("=" * 72)
    print()
    print(f"{'metric':<42} {'value':>10}  {'note'}")
    print("-" * 72)
    print(
        f"{'(a) consec_para_cos G=5':<42} {_fmt(a5['consec_para_cos']):>10}  "
        f"n_pairs={a5['n_consec_pairs']}"
    )
    print(
        f"{'(a) cross_para_cos G=5':<42} {_fmt(a5['cross_para_cos']):>10}  "
        f"control"
    )
    print(
        f"{'(a) delta G=5 (consec-cross)':<42} {_fmt(a5['delta']):>10}  "
        f"{'PASS' if a5['pass'] else 'KILL'}  (need >= +0.10)"
    )
    print(
        f"{'(a) consec_para_cos G=10':<42} {_fmt(a10['consec_para_cos']):>10}  "
        f"n_pairs={a10['n_consec_pairs']}  (context)"
    )
    print(
        f"{'(a) cross_para_cos G=10':<42} {_fmt(a10['cross_para_cos']):>10}  "
        f"control (context)"
    )
    print(
        f"{'(a) delta G=10 (consec-cross)':<42} {_fmt(a10['delta']):>10}  "
        f"context only"
    )
    print(
        f"{'(b) doc_half_cos':<42} {_fmt(b['doc_half_cos']):>10}  "
        f"n_docs={b['n_docs']}"
    )
    print(
        f"{'(b) cross_doc_centroid_cos':<42} {_fmt(b['cross_doc_centroid_cos']):>10}  "
        f"control"
    )
    print(
        f"{'(b) delta (half-cross)':<42} {_fmt(b['delta']):>10}  "
        f"{'PASS' if b['pass'] else 'KILL'}  (need >= +0.15)"
    )
    print(
        f"{'(c) consec sentence-delta cos':<42} {_fmt(c['delta_cos']):>10}  "
        f"expect < -0.2; n={c['n_triples']}"
    )
    print("-" * 72)
    print(
        f"Overall verdict: {v['overall']}  "
        f"(GO if a>=+0.10 or b>=+0.15; KILL if a<+0.05 and b<+0.10)"
    )
    print(
        f"  a_pass={v['a_pass']} (delta={_fmt(v['a_delta'])})  "
        f"b_pass={v['b_pass']} (delta={_fmt(v['b_delta'])})"
    )
    print("=" * 72)


# ---------------------------------------------------------------------------
# P2 — concept-space anchored rollout (no writer / no re-encode)
# ---------------------------------------------------------------------------


def _l2_torch(x):
    import torch
    import torch.nn.functional as F

    return F.normalize(x, dim=-1)


def run_p2(
    corpus: SentenceCorpus,
    mdn_path: str,
    seeds_path: str,
    device: str = "auto",
    steps: int = 8,
    n_seeds: int = 20,
    context_len: int = 8,
) -> dict:
    """Closed loop fed:=concept; flat vs anchored; median cos(concept_k, a0) at k=8."""
    import torch

    from experiments.sentence_pred.coarse_steer import (
        CoarseSteerConfig,
        anchored_sample,
        slerp,
    )
    from experiments.sentence_pred.forward_gen import load_mdn_predictor
    from experiments.sentence_pred.heads import sample_vmf

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = load_mdn_predictor(mdn_path, device=device)
    seeds_file = Path(seeds_path)
    if seeds_file.is_file():
        seed_texts = json.loads(seeds_file.read_text(encoding="utf-8"))[:n_seeds]
    else:
        from experiments.sentence_pred.forward_gen import _pick_seeds

        seed_texts = _pick_seeds(corpus, n_seeds, seed=1234)

    # Map seed texts → corpus vectors when possible; else encode via nearest sentence
    # Prefer exact string match in corpus; fallback to using held mid-doc vectors.
    text_to_vec = {}
    for i, t in enumerate(corpus.sentences):
        text_to_vec[t.strip()] = corpus.sent_vecs[i]

    seed_vecs = []
    for t in seed_texts:
        key = t.strip()
        if key in text_to_vec:
            seed_vecs.append(_l2_normalize(text_to_vec[key]))
        else:
            # fuzzy: use first held mid-doc sentence as stand-in (should not happen)
            raise RuntimeError(f"Seed text not found in corpus: {t[:80]!r}")

    cfg_anchored = CoarseSteerConfig(eta=0.35, beta=8.0, anchor_lambda=0.10, n_cand=8)
    flat_final: list[float] = []
    anc_final: list[float] = []
    flat_curves: list[list[float]] = []
    anc_curves: list[list[float]] = []

    def _rollout(seed_vec: np.ndarray, anchored: bool) -> list[float]:
        device_t = next(predictor.parameters()).device
        a0 = torch.from_numpy(seed_vec[None, :].astype(np.float32)).to(device_t)
        a = a0.clone()
        history = [seed_vec.astype(np.float32)]
        cos_to_a0: list[float] = []
        for step in range(steps):
            window = history[-context_len:]
            ctx = (
                torch.from_numpy(np.stack(window).astype(np.float32))
                .unsqueeze(0)
                .to(device_t)
            )
            pad = torch.zeros(1, ctx.size(1), dtype=torch.bool, device=device_t)
            if anchored:
                samp = anchored_sample(predictor, ctx, pad, a, cfg_anchored)
                concept = samp["concept"]
                # EMA update with concept (no writer): u = concept
                lam = 0.30 if step < 2 else float(cfg_anchored.anchor_lambda)
                a = _l2_torch((1.0 - lam) * a + lam * concept)
            else:
                # flat MDN sample
                sample = predictor.sample(ctx, pad, n_samples=1, temperature=1.0)
                concept = _l2_torch(sample[:, 0, :])
            c_np = concept.detach().float().cpu().numpy()[0]
            cos_to_a0.append(float(np.dot(_l2_normalize(c_np), _l2_normalize(seed_vec))))
            history.append(_l2_normalize(c_np))
        return cos_to_a0

    for i, sv in enumerate(seed_vecs):
        print(f"[P2] seed {i + 1}/{len(seed_vecs)}", flush=True)
        fc = _rollout(sv, anchored=False)
        ac = _rollout(sv, anchored=True)
        flat_curves.append(fc)
        anc_curves.append(ac)
        flat_final.append(fc[-1])
        anc_final.append(ac[-1])

    med_flat = float(np.median(flat_final))
    med_anc = float(np.median(anc_final))
    delta = med_anc - med_flat
    passed = bool(med_anc >= med_flat + 0.20 and med_anc >= 0.50)
    kill = bool(med_anc <= med_flat + 0.05)

    return {
        "probe": "P2",
        "n_seeds": len(seed_vecs),
        "steps": steps,
        "device": device,
        "median_cos_k8_flat": med_flat,
        "median_cos_k8_anchored": med_anc,
        "delta_anchored_minus_flat": float(delta),
        "mean_cos_k8_flat": float(np.mean(flat_final)),
        "mean_cos_k8_anchored": float(np.mean(anc_final)),
        "curve_mean_flat": [float(np.mean([c[k] for c in flat_curves])) for k in range(steps)],
        "curve_mean_anchored": [
            float(np.mean([c[k] for c in anc_curves])) for k in range(steps)
        ],
        "verdict": "PASS" if passed else ("KILL" if kill else "FAIL"),
        "pass": passed,
        "kill": kill,
        "rules": {
            "pass_if": "anchored_median >= flat_median + 0.20 AND anchored_median >= 0.50",
            "kill_if": "anchored_median <= flat_median + 0.05",
        },
    }


# ---------------------------------------------------------------------------
# P1 — FiLM legibility of pooled / slerped concepts
# ---------------------------------------------------------------------------


def run_p1(
    corpus: SentenceCorpus,
    mdn_path: str,
    writer_path: str,
    jepa_path: str,
    device: str = "auto",
    n_items: int = 200,
    g: int = 5,
    seed: int = 1234,
) -> dict:
    """Render with same-z / pooled G=5 / slerp(centroid, MDN, 0.35); MiniLM vs paragraph."""
    import torch

    from conceptpoc.codec.jepa_codec import JepaCodec
    from conceptpoc.jepa.tokenizer import Tokenizer
    from conceptpoc.persistence import load_joint, load_writer
    from experiments.sentence_pred.coarse_steer import slerp
    from experiments.sentence_pred.forward_gen import (
        _l2,
        _load_minilm,
        _render_sentence,
        load_mdn_predictor,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = load_mdn_predictor(mdn_path, device=device)
    writer = load_writer(writer_path, device=device)
    model, tok = load_joint(jepa_path, device=device)
    tokenizer = (
        tok if getattr(tok, "vocab_size", 0) == writer.vocab_size else Tokenizer("gpt2")
    )
    minilm = _load_minilm()

    held = np.asarray(corpus.held_doc_ids, dtype=np.int64)
    rng = np.random.RandomState(seed)
    items: list[dict] = []
    for doc_id in held:
        s, e = corpus.doc_slice(int(doc_id))
        n = e - s
        if n < g + 1:
            continue
        # non-overlapping first G-block that has a previous sentence
        # use sentences [1:1+G] so prefix = sentences[0]
        if n < g + 1:
            continue
        start = 1
        idxs = list(range(s + start, s + start + g))
        vecs = corpus.sent_vecs[idxs]
        texts = [corpus.sentences[i] for i in idxs]
        prefix = corpus.sentences[s + start - 1]
        para_text = " ".join(texts)
        target_i = idxs[g // 2]  # mid sentence of block
        items.append(
            {
                "prefix": prefix,
                "para_text": para_text,
                "z_same": _l2_normalize(corpus.sent_vecs[target_i]),
                "centroid": _mean_pool_normalize(vecs),
                "ctx_vecs": corpus.sent_vecs[max(s, target_i - 8) : target_i],
            }
        )
    rng.shuffle(items)
    items = items[:n_items]
    if len(items) < n_items:
        print(f"[P1] warning: only {len(items)} paragraph items available", flush=True)

    scores = {"same": [], "pooled": [], "slerp": []}
    for i, it in enumerate(items):
        if (i + 1) % 25 == 0:
            print(f"[P1] item {i + 1}/{len(items)}", flush=True)
        para_emb = _l2(np.asarray(minilm.encode([it["para_text"]]), dtype=np.float32)[0])
        # MDN sample from context preceding target
        ctx_np = it["ctx_vecs"]
        if ctx_np.shape[0] == 0:
            ctx_np = it["z_same"][None, :]
        ctx = torch.from_numpy(ctx_np.astype(np.float32)).unsqueeze(0).to(device)
        pad = torch.zeros(1, ctx.size(1), dtype=torch.bool, device=device)
        with torch.no_grad():
            samp = predictor.sample(ctx, pad, n_samples=1, temperature=1.0)
            mdn_s = samp[:, 0, :]
            centroid_t = torch.from_numpy(it["centroid"][None, :]).to(device)
            slerp_c = slerp(centroid_t, mdn_s, 0.35)

        concepts = {
            "same": torch.from_numpy(it["z_same"][None, :]).to(device),
            "pooled": torch.from_numpy(it["centroid"][None, :]).to(device),
            "slerp": slerp_c,
        }
        for name, concept in concepts.items():
            text = _render_sentence(
                writer,
                tokenizer,
                concept,
                it["prefix"],
                temperature=0.8,
                top_k=50,
                max_tokens=96,
            )
            rend_emb = _l2(np.asarray(minilm.encode([text]), dtype=np.float32)[0])
            scores[name].append(float(np.dot(rend_emb, para_emb)))

    means = {k: float(np.mean(v)) if v else float("nan") for k, v in scores.items()}
    medians = {k: float(np.median(v)) if v else float("nan") for k, v in scores.items()}
    pooled_pass = bool(means["pooled"] >= 0.25)
    pooled_kill = bool(means["pooled"] <= 0.10)

    return {
        "probe": "P1",
        "n_items": len(items),
        "G": g,
        "device": device,
        "mean_cos": means,
        "median_cos": medians,
        "verdict": "PASS" if pooled_pass else ("KILL" if pooled_kill else "FAIL"),
        "pass": pooled_pass,
        "kill": pooled_kill,
        "rules": {
            "pass_if": "pooled mean MiniLM cos(render, paragraph) >= 0.25",
            "kill_if": "pooled <= 0.10",
            "reference_note": "same-sentence concept-only renderability historically ~0.395",
        },
        "recommendation": (
            "anchor_source=fed, eta default ok"
            if pooled_pass
            else (
                "prefer anchor_source=concept and/or eta>=0.5; consider v2b dual-FiLM"
                if pooled_kill
                else "pooled marginal; ablate anchor_source=concept"
            )
        ),
    }


def print_p2(results: dict) -> None:
    print("=" * 72)
    print("P2 — Concept-space anchored rollout (no writer)")
    print(f"seeds={results['n_seeds']} steps={results['steps']} device={results['device']}")
    print("=" * 72)
    print(
        f"  median cos(concept_8, a0) flat     = {_fmt(results['median_cos_k8_flat'])}"
    )
    print(
        f"  median cos(concept_8, a0) anchored = {_fmt(results['median_cos_k8_anchored'])}"
    )
    print(
        f"  delta (anchored - flat)            = {_fmt(results['delta_anchored_minus_flat'])}"
    )
    print(f"  verdict: {results['verdict']}  (PASS if anc>=flat+0.20 and anc>=0.50)")
    print("=" * 72)


def print_p1(results: dict) -> None:
    print("=" * 72)
    print("P1 — FiLM legibility of pooled/slerped concepts")
    print(f"n_items={results['n_items']} G={results['G']} device={results['device']}")
    print("=" * 72)
    for name in ("same", "pooled", "slerp"):
        print(
            f"  {name:<8} mean={_fmt(results['mean_cos'][name])}  "
            f"median={_fmt(results['median_cos'][name])}"
        )
    print(f"  verdict: {results['verdict']}  (PASS if pooled>=0.25; KILL if <=0.10)")
    print(f"  recommendation: {results['recommendation']}")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Coarse-steer probes (Fable §7)")
    p.add_argument("--probe", required=True, choices=["P0", "P1", "P2"], help="Probe id")
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="SentenceCorpus cache dir")
    p.add_argument("--out", default=None, help="JSON output path")
    p.add_argument("--max-held-docs", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mdn", default=DEFAULT_MDN)
    p.add_argument("--writer", default=DEFAULT_WRITER)
    p.add_argument("--jepa", default=DEFAULT_JEPA)
    p.add_argument("--seeds-json", default=SEEDS_PATH)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--n-items", type=int, default=200)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    corpus_path = Path(args.corpus)
    print(f"Loading corpus from {corpus_path} ...", flush=True)
    corpus = load_sentence_corpus(corpus_path)
    print(
        f"  n_docs={corpus.n_docs} n_sent={corpus.n_sentences} "
        f"held={len(corpus.held_doc_ids)}",
        flush=True,
    )

    if args.probe == "P0":
        results = run_p0(corpus, max_held_docs=args.max_held_docs, seed=args.seed)
        results["corpus"] = str(corpus_path)
        out_path = Path(args.out or "artifacts/sentence_pred/coarse_steer/probes_P0.json")
        print_p0_table(results)
    elif args.probe == "P2":
        results = run_p2(
            corpus,
            mdn_path=args.mdn,
            seeds_path=args.seeds_json,
            device=args.device,
            steps=args.steps,
            n_seeds=args.n_seeds,
        )
        out_path = Path(args.out or "artifacts/sentence_pred/coarse_steer/probes_P12.json")
        # merge with existing P1 if present
        if out_path.is_file():
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(prev, dict) and "P1" in prev:
                    results = {"P1": prev["P1"], "P2": results}
                elif isinstance(prev, dict) and prev.get("probe") == "P1":
                    results = {"P1": prev, "P2": results}
                else:
                    results = {"P2": results}
            except Exception:
                results = {"P2": results}
        else:
            results = {"P2": results}
        print_p2(results["P2"] if "P2" in results else results)
    else:  # P1
        results = run_p1(
            corpus,
            mdn_path=args.mdn,
            writer_path=args.writer,
            jepa_path=args.jepa,
            device=args.device,
            n_items=args.n_items,
            seed=args.seed,
        )
        out_path = Path(args.out or "artifacts/sentence_pred/coarse_steer/probes_P12.json")
        if out_path.is_file():
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(prev, dict) and "P2" in prev:
                    results = {"P1": results, "P2": prev["P2"]}
                elif isinstance(prev, dict) and prev.get("probe") == "P2":
                    results = {"P1": results, "P2": prev}
                else:
                    results = {"P1": results}
            except Exception:
                results = {"P1": results}
        else:
            results = {"P1": results}
        print_p1(results["P1"] if "P1" in results else results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
