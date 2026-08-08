"""Coarse-steer anchored sampler (Fable §3): training-free two-timescale forward gen.

Reuses mdn.pt + writer_two_channel.pt + repaired codec. No retraining.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

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

from conceptpoc.eval.harness import distinct_2  # noqa: E402
from conceptpoc.persistence import load_joint, load_writer  # noqa: E402
from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.jepa.tokenizer import Tokenizer  # noqa: E402

from experiments.sentence_pred.data import load_sentence_corpus  # noqa: E402
from experiments.sentence_pred.evalsuite import (  # noqa: E402
    centering,
    load_split,
    novelty_scores,
    reference_distribution,
)
from experiments.sentence_pred.forward_gen import (  # noqa: E402
    ForwardGenResult,
    _agg_retention_curve,
    _encode_one,
    _l2,
    _load_minilm,
    _novelty_summary,
    _pairwise_text_divergence,
    _pick_seeds,
    _render_sentence,
    _shuffled_retention_curve,
    generate_document_twochannel,
    load_mdn_predictor,
)
from experiments.sentence_pred.heads import sample_vmf  # noqa: E402

DEFAULT_OUT = _ROOT / "artifacts/sentence_pred/coarse_steer"
SEEDS_PATH = _ROOT / "artifacts/sentence_pred/forward_gen/seeds.json"
MAX_NEW_TOKENS = 96
_WORD_RE = re.compile(r"[a-z0-9']+", re.I)


@dataclass
class CoarseSteerConfig:
    anchor_mode: Literal["ema", "paragraph"] = "ema"
    anchor_lambda: float = 0.10
    anchor_every: int = 4
    anchor_source: Literal["fed", "concept"] = "fed"
    beta: float = 8.0
    n_cand: int = 8
    eta: float = 0.35
    use_novelty_guard: bool = False
    steer_temp: float = 1.0
    context_len: int = 8


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation. a,b unit; returns unit (same shape as a)."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    near = sin_omega.abs() < 1e-4
    out = (torch.sin((1.0 - t) * omega) * a + torch.sin(t * omega) * b) / sin_omega.clamp_min(
        1e-8
    )
    lerp = F.normalize((1.0 - t) * a + t * b, dim=-1)
    return torch.where(near, lerp, out)


@torch.no_grad()
def anchored_sample(
    predictor,
    ctx: torch.Tensor,
    pad: torch.Tensor,
    anchor: torch.Tensor,
    cfg: CoarseSteerConfig,
    bank=None,
    ref_dist: dict | None = None,
    mu_bank: np.ndarray | None = None,
    bank_c: np.ndarray | None = None,
) -> dict[str, Any]:
    """Core sampler (§3.2). Returns concept (1,D) plus instrumentation fields."""
    pi_logits, mu, kappa = predictor.mixture_params_last(ctx, pad)
    kappa = kappa / float(cfg.steer_temp)
    a = F.normalize(anchor, dim=-1)
    if a.dim() == 1:
        a = a.unsqueeze(0)

    align = (mu * a.unsqueeze(1)).sum(dim=-1)  # (B, K)
    flat_argmax = pi_logits.argmax(dim=-1)
    tilted = torch.log_softmax(pi_logits, dim=-1) + float(cfg.beta) * align
    pi_t = torch.softmax(tilted, dim=-1)
    tilted_argmax = pi_t.argmax(dim=-1)
    tilt_moved = bool((tilted_argmax != flat_argmax).any().item())

    n_cand = int(cfg.n_cand)
    comps = torch.multinomial(pi_t, n_cand, replacement=True)  # (B, n_cand)
    # v1 batch size is 1 in generation; keep general for B>=1 via first row API match
    b0 = 0
    mu_sel = mu[b0, comps[b0]]  # (n_cand, D)
    k_sel = kappa[b0, comps[b0]]  # (n_cand,)
    s = sample_vmf(mu_sel, k_sel)  # (n_cand, D)
    score = (s * a[b0]).sum(dim=-1)  # (n_cand,)

    keep = torch.ones(n_cand, dtype=torch.bool, device=s.device)
    if (
        cfg.use_novelty_guard
        and mu_bank is not None
        and bank_c is not None
        and ref_dist is not None
    ):
        s_np = s.detach().float().cpu().numpy()
        n_scores = novelty_scores(s_np, bank_c, mu_bank)
        med = float(ref_dist["median"])
        keep = torch.from_numpy(n_scores <= med).to(device=s.device)
        if not bool(keep.any()):
            keep = torch.ones(n_cand, dtype=torch.bool, device=s.device)

    score_masked = score.clone()
    score_masked[~keep] = -1e9
    best_i = int(score_masked.argmax().item())
    s_star = s[best_i : best_i + 1]  # (1, D)
    concept = slerp(a[b0 : b0 + 1], s_star, float(cfg.eta))

    return {
        "concept": concept,
        "s_star": s_star,
        "chosen_component": int(comps[b0, best_i].item()),
        "tilt_moved_argmax": tilt_moved,
        "sample_vs_anchor_cos": float((s_star * a[b0]).sum().item()),
        "concept_vs_anchor_cos": float((concept * a[b0]).sum().item()),
        "flat_argmax": int(flat_argmax[b0].item()),
        "tilted_argmax": int(tilted_argmax[b0].item()),
    }


@torch.no_grad()
def generate_document_coarse(
    seed_text: str,
    predictor_mdn,
    writer,
    tokenizer,
    codec,
    steps: int = 8,
    cfg: CoarseSteerConfig | None = None,
    train_bank: np.ndarray | None = None,
    ref_dist: dict | None = None,
    minilm=None,
    writer_temperature: float = 0.8,
    writer_top_k: int = 50,
    max_tokens: int = MAX_NEW_TOKENS,
) -> ForwardGenResult:
    """Anchored coarse-steer rollout; mirrors generate_document_twochannel return type."""
    cfg = cfg or CoarseSteerConfig()
    device = next(predictor_mdn.parameters()).device
    seed_vec = _encode_one(codec, seed_text)
    history_vecs = [seed_vec]
    history_texts = [seed_text]
    a0 = torch.from_numpy(seed_vec[None, :]).to(device)
    a = F.normalize(a0.clone(), dim=-1)
    recent_u: list[torch.Tensor] = []
    instr: list[dict[str, Any]] = []

    mu_bank = bank_c = None
    if train_bank is not None and len(train_bank):
        mu_bank, bank_c = centering(train_bank)

    seed_minilm = None
    if minilm is not None:
        seed_minilm = _l2(np.asarray(minilm.encode([seed_text]), dtype=np.float32)[0])
    prev_minilm = seed_minilm

    for step in range(int(steps)):
        window = history_vecs[-int(cfg.context_len) :]
        ctx = (
            torch.from_numpy(np.stack(window).astype(np.float32))
            .unsqueeze(0)
            .to(device)
        )
        pad = torch.zeros(1, ctx.size(1), dtype=torch.bool, device=device)
        z_t = torch.from_numpy(history_vecs[-1][None, :]).to(device)

        samp = anchored_sample(
            predictor_mdn,
            ctx,
            pad,
            a,
            cfg,
            bank=train_bank,
            ref_dist=ref_dist,
            mu_bank=mu_bank,
            bank_c=bank_c,
        )
        concept = samp["concept"]
        s_star = samp["s_star"]
        concept_np = concept.detach().float().cpu().numpy()[0]
        s_star_np = s_star.detach().float().cpu().numpy()[0]

        text = _render_sentence(
            writer,
            tokenizer,
            concept,
            history_texts[-1],
            temperature=writer_temperature,
            top_k=writer_top_k,
            max_tokens=max_tokens,
        )
        fed = _encode_one(codec, text) if codec is not None else _l2(concept_np)
        fed_t = torch.from_numpy(fed[None, :]).to(device)

        u = fed_t if cfg.anchor_source == "fed" else concept
        u = F.normalize(u, dim=-1)
        recent_u.append(u.detach())
        if cfg.anchor_mode == "ema":
            lam = float(cfg.anchor_lambda)
            # Warm-start early steps if seed residual is noisy (spec §7 note)
            if step < 2 and lam < 0.3:
                lam_eff = max(lam, 0.3)
            else:
                lam_eff = lam
            a = F.normalize((1.0 - lam_eff) * a + lam_eff * u, dim=-1)
        else:  # paragraph
            if (step + 1) % int(cfg.anchor_every) == 0:
                block = torch.cat(recent_u[-int(cfg.anchor_every) :], dim=0)
                a = F.normalize(block.mean(dim=0, keepdim=True), dim=-1)

        row: dict[str, Any] = {
            "step": step,
            "text": text,
            "steer_mode": "coarse",
            "steer_temp": float(cfg.steer_temp),
            "anchor_vs_seed_cos": float((a * a0).sum().item()),
            "concept_vs_anchor_cos": float(samp["concept_vs_anchor_cos"]),
            "sample_vs_anchor_cos": float(samp["sample_vs_anchor_cos"]),
            "fed_vs_concept_cos": float(
                np.dot(_l2(fed), _l2(concept_np))
            ),
            "chosen_component": int(samp["chosen_component"]),
            "tilt_moved_argmax": bool(samp["tilt_moved_argmax"]),
            "eta": float(cfg.eta),
            "beta": float(cfg.beta),
            "anchor_lambda": float(cfg.anchor_lambda),
            "anchor_source": cfg.anchor_source,
            "anchor_mode": cfg.anchor_mode,
        }
        if mu_bank is not None and bank_c is not None:
            n_x = float(novelty_scores(concept_np[None, :], bank_c, mu_bank)[0])
            row["n_x_steer"] = n_x
            row["n_x_raw_sample"] = float(
                novelty_scores(s_star_np[None, :], bank_c, mu_bank)[0]
            )
            if ref_dist is not None:
                row["novel_vs_median"] = bool(n_x <= float(ref_dist["median"]))
                row["copy_vs_p99"] = bool(n_x >= float(ref_dist["p99"]))
            row["n_x_fed"] = float(novelty_scores(fed[None, :], bank_c, mu_bank)[0])
        row["consecutive_cos"] = float(np.dot(_l2(fed), _l2(history_vecs[-1])))
        row["steer_vs_zt_cos"] = float(np.dot(_l2(concept_np), _l2(history_vecs[-1])))

        if minilm is not None:
            mv = _l2(np.asarray(minilm.encode([text]), dtype=np.float32)[0])
            if seed_minilm is not None:
                row["minilm_retention"] = float(np.dot(mv, seed_minilm))
            if prev_minilm is not None:
                row["minilm_consecutive"] = float(np.dot(mv, prev_minilm))
            prev_minilm = mv

        instr.append(row)
        history_vecs.append(fed)
        history_texts.append(text)

    return ForwardGenResult(sentences=history_texts[1:], instrumentation=instr)


def _arm_sane_band(rollouts: list[ForwardGenResult]) -> dict[str, float]:
    cons: list[float] = []
    for r in rollouts:
        for row in r.instrumentation:
            if "minilm_consecutive" in row:
                cons.append(float(row["minilm_consecutive"]))
    arr = np.asarray(cons, dtype=np.float32)
    texts = [" ".join(r.sentences) for r in rollouts]
    d2 = [distinct_2(t) for t in texts]
    return {
        "mean_consec": float(arr.mean()) if arr.size else float("nan"),
        "p90_consec": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        "jump_rate": float((arr < 0.05).mean()) if arr.size else float("nan"),
        "distinct2": float(np.mean(d2)) if d2 else float("nan"),
    }


def _fed8_vs_a0(rollouts: list[ForwardGenResult], seeds: list[str], codec) -> float:
    """Median cos(fed_8, a_0) where a_0 = encode(seed)."""
    vals = []
    for seed, r in zip(seeds, rollouts):
        if len(r.sentences) < 8:
            continue
        a0 = _encode_one(codec, seed)
        # last instrumentation consecutive uses fed; recover fed via encode of last sent
        fed8 = _encode_one(codec, r.sentences[7])
        vals.append(float(np.dot(_l2(fed8), _l2(a0))))
    return float(np.median(vals)) if vals else float("nan")


def _load_assets(args: argparse.Namespace):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_mdn_predictor(args.mdn, device=device)
    writer = load_writer(args.writer, device=device)
    model, tok = load_joint(args.jepa, device=device)
    tokenizer = tok if getattr(tok, "vocab_size", 0) == writer.vocab_size else Tokenizer("gpt2")
    codec = JepaCodec(model, tok, max_tokens=128, device=device)
    corpus = load_sentence_corpus(args.corpus)
    bank_train, held_vectors, _, _, _ = load_split(corpus, "")
    rng = np.random.RandomState(1234)
    if bank_train.shape[0] > args.bank_cap:
        idx = rng.choice(bank_train.shape[0], size=args.bank_cap, replace=False)
        bank = bank_train[idx]
    else:
        bank = bank_train
    if held_vectors.shape[0] > args.held_cap:
        hidx = rng.choice(held_vectors.shape[0], size=args.held_cap, replace=False)
        held = held_vectors[hidx]
    else:
        held = held_vectors
    mu, bank_c = centering(bank)
    ref = reference_distribution(held, bank_c, mu)
    minilm = _load_minilm()
    return {
        "device": device,
        "predictor": predictor,
        "writer": writer,
        "tokenizer": tokenizer,
        "codec": codec,
        "corpus": corpus,
        "bank": bank,
        "ref": ref,
        "minilm": minilm,
    }


def _cfg_from_args(args: argparse.Namespace) -> CoarseSteerConfig:
    return CoarseSteerConfig(
        anchor_mode=args.anchor_mode,
        anchor_lambda=float(args.anchor_lambda),
        anchor_every=int(args.anchor_every),
        anchor_source=args.anchor_source,
        beta=float(args.beta),
        n_cand=int(args.n_cand),
        eta=float(args.eta),
        use_novelty_guard=bool(args.use_novelty_guard),
        steer_temp=float(args.steer_temp),
        context_len=int(args.context_len),
    )


def _sweep_grid() -> list[CoarseSteerConfig]:
    etas = [0.25, 0.35, 0.5, 1.0]
    betas = [0.0, 4.0, 8.0, 16.0]
    lams = [0.05, 0.10, 0.25]
    configs: list[CoarseSteerConfig] = []
    for eta, beta, lam in itertools.product(etas, betas, lams):
        configs.append(CoarseSteerConfig(eta=eta, beta=beta, anchor_lambda=lam))
    # Option-D row: window=1 ⇒ lambda=1.0 (anchor = previous concept/fed fully)
    configs.append(CoarseSteerConfig(eta=0.35, beta=8.0, anchor_lambda=1.0))
    return configs


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = _load_assets(args)
    device = assets["device"]
    print(f"[coarse_steer/sweep] device={device}")
    seeds = _pick_seeds(assets["corpus"], args.n_seeds, seed=args.seed)
    (out_dir / "sweep_seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    frontier_path = out_dir / "frontier_coarse.jsonl"
    if frontier_path.exists():
        frontier_path.unlink()

    configs = _sweep_grid()
    print(f"[coarse_steer/sweep] {len(configs)} configs x {len(seeds)} seeds x {args.steps} steps")
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for ci, cfg in enumerate(configs):
        print(
            f"[coarse_steer/sweep] cfg {ci + 1}/{len(configs)} "
            f"eta={cfg.eta} beta={cfg.beta} lam={cfg.anchor_lambda}",
            flush=True,
        )
        rollouts: list[ForwardGenResult] = []
        for seed_text in seeds:
            res = generate_document_coarse(
                seed_text,
                assets["predictor"],
                assets["writer"],
                assets["tokenizer"],
                assets["codec"],
                steps=args.steps,
                cfg=cfg,
                train_bank=assets["bank"],
                ref_dist=assets["ref"],
                minilm=assets["minilm"],
                writer_temperature=args.writer_temperature,
                writer_top_k=args.writer_top_k,
                max_tokens=args.max_tokens,
            )
            rollouts.append(res)
        ret = _agg_retention_curve(rollouts, args.steps)
        shuf = _shuffled_retention_curve(
            rollouts, seeds, assets["minilm"], args.steps, rng_seed=args.seed + 7
        )
        nov = _novelty_summary(rollouts, assets["ref"])
        delta = float(ret[-1] - shuf[-1]) if ret and shuf else float("nan")
        row = {
            "eta": cfg.eta,
            "beta": cfg.beta,
            "anchor_lambda": cfg.anchor_lambda,
            "anchor_mode": cfg.anchor_mode,
            "anchor_source": cfg.anchor_source,
            "n_cand": cfg.n_cand,
            "retain_k8": float(ret[-1]),
            "retain_shuf_k8": float(shuf[-1]),
            "delta_retain": delta,
            "novelty_rate": float(nov.get("novelty_rate", float("nan"))),
            "copy_rate": float(nov.get("copy_rate", float("nan"))),
            "mean_n_x": float(nov.get("mean_n_x", float("nan"))),
            "n_seeds": len(seeds),
            "steps": args.steps,
            "seed": args.seed,
        }
        with open(frontier_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        nov_ok = np.isfinite(row["novelty_rate"]) and row["novelty_rate"] >= 0.5
        if nov_ok and np.isfinite(delta):
            if best is None or delta > best["delta_retain"]:
                best = dict(row)

    # Eligible = novelty_rate >= 0.5. Prefer eta=1.0 pure-rejection if it clears
    # both gates (delta>=0.10); else maximize delta_retain (spec §3.3).
    eligible = [
        r
        for r in rows
        if np.isfinite(r["novelty_rate"])
        and r["novelty_rate"] >= 0.5
        and np.isfinite(r["delta_retain"])
    ]
    clears = [r for r in eligible if r["delta_retain"] >= 0.10]
    pure_rej = [r for r in clears if r["eta"] == 1.0 and r["beta"] > 0]
    if pure_rej:
        chosen = max(pure_rej, key=lambda r: r["delta_retain"])
    elif clears:
        chosen = max(clears, key=lambda r: r["delta_retain"])
    elif eligible:
        chosen = max(eligible, key=lambda r: r["delta_retain"])
    else:
        chosen = max(
            rows,
            key=lambda r: r["delta_retain"] if np.isfinite(r["delta_retain"]) else -1e9,
        )

    (out_dir / "sweep_chosen.json").write_text(
        json.dumps({"chosen": chosen, "n_eligible": len(eligible), "n_rows": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"[coarse_steer/sweep] wrote {frontier_path}")
    print(f"[coarse_steer/sweep] chosen={chosen}")
    return {"frontier": str(frontier_path), "chosen": chosen, "rows": rows}


def _eval_arms(
    seeds: list[str],
    arms: dict[str, list[ForwardGenResult]],
    minilm,
    ref,
    steps: int,
    seed: int,
    codec,
) -> dict[str, Any]:
    retention = {
        name: _agg_retention_curve(arms[name], steps) for name in arms
    }
    for name in list(arms.keys()):
        retention[f"{name}_shuf"] = _shuffled_retention_curve(
            arms[name], seeds, minilm, steps, rng_seed=seed + 7
        )
    novelty = {name: _novelty_summary(arms[name], ref) for name in arms}
    sane = {name: _arm_sane_band(arms[name]) for name in arms}
    div = {
        "coarse_vs_sample": _pairwise_text_divergence(
            arms["coarse"], arms["sample"], minilm
        ),
        "coarse_vs_identity": _pairwise_text_divergence(
            arms["coarse"], arms["identity"], minilm
        ),
        "coarse_vs_zero": _pairwise_text_divergence(
            arms["coarse"], arms["zero"], minilm
        ),
        "sample_vs_identity": _pairwise_text_divergence(
            arms["sample"], arms["identity"], minilm
        ),
        "sample_vs_zero": _pairwise_text_divergence(
            arms["sample"], arms["zero"], minilm
        ),
    }
    fed8 = {
        name: _fed8_vs_a0(arms[name], seeds, codec)
        for name in ("coarse", "sample")
        if name in arms
    }
    return {
        "retention": retention,
        "novelty": novelty,
        "sane": sane,
        "div": div,
        "fed8_vs_a0": fed8,
    }


def _gates(metrics: dict[str, Any]) -> dict[str, Any]:
    ret_c = metrics["retention"]["coarse"]
    shuf_c = metrics["retention"]["coarse_shuf"]
    delta = float(ret_c[-1] - shuf_c[-1])
    nov = metrics["novelty"]["coarse"]
    sane = metrics["sane"]["coarse"]
    div_cz = metrics["div"]["coarse_vs_zero"]["mean_cos"]
    div_ci = metrics["div"]["coarse_vs_identity"]["mean_cos"]
    fed8_c = metrics["fed8_vs_a0"].get("coarse", float("nan"))
    fed8_s = metrics["fed8_vs_a0"].get("sample", float("nan"))

    g1 = bool(delta >= 0.10)
    g2 = bool(
        float(nov.get("novelty_rate", 0)) >= 0.50
        and float(nov.get("copy_rate", 1)) <= 0.02
    )
    mc = float(sane["mean_consec"])
    p90 = float(sane["p90_consec"])
    jr = float(sane["jump_rate"])
    d2 = float(sane["distinct2"])
    g3 = bool(
        0.15 <= mc <= 0.60
        and p90 <= 0.90
        and jr <= 0.15
        and d2 >= 0.30
    )
    g4 = bool(div_cz <= 0.85 and div_ci <= 0.85)
    # G-CS5: computed half (fed8) + qualitative placeholder for blind domain flips
    g5_fed = bool(
        np.isfinite(fed8_c)
        and np.isfinite(fed8_s)
        and fed8_c >= fed8_s + 0.15
    )
    return {
        "G-CS1": {
            "pass": g1,
            "delta_retain": delta,
            "threshold": 0.10,
            "retain_k8": float(ret_c[-1]),
            "shuf_k8": float(shuf_c[-1]),
        },
        "G-CS2": {
            "pass": g2,
            "novelty_rate": float(nov.get("novelty_rate", float("nan"))),
            "copy_rate": float(nov.get("copy_rate", float("nan"))),
            "thresholds": {"novelty_rate": 0.50, "copy_rate": 0.02},
        },
        "G-CS3": {
            "pass": g3,
            "mean_consec": mc,
            "p90_consec": p90,
            "jump_rate": jr,
            "distinct2": d2,
            "bands": {
                "mean_consec": [0.15, 0.60],
                "p90_consec": 0.90,
                "jump_rate": 0.15,
                "distinct2": 0.30,
            },
        },
        "G-CS4": {
            "pass": g4,
            "coarse_vs_zero": float(div_cz),
            "coarse_vs_identity": float(div_ci),
            "threshold": 0.85,
        },
        "G-CS5": {
            "pass": g5_fed,  # blind domain-flip half left for human; fed8 computed
            "median_fed8_vs_a0_coarse": float(fed8_c),
            "median_fed8_vs_a0_sample": float(fed8_s),
            "fed8_delta": float(fed8_c - fed8_s) if np.isfinite(fed8_c) and np.isfinite(fed8_s) else float("nan"),
            "fed8_threshold_vs_sample": 0.15,
            "note": "blind domain-flip count requires human read; fed8 half computed in-run",
        },
        "PASS_core": bool(g1 and g2 and g3 and g4),
    }


def write_results_md(
    out_dir: Path,
    seeds: list[str],
    arms: dict[str, list[ForwardGenResult]],
    metrics: dict[str, Any],
    gates: dict[str, Any],
    cfg: CoarseSteerConfig,
    frontier_note: str,
) -> Path:
    path = out_dir / "RESULTS_coarse.md"
    lines: list[str] = []
    lines.append("# Coarse-steer anchored sampler RESULTS")
    lines.append("")
    lines.append(
        "Training-free v1: MDN tilt + best-of-n + slerp toward slow EMA/paragraph anchor. "
        "Baselines via existing `generate_document_twochannel`."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- seeds: {len(seeds)} x steps: {metrics['steps']}")
    lines.append(f"- frozen cfg: `{json.dumps(asdict(cfg))}`")
    lines.append(f"- writer: `{metrics['writer_path']}`")
    lines.append(f"- mdn: `{metrics['mdn_path']}`")
    lines.append(f"- codec: `{metrics['jepa_path']}`")
    lines.append(f"- corpus: `{metrics['corpus_path']}`")
    lines.append(f"- device: {metrics['device']}")
    lines.append(f"- sweep note: {frontier_note}")
    lines.append("")
    lines.append("## Head-to-head table (8 steps)")
    lines.append("")
    lines.append(
        "| Arm | retain@k8 | shuf@k8 | Δretain | novelty_rate | copy_rate | "
        "mean consec | p90 consec | jump_rate | distinct2 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in ("coarse", "sample", "identity", "zero"):
        ret = metrics["retention"][name]
        shuf = metrics["retention"][f"{name}_shuf"]
        nov = metrics["novelty"][name]
        sane = metrics["sane"][name]
        delta = ret[-1] - shuf[-1]
        lines.append(
            f"| {name} | {ret[-1]:.4f} | {shuf[-1]:.4f} | {delta:+.4f} | "
            f"{nov.get('novelty_rate', float('nan')):.4f} | "
            f"{nov.get('copy_rate', float('nan')):.4f} | "
            f"{sane['mean_consec']:.4f} | {sane['p90_consec']:.4f} | "
            f"{sane['jump_rate']:.4f} | {sane['distinct2']:.4f} |"
        )
    lines.append("")
    lines.append("Baseline quotes (RESULTS_forward.md): sample Δ=+0.034 / nov 1.00; "
                 "identity Δ=+0.087 / nov 0.344; zero Δ=+0.029 / nov 1.00; "
                 "mean consec 0.255, p90 0.456.")
    lines.append("")
    lines.append("## Retention curves")
    lines.append("")
    for name in ("coarse", "sample", "identity", "zero"):
        curve = ", ".join(f"{v:.3f}" for v in metrics["retention"][name])
        lines.append(f"- {name}: {curve}")
    lines.append("")
    lines.append("## Cross-arm divergence (MiniLM)")
    lines.append("")
    for k, v in metrics["div"].items():
        lines.append(f"- {k}: mean_cos={v['mean_cos']:.4f}")
    lines.append("")
    lines.append("## Gates (§6)")
    lines.append("")
    lines.append("| Gate | Pass | Measured | Threshold |")
    lines.append("|---|---|---|---|")
    g = gates
    lines.append(
        f"| G-CS1 coherence | {'PASS' if g['G-CS1']['pass'] else 'FAIL'} | "
        f"Δ={g['G-CS1']['delta_retain']:+.4f} "
        f"(ret={g['G-CS1']['retain_k8']:.4f}, shuf={g['G-CS1']['shuf_k8']:.4f}) | ≥ +0.10 |"
    )
    lines.append(
        f"| G-CS2 novelty | {'PASS' if g['G-CS2']['pass'] else 'FAIL'} | "
        f"nov={g['G-CS2']['novelty_rate']:.4f}, copy={g['G-CS2']['copy_rate']:.4f} | "
        f"≥0.50 / ≤0.02 |"
    )
    lines.append(
        f"| G-CS3 sane band | {'PASS' if g['G-CS3']['pass'] else 'FAIL'} | "
        f"consec={g['G-CS3']['mean_consec']:.3f}, p90={g['G-CS3']['p90_consec']:.3f}, "
        f"jump={g['G-CS3']['jump_rate']:.3f}, d2={g['G-CS3']['distinct2']:.3f} | "
        f"[0.15,0.60] / ≤0.90 / ≤0.15 / ≥0.30 |"
    )
    lines.append(
        f"| G-CS4 steer alive | {'PASS' if g['G-CS4']['pass'] else 'FAIL'} | "
        f"coarse↔zero={g['G-CS4']['coarse_vs_zero']:.4f}, "
        f"coarse↔identity={g['G-CS4']['coarse_vs_identity']:.4f} | both ≤0.85 |"
    )
    lines.append(
        f"| G-CS5 genre-hold (fed8 half) | {'PASS' if g['G-CS5']['pass'] else 'FAIL'} | "
        f"fed8 coarse={g['G-CS5']['median_fed8_vs_a0_coarse']:.4f}, "
        f"sample={g['G-CS5']['median_fed8_vs_a0_sample']:.4f}, "
        f"Δ={g['G-CS5']['fed8_delta']:+.4f} | coarse ≥ sample+0.15 |"
    )
    lines.append("")
    lines.append(
        f"**Core PASS (G-CS1∧2∧3∧4) = {gates['PASS_core']}**"
    )
    lines.append("")
    if "steps16" in metrics:
        lines.append("## Secondary table (16 steps, coarse arm)")
        lines.append("")
        m16 = metrics["steps16"]
        lines.append(
            f"- retain@k16={m16['retain'][-1]:.4f}, shuf={m16['shuf'][-1]:.4f}, "
            f"Δ={m16['retain'][-1]-m16['shuf'][-1]:+.4f}, "
            f"novelty_rate={m16['novelty'].get('novelty_rate', float('nan')):.4f}"
        )
        lines.append("")
    lines.append("## Coarse-arm continuation excerpts (8 seeds)")
    lines.append("")
    for i in range(min(8, len(seeds))):
        lines.append(f"### Seed {i + 1}")
        lines.append(f"- seed: {json.dumps(seeds[i])}")
        for sent in arms["coarse"][i].sentences:
            clipped = sent if len(sent) < 240 else sent[:237] + "..."
            lines.append(f"  - {clipped}")
        lines.append("")
    lines.append("## Honest verdict")
    lines.append("")
    lines.append(metrics.get("conclusion", ""))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _honest_conclusion(gates: dict, metrics: dict) -> str:
    g1 = gates["G-CS1"]
    g2 = gates["G-CS2"]
    parts = []
    if gates["PASS_core"]:
        parts.append(
            f"PASS: coarse arm clears G-CS1–4 "
            f"(Δretain={g1['delta_retain']:+.3f}, novelty={g2['novelty_rate']:.3f})."
        )
    else:
        fails = [k for k in ("G-CS1", "G-CS2", "G-CS3", "G-CS4") if not gates[k]["pass"]]
        parts.append(
            f"FAIL core gates: {', '.join(fails)}. "
            f"Δretain={g1['delta_retain']:+.3f} (need ≥+0.10), "
            f"novelty_rate={g2['novelty_rate']:.3f} (need ≥0.50)."
        )
    # Frontier finding
    nov = g2["novelty_rate"]
    delta = g1["delta_retain"]
    if delta >= 0.10 and nov < 0.50:
        parts.append(
            "Coherence budget exists but novelty collapses under the leash — "
            "same trade as identity, not a new frontier corner."
        )
    elif nov >= 0.50 and delta < 0.10:
        parts.append(
            "Novelty holds but retention does not clear +0.10 — tilt+rejection did not "
            "buy enough coherence beyond flat sample."
        )
    elif not (delta >= 0.10 and nov >= 0.50):
        parts.append(
            "No operating point jointly clears G-CS1∧G-CS2 on this frozen run; "
            "see frontier_coarse.jsonl for the full (novelty, Δretain) surface."
        )
    return " ".join(parts)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = _load_assets(args)
    device = assets["device"]
    print(f"[coarse_steer/run] device={device}")

    # Load frozen seeds from forward_gen
    seeds_path = Path(args.seeds_json)
    if seeds_path.is_file():
        seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    else:
        seeds = _pick_seeds(assets["corpus"], args.n_seeds, seed=args.seed)
    seeds = seeds[: args.n_seeds]
    (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")

    # Config: prefer sweep_chosen if present, else CLI
    chosen_path = out_dir / "sweep_chosen.json"
    frontier_note = "defaults / CLI"
    if chosen_path.is_file() and not args.ignore_sweep:
        blob = json.loads(chosen_path.read_text(encoding="utf-8"))
        ch = blob.get("chosen") or {}
        cfg = CoarseSteerConfig(
            eta=float(ch.get("eta", args.eta)),
            beta=float(ch.get("beta", args.beta)),
            anchor_lambda=float(ch.get("anchor_lambda", args.anchor_lambda)),
            anchor_mode=ch.get("anchor_mode", args.anchor_mode),
            anchor_source=ch.get("anchor_source", args.anchor_source),
            n_cand=int(ch.get("n_cand", args.n_cand)),
            steer_temp=float(args.steer_temp),
            context_len=int(args.context_len),
        )
        frontier_note = f"from sweep_chosen.json: eta={cfg.eta} beta={cfg.beta} lam={cfg.anchor_lambda}"
    else:
        cfg = _cfg_from_args(args)
        frontier_note = f"CLI cfg: eta={cfg.eta} beta={cfg.beta} lam={cfg.anchor_lambda}"

    print(f"[coarse_steer/run] cfg={asdict(cfg)}")
    print(f"[coarse_steer/run] {len(seeds)} seeds x {args.steps} steps x 4 arms")

    arms: dict[str, list[ForwardGenResult]] = {
        "coarse": [],
        "sample": [],
        "identity": [],
        "zero": [],
    }
    for i, seed_text in enumerate(seeds):
        print(f"[coarse_steer/run] seed {i + 1}/{len(seeds)}", flush=True)
        arms["coarse"].append(
            generate_document_coarse(
                seed_text,
                assets["predictor"],
                assets["writer"],
                assets["tokenizer"],
                assets["codec"],
                steps=args.steps,
                cfg=cfg,
                train_bank=assets["bank"],
                ref_dist=assets["ref"],
                minilm=assets["minilm"],
                writer_temperature=args.writer_temperature,
                writer_top_k=args.writer_top_k,
                max_tokens=args.max_tokens,
            )
        )
        for mode in ("sample", "identity", "zero"):
            arms[mode].append(
                generate_document_twochannel(
                    seed_text,
                    assets["predictor"],
                    assets["writer"],
                    assets["tokenizer"],
                    assets["codec"],
                    steps=args.steps,
                    steer_temp=cfg.steer_temp,
                    steer_mode=mode,  # type: ignore[arg-type]
                    context_len=cfg.context_len,
                    train_bank=assets["bank"],
                    ref_dist=assets["ref"],
                    minilm=assets["minilm"],
                    writer_temperature=args.writer_temperature,
                    writer_top_k=args.writer_top_k,
                    max_tokens=args.max_tokens,
                )
            )

    metrics = _eval_arms(
        seeds, arms, assets["minilm"], assets["ref"], args.steps, args.seed, assets["codec"]
    )
    metrics.update(
        {
            "steps": args.steps,
            "n_seeds": len(seeds),
            "device": device,
            "writer_path": str(args.writer),
            "mdn_path": str(args.mdn),
            "jepa_path": str(args.jepa),
            "corpus_path": str(args.corpus),
            "cfg": asdict(cfg),
            "ref_median": float(assets["ref"]["median"]),
            "ref_p99": float(assets["ref"]["p99"]),
        }
    )

    # Secondary 16-step coarse table
    if args.steps_secondary and args.steps_secondary > args.steps:
        print(f"[coarse_steer/run] secondary {args.steps_secondary}-step coarse rollouts")
        roll16: list[ForwardGenResult] = []
        for seed_text in seeds:
            roll16.append(
                generate_document_coarse(
                    seed_text,
                    assets["predictor"],
                    assets["writer"],
                    assets["tokenizer"],
                    assets["codec"],
                    steps=args.steps_secondary,
                    cfg=cfg,
                    train_bank=assets["bank"],
                    ref_dist=assets["ref"],
                    minilm=assets["minilm"],
                    writer_temperature=args.writer_temperature,
                    writer_top_k=args.writer_top_k,
                    max_tokens=args.max_tokens,
                )
            )
        ret16 = _agg_retention_curve(roll16, args.steps_secondary)
        shuf16 = _shuffled_retention_curve(
            roll16, seeds, assets["minilm"], args.steps_secondary, rng_seed=args.seed + 7
        )
        metrics["steps16"] = {
            "retain": ret16,
            "shuf": shuf16,
            "novelty": _novelty_summary(roll16, assets["ref"]),
            "sane": _arm_sane_band(roll16),
        }

    gates = _gates(metrics)
    metrics["conclusion"] = _honest_conclusion(gates, metrics)
    metrics["gates"] = gates

    # Blind sheet
    blind_items = []
    blind_key = []
    rng = np.random.RandomState(42)
    for i, seed in enumerate(seeds[:10]):
        for arm_name in ("coarse", "sample", "identity", "zero"):
            blind_items.append(
                {
                    "id": len(blind_items),
                    "seed": seed,
                    "text": " ".join(arms[arm_name][i].sentences),
                }
            )
            blind_key.append({"id": blind_items[-1]["id"], "arm": arm_name, "seed_idx": i})
    order = rng.permutation(len(blind_items))
    blind_shuffled = [blind_items[int(j)] for j in order]
    key_shuffled = [blind_key[int(j)] for j in order]
    (out_dir / "blind_sheet.json").write_text(
        json.dumps(blind_shuffled, indent=2), encoding="utf-8"
    )
    (out_dir / "blind_key.json").write_text(
        json.dumps(key_shuffled, indent=2), encoding="utf-8"
    )

    serial = {
        "seeds": seeds,
        "cfg": asdict(cfg),
        "arms": {
            name: [
                {"sentences": r.sentences, "instrumentation": r.instrumentation}
                for r in arms[name]
            ]
            for name in arms
        },
        "metrics": {
            k: v
            for k, v in metrics.items()
            if k not in ("conclusion",)
        },
        "gates": gates,
        "conclusion": metrics["conclusion"],
    }
    (out_dir / "rollouts.json").write_text(
        json.dumps(serial, indent=2, default=float), encoding="utf-8"
    )
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    report = write_results_md(
        out_dir, seeds, arms, metrics, gates, cfg, frontier_note
    )
    print(f"[coarse_steer/run] wrote {report}")
    print(f"[coarse_steer/run] conclusion: {metrics['conclusion']}")
    return {"report": str(report), "gates": gates, "metrics": metrics, "cfg": asdict(cfg)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true", help="Run §3.3 config grid")
    p.add_argument("--run", action="store_true", help="Frozen head-to-head run")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--mdn", default=str(_ROOT / "artifacts/sentence_pred/mdn.pt"))
    p.add_argument(
        "--writer", default=str(_ROOT / "artifacts/writer_p25/writer_two_channel.pt")
    )
    p.add_argument(
        "--jepa", default=str(_ROOT / "artifacts/jepa_repair_1m/jepa/joint.pt")
    )
    p.add_argument(
        "--corpus",
        default=str(
            _ROOT / "artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1"
        ),
    )
    p.add_argument("--seeds-json", default=str(SEEDS_PATH))
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--steps-secondary", type=int, default=16)
    p.add_argument("--steer-temp", type=float, default=1.0)
    p.add_argument("--context-len", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.35)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--anchor-lambda", type=float, default=0.10)
    p.add_argument("--anchor-every", type=int, default=4)
    p.add_argument("--anchor-mode", choices=["ema", "paragraph"], default="ema")
    p.add_argument("--anchor-source", choices=["fed", "concept"], default="fed")
    p.add_argument("--n-cand", type=int, default=8)
    p.add_argument("--use-novelty-guard", action="store_true")
    p.add_argument("--writer-temperature", type=float, default=0.8)
    p.add_argument("--writer-top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--bank-cap", type=int, default=50000)
    p.add_argument("--held-cap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ignore-sweep", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    if args.sweep:
        args.n_seeds = min(args.n_seeds, 8) if args.n_seeds == 20 else args.n_seeds
        if args.seed == 1234:
            args.seed = 777
        # sweep uses 8 seeds
        if "--n-seeds" not in (argv or sys.argv[1:]):
            args.n_seeds = 8
        run_sweep(args)
        return 0
    if args.run:
        run_experiment(args)
        return 0
    p.error("Specify --sweep or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
