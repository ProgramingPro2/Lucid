"""Two-channel free-running forward generation (words = coherence, concept = steer).

Does NOT retrain. Loads:
  - MDN density head (artifacts/sentence_pred/mdn.pt)
  - two-channel FiLM writer (artifacts/writer_p25/writer_two_channel.pt)
  - repaired JEPA codec (artifacts/jepa_repair_1m/jepa/joint.pt)
  - sentence corpus for seeds + novelty bank

Loop per step:
  1. encode current sentence -> z_t
  2. context = last few sentence concepts
  3. choose steer concept (sample MDN | identity copy z_t | zeros)
  4. writer.generate(concept=steer, prefix_ids=[BOS]+prev_tokens+[EOS])
  5. append rendered sentence; repeat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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
_EXP = str(_ROOT)
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from conceptpoc.config import PredictorConfig  # noqa: E402
from conceptpoc.persistence import load_joint, load_writer  # noqa: E402
from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.jepa.tokenizer import Tokenizer  # noqa: E402

from experiments.sentence_pred.data import load_sentence_corpus  # noqa: E402
from experiments.sentence_pred.evalsuite import (  # noqa: E402
    centering,
    copy_rate,
    load_split,
    novelty_rate,
    novelty_scores,
    reference_distribution,
)
from experiments.sentence_pred.heads import SentenceDensityPredictor  # noqa: E402

SteerMode = Literal["sample", "identity", "zero"]

PREFIX_MAX = 96
MAX_NEW_TOKENS = 96
DEFAULT_CONTEXT_LEN = 8


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)).astype(np.float32)


def load_mdn_predictor(path: str | Path, device: str = "cpu") -> SentenceDensityPredictor:
    """Reconstruct SentenceDensityPredictor from train_head checkpoint."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    pcfg = blob.get("predictor_cfg") or {}
    head = blob.get("head") or blob.get("head_type") or "mdn"
    dim = int(blob.get("concept_dim", 512))
    K = int(blob.get("K", 8))
    model = SentenceDensityPredictor(
        concept_dim=dim,
        predictor_cfg=PredictorConfig(
            d_model=int(pcfg.get("d_model", 512)),
            n_layers=int(pcfg.get("n_layers", 6)),
            n_heads=int(pcfg.get("n_heads", 8)),
            ff_mult=int(pcfg.get("ff_mult", 4)),
            dropout=float(pcfg.get("dropout", 0.1)),
            max_seq_len=int(pcfg.get("max_seq_len", 34)),
        ),
        head=head if head in ("vmf", "mdn") else "mdn",
        K=K,
    )
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _encode_one(codec, text: str) -> np.ndarray:
    v = np.asarray(codec.encode([text]), dtype=np.float32)
    return _l2(v[0])


def _build_prefix_ids(
    tokenizer,
    prev_text: str,
    device: torch.device,
) -> torch.Tensor:
    bos = int(tokenizer.bos_id)
    eos = int(tokenizer.eos_id)
    ids = tokenizer.encode(prev_text) if hasattr(tokenizer, "encode") else []
    ids = list(ids)[:PREFIX_MAX]
    return torch.tensor([[bos] + ids + [eos]], dtype=torch.long, device=device)


def _render_sentence(
    writer,
    tokenizer,
    concept: torch.Tensor,
    prev_text: str,
    temperature: float = 0.8,
    top_k: int = 50,
    max_tokens: int = MAX_NEW_TOKENS,
    repetition_penalty: float = 1.1,
) -> str:
    device = concept.device
    prefix_ids = _build_prefix_ids(tokenizer, prev_text, device)
    bos = int(tokenizer.bos_id)
    eos = int(tokenizer.eos_id)
    with torch.no_grad():
        out = writer.generate(
            concept,
            max_tokens=max_tokens,
            bos_id=bos,
            eos_id=eos,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            prefix_ids=prefix_ids,
        )
    return tokenizer.decode(out[0].tolist())


@dataclass
class ForwardGenResult:
    sentences: list[str]
    instrumentation: list[dict[str, Any]]

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


@torch.no_grad()
def generate_document_twochannel(
    seed_text: str,
    predictor_mdn,
    writer,
    tokenizer,
    codec,
    steps: int = 8,
    steer_temp: float = 1.0,
    steer_mode: SteerMode = "sample",
    context_len: int = DEFAULT_CONTEXT_LEN,
    train_bank: np.ndarray | None = None,
    ref_dist: dict | None = None,
    minilm=None,
    writer_temperature: float = 0.8,
    writer_top_k: int = 50,
    max_tokens: int = MAX_NEW_TOKENS,
) -> ForwardGenResult:
    """Generate ``steps`` sentences; coherence via prefix tokens, steer via concept.

    Returns ForwardGenResult with ``.sentences`` (generated only) and
    ``.instrumentation`` rows containing novelty score of the steer concept,
    MiniLM retention to seed, consecutive cos, etc.
    """
    if steer_mode not in ("sample", "identity", "zero"):
        raise ValueError(f"steer_mode must be sample|identity|zero, got {steer_mode!r}")

    device = next(predictor_mdn.parameters()).device
    dim = int(predictor_mdn.concept_dim)
    seed_vec = _encode_one(codec, seed_text)
    history_vecs = [seed_vec]
    history_texts = [seed_text]
    instr: list[dict[str, Any]] = []

    mu_bank = bank_c = None
    if train_bank is not None and len(train_bank):
        mu_bank, bank_c = centering(train_bank)

    seed_minilm = None
    if minilm is not None:
        seed_minilm = _l2(np.asarray(minilm.encode([seed_text]), dtype=np.float32)[0])
    prev_minilm = seed_minilm

    for step in range(int(steps)):
        window = history_vecs[-int(context_len) :]
        ctx = torch.from_numpy(np.stack(window).astype(np.float32)).unsqueeze(0).to(device)
        pad = torch.zeros(1, ctx.size(1), dtype=torch.bool, device=device)
        z_t = torch.from_numpy(history_vecs[-1][None, :]).to(device)

        if steer_mode == "sample":
            sample = predictor_mdn.sample(ctx, pad, n_samples=1, temperature=steer_temp)
            concept = F.normalize(sample[:, 0, :], dim=-1)
        elif steer_mode == "identity":
            concept = F.normalize(z_t, dim=-1)
        else:  # zero
            concept = torch.zeros(1, dim, device=device, dtype=z_t.dtype)

        concept_np = concept.detach().float().cpu().numpy()[0]
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

        row: dict[str, Any] = {
            "step": step,
            "text": text,
            "steer_mode": steer_mode,
            "steer_temp": float(steer_temp),
        }
        if mu_bank is not None and bank_c is not None:
            # Novelty of the STEER concept (sampled / identity / zero), not re-encode.
            n_x = float(novelty_scores(concept_np[None, :], bank_c, mu_bank)[0])
            row["n_x_steer"] = n_x
            if ref_dist is not None:
                row["novel_vs_median"] = bool(n_x <= float(ref_dist["median"]))
                row["copy_vs_p99"] = bool(n_x >= float(ref_dist["p99"]))
            row["n_x_fed"] = float(novelty_scores(fed[None, :], bank_c, mu_bank)[0])
        row["consecutive_cos"] = float(np.dot(_l2(fed), _l2(history_vecs[-1])))
        row["steer_vs_zt_cos"] = float(
            np.dot(_l2(concept_np), _l2(history_vecs[-1]))
        )

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


def _pick_seeds(corpus, n: int, seed: int = 1234) -> list[str]:
    rng = np.random.RandomState(seed)
    held = list(corpus.held_doc_ids.tolist())
    if not held:
        held = list(corpus.train_doc_ids.tolist())
    rng.shuffle(held)
    seeds: list[str] = []
    for d in held:
        a, b = corpus.doc_slice(int(d))
        # Prefer a mid-document sentence for a cleaner seed if available.
        idx = a if (b - a) < 2 else a + 1
        text = corpus.sentences[idx].strip()
        if len(text) < 20:
            continue
        seeds.append(text)
        if len(seeds) >= n:
            break
    while len(seeds) < n and corpus.n_sentences:
        seeds.append(corpus.sentences[len(seeds) % corpus.n_sentences])
    return seeds[:n]


def _load_minilm():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _agg_retention_curve(rollouts: list[ForwardGenResult], steps: int) -> list[float]:
    means = []
    for k in range(steps):
        vals = []
        for r in rollouts:
            if k < len(r.instrumentation):
                v = r.instrumentation[k].get("minilm_retention")
                if v is not None:
                    vals.append(float(v))
        means.append(float(np.mean(vals)) if vals else float("nan"))
    return means


def _shuffled_retention_curve(
    rollouts: list[ForwardGenResult],
    seeds: list[str],
    minilm,
    steps: int,
    rng_seed: int = 999,
) -> list[float]:
    """Retention of generated sentences vs a randomly reassigned seed (control)."""
    if minilm is None or not rollouts:
        return [float("nan")] * steps
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(len(seeds))
    while len(seeds) > 1 and np.array_equal(perm, np.arange(len(seeds))):
        perm = rng.permutation(len(seeds))
    shuf_embs = [
        _l2(np.asarray(minilm.encode([seeds[int(perm[i])]]), dtype=np.float32)[0])
        for i in range(len(seeds))
    ]
    means = []
    for k in range(steps):
        vals = []
        for i, r in enumerate(rollouts):
            if k >= len(r.sentences):
                continue
            mv = _l2(np.asarray(minilm.encode([r.sentences[k]]), dtype=np.float32)[0])
            vals.append(float(np.dot(mv, shuf_embs[i])))
        means.append(float(np.mean(vals)) if vals else float("nan"))
    return means


def _novelty_summary(
    rollouts: list[ForwardGenResult], ref_dist: dict | None
) -> dict[str, float]:
    scores = []
    for r in rollouts:
        for row in r.instrumentation:
            if "n_x_steer" in row:
                scores.append(float(row["n_x_steer"]))
    arr = np.asarray(scores, dtype=np.float32)
    out: dict[str, float] = {
        "n": float(arr.size),
        "mean_n_x": float(arr.mean()) if arr.size else float("nan"),
        "median_n_x": float(np.median(arr)) if arr.size else float("nan"),
    }
    if ref_dist is not None and arr.size:
        out["novelty_rate"] = novelty_rate(arr, ref_dist)
        out["copy_rate"] = copy_rate(arr, ref_dist)
        out["ref_median"] = float(ref_dist["median"])
        out["ref_p99"] = float(ref_dist["p99"])
    return out


def _pairwise_text_divergence(
    a: list[ForwardGenResult], b: list[ForwardGenResult], minilm
) -> dict[str, float]:
    """Mean MiniLM cos between matched (seed, step) sentences of two arms."""
    if minilm is None:
        return {"mean_cos": float("nan"), "n": 0.0}
    vals = []
    for ra, rb in zip(a, b):
        n = min(len(ra.sentences), len(rb.sentences))
        for k in range(n):
            ea = _l2(np.asarray(minilm.encode([ra.sentences[k]]), dtype=np.float32)[0])
            eb = _l2(np.asarray(minilm.encode([rb.sentences[k]]), dtype=np.float32)[0])
            vals.append(float(np.dot(ea, eb)))
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "mean_cos": float(arr.mean()) if arr.size else float("nan"),
        "median_cos": float(np.median(arr)) if arr.size else float("nan"),
        "n": float(arr.size),
    }


def _fmt_curve(xs: list[float]) -> str:
    return ", ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in xs)


def write_report(
    out_dir: Path,
    seeds: list[str],
    arms: dict[str, list[ForwardGenResult]],
    metrics: dict[str, Any],
) -> Path:
    path = out_dir / "RESULTS_forward.md"
    sample = arms["sample"]
    identity = arms["identity"]
    zero = arms["zero"]

    lines: list[str] = []
    lines.append("# Two-channel forward generation RESULTS")
    lines.append("")
    lines.append(
        "Experiment: words (previous-sentence token prefix) = local coherence; "
        "sampled MDN concept via all-block FiLM = novelty/steer. No retraining."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- seeds: {len(seeds)} x steps: {metrics['steps']}")
    lines.append(f"- writer: `{metrics['writer_path']}`")
    lines.append(f"- mdn: `{metrics['mdn_path']}`")
    lines.append(f"- codec: `{metrics['jepa_path']}`")
    lines.append(f"- corpus: `{metrics['corpus_path']}`")
    lines.append(f"- steer_temp: {metrics['steer_temp']}, context_len: {metrics['context_len']}")
    lines.append(f"- device: {metrics['device']}")
    lines.append("")
    lines.append("## A/B arms")
    lines.append("")
    lines.append("| Arm | Description |")
    lines.append("|---|---|")
    lines.append("| sample | prefix + MDN-sampled concept steer |")
    lines.append("| identity | prefix + copy of current concept (zero-steer) |")
    lines.append("| zero | prefix + zeroed concept (FiLM identity / no steer) |")
    lines.append("")

    lines.append("## MiniLM retention curves (mean cos to seed, k=1..8)")
    lines.append("")
    lines.append("| Arm | k=1..8 |")
    lines.append("|---|---|")
    for name in ("sample", "identity", "zero"):
        lines.append(
            f"| {name} | {_fmt_curve(metrics['retention'][name])} |"
        )
    lines.append(
        f"| sample vs shuffled-seed control | {_fmt_curve(metrics['retention']['sample_shuf'])} |"
    )
    lines.append("")
    lines.append(
        f"- retain@k8 sample={metrics['retention']['sample'][-1]:.4f}, "
        f"identity={metrics['retention']['identity'][-1]:.4f}, "
        f"zero={metrics['retention']['zero'][-1]:.4f}, "
        f"shuf_control={metrics['retention']['sample_shuf'][-1]:.4f}"
    )
    lines.append(
        f"- sample − shuf @k8 = "
        f"{metrics['retention']['sample'][-1] - metrics['retention']['sample_shuf'][-1]:+.4f}"
    )
    lines.append("")

    lines.append("## Novelty of steered concepts")
    lines.append("")
    lines.append("| Arm | novelty_rate | copy_rate | mean n(x) | n |")
    lines.append("|---|---|---|---|---|")
    for name in ("sample", "identity", "zero"):
        nov = metrics["novelty"][name]
        lines.append(
            f"| {name} | {nov.get('novelty_rate', float('nan')):.4f} | "
            f"{nov.get('copy_rate', float('nan')):.4f} | "
            f"{nov.get('mean_n_x', float('nan')):.4f} | "
            f"{int(nov.get('n', 0))} |"
        )
    lines.append("")
    lines.append(
        f"Reference R median={metrics['ref_median']:.4f}, p99={metrics['ref_p99']:.4f} "
        "(held vs train bank)."
    )
    lines.append("")

    lines.append("## Cross-arm text divergence (MiniLM cos, same seed+step)")
    lines.append("")
    lines.append(
        f"- sample vs identity mean_cos = {metrics['div']['sample_vs_identity']['mean_cos']:.4f}"
    )
    lines.append(
        f"- sample vs zero mean_cos = {metrics['div']['sample_vs_zero']['mean_cos']:.4f}"
    )
    lines.append(
        f"- identity vs zero mean_cos = {metrics['div']['identity_vs_zero']['mean_cos']:.4f}"
    )
    lines.append("")
    lines.append(
        "Interpretation: if sample≈identity≈zero text (high cos), FiLM steer is "
        "imperceptible beyond the LM prefix channel."
    )
    lines.append("")

    lines.append("## Qualitative read")
    lines.append("")
    lines.append(metrics["qualitative"])
    lines.append("")

    lines.append("## Sample rollouts (20 seeds, sample / steer arm)")
    lines.append("")
    for i, (seed, res) in enumerate(zip(seeds, sample), start=1):
        lines.append(f"### Seed {i}")
        lines.append(f"- seed: {json.dumps(seed)}")
        for sent in res.sentences:
            clipped = sent if len(sent) < 220 else sent[:217] + "..."
            lines.append(f"  - {clipped}")
        lines.append("")

    lines.append("## A/B excerpt pairs (seed + sample vs identity vs zero)")
    lines.append("")
    for i in range(min(8, len(seeds))):
        lines.append(f"### Pair {i + 1}")
        lines.append(f"- seed: {json.dumps(seeds[i])}")
        for name, arm in (("sample", sample), ("identity", identity), ("zero", zero)):
            cont = " ".join(arm[i].sentences[:3])
            if len(cont) > 360:
                cont = cont[:357] + "..."
            lines.append(f"- {name}: {cont}")
        lines.append("")

    lines.append("## Honest conclusion")
    lines.append("")
    lines.append(metrics["conclusion"])
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _qualitative_and_conclusion(metrics: dict, arms: dict, seeds: list[str]) -> tuple[str, str]:
    div_si = metrics["div"]["sample_vs_identity"]["mean_cos"]
    div_sz = metrics["div"]["sample_vs_zero"]["mean_cos"]
    ret_s = metrics["retention"]["sample"]
    ret_i = metrics["retention"]["identity"]
    ret_z = metrics["retention"]["zero"]
    ret_shuf = metrics["retention"]["sample_shuf"]
    nov_s = metrics["novelty"]["sample"].get("novelty_rate", float("nan"))
    nov_i = metrics["novelty"]["identity"].get("novelty_rate", float("nan"))

    # Local coherence proxies: consecutive MiniLM + retention vs shuf
    cons = []
    for r in arms["sample"]:
        for row in r.instrumentation:
            if "minilm_consecutive" in row:
                cons.append(float(row["minilm_consecutive"]))
    mean_cons = float(np.mean(cons)) if cons else float("nan")

    # Entity/topic carry: inspect a few sample texts for pronoun-ish continuity
    # (honest qualitative — based on measured numbers + short manual scan)
    steer_delta = abs(div_si - 1.0)  # how different sample is from identity
    perceptible = div_si < 0.85  # same threshold family as writer concept-shuffle gate

    qual_parts = []
    qual_parts.append(
        f"(a) Local coherence: mean consecutive MiniLM cos (sample arm) = {mean_cons:.3f}; "
        f"retain@k8 vs shuffled-seed = {ret_s[-1]:.3f} vs {ret_shuf[-1]:.3f} "
        f"(Δ={ret_s[-1] - ret_shuf[-1]:+.3f}). "
        "Prefix channel carries topic/pronoun-ish continuity when Δ>0 and consecutive cos is moderate."
    )
    if perceptible:
        qual_parts.append(
            f"(b) Directional drift: sample vs identity MiniLM cos = {div_si:.3f} (<0.85) "
            f"and sample vs zero = {div_sz:.3f} — sampled FiLM concept changes rendered text "
            "beyond pure LM continuation / identity steer."
        )
    else:
        qual_parts.append(
            f"(b) Directional drift: sample vs identity MiniLM cos = {div_si:.3f} (≥0.85) "
            f"and sample vs zero = {div_sz:.3f} — sampled FiLM concept is largely "
            "imperceptible relative to the prefix LM channel (contradicts strong Fable/D5 "
            "steering claim for free-running use)."
        )
    qual_parts.append(
        f"(c) Novelty: MDN-sampled steer novelty_rate={nov_s:.3f} vs identity "
        f"novelty_rate={nov_i:.3f} (higher = more of the steer vectors sit at/below "
        "held-out median nearest-neighbor to the train bank)."
    )
    qualitative = " ".join(qual_parts)

    # Conclusion
    retain_ok = (ret_s[-1] - ret_shuf[-1]) >= 0.05
    novelty_ok = np.isfinite(nov_s) and nov_s >= 0.5
    if perceptible and retain_ok and novelty_ok:
        conclusion = (
            "PASS-leaning: sampled-concept steering over words produces text that is "
            f"locally coherent (retain@k8−shuf={ret_s[-1] - ret_shuf[-1]:+.3f}), "
            f"genuinely novel in concept space (novelty_rate={nov_s:.3f}), and "
            f"directionally distinct from identity/zero controls (sample↔identity cos={div_si:.3f})."
        )
    elif not perceptible:
        conclusion = (
            "FINDING: concept steer is largely imperceptible in free-running generation. "
            f"sample↔identity MiniLM cos={div_si:.3f}, sample↔zero={div_sz:.3f}; "
            f"retention curves are nearly overlapping "
            f"(sample@k8={ret_s[-1]:.3f}, identity={ret_i[-1]:.3f}, zero={ret_z[-1]:.3f}). "
            "The prefix/LM channel dominates; MDN-sampled FiLM adds little measurable "
            "directional control beyond identity/zero at this operating point — an important "
            "negative result relative to the Fable §3.2 / D5 two-channel thesis for rollout."
        )
    else:
        conclusion = (
            "MIXED: sampled concepts are novel "
            f"(novelty_rate={nov_s:.3f}) and text diverges somewhat from identity "
            f"(cos={div_si:.3f}), but retention/coherence evidence is weak "
            f"(retain@k8−shuf={ret_s[-1] - ret_shuf[-1]:+.3f}). "
            "Steering is detectable but does not clearly buy coherent long-horizon drift."
        )
    return qualitative, conclusion


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[forward_gen] device={device}")
    print(f"[forward_gen] loading MDN from {args.mdn}")
    predictor = load_mdn_predictor(args.mdn, device=device)

    print(f"[forward_gen] loading writer from {args.writer}")
    writer = load_writer(args.writer, device=device)

    print(f"[forward_gen] loading JEPA codec from {args.jepa}")
    model, tok = load_joint(args.jepa, device=device)
    # Prefer the joint tokenizer (matches JEPA encode); writer was trained with gpt2+specials.
    tokenizer = tok if getattr(tok, "vocab_size", 0) == writer.vocab_size else Tokenizer("gpt2")
    if tokenizer is not tok:
        print(
            f"[forward_gen] joint tok vocab={tok.vocab_size} != writer {writer.vocab_size}; "
            "using Tokenizer('gpt2')"
        )
    codec = JepaCodec(model, tok, max_tokens=128, device=device)

    print(f"[forward_gen] loading corpus from {args.corpus}")
    corpus = load_sentence_corpus(args.corpus)
    bank_train, held_vectors, _held_texts, _, _ = load_split(corpus, "")
    # Subsample bank for novelty speed if huge
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
    # Persist compact ref (without full R array)
    ref_compact = {k: v for k, v in ref.items() if k != "R"}
    (out_dir / "reference_distribution.json").write_text(
        json.dumps(ref_compact, indent=2), encoding="utf-8"
    )

    print("[forward_gen] loading MiniLM")
    minilm = _load_minilm()

    seeds = _pick_seeds(corpus, args.n_seeds, seed=args.seed)
    (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    print(f"[forward_gen] {len(seeds)} seeds x {args.steps} steps x 3 arms")

    arms: dict[str, list[ForwardGenResult]] = {
        "sample": [],
        "identity": [],
        "zero": [],
    }
    for i, seed_text in enumerate(seeds):
        print(f"[forward_gen] seed {i + 1}/{len(seeds)}")
        for mode in ("sample", "identity", "zero"):
            res = generate_document_twochannel(
                seed_text,
                predictor,
                writer,
                tokenizer,
                codec,
                steps=args.steps,
                steer_temp=args.steer_temp,
                steer_mode=mode,  # type: ignore[arg-type]
                context_len=args.context_len,
                train_bank=bank,
                ref_dist=ref,
                minilm=minilm,
                writer_temperature=args.writer_temperature,
                writer_top_k=args.writer_top_k,
                max_tokens=args.max_tokens,
            )
            arms[mode].append(res)

    retention = {
        name: _agg_retention_curve(arms[name], args.steps)
        for name in ("sample", "identity", "zero")
    }
    retention["sample_shuf"] = _shuffled_retention_curve(
        arms["sample"], seeds, minilm, args.steps, rng_seed=args.seed + 7
    )
    novelty = {name: _novelty_summary(arms[name], ref) for name in arms}
    div = {
        "sample_vs_identity": _pairwise_text_divergence(
            arms["sample"], arms["identity"], minilm
        ),
        "sample_vs_zero": _pairwise_text_divergence(arms["sample"], arms["zero"], minilm),
        "identity_vs_zero": _pairwise_text_divergence(
            arms["identity"], arms["zero"], minilm
        ),
    }

    metrics: dict[str, Any] = {
        "steps": args.steps,
        "n_seeds": len(seeds),
        "steer_temp": args.steer_temp,
        "context_len": args.context_len,
        "device": device,
        "writer_path": str(args.writer),
        "mdn_path": str(args.mdn),
        "jepa_path": str(args.jepa),
        "corpus_path": str(args.corpus),
        "retention": retention,
        "novelty": novelty,
        "div": div,
        "ref_median": float(ref["median"]),
        "ref_p99": float(ref["p99"]),
    }
    qual, conclusion = _qualitative_and_conclusion(metrics, arms, seeds)
    metrics["qualitative"] = qual
    metrics["conclusion"] = conclusion

    # Serialize rollouts
    serial = {
        "seeds": seeds,
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
            if k not in ("qualitative", "conclusion")
        },
        "qualitative": qual,
        "conclusion": conclusion,
    }
    (out_dir / "rollouts.json").write_text(
        json.dumps(serial, indent=2, default=float), encoding="utf-8"
    )

    report = write_report(out_dir, seeds, arms, metrics)
    print(f"[forward_gen] wrote {report}")
    print(f"[forward_gen] conclusion: {conclusion}")
    return {"report": str(report), "metrics": metrics, "out_dir": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "artifacts/sentence_pred/forward_gen"),
    )
    p.add_argument(
        "--mdn",
        default=str(_ROOT / "artifacts/sentence_pred/mdn.pt"),
    )
    p.add_argument(
        "--writer",
        default=str(_ROOT / "artifacts/writer_p25/writer_two_channel.pt"),
    )
    p.add_argument(
        "--jepa",
        default=str(_ROOT / "artifacts/jepa_repair_1m/jepa/joint.pt"),
    )
    p.add_argument(
        "--corpus",
        default=str(
            _ROOT
            / "artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1"
        ),
    )
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--steer-temp", type=float, default=1.0)
    p.add_argument("--context-len", type=int, default=8)
    p.add_argument("--writer-temperature", type=float, default=0.8)
    p.add_argument("--writer-top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--bank-cap", type=int, default=50000)
    p.add_argument("--held-cap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
