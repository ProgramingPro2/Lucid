"""Closed-loop document generation for per-sentence prediction (Fable §8)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from experiments.sentence_pred.evalsuite import (  # noqa: E402
    centering,
    novelty_scores,
)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)).astype(np.float32)


def _encode_one(codec, text: str) -> np.ndarray:
    v = np.asarray(codec.encode([text]), dtype=np.float32)
    return _l2(v[0])


def _render(writer, tokenizer, concept: torch.Tensor, prefix_text: str | None) -> str:
    """Render one sentence from a concept (+ optional token prefix)."""
    if writer is None or tokenizer is None:
        # Smoke stub: deterministic pseudo-text from concept hash.
        h = int(abs(concept.detach().cpu().numpy().sum()) * 1e6) % 100000
        return f"generated sentence {h}."
    device = concept.device
    bos = getattr(tokenizer, "bos_id", 50256)
    eos = getattr(tokenizer, "eos_id", 50256)
    prefix_ids = None
    if prefix_text:
        ids = tokenizer.encode(prefix_text) if hasattr(tokenizer, "encode") else []
        prefix_ids = torch.tensor([[bos] + list(ids)[:96] + [eos]], device=device)
    with torch.no_grad():
        out = writer.generate(
            concept,
            max_tokens=64,
            bos_id=bos,
            eos_id=eos,
            temperature=0.8,
            prefix_ids=prefix_ids,
        )
    return tokenizer.decode(out[0].tolist()) if hasattr(tokenizer, "decode") else ""


def generate_document(
    seed_text: str,
    predictor,
    writer,
    tokenizer,
    codec,
    steps: int = 8,
    temperature: float = 1.0,
    feedback: str = "free",
    train_bank: np.ndarray | None = None,
    minilm=None,
    context_len: int = 32,
) -> list[str]:
    """Roll out ``steps`` sentences from ``seed_text``.

    feedback:
      - ``free``: re-encode rendered sentences and feed back (closed loop)
      - ``latent``: feed the sampled concept vector back (open latent loop)

    Per-step instrumentation attached as attribute ``.instrumentation`` on the
    returned list (MiniLM retention, n(x), consecutive cos).
    """
    if feedback not in ("free", "latent"):
        raise ValueError(f"feedback must be 'free'|'latent', got {feedback!r}")

    device = next(predictor.parameters()).device
    seed_vec = _encode_one(codec, seed_text) if codec is not None else _l2(
        np.random.RandomState(1234).randn(predictor.concept_dim).astype(np.float32)
    )
    history_vecs = [seed_vec]
    history_texts = [seed_text]
    instr: list[dict[str, Any]] = []

    mu_bank = None
    bank_c = None
    if train_bank is not None and len(train_bank):
        mu_bank, bank_c = centering(train_bank)

    seed_minilm = None
    if minilm is not None:
        try:
            seed_minilm = _l2(np.asarray(minilm.encode([seed_text]), dtype=np.float32)[0])
        except Exception:
            seed_minilm = None

    prev_minilm = seed_minilm
    for step in range(int(steps)):
        # Build context window from history (up to context_len).
        window = history_vecs[-context_len:]
        ctx = torch.from_numpy(np.stack(window).astype(np.float32)).unsqueeze(0).to(device)
        pad = torch.zeros(1, ctx.size(1), dtype=torch.bool, device=device)
        with torch.no_grad():
            sample = predictor.sample(ctx, pad, n_samples=1, temperature=temperature)
            concept = F.normalize(sample[:, 0, :], dim=-1)  # (1, D)

        concept_np = concept.detach().cpu().numpy()[0]
        text = _render(writer, tokenizer, concept, history_texts[-1])

        if feedback == "free" and codec is not None:
            fed = _encode_one(codec, text)
        else:
            fed = _l2(concept_np)

        # Instrumentation
        row: dict[str, Any] = {
            "step": step,
            "text": text,
            "feedback": feedback,
            "temperature": float(temperature),
        }
        if mu_bank is not None and bank_c is not None:
            row["n_x"] = float(novelty_scores(fed[None, :], bank_c, mu_bank)[0])
        if len(history_vecs) >= 1:
            row["consecutive_cos"] = float(np.dot(_l2(fed), _l2(history_vecs[-1])))
        if minilm is not None:
            try:
                mv = _l2(np.asarray(minilm.encode([text]), dtype=np.float32)[0])
                if seed_minilm is not None:
                    row["minilm_retention"] = float(np.dot(mv, seed_minilm))
                if prev_minilm is not None:
                    row["minilm_consecutive"] = float(np.dot(mv, prev_minilm))
                prev_minilm = mv
            except Exception as e:
                row["minilm_error"] = str(e)

        instr.append(row)
        history_vecs.append(fed)
        history_texts.append(text)

    class _SentList(list):
        instrumentation: list[dict[str, Any]]

    out: _SentList = _SentList(history_texts[1:])  # generated only (exclude seed)
    out.instrumentation = instr
    return out


def rollout_batch(
    seeds: list[str],
    predictor,
    writer,
    tokenizer,
    codec,
    steps: int = 8,
    temperature: float = 1.0,
    feedback: str = "free",
    train_bank: np.ndarray | None = None,
    minilm=None,
) -> list[list[str]]:
    """Roll out many seed documents; returns list of sentence lists."""
    return [
        generate_document(
            s,
            predictor,
            writer,
            tokenizer,
            codec,
            steps=steps,
            temperature=temperature,
            feedback=feedback,
            train_bank=train_bank,
            minilm=minilm,
        )
        for s in seeds
    ]
