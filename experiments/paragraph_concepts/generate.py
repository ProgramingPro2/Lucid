"""Closed-loop paragraph-scale generation via ranking reasoner + FiLM writer.

Encode a seed paragraph, repeatedly predict the next paragraph key with the
trained ranking reasoner, retrieve the nearest corpus paragraph concept (or
score a candidate bank), render through the FiLM Phase-2 writer, and feed back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.data.segment import segment_sentences  # noqa: E402
from conceptpoc.persistence import load_joint, load_writer  # noqa: E402

from .data import CachedParagraphCorpus, _l2_normalize, load_and_build  # noqa: E402
from .reasoner import RankingReasoner, RankingReasonerConfig  # noqa: E402


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_reasoner(path: str | Path, device: str | torch.device = "cpu") -> RankingReasoner:
    from dataclasses import fields as dc_fields

    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw = blob.get("config") or {}
    allowed = {f.name for f in dc_fields(RankingReasonerConfig)}
    cfg = RankingReasonerConfig(**{k: v for k, v in raw.items() if k in allowed})
    model = RankingReasoner(int(blob["concept_dim"]), cfg)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def retrieve_nearest(
    query: torch.Tensor,
    bank: torch.Tensor,
    *,
    top_k: int = 1,
    exclude: set[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Return top-k nearest bank vectors (cosine) and their indices."""
    q = F.normalize(query.view(1, -1), dim=-1)
    b = F.normalize(bank, dim=-1)
    sims = (b @ q.squeeze(0))
    if exclude:
        for i in exclude:
            if 0 <= i < sims.numel():
                sims[i] = -2.0
    k = min(top_k, sims.numel())
    vals, idxs = torch.topk(sims, k=k)
    return bank[idxs], idxs.tolist()


@torch.no_grad()
def render_paragraph(
    writer,
    tokenizer,
    concept: torch.Tensor,
    *,
    max_tokens: int = 96,
    temperature: float = 0.7,
    n_sentences: int = 5,
) -> str:
    """Render one concept vector into a multi-sentence paragraph via FiLM writer.

    The writer is sentence-conditioned; we sample ``n_sentences`` times from the
    same paragraph concept (with mild noise) and join them.
    """
    texts: list[str] = []
    base = F.normalize(concept.view(1, -1), dim=-1)
    for i in range(n_sentences):
        c = base
        if i > 0:
            noise = torch.randn_like(c) * 0.05
            c = F.normalize(c + noise, dim=-1)
        token_ids = writer.generate(
            c,
            max_tokens=max_tokens,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
        )[0].tolist()
        text = tokenizer.decode(token_ids).strip()
        if text:
            texts.append(text)
    return " ".join(texts)


@torch.no_grad()
def generate_continuation(
    seed_text: str,
    reasoner: RankingReasoner,
    codec: JepaCodec,
    writer,
    tokenizer,
    bank_vectors: np.ndarray,
    *,
    steps: int = 3,
    window: int = 8,
    group_size: int = 5,
    device: torch.device | str = "cpu",
    temperature: float = 0.7,
) -> list[str]:
    """Closed-loop: seed -> reasoner key -> retrieve -> writer -> encode -> repeat."""
    dev = torch.device(device) if isinstance(device, str) else device
    reasoner.to(dev).eval()
    writer.to(dev).eval()

    seed_sents = segment_sentences(seed_text) or [seed_text]
    # Encode whatever sentences we have; do not pad by repeating the prompt.
    seed_vecs = codec.encode(seed_sents)
    para = _l2_normalize(seed_vecs.mean(axis=0))

    bank = torch.from_numpy(np.ascontiguousarray(bank_vectors, dtype=np.float32)).to(dev)
    context_vecs = [para]
    paragraphs = [seed_text.strip()]
    used: set[int] = set()

    for _ in range(steps):
        window_np = np.stack(context_vecs[-window:]).astype(np.float32)
        ctx = torch.from_numpy(window_np).unsqueeze(0).to(dev)
        key = reasoner(ctx).squeeze(0)
        nearest, idxs = retrieve_nearest(key, bank, top_k=5, exclude=used)
        chosen = nearest[0]
        used.add(idxs[0])
        text = render_paragraph(
            writer,
            tokenizer,
            chosen,
            n_sentences=group_size,
            temperature=temperature,
        )
        paragraphs.append(text)
        # Feed back: re-encode rendered text into a paragraph concept
        rendered_sents = segment_sentences(text) or [text]
        if len(rendered_sents) < 1:
            rendered_sents = [text]
        enc = codec.encode(rendered_sents[: max(group_size, len(rendered_sents))])
        feedback = _l2_normalize(enc.mean(axis=0))
        context_vecs.append(feedback)

    return paragraphs


def run_samples(
    reasoner_path: str = "artifacts/paragraph_reasoner.pt",
    writer_path: str = "artifacts/writer_p2_film/writer_cond.pt",
    jepa_path: str = "artifacts/jepa_v3_128t/jepa/joint_robust.pt",
    corpus_cache: str | None = None,
    seed_prompt: str = "The scientist discovered that",
    n_samples: int = 2,
    steps: int = 3,
    device: str = "auto",
    max_docs: int = 2000,
    group_size: int = 5,
) -> list[list[str]]:
    dev = _device(device)
    print(f"[generate] device={dev}")

    reasoner = load_reasoner(reasoner_path, device=dev)
    blob = torch.load(reasoner_path, map_location="cpu", weights_only=False)
    group_size = int(blob.get("group_size", group_size))
    window = int(blob.get("window", 8))

    model, tok = load_joint(jepa_path, device=str(dev))
    codec = JepaCodec(model, tok, 128, device=str(dev))
    writer = load_writer(writer_path, device=str(dev))

    # Candidate bank: paragraph vectors from corpus cache (or rebuild lightly)
    bank_vecs: np.ndarray
    if corpus_cache and Path(corpus_cache).exists():
        cached = CachedParagraphCorpus.load(corpus_cache)
        bank_vecs = cached.vectors
        group_size = cached.group_size
    else:
        # Prefer any cache in default dir
        cache_dir = Path("artifacts/paragraph_concepts_cache")
        caches = sorted(cache_dir.glob(f"paragraphs_G{group_size}_*.npz")) if cache_dir.exists() else []
        if caches:
            cached = CachedParagraphCorpus.load(caches[-1])
            bank_vecs = cached.vectors
        else:
            print("[generate] building paragraph bank (may take a bit)...")
            cached = load_and_build(
                group_size=group_size,
                max_docs=max_docs,
                device=str(dev),
                cache_dir=str(cache_dir),
            )
            bank_vecs = cached.vectors

    print(f"[generate] bank size={len(bank_vecs)}  G={group_size}  window={window}")
    print(f"[generate] seed: {seed_prompt!r}\n")

    all_samples: list[list[str]] = []
    for s in range(n_samples):
        # Slight temperature jitter across samples
        temp = 0.6 + 0.2 * s
        paras = generate_continuation(
            seed_prompt,
            reasoner,
            codec,
            writer,
            tok,
            bank_vecs,
            steps=steps,
            window=window,
            group_size=group_size,
            device=dev,
            temperature=temp,
        )
        all_samples.append(paras)
        print("=" * 60)
        print(f"SAMPLE {s + 1}")
        print("=" * 60)
        for i, p in enumerate(paras):
            label = "SEED" if i == 0 else f"PARA+{i}"
            print(f"[{label}] {p}\n")
    return all_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paragraph-scale closed-loop generation")
    p.add_argument("--reasoner", default="artifacts/paragraph_reasoner.pt")
    p.add_argument("--writer", default="artifacts/writer_p2_film/writer_cond.pt")
    p.add_argument("--jepa", default="artifacts/jepa_v3_128t/jepa/joint_robust.pt")
    p.add_argument("--seed", default="The scientist discovered that")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--dev", default="auto")
    p.add_argument("--group-size", type=int, default=5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> list[list[str]]:
    args = parse_args(argv)
    return run_samples(
        reasoner_path=args.reasoner,
        writer_path=args.writer,
        jepa_path=args.jepa,
        seed_prompt=args.seed,
        n_samples=args.samples,
        steps=args.steps,
        device=args.dev,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
