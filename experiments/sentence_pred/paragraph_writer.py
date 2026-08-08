"""Render G sentences under a single frozen paragraph concept (phase-3 contract).

Inference mirror of phase-3 training: one pooled topic vector steers FiLM for
every sentence; local coherence comes from chaining each rendered sentence as
the next prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.writer.train_phase2 import PREFIX_MAX  # noqa: E402

G_DEFAULT = 5
MAX_NEW_TOKENS = 96


def _as_concept_batch(paragraph_concept: torch.Tensor, device: torch.device) -> torch.Tensor:
    c = paragraph_concept
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c, dtype=torch.float32)
    c = c.detach().to(device=device, dtype=torch.float32)
    if c.dim() == 1:
        c = c.unsqueeze(0)
    # L2-normalize for safety (FiLM expects unit-ish concepts).
    n = c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return c / n


def _prefix_ids(
    tokenizer,
    prev_text: str | None,
    device: torch.device,
) -> torch.Tensor:
    """Build [BOS]+prev[:96]+[EOS], or [BOS] alone when prev_text is None/empty."""
    bos = int(tokenizer.bos_id)
    eos = int(tokenizer.eos_id)
    if prev_text is None or not str(prev_text).strip():
        return torch.tensor([[bos]], dtype=torch.long, device=device)
    ids = list(tokenizer.encode(str(prev_text)))[:PREFIX_MAX]
    return torch.tensor([[bos] + ids + [eos]], dtype=torch.long, device=device)


@torch.no_grad()
def render_paragraph(
    writer,
    tokenizer,
    paragraph_concept: torch.Tensor,
    first_prefix: str | None = None,
    n_sentences: int = G_DEFAULT,
    temperature: float = 0.8,
    top_k: int = 50,
    max_tokens: int = MAX_NEW_TOKENS,
    repetition_penalty: float = 1.1,
) -> list[str]:
    """Render ``n_sentences`` under one frozen paragraph concept with prefix chaining.

    Sentence 0 uses ``first_prefix`` (or BOS-only if None). Each subsequent
    sentence uses the previous rendered sentence text as its prefix. Returns a
    list of detokenized strings of length ``n_sentences``.
    """
    writer.eval()
    device = next(writer.parameters()).device
    concept = _as_concept_batch(paragraph_concept, device)
    bos = int(tokenizer.bos_id)
    eos = int(tokenizer.eos_id)

    texts: list[str] = []
    prev: str | None = first_prefix
    for _ in range(int(n_sentences)):
        prefix = _prefix_ids(tokenizer, prev, device)
        out = writer.generate(
            concept,
            max_tokens=int(max_tokens),
            bos_id=bos,
            eos_id=eos,
            temperature=float(temperature),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            prefix_ids=prefix,
        )
        # generate returns newly produced ids only (may include trailing EOS).
        text = tokenizer.decode(out[0].tolist()).strip()
        texts.append(text)
        prev = text
    return texts


def join_paragraph(sentences: Sequence[str]) -> str:
    return " ".join(s.strip() for s in sentences if s and str(s).strip())
