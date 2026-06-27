"""Instruction/theme conditioning data (E10).

Builds and caches ``(instruction_vecs, response_vecs, segment_ids)`` windows for
teacher-forced conditioned training. Two builders are provided:

- ``build_synthetic_instructions``: derive themed/template instructions from a
  local corpus (no network). Optionally author instructions with a local LLM
  via the lemonade client wiring (lazy, opt-in).
- ``build_borrowed_instructions``: load a small subset of an open instruction
  set (Dolly/Alpaca) via HF ``datasets`` (lazy import, opt-in).

All network/HF/LLM access is lazy and behind explicit flags so importing this
module (and running tests) never downloads anything or contacts a server.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..codec.base import ConceptCodec
from .corpus import Document
from .segment import segment_sentences


@dataclass
class InstructionPair:
    instruction_vecs: np.ndarray  # (Ni, dim) float32
    response_vecs: np.ndarray  # (Nr, dim) float32
    theme_vec: np.ndarray | None = None  # (dim,) float32, optional


class InstructionConceptDataset(Dataset):
    """Sequence pairs for conditioned next-vector prediction.

    Each item is ``(instruction_vecs, response_vecs, segment_ids)`` where
    ``segment_ids`` is 0 over the instruction span and 1 over the response span.
    An optional per-pair theme vector is carried alongside.
    """

    def __init__(self, pairs: list[InstructionPair]) -> None:
        if not pairs:
            raise ValueError("InstructionConceptDataset requires at least one pair")
        self.pairs = pairs
        self._dim = int(pairs[0].instruction_vecs.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def has_theme(self) -> bool:
        return self.pairs[0].theme_vec is not None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        instr = torch.from_numpy(np.ascontiguousarray(pair.instruction_vecs))
        resp = torch.from_numpy(np.ascontiguousarray(pair.response_vecs))
        segment_ids = torch.cat(
            [
                torch.zeros(instr.shape[0], dtype=torch.long),
                torch.ones(resp.shape[0], dtype=torch.long),
            ]
        )
        theme = (
            torch.from_numpy(np.ascontiguousarray(pair.theme_vec))
            if pair.theme_vec is not None
            else None
        )
        return instr, resp, segment_ids, theme

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        instr = np.concatenate([p.instruction_vecs for p in self.pairs], axis=0)
        resp = np.concatenate([p.response_vecs for p in self.pairs], axis=0)
        instr_off = np.cumsum([0] + [p.instruction_vecs.shape[0] for p in self.pairs])
        resp_off = np.cumsum([0] + [p.response_vecs.shape[0] for p in self.pairs])
        arrays = dict(
            instr=instr.astype(np.float32),
            resp=resp.astype(np.float32),
            instr_off=instr_off.astype(np.int64),
            resp_off=resp_off.astype(np.int64),
        )
        if self.has_theme:
            arrays["themes"] = np.stack(
                [p.theme_vec for p in self.pairs]
            ).astype(np.float32)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "InstructionConceptDataset":
        data = np.load(Path(path), allow_pickle=False)
        instr, resp = data["instr"], data["resp"]
        instr_off, resp_off = data["instr_off"], data["resp_off"]
        themes = data["themes"] if "themes" in data.files else None
        pairs: list[InstructionPair] = []
        for i in range(len(instr_off) - 1):
            pairs.append(
                InstructionPair(
                    instruction_vecs=instr[instr_off[i] : instr_off[i + 1]],
                    response_vecs=resp[resp_off[i] : resp_off[i + 1]],
                    theme_vec=themes[i] if themes is not None else None,
                )
            )
        return cls(pairs)


def collate_instructions(batch):
    """Right-pad pairs into one ``[instruction | response]`` causal sequence.

    Returns ``(seq, pad_mask, segment_ids, targets, loss_mask, theme)``:
    - ``seq`` (B, L, dim): the concatenated, right-padded sequence.
    - ``pad_mask`` (B, L): True at padding positions.
    - ``segment_ids`` (B, L): 0 over instruction, 1 over response (0 at pad).
    - ``targets`` (B, L, dim): ``seq`` shifted left by one (next vector).
    - ``loss_mask`` (B, L): True where the predicted next vector is a response
      vector (supervise response positions only).
    - ``theme`` (B, dim) or None.
    """
    instrs, resps, seg_ids, themes = zip(*batch)
    dim = instrs[0].shape[1]
    lengths = [i.shape[0] + r.shape[0] for i, r in zip(instrs, resps)]
    max_len = max(lengths)
    bsz = len(batch)

    seq = torch.zeros(bsz, max_len, dim, dtype=torch.float32)
    targets = torch.zeros(bsz, max_len, dim, dtype=torch.float32)
    pad_mask = torch.ones(bsz, max_len, dtype=torch.bool)
    segment_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    loss_mask = torch.zeros(bsz, max_len, dtype=torch.bool)

    for b, (instr, resp, seg) in enumerate(zip(instrs, resps, seg_ids)):
        ni, nr = instr.shape[0], resp.shape[0]
        length = ni + nr
        full = torch.cat([instr, resp], dim=0)
        seq[b, :length] = full
        pad_mask[b, :length] = False
        segment_ids[b, :length] = seg
        # Position t predicts seq[t+1]; supervise where seq[t+1] is a response
        # vector, i.e. t in [ni-1, length-2].
        targets[b, : length - 1] = full[1:length]
        loss_mask[b, ni - 1 : length - 1] = True

    theme = None
    if themes[0] is not None:
        theme = torch.stack(themes)
    return seq, pad_mask, segment_ids, targets, loss_mask, theme


def _encode_sentences(
    codec: ConceptCodec, text: str, max_sentences: int, min_chars: int
) -> np.ndarray:
    sents = segment_sentences(text, min_chars=min_chars) or [text.strip()]
    sents = [s for s in sents if s][:max_sentences]
    if not sents:
        sents = [text.strip() or "."]
    return codec.encode(sents).astype(np.float32)


def build_synthetic_instructions(
    docs: list[Document],
    codec: ConceptCodec,
    templates: list[str],
    themes: list[str],
    *,
    max_response_sentences: int = 8,
    max_instruction_sentences: int = 4,
    min_chars: int = 1,
    author_with_llm: bool = False,
    llm_author=None,
) -> InstructionConceptDataset:
    """Pair corpus passages with template/themed instructions (no network).

    When ``author_with_llm`` is True an ``llm_author`` callable (or the lazily
    built lemonade author) rewrites each template into a natural instruction;
    this is opt-in and never triggered during import/tests.
    """
    if author_with_llm and llm_author is None:
        llm_author = build_lemonade_author()

    pairs: list[InstructionPair] = []
    for i, doc in enumerate(docs):
        if len(doc.sentences) < 2:
            continue
        template = templates[i % len(templates)]
        theme = themes[i % len(themes)] if themes else ""
        instruction_text = template.format(theme=theme)
        if author_with_llm and llm_author is not None:
            instruction_text = llm_author(instruction_text, doc.sentences)
        instr_vecs = _encode_sentences(
            codec, instruction_text, max_instruction_sentences, min_chars
        )
        resp_vecs = codec.encode(doc.sentences[:max_response_sentences]).astype(
            np.float32
        )
        theme_vec = (
            codec.encode([theme])[0].astype(np.float32)
            if "{theme}" in template and theme
            else None
        )
        pairs.append(
            InstructionPair(
                instruction_vecs=instr_vecs,
                response_vecs=resp_vecs,
                theme_vec=theme_vec,
            )
        )
    if not pairs:
        raise ValueError("no synthetic instruction pairs produced from corpus")
    # Themes must be uniform across pairs for batching; drop them if mixed.
    if any(p.theme_vec is None for p in pairs):
        for p in pairs:
            p.theme_vec = None
    return InstructionConceptDataset(pairs)


def build_borrowed_instructions(
    codec: ConceptCodec,
    hf_dataset: str,
    *,
    split: str = "train",
    max_pairs: int = 256,
    instruction_field: str = "instruction",
    response_field: str = "response",
    max_response_sentences: int = 8,
    max_instruction_sentences: int = 4,
) -> InstructionConceptDataset:
    """Load a small subset of an open instruction set via HF ``datasets``.

    The HF import and download happen here only (lazy); callers opt in via the
    ``build-instructions --source borrow`` CLI path.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split=split)
    pairs: list[InstructionPair] = []
    for record in ds:
        instruction = str(record.get(instruction_field, "")).strip()
        response = str(record.get(response_field, "")).strip()
        if not instruction or not response:
            continue
        instr_vecs = _encode_sentences(
            codec, instruction, max_instruction_sentences, min_chars=1
        )
        resp_sents = segment_sentences(response, min_chars=1) or [response]
        resp_vecs = codec.encode(resp_sents[:max_response_sentences]).astype(
            np.float32
        )
        pairs.append(
            InstructionPair(instruction_vecs=instr_vecs, response_vecs=resp_vecs)
        )
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise ValueError(f"no usable instruction pairs in {hf_dataset}:{split}")
    return InstructionConceptDataset(pairs)


def combine_instruction_datasets(
    datasets: list[InstructionConceptDataset],
) -> InstructionConceptDataset:
    """Concatenate datasets into one, keeping themes uniform for batching.

    If any source pair carries a theme vector, every pair without one is given a
    zero theme vector (a learned "no specific theme" offset) so the combined set
    is theme-uniform; otherwise themes are left absent. This lets a borrowed set
    (no themes) and a themed synthetic set train together under ``use_theme``.
    """
    datasets = [d for d in datasets if d is not None and len(d) > 0]
    if not datasets:
        raise ValueError("combine_instruction_datasets requires a non-empty dataset")
    dim = datasets[0].dim
    pairs: list[InstructionPair] = []
    for ds in datasets:
        if ds.dim != dim:
            raise ValueError(f"dim mismatch when combining: {ds.dim} != {dim}")
        pairs.extend(ds.pairs)
    if any(p.theme_vec is not None for p in pairs):
        zero = np.zeros(dim, dtype=np.float32)
        for p in pairs:
            if p.theme_vec is None:
                p.theme_vec = zero
    return InstructionConceptDataset(pairs)


def build_lemonade_author(model: str | None = None, base_url: str | None = None):
    """Lazily build an LLM instruction author reusing the lemonade wiring.

    Mirrors ``reward.judge.LemonadeJudge``: the OpenAI client is imported and
    constructed only when this is called, so nothing connects at import time.
    """
    import os

    from openai import OpenAI

    client = OpenAI(
        base_url=base_url
        or os.environ.get("LEMONADE_BASE_URL", "http://localhost:13305/api/v1"),
        api_key=os.environ.get("LEMONADE_API_KEY", "lemonade"),
    )
    chosen = model or os.environ.get("LEMONADE_JUDGE_MODEL", "")
    if not chosen:
        raise ValueError("LEMONADE_JUDGE_MODEL must be set for LLM-authored instructions")

    def author(template: str, response_sentences: list[str]) -> str:
        passage = " ".join(response_sentences[:4])
        resp = client.chat.completions.create(
            model=chosen,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a single concise instruction that the following "
                        "passage would be a good response to. Reply with only the "
                        "instruction."
                    ),
                },
                {"role": "user", "content": f"Template hint: {template}\nPassage: {passage}"},
            ],
            temperature=0.0,
            max_completion_tokens=64,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or template

    return author
