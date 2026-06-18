"""Diagnostics for the codec and predictor.

- ``reconstruction_quality``: how well encode -> decode -> re-encode preserves a
  concept vector (cosine). With the retrieval decoder this measures how densely
  the bank covers the concept space.
- ``next_vector_accuracy``: predictor quality on held-out windows, reported as
  mean cosine to the true next vector and top-1 retrieval accuracy against the
  decoder bank.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..codec.base import ConceptCodec
from ..data.dataset import ConceptSequenceDataset, collate_sequences
from ..predictor.model import ConceptPredictor


@dataclass
class ReconReport:
    mean_cosine: float
    n: int


def reconstruction_quality(codec: ConceptCodec, texts: list[str]) -> ReconReport:
    original = codec.encode(texts)
    decoded = codec.decode(original)
    re_encoded = codec.encode(decoded)
    cos = np.sum(original * re_encoded, axis=1)
    return ReconReport(mean_cosine=float(np.mean(cos)), n=len(texts))


@dataclass
class PredictReport:
    mean_cosine: float
    retrieval_top1: float
    n: int


@torch.no_grad()
def next_vector_accuracy(
    model: ConceptPredictor,
    dataset: ConceptSequenceDataset,
    bank_vectors: np.ndarray | None = None,
    device: str = "cpu",
    max_items: int = 512,
) -> PredictReport:
    from torch.utils.data import DataLoader, Subset

    model.eval().to(device)
    n = min(len(dataset), max_items)
    subset = Subset(dataset, list(range(n)))
    dl = DataLoader(subset, batch_size=64, collate_fn=collate_sequences)

    bank = (
        torch.from_numpy(np.ascontiguousarray(bank_vectors)).to(device)
        if bank_vectors is not None
        else None
    )
    cos_sum, correct, count = 0.0, 0, 0
    for context, mask, target in dl:
        context, mask, target = context.to(device), mask.to(device), target.to(device)
        pred = model(context, mask)
        cos_sum += torch.cosine_similarity(pred, target, dim=-1).sum().item()
        if bank is not None:
            pred_idx = (pred @ bank.T).argmax(1)
            tgt_idx = (target @ bank.T).argmax(1)
            correct += (pred_idx == tgt_idx).sum().item()
        count += context.size(0)

    return PredictReport(
        mean_cosine=cos_sum / max(1, count),
        retrieval_top1=correct / max(1, count) if bank is not None else float("nan"),
        n=count,
    )
