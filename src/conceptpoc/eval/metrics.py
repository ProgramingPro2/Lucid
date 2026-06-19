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
from ..progress import iter_bar

_EVAL_BATCH = 64


@dataclass
class ReconReport:
    mean_cosine: float
    n: int


def reconstruction_quality(
    codec: ConceptCodec,
    texts: list[str],
    progress: bool = True,
    batch_size: int = _EVAL_BATCH,
) -> ReconReport:
    if not texts:
        return ReconReport(mean_cosine=float("nan"), n=0)

    cos_values: list[float] = []
    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in iter_bar(
        chunks,
        desc="eval: reconstruction",
        total=len(chunks),
        enabled=progress,
        unit="batch",
    ):
        original = codec.encode(batch)
        decoded = codec.decode(original)
        re_encoded = codec.encode(decoded)
        cos_values.extend(np.sum(original * re_encoded, axis=1).tolist())
    return ReconReport(mean_cosine=float(np.mean(cos_values)), n=len(texts))


@dataclass
class PredictReport:
    # Raw mean cosine between predicted and true next vector. In an anisotropic
    # space this is inflated by the shared mean direction (everything is close
    # to everything), so it is reported alongside the honest metrics below.
    mean_cosine: float
    # Mean cosine after subtracting the global mean direction. This removes the
    # anisotropy floor and reflects genuine directional prediction skill.
    centered_cosine: float
    # Mean cosine between random pairs of bank vectors (the anisotropy floor).
    # If this is close to mean_cosine, the raw cosine carries little signal.
    anisotropy: float
    # Rank-based retrieval against the (mean-centered) bank: fraction of queries
    # whose true next vector is ranked top-1 / top-5, and mean reciprocal rank.
    # These cannot be gamed by predicting the mean vector.
    retrieval_top1: float
    retrieval_top5: float
    mrr: float
    n: int


def _anisotropy(bank: torch.Tensor, sample: int = 4096) -> float:
    """Mean cosine of random distinct bank pairs (the anisotropy floor)."""
    m = bank.size(0)
    if m < 2:
        return float("nan")
    g = torch.Generator(device="cpu").manual_seed(0)
    k = min(sample, m)
    idx = torch.randperm(m, generator=g)[:k].to(bank.device)
    z = torch.nn.functional.normalize(bank[idx], dim=-1)
    sims = z @ z.T
    off = (sims.sum() - k) / (k * (k - 1))
    return float(off.item())


@torch.no_grad()
def next_vector_accuracy(
    model: ConceptPredictor,
    dataset: ConceptSequenceDataset,
    bank_vectors: np.ndarray | None = None,
    device: str = "cpu",
    max_items: int = 512,
    progress: bool = True,
) -> PredictReport:
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    model.eval().to(device)
    n = min(len(dataset), max_items)
    subset = Subset(dataset, list(range(n)))
    dl = DataLoader(subset, batch_size=64, collate_fn=collate_sequences)

    bank = bankc = mu = None
    anisotropy = float("nan")
    if bank_vectors is not None:
        bank = torch.from_numpy(np.ascontiguousarray(bank_vectors)).float().to(device)
        anisotropy = _anisotropy(bank)
        # Centered bank: subtract the global mean direction, then renormalize so
        # cosine ranking reflects direction rather than the shared offset.
        mu = bank.mean(dim=0, keepdim=True)
        bankc = F.normalize(bank - mu, dim=-1)

    cos_sum = ccos_sum = mrr_sum = 0.0
    top1 = top5 = count = 0
    for context, mask, target in iter_bar(
        dl,
        desc="eval: next-vector",
        total=len(dl),
        enabled=progress,
        unit="batch",
    ):
        context, mask, target = context.to(device), mask.to(device), target.to(device)
        pred = model(context, mask)
        cos_sum += torch.cosine_similarity(pred, target, dim=-1).sum().item()

        if bankc is not None:
            pred_c = F.normalize(pred - mu, dim=-1)
            tgt_c = F.normalize(target - mu, dim=-1)
            ccos_sum += torch.cosine_similarity(pred_c, tgt_c, dim=-1).sum().item()

            # True bank index for each target is its own nearest centered vector.
            true_idx = (tgt_c @ bankc.T).argmax(1)
            sims = pred_c @ bankc.T  # (B, M)
            true_sim = sims.gather(1, true_idx.unsqueeze(1))
            rank = (sims >= true_sim).sum(1)  # 1 = best
            top1 += (rank == 1).sum().item()
            top5 += (rank <= 5).sum().item()
            mrr_sum += (1.0 / rank.float()).sum().item()
        count += context.size(0)

    denom = max(1, count)
    has_bank = bankc is not None
    return PredictReport(
        mean_cosine=cos_sum / denom,
        centered_cosine=ccos_sum / denom if has_bank else float("nan"),
        anisotropy=anisotropy,
        retrieval_top1=top1 / denom if has_bank else float("nan"),
        retrieval_top5=top5 / denom if has_bank else float("nan"),
        mrr=mrr_sum / denom if has_bank else float("nan"),
        n=count,
    )
