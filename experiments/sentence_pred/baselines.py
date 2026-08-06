"""Baseline query_fns for anti-mirage M3 controls (Fable §7.2).

All callables match ``query_fn(context, pad_mask) -> (B, D)`` unit vectors so
they plug directly into ``evalsuite.m3_eval``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _last_real(context: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Last non-pad context vector per row: (B, D)."""
    b, t, d = context.shape
    if pad_mask is None:
        return context[:, -1, :]
    # pad_mask True = pad; last real index = (#real) - 1
    n_real = (~pad_mask).sum(dim=1).clamp_min(1)  # (B,)
    idx = (n_real - 1).long()
    return context[torch.arange(b, device=context.device), idx]


def identity(context: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Killer control: query = z_t (last context vector)."""
    return F.normalize(_last_real(context, pad_mask), dim=-1)


def prefix_mean(context: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """query = normalize(mean of real context vectors)."""
    if pad_mask is None:
        return F.normalize(context.mean(dim=1), dim=-1)
    mask = (~pad_mask).to(dtype=context.dtype).unsqueeze(-1)  # (B,T,1)
    s = (context * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(s / denom, dim=-1)


def make_knn_transition(
    train_bank: np.ndarray | torch.Tensor,
    train_successors: np.ndarray | torch.Tensor,
) -> callable:
    """Nearest train-bank z to z_t → that sentence's true successor vector.

    ``train_bank`` (N, D) and ``train_successors`` (N, D) are aligned: row i of
    successors is the true next vector of bank row i (or the row itself if last).
    """
    bank = torch.as_tensor(np.asarray(train_bank, dtype=np.float32))
    bank = F.normalize(bank, dim=-1)
    succ = torch.as_tensor(np.asarray(train_successors, dtype=np.float32))
    succ = F.normalize(succ, dim=-1)

    def knn_transition(
        context: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        zt = F.normalize(_last_real(context, pad_mask), dim=-1)  # (B, D)
        b_dev = bank.to(device=zt.device, dtype=zt.dtype)
        s_dev = succ.to(device=zt.device, dtype=zt.dtype)
        sims = zt @ b_dev.T  # (B, N)
        nn = sims.argmax(dim=-1)
        return s_dev[nn]

    return knn_transition


def make_random_bf(dim: int = 512, seed: int = 1234) -> callable:
    """Random unit vector baseline (harness self-test; must land ~1/n_cand)."""
    rng = torch.Generator().manual_seed(int(seed))

    def random_bf(
        context: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b = context.size(0)
        d = context.size(-1) if context.dim() == 3 else dim
        g = torch.randn(b, d, generator=rng, dtype=context.dtype)
        return F.normalize(g.to(device=context.device), dim=-1)

    return random_bf


def build_train_successor_bank(corpus) -> tuple[np.ndarray, np.ndarray]:
    """Build (bank, successors) over TRAIN docs only (document-disjoint).

    For each train sentence that has a next sentence in the same doc, bank row
    is that sentence and successor is the next. Last-in-doc sentences omitted.
    """
    bank_rows: list[np.ndarray] = []
    succ_rows: list[np.ndarray] = []
    for doc_id in np.asarray(corpus.train_doc_ids, dtype=np.int64).tolist():
        a, b = corpus.doc_slice(int(doc_id))
        if b - a < 2:
            continue
        vecs = corpus.sent_vecs[a:b]
        bank_rows.append(vecs[:-1])
        succ_rows.append(vecs[1:])
    if not bank_rows:
        d = int(corpus.sent_vecs.shape[1])
        return np.zeros((0, d), np.float32), np.zeros((0, d), np.float32)
    return (
        np.concatenate(bank_rows, axis=0).astype(np.float32),
        np.concatenate(succ_rows, axis=0).astype(np.float32),
    )


def all_baseline_fns(corpus, seed: int = 1234) -> dict[str, callable]:
    """Named baseline query_fns ready for ``m3_eval``."""
    bank, succ = build_train_successor_bank(corpus)
    dim = int(corpus.sent_vecs.shape[1])
    return {
        "identity": identity,
        "prefix_mean": prefix_mean,
        "knn_transition": make_knn_transition(bank, succ),
        "random_bf": make_random_bf(dim=dim, seed=seed),
    }
