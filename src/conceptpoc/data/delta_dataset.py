"""Delta-sequence dataset for forward JEPA prediction.

Core result this is designed for (see experiments/test_delta_verify.py): text
*oscillates* — you can't predict the exact next concept vector (adjacent < 0.05
cosine, anti-predictive), but you CAN predict the next *change direction* from
the recent changes (ridge gets 0.46 test cosine vs baselines). Each example:
window of prior concept-*deltas* -> the next delta (normalized direction).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CachedCorpus


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


class DeltaSequenceDataset(Dataset):
    """Predict next change-direction from a window of prior change-directions.

    ``vectors`` are the (doc-contiguous) concept vectors. delta at position i
    (within a doc) is d_i = z[i+1] - z[i]. We predict d_t from the window of
    deltas *before* t. Deltas are L2-normalized (direction-only), matching the
    experiment that found the signal.
    """

    def __init__(self, corpus: CachedCorpus, context_len: int = 16) -> None:
        self.context_len = context_len
        # We need at least 2 deltas for context + 1 target => >= 4 sentences.
        self.vectors_t = torch.from_numpy(np.ascontiguousarray(corpus.vectors))
        self._windows: list[tuple[int, int]] = []  # (ctx_start, target_idx)
        for start, end in corpus.doc_slices():
            n = end - start
            if n < 4:
                continue
            # target delta index within doc is i (d_i = vec[i+1]-vec[i]); we
            # predict d_target from deltas d_target-len .. d_target-1, so the
            # first valid target needs target >= 1 (one prior delta) -> use >=2
            # with absolute index into the doc (start..end-2).
            first_tgt = start + 1
            last_tgt = end - 2  # need vec[target+1]
            for target in range(first_tgt, last_tgt + 1):
                # context deltas: d_{target-ctxlen} .. d_{target-1}
                ctx_beg = max(start, target - context_len)
                if target - ctx_beg < 2:  # need >=2 prior deltas
                    continue
                self._windows.append((ctx_beg, target))

    @property
    def dim(self) -> int:
        return int(self.vectors_t.shape[1])

    def _delta(self, i: int) -> torch.Tensor:
        """Delta direction at absolute index i (vec[i+1]-vec[i] normalized)."""
        d = self.vectors_t[i + 1] - self.vectors_t[i]
        return normalize_rows(d.unsqueeze(0)).squeeze(0)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        ctx_beg, target = self._windows[idx]
        # prior deltas: d_ctx_beg .. d_{target-1}
        deltas = [self._delta(i) for i in range(ctx_beg, target)]
        ctx = torch.stack(deltas)          # (L, dim)
        tgt = self._delta(target)          # (dim,)
        return ctx, tgt


def collate_deltas(batch):
    """Right-pad context delta windows; return (padded, mask, target_deltas)."""
    contexts, targets = zip(*batch)
    dim = contexts[0].shape[1]
    max_len = max(c.shape[0] for c in contexts)
    padded = torch.zeros(len(contexts), max_len, dim, dtype=torch.float32)
    mask = torch.ones(len(contexts), max_len, dtype=torch.bool)  # True = pad
    for i, c in enumerate(contexts):
        length = c.shape[0]
        padded[i, :length] = c
        mask[i, :length] = False
    return padded, mask, torch.stack(targets)
