"""Concept-sequence dataset.

Turns the cached per-document sentence vectors into fixed-length training
windows. Each example is a context of concept vectors and the next concept
vector to predict. Windows never cross document boundaries.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CachedCorpus


class ConceptSequenceDataset(Dataset):
    def __init__(self, corpus: CachedCorpus, context_len: int = 16) -> None:
        self.vectors = torch.from_numpy(np.ascontiguousarray(corpus.vectors))
        self.context_len = context_len
        self._windows: list[tuple[int, int]] = []
        for start, end in corpus.doc_slices():
            n = end - start
            if n < 2:
                continue
            # Predict each position from its prefix (min context of 1).
            for target in range(start + 1, end):
                ctx_start = max(start, target - context_len)
                self._windows.append((ctx_start, target))

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        ctx_start, target = self._windows[idx]
        context = self.vectors[ctx_start:target]
        next_vec = self.vectors[target]
        return context, next_vec


def collate_sequences(batch):
    """Right-pad variable-length contexts; return a key-padding mask (True=pad).

    Right padding keeps every real query position able to attend to at least
    itself under a causal mask, which avoids fully-masked attention rows (a
    source of NaNs). The model gathers each sequence's last real position.
    """
    contexts, targets = zip(*batch)
    dim = contexts[0].shape[1]
    max_len = max(c.shape[0] for c in contexts)
    padded = torch.zeros(len(contexts), max_len, dim, dtype=torch.float32)
    mask = torch.ones(len(contexts), max_len, dtype=torch.bool)  # True = pad
    for i, ctx in enumerate(contexts):
        length = ctx.shape[0]
        padded[i, :length] = ctx
        mask[i, :length] = False
    return padded, mask, torch.stack(targets)
