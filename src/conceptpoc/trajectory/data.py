"""K-step latent trajectory dataset.

Each item is (context_window, target_next_K): the previous C sentence concept
vectors, and the next K concept vectors the planner must learn to hit. The
targets come from the frozen JEPA encoder (ground truth from real text), so the
planner learns a forward-predictable objective rather than one-step fuzziness.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..codec.jepa_codec import JepaCodec
from ..data.corpus import load_documents


class TrajectoryDataset(Dataset):
    """In-memory K-step trajectory dataset."""

    def __init__(self, corpus_path: str, codec: JepaCodec, C: int = 8,
                 K: int = 5, max_docs: int | None = None) -> None:
        docs = load_documents(corpus_path)
        self.C, self.K = C, K
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for d in docs[: max_docs or len(docs)]:
            sents = [s for s in d.sentences if s.strip()]
            if len(sents) < C + K:
                continue
            vecs = codec.encode(sents)  # (N, dim)
            for i in range(len(vecs) - C - K + 1):
                ctx = vecs[i:i + C]
                tgt = vecs[i + C:i + C + K]
                self.items.append((ctx, tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


class TrajectoryIterable(IterableDataset):
    """Streaming version for large corpora: encodes doc-by-doc, yields windows."""

    def __init__(self, corpus_path: str, codec: JepaCodec, C: int = 8,
                 K: int = 5, max_docs: int | None = None) -> None:
        self.corpus_path = corpus_path
        self.codec = codec
        self.C, self.K = C, K
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        docs = load_documents(self.corpus_path)
        if self.max_docs is not None:
            docs = docs[: self.max_docs]
        for d in docs:
            sents = [s for s in d.sentences if s.strip()]
            if len(sents) < self.C + self.K:
                continue
            vecs = self.codec.encode(sents)
            for i in range(len(vecs) - self.C - self.K + 1):
                yield (vecs[i:i + self.C], vecs[i + self.C:i + self.C + self.K])


def collate(batch):
    ctxs = torch.stack([torch.from_numpy(b[0]) for b in batch])
    tgts = torch.stack([torch.from_numpy(b[1]) for b in batch])
    return ctxs, tgts


def make_trajectory_loader(corpus_path: str, codec: JepaCodec, C: int = 8,
                           K: int = 5, batch_size: int = 64, num_workers: int = 0,
                           shuffle: bool = True, max_docs: int | None = None,
                           streaming: bool = True):
    if streaming:
        ds = TrajectoryIterable(corpus_path, codec, C=C, K=K, max_docs=max_docs)
        from torch.utils.data import DataLoader
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          collate_fn=collate, drop_last=True)
    ds = TrajectoryDataset(corpus_path, codec, C=C, K=K, max_docs=max_docs)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate, drop_last=True)
