"""Windowed sentence dataset for joint training.

Each example is a window of up to ``window`` consecutive sentences from a single
document (non-overlapping windows; windows with fewer than two sentences are
dropped). The collate function tokenizes and packs a batch into:

- ``sent_ids``  (B, S, L) token ids
- ``sent_pad``  (B, S, L) True where pad
- ``sent_valid`` (B, S) True where the slot holds a real sentence
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..data.corpus import Document
from .tokenizer import Tokenizer


class JepaWindowDataset(Dataset):
    def __init__(self, docs: list[Document], window: int) -> None:
        self.window = window
        self._windows: list[list[str]] = []
        for doc in docs:
            sents = doc.sentences
            for start in range(0, len(sents), window):
                chunk = sents[start : start + window]
                if len(chunk) >= 2:
                    self._windows.append(chunk)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> list[str]:
        return self._windows[idx]


def make_collate(tokenizer: Tokenizer, window: int, max_tokens: int):
    def collate(batch: list[list[str]]):
        flat_sentences: list[str] = []
        positions: list[tuple[int, int]] = []
        for b, window_sents in enumerate(batch):
            for s, sent in enumerate(window_sents[:window]):
                flat_sentences.append(sent)
                positions.append((b, s))

        ids, pad = tokenizer.encode_batch(flat_sentences, max_tokens)
        length = ids.size(1)
        bsz = len(batch)

        sent_ids = torch.full((bsz, window, length), tokenizer.pad_id, dtype=torch.long)
        sent_pad = torch.ones(bsz, window, length, dtype=torch.bool)
        sent_valid = torch.zeros(bsz, window, dtype=torch.bool)
        for n, (b, s) in enumerate(positions):
            sent_ids[b, s] = ids[n]
            sent_pad[b, s] = pad[n]
            sent_valid[b, s] = True
        return sent_ids, sent_pad, sent_valid

    return collate
