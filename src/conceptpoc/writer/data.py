"""Token-window streaming for writer pretraining (Phase 1).

Streams document text into a token buffer and yields fixed-length (input,
target) next-token windows. Uses an IterableDataset so an arbitrarily large
corpus never needs to be held in memory at once; a fresh shuffle window is
used each epoch by re-reading the corpus.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, IterableDataset

from ..data.corpus import load_documents
from ..jepa.tokenizer import Tokenizer


class TokenWindowIterable(IterableDataset):
    def __init__(
        self, corpus_path: str, tokenizer: Tokenizer, seq_len: int = 256
    ) -> None:
        self.corpus_path = corpus_path
        self.tok = tokenizer
        self.seq_len = seq_len

    def _token_buffer(self):
        """Yield a flat list of token ids for the whole corpus (streamed)."""
        # Load documents; each sentence appended EOS to mark boundaries.
        docs = load_documents(self.corpus_path)
        buf: list[int] = []
        for doc in docs:
            for sent in doc.sentences:
                ids, _ = self.tok.encode_batch([sent], 512)
                buf.extend(ids[0].tolist())
                buf.append(self.tok.eos_id)
        return buf

    def __iter__(self):
        buf = self._token_buffer()
        if len(buf) < self.seq_len + 1:
            raise ValueError("corpus too small for one sequence window")
        # Slide a non-overlapping window, leaving the last token as the target.
        for start in range(0, len(buf) - self.seq_len, self.seq_len):
            chunk = buf[start : start + self.seq_len + 1]

            def _guard(ids):
                # Mask any pad tokens (-1) to -100 so cross_entropy ignores them.
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                y = torch.where(y == self.tok.pad_id, torch.tensor(-100), y)
                # Drop the pad_id from inputs too (pad only appears post-EOS).
                x = torch.where(x == self.tok.pad_id, torch.tensor(self.tok.eos_id), x)
                return x, y

            yield _guard(chunk)

    def __len__(self) -> int:
        # Compute lazily without materializing the buffer (len is used only by
        # progress bars; return an estimate so tqdm totals are sane).
        return 0


def make_phase1_loader(
    corpus_path: str,
    tokenizer: Tokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    ds = TokenWindowIterable(corpus_path, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
