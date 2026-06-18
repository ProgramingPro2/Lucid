"""Tokenizer wrapper with two backends.

- A pretrained BPE tokenizer (default ``gpt2``) via ``transformers``. Special
  pad/bos tokens are added so loss masking and decoder priming are unambiguous.
- A dependency-free char-level fallback (``"char"``) for offline tests.

Both expose the same interface: ``vocab_size``, ``pad_id``, ``bos_id``,
``eos_id``, ``encode``/``decode`` (no special tokens), and ``encode_batch``
which appends EOS and right-pads to a tensor.
"""
from __future__ import annotations

import torch

_CHAR_VOCAB = (
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".,;:!?'\"()-"
)


class Tokenizer:
    def __init__(self, name: str = "gpt2") -> None:
        self.name = name
        if name == "char":
            self._init_char()
        else:
            self._init_bpe(name)

    # -- char backend ---------------------------------------------------------
    def _init_char(self) -> None:
        self._backend = "char"
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        self._itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + list(_CHAR_VOCAB)
        self._stoi = {c: i for i, c in enumerate(self._itos)}
        self.vocab_size = len(self._itos)

    # -- bpe backend ----------------------------------------------------------
    def _init_bpe(self, name: str) -> None:
        from transformers import AutoTokenizer

        self._backend = "bpe"
        tok = AutoTokenizer.from_pretrained(name)
        added = {}
        if tok.pad_token is None:
            added["pad_token"] = "[PAD]"
        if tok.bos_token is None:
            added["bos_token"] = "[BOS]"
        if tok.eos_token is None:
            added["eos_token"] = "[EOS]"
        if added:
            tok.add_special_tokens(added)
        self._tok = tok
        self.pad_id = tok.pad_token_id
        self.bos_id = tok.bos_token_id
        self.eos_id = tok.eos_token_id
        self.vocab_size = len(tok)

    # -- shared API -----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        if self._backend == "char":
            return [self._stoi.get(c, self.unk_id) for c in text]
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        ids = [int(i) for i in ids if int(i) not in specials]
        if self._backend == "char":
            return "".join(self._itos[i] for i in ids if 0 <= i < self.vocab_size)
        return self._tok.decode(ids, skip_special_tokens=True).strip()

    def encode_batch(
        self, texts: list[str], max_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ids (B, L), pad_mask (B, L) True=pad). EOS is appended."""
        seqs = []
        for t in texts:
            ids = self.encode(t)[: max_tokens - 1] + [self.eos_id]
            seqs.append(ids)
        length = max(len(s) for s in seqs) if seqs else 1
        ids = torch.full((len(seqs), length), self.pad_id, dtype=torch.long)
        mask = torch.ones(len(seqs), length, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = False
        return ids, mask
