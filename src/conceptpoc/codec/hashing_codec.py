"""Deterministic hashing codec for tests and offline development.

Encodes text into a fixed-dimensional vector via hashed token features, with no
model downloads. Decoding uses the same retrieval-bank mechanism as the real
codec, so the full pipeline can be exercised deterministically.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .base import ConceptCodec
from .retrieval_decoder import RetrievalDecoder

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingCodec(ConceptCodec):
    def __init__(self, dim: int = 256) -> None:
        self._dim = int(dim)
        self._decoder = RetrievalDecoder()

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            tokens = ["<empty>"]
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.stack([self._encode_one(t) for t in texts])
        return self.normalize(out)

    def fit_decoder(self, texts: list[str], vectors: np.ndarray | None = None) -> None:
        if vectors is None:
            vectors = self.encode(texts)
        self._decoder.fit(texts, vectors)

    def decode(self, vectors: np.ndarray) -> list[str]:
        return self._decoder.decode(vectors)

    @property
    def decoder(self) -> RetrievalDecoder:
        return self._decoder
