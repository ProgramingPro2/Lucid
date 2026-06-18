"""Retrieval (nearest-neighbor) decoder.

Decoding an arbitrary concept vector back to fluent text is the hard half of the
problem. Generative decoders (SONAR) exist but are heavy and degrade on vectors
the predictor never saw during training. A robust, dependency-light alternative
is to keep a bank of real sentences and their encodings, then "decode" a vector
by returning the nearest sentence in the bank. This is the snap-to-nearest
mitigation called out in the plan and always produces grammatical text.
"""
from __future__ import annotations

import numpy as np


class RetrievalDecoder:
    def __init__(self) -> None:
        self._bank_texts: list[str] = []
        self._bank_vecs: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._bank_texts)

    def fit(self, texts: list[str], vectors: np.ndarray) -> "RetrievalDecoder":
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must align")
        self._bank_texts = list(texts)
        self._bank_vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        return self

    def decode(self, vectors: np.ndarray) -> list[str]:
        if self._bank_vecs is None or self.size == 0:
            raise RuntimeError("RetrievalDecoder is empty; call fit() first")
        query = np.asarray(vectors, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        # Vectors are L2-normalized, so a dot product is cosine similarity.
        sims = query @ self._bank_vecs.T
        idx = np.argmax(sims, axis=1)
        return [self._bank_texts[i] for i in idx]

    def state_dict(self) -> dict:
        return {"texts": self._bank_texts, "vecs": self._bank_vecs}

    def load_state_dict(self, state: dict) -> "RetrievalDecoder":
        self._bank_texts = list(state["texts"])
        self._bank_vecs = np.asarray(state["vecs"], dtype=np.float32)
        return self
