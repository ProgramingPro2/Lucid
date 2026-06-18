"""Abstract concept codec interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ConceptCodec(ABC):
    """Maps text to/from a fixed-dimensional concept space.

    Vectors are returned L2-normalized by convention so that dot products are
    cosine similarities throughout the pipeline.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the concept space."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode sentences into an ``(n, dim)`` float32 array (L2-normalized)."""

    @abstractmethod
    def decode(self, vectors: np.ndarray) -> list[str]:
        """Decode an ``(n, dim)`` array of concept vectors back to text."""

    def round_trip(self, texts: list[str]) -> list[str]:
        """Convenience: encode then decode, for reconstruction diagnostics."""
        return self.decode(self.encode(texts))

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms
