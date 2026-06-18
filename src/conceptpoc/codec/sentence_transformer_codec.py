"""Sentence-transformer codec: frozen encoder + retrieval decoder."""
from __future__ import annotations

import numpy as np

from .base import ConceptCodec
from .retrieval_decoder import RetrievalDecoder


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SentenceTransformerCodec(ConceptCodec):
    """Encodes with a frozen sentence-transformer; decodes via nearest neighbor.

    The decoder bank is populated by :meth:`fit_decoder` (typically with the
    training corpus). Until then, :meth:`decode` raises.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._device = _resolve_device(device)
        self._model = SentenceTransformer(model_name, device=self._device)
        self._model.eval()
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self._dim = int(get_dim())
        self._decoder = RetrievalDecoder()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def fit_decoder(self, texts: list[str], vectors: np.ndarray | None = None) -> None:
        if vectors is None:
            vectors = self.encode(texts)
        self._decoder.fit(texts, vectors)

    def decode(self, vectors: np.ndarray) -> list[str]:
        return self._decoder.decode(vectors)

    @property
    def decoder(self) -> RetrievalDecoder:
        return self._decoder
