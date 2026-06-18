"""SONAR codec: frozen encoder + generative decoder (optional install).

This is the closest match to Meta's Large Concept Model setup: SONAR encodes a
sentence to a vector and can generatively decode a vector back to a sentence in
many languages. Requires ``sonar-space`` + ``fairseq2`` (see
requirements-sonar.txt). If those are not installed, construction raises with a
clear message and the caller should fall back to the sentence-transformer codec.
"""
from __future__ import annotations

import numpy as np

from .base import ConceptCodec


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SonarCodec(ConceptCodec):
    def __init__(
        self,
        encoder_name: str = "text_sonar_basic_encoder",
        decoder_name: str = "text_sonar_basic_decoder",
        lang: str = "eng_Latn",
        device: str = "auto",
    ) -> None:
        try:
            from sonar.inference_pipelines.text import (
                EmbeddingToTextModelPipeline,
                TextToEmbeddingModelPipeline,
            )
        except Exception as exc:  # pragma: no cover - exercised only without SONAR
            raise ImportError(
                "SONAR is not installed. Install with "
                "`pip install -r requirements-sonar.txt` or use the "
                "sentence-transformer codec instead."
            ) from exc

        self._lang = lang
        self._device = _resolve_device(device)
        self._t2vec = TextToEmbeddingModelPipeline(
            encoder=encoder_name, tokenizer=encoder_name, device=self._device
        )
        self._vec2t = EmbeddingToTextModelPipeline(
            decoder=decoder_name, tokenizer=encoder_name, device=self._device
        )
        probe = self._t2vec.predict(["probe"], source_lang=lang)
        self._dim = int(probe.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        emb = self._t2vec.predict(texts, source_lang=self._lang)
        arr = emb.detach().cpu().numpy().astype(np.float32)
        return self.normalize(arr)

    def decode(self, vectors: np.ndarray) -> list[str]:
        import torch

        if len(vectors) == 0:
            return []
        tensor = torch.as_tensor(np.asarray(vectors, dtype=np.float32))
        return list(self._vec2t.predict(tensor, target_lang=self._lang, max_seq_len=256))
