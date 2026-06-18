"""Codec backed by a jointly-trained JEPA encoder/decoder.

Wraps a trained ``JointJepa`` so it satisfies the ``ConceptCodec`` interface.
Unlike the retrieval decoder, decoding is generative: a predicted concept vector
is turned into text token-by-token, so it can produce sentences never seen in
training. Exposes the jointly-trained ``predictor`` for downstream use.
"""
from __future__ import annotations

import numpy as np
import torch

from ..jepa.joint_model import JointJepa
from ..jepa.tokenizer import Tokenizer
from .base import ConceptCodec


class JepaCodec(ConceptCodec):
    def __init__(
        self,
        model: JointJepa,
        tokenizer: Tokenizer,
        max_tokens: int,
        device: str = "cpu",
        decode_temperature: float = 0.0,
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._tok = tokenizer
        self._max_tokens = max_tokens
        self._device = device
        self._decode_temperature = decode_temperature

    @property
    def dim(self) -> int:
        return self._model.cfg.dim

    @property
    def predictor(self):
        return self._model.predictor

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        ids, pad = self._tok.encode_batch(texts, self._max_tokens)
        ids, pad = ids.to(self._device), pad.to(self._device)
        z = self._model.encoder(ids, pad)
        return np.ascontiguousarray(z.cpu().numpy(), dtype=np.float32)

    @torch.no_grad()
    def decode(self, vectors: np.ndarray) -> list[str]:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 0:
            return []
        z = torch.from_numpy(arr).to(self._device)
        token_lists = self._model.decoder.generate(
            z,
            bos_id=self._tok.bos_id,
            eos_id=self._tok.eos_id,
            max_tokens=self._max_tokens,
            temperature=self._decode_temperature,
        )
        return [self._tok.decode(ids) for ids in token_lists]
