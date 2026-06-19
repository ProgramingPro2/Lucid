"""Build a codec from configuration, with a safe fallback."""
from __future__ import annotations

import warnings

from ..config import CodecConfig
from .base import ConceptCodec


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_codec(cfg: CodecConfig) -> ConceptCodec:
    kind = cfg.kind.lower()

    if kind == "hashing":
        from .hashing_codec import HashingCodec

        return HashingCodec(dim=cfg.hashing_dim)

    if kind == "jepa":
        if not cfg.jepa_checkpoint:
            raise ValueError(
                "codec kind 'jepa' requires jepa_checkpoint; train one with "
                "`python -m conceptpoc.cli train-joint` first."
            )
        from ..persistence import load_joint
        from .jepa_codec import JepaCodec

        device = _resolve_device(cfg.device)
        model, tokenizer = load_joint(cfg.jepa_checkpoint, device=device)
        return JepaCodec(
            model,
            tokenizer,
            max_tokens=model.cfg.max_tokens,
            device=device,
            decode_temperature=cfg.decode_temperature,
            decode_top_k=cfg.decode_top_k,
            decode_top_p=cfg.decode_top_p,
        )

    if kind == "sonar":
        try:
            from .sonar_codec import SonarCodec

            return SonarCodec(device=cfg.device)
        except ImportError as exc:
            warnings.warn(
                f"Falling back to sentence-transformer codec: {exc}", stacklevel=2
            )
            kind = "sentence-transformer"

    if kind == "sentence-transformer":
        from .sentence_transformer_codec import SentenceTransformerCodec

        return SentenceTransformerCodec(model_name=cfg.model_name, device=cfg.device)

    raise ValueError(f"Unknown codec kind: {cfg.kind!r}")
