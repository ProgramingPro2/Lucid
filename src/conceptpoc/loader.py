"""Shared codec + predictor loading helpers.

Extracted from ``cli.py`` so the eval harness (and other tools) can load a
codec and predictor from a :class:`Config` without depending on the CLI module.
``cli.py`` re-exports these under its historical private names for backward
compatibility.
"""
from __future__ import annotations

from pathlib import Path

from .codec import build_codec
from .config import Config
from .persistence import load_decoder_bank, load_predictor


def artifact_paths(cfg: Config) -> dict[str, Path]:
    a = cfg.artifacts_dir
    return {
        "predictor": a / "predictor.pt",
        "conditioned_predictor": a / "predictor_conditioned.pt",
        "bank": a / "decoder_bank.npz",
        "metrics": a / "metrics.json",
    }


def codec_and_conditioned_predictor(cfg: Config, fit_texts: list[str] | None):
    """Return ``(codec, conditioned_predictor)`` for E10 conditioned eval.

    The codec is the frozen JEPA codec (same concept space as training); the
    predictor is the segment/theme-conditioned one saved by ``train-conditioned``
    (not the codec's own continuation predictor).
    """
    from .persistence import load_predictor

    codec = build_codec_with_bank(cfg, fit_texts)
    path = artifact_paths(cfg)["conditioned_predictor"]
    if not path.exists():
        raise FileNotFoundError(
            f"no conditioned predictor at {path}; run 'train-conditioned' first"
        )
    return codec, load_predictor(path)


def is_jepa(cfg: Config) -> bool:
    return cfg.codec.kind.lower() == "jepa"


def ensure_jepa_checkpoint(cfg: Config) -> None:
    if not cfg.codec.jepa_checkpoint:
        cfg.codec.jepa_checkpoint = str(cfg.artifacts_dir / "jepa" / "joint.pt")


def build_codec_with_bank(cfg: Config, fit_texts: list[str] | None):
    if is_jepa(cfg):
        ensure_jepa_checkpoint(cfg)
        return build_codec(cfg.codec)
    codec = build_codec(cfg.codec)
    if hasattr(codec, "fit_decoder"):
        if fit_texts is not None:
            bank = fit_texts[: cfg.codec.decoder_bank_size]
            codec.fit_decoder(bank)
        else:
            paths = artifact_paths(cfg)
            if paths["bank"].exists():
                texts, vecs = load_decoder_bank(paths["bank"])
                codec.decoder.fit(texts, vecs)
    return codec


def codec_and_predictor(cfg: Config, fit_texts: list[str] | None):
    """Return ``(codec, predictor)``.

    For ``kind == 'jepa'`` the predictor is the jointly-trained one carried by
    the codec; otherwise it is loaded from ``predictor.pt``.
    """
    codec = build_codec_with_bank(cfg, fit_texts)
    if is_jepa(cfg):
        return codec, codec.predictor
    return codec, load_predictor(artifact_paths(cfg)["predictor"])
