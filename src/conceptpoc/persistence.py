"""Save/load trained artifacts (predictor + decoder bank)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import JepaConfig, PredictorConfig
from .predictor.model import ConceptPredictor


def save_predictor(path: str | Path, model: ConceptPredictor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "concept_dim": model.concept_dim,
            "predictor_cfg": model.cfg.__dict__,
        },
        path,
    )


def load_predictor(path: str | Path, device: str = "cpu") -> ConceptPredictor:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = PredictorConfig(**blob["predictor_cfg"])
    model = ConceptPredictor(blob["concept_dim"], cfg)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model


def save_decoder_bank(path: str | Path, texts: list[str], vectors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sentences=np.array(texts, dtype=object), vectors=vectors)


def load_decoder_bank(path: str | Path) -> tuple[list[str], np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return list(data["sentences"]), data["vectors"].astype(np.float32)


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def save_joint(path: str | Path, model, tokenizer_name: str) -> None:
    """Save a jointly-trained JointJepa model and the tokenizer it used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": model.decoder.vocab_size,
            "pad_id": model.pad_id,
            "bos_id": model.bos_id,
            "jepa_cfg": model.cfg.__dict__,
            "predictor_cfg": model.predictor.cfg.__dict__,
            "tokenizer_name": tokenizer_name,
        },
        path,
    )


def load_joint(path: str | Path, device: str = "cpu"):
    """Rebuild (JointJepa, Tokenizer) from a checkpoint."""
    from .jepa.joint_model import JointJepa
    from .jepa.tokenizer import Tokenizer

    blob = torch.load(Path(path), map_location=device, weights_only=False)
    jepa_cfg = JepaConfig(**blob["jepa_cfg"])
    predictor_cfg = PredictorConfig(**blob["predictor_cfg"])
    tokenizer = Tokenizer(blob["tokenizer_name"])
    model = JointJepa(
        blob["vocab_size"], blob["pad_id"], blob["bos_id"], jepa_cfg, predictor_cfg
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer
