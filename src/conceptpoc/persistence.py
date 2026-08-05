"""Save/load trained artifacts (predictor + decoder bank)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .config import DiffusionConfig, JepaConfig, PredictorConfig
from .predictor.diffusion import DiffusionConceptPredictor
from .predictor.model import ConceptPredictor, remap_legacy_predictor_state

PredictorModel = Union[ConceptPredictor, DiffusionConceptPredictor]


def _predictor_cfg_from_blob(raw: dict) -> PredictorConfig:
    data = dict(raw)
    if "diffusion" in data and isinstance(data["diffusion"], dict):
        data["diffusion"] = DiffusionConfig(**data["diffusion"])
    elif "diffusion" not in data:
        data.setdefault("head", "regression")
    return PredictorConfig(**data)


def save_predictor(path: str | Path, model: ConceptPredictor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_type": "regression",
            "state_dict": model.state_dict(),
            "concept_dim": model.concept_dim,
            "predictor_cfg": asdict(model.cfg),
        },
        path,
    )


def save_diffusion_predictor(path: str | Path, model: DiffusionConceptPredictor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_type": "diffusion",
            "state_dict": model.state_dict(),
            "concept_dim": model.concept_dim,
            "predictor_cfg": asdict(model.cfg),
            "diffusion_cfg": asdict(model.diff_cfg),
        },
        path,
    )


def load_predictor(path: str | Path, device: str = "cpu") -> ConceptPredictor:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    if blob.get("head_type") == "diffusion":
        raise TypeError(
            f"{path} is a diffusion checkpoint; use load_diffusion_predictor instead"
        )
    cfg = _predictor_cfg_from_blob(blob["predictor_cfg"])
    model = ConceptPredictor(blob["concept_dim"], cfg)
    state = remap_legacy_predictor_state(blob["state_dict"])
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_diffusion_predictor(
    path: str | Path, device: str = "cpu"
) -> DiffusionConceptPredictor:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = _predictor_cfg_from_blob(blob["predictor_cfg"])
    diff_raw = blob.get("diffusion_cfg", asdict(cfg.diffusion))
    if isinstance(diff_raw, DiffusionConfig):
        diff_cfg = diff_raw
    else:
        diff_cfg = DiffusionConfig(**diff_raw)
    model = DiffusionConceptPredictor(blob["concept_dim"], cfg, diff_cfg)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_any_predictor(path: str | Path, device: str = "cpu") -> PredictorModel:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if blob.get("head_type") == "diffusion":
        return load_diffusion_predictor(path, device)
    return load_predictor(path, device)


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
            "predictor_cfg": asdict(model.predictor.cfg),
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
    predictor_cfg = _predictor_cfg_from_blob(blob["predictor_cfg"])
    tokenizer = Tokenizer(blob["tokenizer_name"])
    model = JointJepa(
        blob["vocab_size"], blob["pad_id"], blob["bos_id"], jepa_cfg, predictor_cfg
    )
    state = blob["state_dict"]
    # Joint checkpoint may use legacy flat predictor keys under ``predictor.*``.
    if any(k.startswith("predictor.in_proj.") for k in state):
        remapped = {}
        for key, value in state.items():
            if key.startswith("predictor.") and not key.startswith("predictor.trunk."):
                sub = key[len("predictor.") :]
                if sub.startswith("out_proj."):
                    remapped[key] = value
                elif not sub.startswith("trunk."):
                    remapped[f"predictor.trunk.{sub}"] = value
                else:
                    remapped[key] = value
            else:
                remapped[key] = value
        state = remapped
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def save_writer(path: str | Path, model, config: dict) -> None:
    """Save a trained FluWriter (hybrid writer head) + its config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_writer(path: str | Path, device: str = "cpu"):
    """Load a FluWriter + config from a checkpoint."""
    from .writer import FluWriter

    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = blob["config"]
    m = FluWriter(
        cfg["vocab_size"],
        dim=cfg.get("dim", 768),
        n_layers=cfg.get("n_layers", 12),
        n_heads=cfg.get("n_heads", 12),
        ff_mult=cfg.get("ff_mult", 4),
        max_seq_len=cfg.get("max_seq_len", 264),
        concept_dim=cfg.get("concept_dim", 512),
        use_concept=cfg.get("use_concept", True),
        use_film=cfg.get("use_film", False),
    )
    m.load_state_dict(blob["state_dict"])
    m.to(device)
    m.eval()
    return m


def save_planner(path: str | Path, model, config: dict) -> None:
    """Save a trained TrajectoryPlanner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_planner(path: str | Path, device: str = "cpu"):
    """Load a TrajectoryPlanner."""
    from .trajectory.planner import TrajectoryPlanner

    dev = torch.device(device)
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = blob.get("config") or blob.get("cfg", {})
    m = TrajectoryPlanner(
        dim=cfg.get("dim", 512),
        C=cfg.get("C", 8),
        K=cfg.get("K", 5),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 8),
        latent_iters=cfg.get("latent_iters", 3),
    )
    m.load_state_dict(blob["state_dict"])
    m.to(dev)
    m.eval()
    return m

