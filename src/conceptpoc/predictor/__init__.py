"""The trainable latent concept predictor."""
from __future__ import annotations

from .model import ConceptPredictor, predictor_loss
from .train import train_predictor

__all__ = ["ConceptPredictor", "predictor_loss", "train_predictor"]
