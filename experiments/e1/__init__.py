"""Experiment E1: is a JEPA block load-bearing? See docs/PROTOCOL.md §5."""

from .arms import Arm, all_arms, total_training_units
from .block import EMATeacher, JEPABlock, whiten
from .objective import combined_loss, context_gain, jepa_loss, vicreg_terms

__all__ = [
    "Arm",
    "all_arms",
    "total_training_units",
    "EMATeacher",
    "JEPABlock",
    "whiten",
    "combined_loss",
    "context_gain",
    "jepa_loss",
    "vicreg_terms",
]
