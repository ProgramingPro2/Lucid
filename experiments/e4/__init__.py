"""Experiment E4: is a draft necessary? See docs/PROTOCOL.md §7."""

from .metrics import (
    entity_recall,
    loo_centroid_cosine,
    normalise_entity,
    retrieval_win_rate,
)
from .necessity import (
    NecessityResult,
    absolute_gate_is_valid,
    combine_endpoints,
    evaluate_necessity,
    fidelity_gate,
)

__all__ = [
    "entity_recall",
    "loo_centroid_cosine",
    "normalise_entity",
    "retrieval_win_rate",
    "NecessityResult",
    "absolute_gate_is_valid",
    "combine_endpoints",
    "evaluate_necessity",
    "fidelity_gate",
]
