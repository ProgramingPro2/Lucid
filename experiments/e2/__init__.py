"""E2 task generation, prompting, solving and verification (PROTOCOL.md §6).

This package produces and audits generated items. It does not run a model, so it
supports no claim about model accuracy or about the C3 hypothesis.
"""

from .tasks import (
    CHANCE,
    FAMILIES,
    SPLIT_RANGES,
    Item,
    chance,
    generate,
    generate_many,
    variant_seed,
    verify,
)

__all__ = [
    "CHANCE",
    "FAMILIES",
    "SPLIT_RANGES",
    "Item",
    "chance",
    "generate",
    "generate_many",
    "variant_seed",
    "verify",
]
