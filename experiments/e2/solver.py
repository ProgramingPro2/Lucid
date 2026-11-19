"""Programmatic oracle for E2 items.

The solver recomputes the answer from ``item.meta`` and never reads
``item.gold``. That independence is the point: it checks the generator instead
of repeating the generator's own bookkeeping.
"""

from __future__ import annotations

from typing import Any

from .tasks import Item


def solve_meta(family: str, meta: dict[str, Any]) -> str:
    """Recompute the gold answer for one family from its public structure."""
    if family == "f1":
        return str(int(meta["a"]) * int(meta["b"]))
    if family == "f2":
        return _solve_relational(meta)
    if family == "f3":
        return _solve_state(meta)
    raise ValueError(f"unknown family {family!r}")


def solve(item: Item) -> str:
    """Oracle answer for ``item``, independent of ``item.gold``."""
    return solve_meta(item.family, item.meta)


def _solve_relational(meta: dict[str, Any]) -> str:
    entities = list(meta["entities"])
    relations = meta["relations"]
    idx = entities.index(meta["start"])
    for name in meta["query_relations"]:
        images = relations[name]
        idx = entities.index(images[idx])
    return entities[idx]


def _solve_state(meta: dict[str, Any]) -> str:
    state = list(meta["tokens"])
    for move in meta["moves"]:
        i = int(move["i"])
        j = int(move["j"])
        if move["op"] == "swap":
            state[i - 1], state[j - 1] = state[j - 1], state[i - 1]
        elif move["op"] == "move":
            token = state.pop(i - 1)
            state.insert(min(j - 1, len(state)), token)
        else:
            raise ValueError(f"unknown move opcode {move['op']!r}")
    return state[int(meta["position"]) - 1]
