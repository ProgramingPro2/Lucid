"""Deterministic E2 task generation and answer verification.

This module builds the three families in docs/PROTOCOL.md §6 and checks text
against a gold answer. It scores no model, so nothing here is evidence about
model performance.

Every item is a pure function of ``(family, split, index)``. The per-item RNG
is seeded from ``sha256(f"{family}:{split}:{index}")``; iteration order and
process state cannot affect an item.
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.protocol.seeds import spawn_key

from . import prompts

FAMILIES: tuple[str, ...] = ("f1", "f2", "f3")

# The protocol reserves generator seeds 0-99 for train and 100-199 for test.
# Each split owns 2,000 item indices, i.e. 20 items per generator seed.
SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 2000),
    "test": (2000, 4000),
}

ITEMS_PER_VARIANT = 20

ENTITIES: tuple[str, ...] = (
    "Ada Lark",
    "Boris Vale",
    "Cira Moss",
    "Dane Orr",
    "Esi Quill",
    "Fen Rook",
    "Gita Marr",
    "Hale Dune",
    "Ivo Pike",
    "Juno Wren",
    "Kael Ash",
    "Lira Voss",
    "Mira Holt",
    "Nial Fenn",
    "Oona Reed",
    "Pask Iver",
    "Quen Sable",
    "Rhea Flint",
    "Sol Tarn",
    "Tova Brine",
)

# Abstract labels rather than real-world predicates: the only property the
# task uses is that each map is a bijection.
RELATIONS: tuple[str, ...] = ("R1", "R2", "R3")

TOKENS: tuple[str, ...] = (
    "amber",
    "basalt",
    "cobalt",
    "coral",
    "denim",
    "ember",
    "fern",
    "garnet",
    "hazel",
    "indigo",
    "ivory",
    "jade",
    "kelp",
    "lilac",
    "maple",
    "onyx",
    "pearl",
    "quartz",
    "russet",
    "slate",
    "topaz",
    "umber",
    "violet",
    "willow",
)

_STATE_LENGTH = 12
_STATE_MOVES = 5
_HOPS = 3

CHANCE: dict[str, float] = {
    "f1": 0.0,
    "f2": 1.0 / len(ENTITIES),
    "f3": 1.0 / _STATE_LENGTH,
}


@dataclass(frozen=True)
class Item:
    """One generated task instance, with everything needed to verify it."""

    id: str
    family: str
    split: str
    prompt_direct: str
    prompt_cot: str
    gold: str
    solution: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "split": self.split,
            "prompt_direct": self.prompt_direct,
            "prompt_cot": self.prompt_cot,
            "gold": self.gold,
            "solution": self.solution,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        """Canonical serialisation: sorted keys, no incidental whitespace."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def item_hash(self) -> str:
        """SHA-256 of the canonical item, including its split.

        The split is part of the identity, so train and test hash namespaces are
        disjoint even when two draws happen to land on the same instance.
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def variant_seed(index: int) -> int:
    """Map an item index onto the protocol's generator-seed space.

    The protocol allows generator seeds 0-99 for train and 100-199 for test
    (2,000 items per split). Index ``i`` belongs to generator variant
    ``i // 20``: indices 0-1999 map to 0-99 and indices 2000-3999 map to
    100-199, with 20 independently keyed items per variant. The variant is
    metadata only; item content is still keyed by the full index.
    """
    if not 0 <= index < 4000:
        raise ValueError(f"index must be in [0, 4000), got {index}")
    return index // ITEMS_PER_VARIANT


def _normalise_family(family: str) -> str:
    key = str(family).lower()
    if key not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    return key


def _item_seed(family: str, split: str, index: int) -> int:
    """Return the 64-bit RNG seed for one item's key.

    ``spawn_key`` already hashes ``f"{seed}:{name}"`` and only accepts registered
    stream names. Folding the required key ``f"{family}:{split}:{index}"``
    through SHA-256 first keeps the item namespace flat and independent of every
    other item while still using the programme's spawn-key helper.
    """
    digest = hashlib.sha256(f"{family}:{split}:{index}".encode("utf-8")).digest()
    return spawn_key(int.from_bytes(digest[:8], "big"), "data")


def _rng(family: str, split: str, index: int) -> np.random.Generator:
    return np.random.default_rng(_item_seed(family, split, index))


def generate(family: str, split: str, index: int) -> Item:
    """Generate one item. The same arguments always yield the same item."""
    family = _normalise_family(family)
    if split not in SPLIT_RANGES:
        raise ValueError(f"unknown split {split!r}; expected train or test")
    lo, hi = SPLIT_RANGES[split]
    if not lo <= index < hi:
        raise ValueError(f"index {index} is outside the {split} range [{lo}, {hi})")

    if family == "f1":
        return _generate_f1(split, index)
    if family == "f2":
        return _generate_f2(split, index)
    return _generate_f3(split, index)


def generate_many(
    family: str,
    split: str,
    n: int,
    start: int | None = None,
) -> list[Item]:
    """Generate ``n`` consecutive items, independent of any RNG global state."""
    family = _normalise_family(family)
    lo, hi = SPLIT_RANGES[split]
    first = lo if start is None else int(start)
    if n < 0 or first < lo or first + n > hi:
        raise ValueError(
            f"cannot generate {n} {family}/{split} items from {first}; "
            f"valid indices are [{lo}, {hi})"
        )
    return [generate(family, split, i) for i in range(first, first + n)]


def chance(family: str) -> float:
    """Probability of a correct answer from a uniform random guess."""
    return CHANCE[_normalise_family(family)]


# ---------------------------------------------------------------------------
# F1: 3-digit x 3-digit arithmetic
# ---------------------------------------------------------------------------


def _arithmetic_solution(a: int, b: int) -> str:
    hundreds = (b // 100) * 100
    tens = ((b // 10) % 10) * 10
    ones = b % 10
    p1, p2, p3 = a * hundreds, a * tens, a * ones
    return (
        f"{a} x {b} = {a} x {hundreds} + {a} x {tens} + {a} x {ones} "
        f"= {p1} + {p2} + {p3} = {a * b}."
    )


def _generate_f1(split: str, index: int) -> Item:
    rng = _rng("f1", split, index)
    a = int(rng.integers(100, 1000))
    b = int(rng.integers(100, 1000))
    gold = str(a * b)
    return Item(
        id=f"f1:{split}:{index:04d}",
        family="f1",
        split=split,
        prompt_direct=prompts.arithmetic_direct(a, b),
        prompt_cot=prompts.arithmetic_cot(a, b),
        gold=gold,
        solution=_arithmetic_solution(a, b),
        meta={"a": a, "b": b},
    )


# ---------------------------------------------------------------------------
# F2: composition of three bijections
# ---------------------------------------------------------------------------


def _relational_solution(
    entities: list[str],
    relations: dict[str, list[str]],
    order: list[str],
    start: str,
) -> str:
    idx = entities.index(start)
    lines = [f"Start at {start}."]
    for step, name in enumerate(order, 1):
        image = relations[name][idx]
        lines.append(f"{step}. {name}: {entities[idx]} -> {image}.")
        idx = entities.index(image)
    lines.append(f"The final entity is {entities[idx]}.")
    return "\n".join(lines)


def _generate_f2(split: str, index: int) -> Item:
    rng = _rng("f2", split, index)
    entities = list(ENTITIES)

    # Each relation is a bijection. Rejecting the identity keeps every table
    # non-trivial; retrying consumes the same local RNG, so it stays keyed.
    relations: dict[str, list[str]] = {}
    identity = np.arange(len(entities))
    for name in RELATIONS:
        perm = rng.permutation(len(entities))
        while np.array_equal(perm, identity):
            perm = rng.permutation(len(entities))
        relations[name] = [entities[int(k)] for k in perm]

    start_idx = int(rng.integers(0, len(entities)))
    # Use each relation type exactly once, in a random order: a three-hop query
    # over the three declared relations, with no repeated hop.
    order = [RELATIONS[int(k)] for k in rng.permutation(_HOPS)]

    idx = start_idx
    for name in order:
        idx = entities.index(relations[name][idx])
    gold = entities[idx]
    start = entities[start_idx]

    return Item(
        id=f"f2:{split}:{index:04d}",
        family="f2",
        split=split,
        prompt_direct=prompts.relational_direct(entities, relations, order, start),
        prompt_cot=prompts.relational_cot(entities, relations, order, start),
        gold=gold,
        solution=_relational_solution(entities, relations, order, start),
        meta={
            "entities": entities,
            "relations": relations,
            "query_relations": order,
            "start": start,
        },
    )


# ---------------------------------------------------------------------------
# F3: state tracking under swaps and moves
# ---------------------------------------------------------------------------


def _apply_moves(
    tokens: list[str], moves: list[dict[str, int | str]]
) -> list[str]:
    """Apply 1-based swap/move operations; used only to build the item."""
    state = list(tokens)
    for move in moves:
        i = int(move["i"])
        j = int(move["j"])
        if move["op"] == "swap":
            state[i - 1], state[j - 1] = state[j - 1], state[i - 1]
        elif move["op"] == "move":
            token = state.pop(i - 1)
            state.insert(min(j - 1, len(state)), token)
        else:  # pragma: no cover - generation only emits the two opcodes
            raise ValueError(f"unknown move {move['op']!r}")
    return state


def _format_state(state: list[str]) -> str:
    return ", ".join(f"{i}. {token}" for i, token in enumerate(state, 1))


def _state_solution(
    tokens: list[str], moves: list[dict[str, int | str]], position: int
) -> str:
    state = list(tokens)
    lines = [f"Start: {_format_state(state)}."]
    for number, move in enumerate(moves, 1):
        i = int(move["i"])
        j = int(move["j"])
        if move["op"] == "swap":
            state[i - 1], state[j - 1] = state[j - 1], state[i - 1]
            lines.append(f"{number}. swap({i}, {j}): {_format_state(state)}.")
        else:
            token = state.pop(i - 1)
            state.insert(min(j - 1, len(state)), token)
            lines.append(f"{number}. move({i} -> {j}): {_format_state(state)}.")
    lines.append(
        f"After all {len(moves)} moves, position {position} holds "
        f"{state[position - 1]}."
    )
    return "\n".join(lines)


def _generate_f3(split: str, index: int) -> Item:
    rng = _rng("f3", split, index)
    tokens = [
        str(token)
        for token in rng.choice(
            np.asarray(TOKENS), size=_STATE_LENGTH, replace=False
        )
    ]

    moves: list[dict[str, int | str]] = []
    for _ in range(_STATE_MOVES):
        op = "swap" if float(rng.random()) < 0.5 else "move"
        i = int(rng.integers(1, _STATE_LENGTH + 1))
        j = int(rng.integers(1, _STATE_LENGTH + 1))
        while j == i:
            j = int(rng.integers(1, _STATE_LENGTH + 1))
        moves.append({"op": op, "i": i, "j": j})

    position = int(rng.integers(1, _STATE_LENGTH + 1))
    state = _apply_moves(tokens, moves)
    gold = state[position - 1]

    return Item(
        id=f"f3:{split}:{index:04d}",
        family="f3",
        split=split,
        prompt_direct=prompts.state_direct(tokens, moves, position),
        prompt_cot=prompts.state_cot(tokens, moves, position),
        gold=gold,
        solution=_state_solution(tokens, moves, position),
        meta={"tokens": tokens, "moves": moves, "position": position},
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_SURROUNDING = set(string.punctuation + string.whitespace + "\u201c\u201d\u2018\u2019")
_ANSWER_MARKER = "answer:"


def _last_answer(text: str) -> str | None:
    """Return the text after the last case-insensitive ``ANSWER:`` marker.

    If the marker is absent, fall back to the last non-empty line. Returning
    ``None`` means there is nothing left to compare.
    """
    idx = text.lower().rfind(_ANSWER_MARKER)
    if idx >= 0:
        rest = text[idx + len(_ANSWER_MARKER) :]
        after_marker = [line.strip() for line in rest.splitlines() if line.strip()]
        if after_marker:
            return after_marker[0]
        return rest.strip()

    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].strip() if lines else None


def _strip_surrounding(text: str) -> str:
    start, end = 0, len(text)
    while start < end and text[start] in _SURROUNDING:
        start += 1
    while end > start and text[end - 1] in _SURROUNDING:
        end -= 1
    return text[start:end]


def _as_int(text: str) -> int | None:
    cleaned = _strip_surrounding(text)
    cleaned = cleaned.replace(",", "").replace(" ", "").replace("_", "")
    while cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return int(cleaned)
    except (TypeError, ValueError):
        return None


def verify(family: str, text: object, gold: object) -> bool:
    """Check a reply against the gold answer, never raising on bad input.

    The rule is deliberately forgiving about prose and punctuation but strict
    about the value: one wrong digit or one wrong entity is a failure.
    """
    try:
        family = _normalise_family(family)
    except (TypeError, ValueError):
        return False
    if not isinstance(text, str) or not isinstance(gold, str):
        return False
    if not text.strip() or not gold.strip():
        return False

    candidate = _last_answer(text)
    if candidate is None:
        return False

    if family == "f1":
        got = _as_int(candidate)
        want = _as_int(gold)
        return got is not None and want is not None and got == want

    got = _strip_surrounding(candidate).casefold()
    want = _strip_surrounding(gold).casefold()
    return bool(got) and got == want
