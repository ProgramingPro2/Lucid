"""Prompt templates for the three E2 task families (docs/PROTOCOL.md §6).

The direct and chain-of-thought prompts describe the *same* instance and expect
the *same* answer; they differ only in how much working the reply may contain.
Every prompt asks for a final line of exactly ``ANSWER: <value>`` because the
verifier keys on that marker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_MARKER = "ANSWER: <value>"

_DIRECT_CLOSE = (
    "Do not show any working. Reply with exactly one line of the form\n"
    f"{_MARKER}\n"
    "and write nothing before or after that line."
)

_COT_CLOSE = (
    "Show your working first. Then finish with exactly one final line of the form\n"
    f"{_MARKER}\n"
    "and write nothing after that line."
)


def arithmetic_direct(a: int, b: int) -> str:
    """Direct prompt: minimum output budget, one answer line."""
    return (
        f"What is {a} multiplied by {b}?\n\n"
        f"{_DIRECT_CLOSE}"
    )


def arithmetic_cot(a: int, b: int) -> str:
    """CoT prompt: same question, room to decompose the product."""
    return (
        f"What is {a} multiplied by {b}?\n\n"
        "Work step by step: split the second factor into hundreds, tens and "
        "ones, compute the three partial products, and add them.\n\n"
        f"{_COT_CLOSE}"
    )


def _relation_table(
    entities: Sequence[str],
    relations: Mapping[str, Sequence[str]],
) -> str:
    """Render each bijection as explicit ``entity -> image`` pairs.

    Enumeration matters: a composition is only well defined for the reader if
    every lookup is written out, not merely described.
    """
    blocks: list[str] = []
    for name, images in relations.items():
        pairs = "\n".join(
            f"  {entity} -> {image}" for entity, image in zip(entities, images)
        )
        blocks.append(f"{name}:\n{pairs}")
    return "\n\n".join(blocks)


def _chain_text(order: Sequence[str]) -> str:
    return ", then ".join(order)


def relational_direct(
    entities: Sequence[str],
    relations: Mapping[str, Sequence[str]],
    order: Sequence[str],
    start: str,
) -> str:
    """Direct prompt with the full relation tables and a three-hop query."""
    table = _relation_table(entities, relations)
    return (
        f"There are {len(entities)} named entities and three relations, "
        f"{', '.join(relations)}. Each relation is a bijection: it maps every "
        "entity to exactly one entity, and no two entities share an image.\n\n"
        f"Entities: {', '.join(entities)}.\n\n"
        f"{table}\n\n"
        f"Starting at {start}, follow {_chain_text(order)} in that order.\n\n"
        f"{_DIRECT_CLOSE}"
    )


def relational_cot(
    entities: Sequence[str],
    relations: Mapping[str, Sequence[str]],
    order: Sequence[str],
    start: str,
) -> str:
    """CoT prompt: same tables, but ask for each intermediate lookup."""
    table = _relation_table(entities, relations)
    return (
        f"There are {len(entities)} named entities and three relations, "
        f"{', '.join(relations)}. Each relation is a bijection: it maps every "
        "entity to exactly one entity, and no two entities share an image.\n\n"
        f"Entities: {', '.join(entities)}.\n\n"
        f"{table}\n\n"
        f"Starting at {start}, follow {_chain_text(order)} in that order.\n\n"
        "Track the current entity one hop at a time, writing the entity after "
        "each hop.\n\n"
        f"{_COT_CLOSE}"
    )


def _moves_text(moves: Sequence[Mapping[str, int | str]]) -> str:
    lines: list[str] = []
    for number, move in enumerate(moves, 1):
        if move["op"] == "swap":
            lines.append(f"{number}. swap({move['i']}, {move['j']})")
        else:
            lines.append(f"{number}. move({move['i']} -> {move['j']})")
    return "\n".join(lines)


def _state_common(
    tokens: Sequence[str],
    moves: Sequence[Mapping[str, int | str]],
    position: int,
) -> str:
    initial = "\n".join(f"{i}. {token}" for i, token in enumerate(tokens, 1))
    return (
        "A list holds 12 distinct tokens. Positions are 1-based.\n"
        "swap(i, j) exchanges the tokens at positions i and j.\n"
        "move(i -> j) removes the token at position i and inserts it at "
        "position j in the resulting 12-token list.\n\n"
        f"Initial list:\n{initial}\n\n"
        f"Moves, applied in order:\n{_moves_text(moves)}\n\n"
        f"After all {len(moves)} moves, which token is at position {position}?"
    )


def state_direct(
    tokens: Sequence[str],
    moves: Sequence[Mapping[str, int | str]],
    position: int,
) -> str:
    """Direct prompt for state tracking: answer only, no trace."""
    return f"{_state_common(tokens, moves, position)}\n\n{_DIRECT_CLOSE}"


def state_cot(
    tokens: Sequence[str],
    moves: Sequence[Mapping[str, int | str]],
    position: int,
) -> str:
    """CoT prompt: ask for the list after every move."""
    return (
        f"{_state_common(tokens, moves, position)}\n\n"
        "Write the full list after each move, in order, then read off the "
        f"token at position {position}.\n\n"
        f"{_COT_CLOSE}"
    )
