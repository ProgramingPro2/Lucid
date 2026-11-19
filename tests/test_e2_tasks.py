"""Tests for E2 task generation and answer verification (PROTOCOL.md §6).

These assert properties of the data pipeline only. No model is run and no
performance claim is made.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from experiments.e2 import solver, tasks
from experiments.e2.tasks import (
    FAMILIES,
    chance,
    generate,
    generate_many,
    variant_seed,
    verify,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_determinism_across_calls_and_shuffled_order(family: str) -> None:
    indices = list(range(2010, 2060))
    first = {i: generate(family, "test", i).to_json() for i in indices}
    second = {i: generate(family, "test", i).to_json() for i in indices}
    assert first == second

    shuffled = indices[:]
    random.Random(7).shuffle(shuffled)
    by_shuffled_order = {
        i: generate(family, "test", i).to_json() for i in shuffled
    }
    assert by_shuffled_order == first


@pytest.mark.parametrize("family", FAMILIES)
def test_same_prompt_always_has_one_gold(family: str) -> None:
    items = generate_many(family, "test", 500)
    for attribute in ("prompt_direct", "prompt_cot"):
        golds_by_prompt: dict[str, set[str]] = {}
        for item in items:
            golds_by_prompt.setdefault(getattr(item, attribute), set()).add(item.gold)
        offenders = {k: v for k, v in golds_by_prompt.items() if len(v) > 1}
        assert not offenders


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_accuracy_is_exactly_one(family: str) -> None:
    items = generate_many(family, "test", 250)
    correct = 0
    for item in items:
        answer = solver.solve(item)
        assert answer == item.gold
        correct += int(verify(family, f"ANSWER: {answer}", item.gold))
    assert correct / len(items) == 1.0


@pytest.mark.parametrize("family", ["f2", "f3"])
def test_uniform_random_answerer_is_near_chance(family: str) -> None:
    items = generate_many(family, "test", 2000)
    rng = np.random.default_rng(20261117)
    correct = 0
    for item in items:
        if family == "f2":
            pool = item.meta["entities"]
        else:
            pool = item.meta["tokens"]
        guess = pool[int(rng.integers(0, len(pool)))]
        correct += int(verify(family, f"ANSWER: {guess}", item.gold))

    accuracy = correct / len(items)
    expected = chance(family)
    assert 0.5 * expected <= accuracy <= 1.5 * expected


def _wrong_answer(item: tasks.Item) -> str:
    if item.family == "f1":
        return str(int(item.gold) + 1)
    if item.family == "f2":
        return next(
            entity
            for entity in item.meta["entities"]
            if entity.casefold() != item.gold.casefold()
        )
    return next(token for token in item.meta["tokens"] if token != item.gold)


@pytest.mark.parametrize("family", FAMILIES)
def test_verifier_accepts_wrapped_correct_answers(family: str) -> None:
    item = generate(family, "test", 2001)
    gold = item.gold
    wrapped = [
        f"Some reasoning that should be ignored.\nANSWER: {gold}",
        f"ANSWER:\n{gold}",
        f"ANSWER: {gold}.",
        f'ANSWER: "{gold}"',
        f"ANSWER: {gold.lower()}",
        f"ANSWER:    {gold}    ",
        f"The result is below.\n{gold}.",
        f"ANSWER: {_wrong_answer(item)}\nCorrection follows.\nANSWER: {gold}",
    ]
    if family == "f1":
        wrapped.extend(
            [f"ANSWER: {int(gold):,}", f"ANSWER: {int(gold):,}."]
        )
    for text in wrapped:
        assert verify(family, text, gold) is True, text


@pytest.mark.parametrize("family", FAMILIES)
def test_verifier_rejects_wrong_or_malformed_answers(family: str) -> None:
    item = generate(family, "test", 2002)
    gold = item.gold
    wrong = _wrong_answer(item)
    assert verify(family, f"ANSWER: {wrong}", gold) is False
    assert verify(family, f"ANSWER: {wrong}.", gold) is False
    assert verify(family, "", gold) is False
    assert verify(family, "ANSWER:", gold) is False
    assert verify(family, "ANSWER: \n", gold) is False
    assert verify(family, "no answer marker and no plausible final line", gold) is False
    assert verify(family, None, gold) is False  # type: ignore[arg-type]
    assert verify(family, b"ANSWER: " + gold.encode(), gold) is False  # type: ignore[arg-type]
    assert verify(family, 12345, gold) is False  # type: ignore[arg-type]
    assert verify(family, "ANSWER: None", gold) is False


@pytest.mark.parametrize("family", FAMILIES)
def test_train_and_test_hashes_are_disjoint(family: str) -> None:
    train = {generate(family, "train", i).item_hash() for i in range(300)}
    test = {generate(family, "test", i).item_hash() for i in range(2000, 2300)}
    assert train.isdisjoint(test)


def test_f1_gold_is_exactly_the_product() -> None:
    items = generate_many("f1", "test", 50)
    for item in items:
        assert int(item.gold) == item.meta["a"] * item.meta["b"]
        assert str(item.meta["a"]) in item.prompt_direct
        assert str(item.meta["b"]) in item.prompt_cot


def test_variant_seed_covers_the_protocol_seed_blocks() -> None:
    assert variant_seed(0) == 0
    assert variant_seed(1999) == 99
    assert variant_seed(2000) == 100
    assert variant_seed(3999) == 199
    with pytest.raises(ValueError):
        variant_seed(4000)
    with pytest.raises(ValueError):
        variant_seed(-1)


def test_splits_are_bounded_and_disjoint() -> None:
    for family in FAMILIES:
        assert generate(family, "train", 1999).split == "train"
        assert generate(family, "test", 2000).split == "test"
        with pytest.raises(ValueError):
            generate(family, "test", 1999)
        with pytest.raises(ValueError):
            generate(family, "train", 2000)
        with pytest.raises(ValueError):
            generate(family, "heldout", 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_prompts_request_the_answer_marker(family: str) -> None:
    item = generate(family, "test", 2003)
    for prompt in (item.prompt_direct, item.prompt_cot):
        assert "ANSWER: <value>" in prompt
        assert len(item.solution) > 0
