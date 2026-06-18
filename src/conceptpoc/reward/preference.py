"""Build preference pairs from scored rollouts.

A "scored item" is a generated continuation with its concept vectors and a
scalar reward (a blend of LLM-judge and quality-controlled human ratings).
Within the candidates for one prompt, we form (chosen, rejected) pairs where the
reward gap exceeds a margin. These pairs train the reward model and drive the
reward-ranked fine-tuning loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoredItem:
    prompt_id: str
    text: str
    vectors: np.ndarray  # generated concept vectors (steps, dim)
    reward: float


@dataclass
class PreferencePair:
    prompt_id: str
    chosen: ScoredItem
    rejected: ScoredItem


def combine_rewards(
    llm_score: float | None,
    human_score: float | None,
    human_weight: float = 0.5,
) -> float:
    if llm_score is None and human_score is None:
        raise ValueError("at least one reward source is required")
    if human_score is None:
        return float(llm_score)
    if llm_score is None:
        return float(human_score)
    return float(human_weight * human_score + (1.0 - human_weight) * llm_score)


def build_preference_pairs(
    items: list[ScoredItem], margin: float = 0.05, max_pairs_per_prompt: int = 4
) -> list[PreferencePair]:
    by_prompt: dict[str, list[ScoredItem]] = {}
    for it in items:
        by_prompt.setdefault(it.prompt_id, []).append(it)

    pairs: list[PreferencePair] = []
    for prompt_id, group in by_prompt.items():
        ordered = sorted(group, key=lambda x: x.reward, reverse=True)
        made = 0
        for i in range(len(ordered)):
            for j in range(len(ordered) - 1, i, -1):
                if made >= max_pairs_per_prompt:
                    break
                if ordered[i].reward - ordered[j].reward >= margin:
                    pairs.append(
                        PreferencePair(prompt_id, ordered[i], ordered[j])
                    )
                    made += 1
    return pairs
