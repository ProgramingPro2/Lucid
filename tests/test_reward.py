from __future__ import annotations

import numpy as np

from conceptpoc.reward.judge import HeuristicJudge
from conceptpoc.reward.preference import (
    ScoredItem,
    build_preference_pairs,
    combine_rewards,
)
from conceptpoc.reward.quality_control import HumanRating, QualityControl
from conceptpoc.reward.reward_model import train_reward_model


def test_heuristic_judge_prefers_varied_text():
    judge = HeuristicJudge()
    good = "The river moved slowly. Birds rose from the reeds. The morning was calm."
    bad = "the the the the the the the the the the"
    assert judge.score("", good).score > judge.score("", bad).score


def test_quality_control_flags_constant_rater():
    ratings = [HumanRating("troll", f"i{i}", 1.0, 0.5) for i in range(5)]
    ratings += [HumanRating("real", f"i{i}", float(i % 2), 8.0) for i in range(5)]
    verdicts = QualityControl().assess(ratings)
    assert verdicts["troll"].weight < verdicts["real"].weight
    assert "constant ratings" in verdicts["troll"].reasons


def test_combine_rewards():
    assert combine_rewards(0.8, None) == 0.8
    assert combine_rewards(None, 0.4) == 0.4
    assert abs(combine_rewards(1.0, 0.0, human_weight=0.5) - 0.5) < 1e-9


def _item(prompt_id, reward, dim=8):
    return ScoredItem(
        prompt_id=prompt_id,
        text="x",
        vectors=np.random.randn(4, dim).astype(np.float32),
        reward=reward,
    )


def test_build_preference_pairs_respects_margin():
    items = [_item("p", 0.9), _item("p", 0.5), _item("p", 0.45)]
    pairs = build_preference_pairs(items, margin=0.1)
    assert all(p.chosen.reward - p.rejected.reward >= 0.1 for p in pairs)
    assert len(pairs) >= 1


def test_reward_model_learns_ordering():
    rng = np.random.default_rng(0)
    # Chosen items have larger mean magnitude in dim 0; reward model should learn it.
    pairs = []
    from conceptpoc.reward.preference import PreferencePair

    for i in range(40):
        chosen = _item("p", 1.0)
        chosen.vectors[:, 0] += 1.0
        rejected = _item("p", 0.0)
        rejected.vectors[:, 0] -= 1.0
        pairs.append(PreferencePair("p", chosen, rejected))
    model, hist = train_reward_model(pairs, concept_dim=8, epochs=60, lr=5e-3)
    assert hist.accuracy[-1] > 0.8
