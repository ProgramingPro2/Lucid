from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conceptpoc.reward.judge import (
    HeuristicJudge,
    LemonadeJudge,
    OpenAIJudge,
    _parse_judge_response,
    build_judge,
)
from conceptpoc.reward.preference import (
    ScoredItem,
    build_preference_pairs,
    combine_rewards,
)
from conceptpoc.reward.quality_control import HumanRating, QualityControl
from conceptpoc.reward.reward_model import train_reward_model


def test_parse_judge_response_valid_json():
    result = _parse_judge_response('{"score": 0.75, "rationale": "clear prose"}')
    assert result.score == 0.75
    assert result.rationale == "clear prose"


def test_parse_judge_response_clips_score():
    result = _parse_judge_response('{"score": 1.5, "rationale": "high"}')
    assert result.score == 1.0


def test_parse_judge_response_embedded_json():
    text = 'Here is my rating:\n{"score": 0.6, "rationale": "okay"}\nThanks.'
    result = _parse_judge_response(text)
    assert result.score == 0.6
    assert result.rationale == "okay"


def test_parse_judge_response_malformed():
    result = _parse_judge_response("not json at all")
    assert result.score == 0.0
    assert result.rationale.startswith("unparseable")


def test_parse_judge_response_code_fenced():
    text = '```json\n{"score": 0.7, "rationale": "good"}\n```'
    result = _parse_judge_response(text)
    assert result.score == 0.7
    assert result.rationale == "good"


def test_parse_judge_response_truncated_unclosed():
    # Chat models emit score before a long rationale; the token cap can truncate
    # the reply before the closing brace. The score must still be recovered.
    text = '```json\n{\n  "score": 0.25,\n  "rationale": "Partially relevant but the'
    result = _parse_judge_response(text)
    assert result.score == 0.25
    assert result.rationale.startswith("Partially relevant")


def test_parse_judge_response_strips_think_block():
    text = (
        "<think>The text rambles but is on-topic.</think>\n"
        '{"score": 0.5, "rationale": "ok"}'
    )
    result = _parse_judge_response(text)
    assert result.score == 0.5
    assert result.rationale == "ok"


def test_parse_judge_response_rescales_0_to_5_scale():
    # A clear 0-5 rating is rescaled; a small overshoot stays clipped to 1.0.
    assert _parse_judge_response('{"score": 4, "rationale": "good"}').score == 0.8
    assert _parse_judge_response('{"score": 1.5, "rationale": "high"}').score == 1.0


def test_parse_judge_response_empty_is_unparseable():
    result = _parse_judge_response("")
    assert result.score == 0.0
    assert "empty" in result.rationale


def test_build_judge_explicit_backends(monkeypatch):
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("LEMONADE_JUDGE_MODEL", raising=False)
    assert isinstance(build_judge(), HeuristicJudge)

    monkeypatch.setenv("JUDGE_BACKEND", "heuristic")
    assert isinstance(build_judge(), HeuristicJudge)

    monkeypatch.setenv("JUDGE_BACKEND", "openai")
    with patch("openai.OpenAI"):
        assert isinstance(build_judge(), OpenAIJudge)

    monkeypatch.setenv("JUDGE_BACKEND", "lemonade")
    monkeypatch.setenv("LEMONADE_JUDGE_MODEL", "user.Flow-Judge")
    with patch("openai.OpenAI"):
        assert isinstance(build_judge(), LemonadeJudge)


def test_build_judge_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unknown JUDGE_BACKEND"):
        build_judge()


def test_lemonade_judge_requires_model(monkeypatch):
    monkeypatch.delenv("LEMONADE_JUDGE_MODEL", raising=False)
    with patch("openai.OpenAI"):
        with pytest.raises(ValueError, match="LEMONADE_JUDGE_MODEL"):
            LemonadeJudge()


def test_lemonade_judge_score_mocked(monkeypatch):
    monkeypatch.setenv("LEMONADE_JUDGE_MODEL", "user.Flow-Judge")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"score": 0.8, "rationale": "good"}'))]
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        judge = LemonadeJudge()
        result = judge.score("Start here.", "The story continued calmly.")

    assert result.score == 0.8
    assert result.rationale == "good"
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "user.Flow-Judge"
    assert kwargs["temperature"] == 0.0
    # Headroom for reasoning-capable judges to finish the JSON; JSON mode keeps
    # the reply free of code fences / prose for reliable parsing.
    assert kwargs["max_completion_tokens"] == 512
    assert kwargs["response_format"] == {"type": "json_object"}


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
    model, hist = train_reward_model(
        pairs, concept_dim=8, epochs=60, lr=5e-3, progress=False
    )
    assert hist.accuracy[-1] > 0.8
