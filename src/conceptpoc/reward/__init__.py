"""Reward layer: LLM-as-judge, human-rating quality control, reward model."""
from __future__ import annotations

from .judge import HeuristicJudge, Judge, JudgeResult, LemonadeJudge, OpenAIJudge, build_judge
from .preference import PreferencePair, build_preference_pairs
from .quality_control import QualityControl, RaterVerdict
from .reward_model import RewardModel, train_reward_model

__all__ = [
    "Judge",
    "JudgeResult",
    "HeuristicJudge",
    "LemonadeJudge",
    "OpenAIJudge",
    "build_judge",
    "PreferencePair",
    "build_preference_pairs",
    "QualityControl",
    "RaterVerdict",
    "RewardModel",
    "train_reward_model",
]
