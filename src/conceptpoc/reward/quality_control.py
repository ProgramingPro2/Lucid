"""LLM quality control over human raters.

The user's idea: let ~20 people rate outputs, but have an LLM double-check that
the human work is not garbage/trolling. This module flags raters whose ratings
look low-effort or adversarial, so their input can be down-weighted or dropped
before it pollutes the reward signal.

Signals (no network required):
- constant ratings (always the same value)
- implausibly fast ratings
- strong, systematic disagreement with the consensus / LLM judge

An optional LLM pass can be layered on top, but the statistical checks alone
catch the most damaging cases and keep the pipeline runnable offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RaterVerdict:
    rater_id: str
    trusted: bool
    weight: float
    reasons: list[str]


@dataclass
class HumanRating:
    rater_id: str
    item_id: str
    score: float  # 0..1
    seconds_spent: float | None = None


class QualityControl:
    def __init__(
        self,
        min_seconds: float = 2.0,
        max_consensus_gap: float = 0.6,
        min_ratings: int = 3,
    ) -> None:
        self.min_seconds = min_seconds
        self.max_consensus_gap = max_consensus_gap
        self.min_ratings = min_ratings

    def assess(
        self, ratings: list[HumanRating], reference: dict[str, float] | None = None
    ) -> dict[str, RaterVerdict]:
        by_rater: dict[str, list[HumanRating]] = {}
        for r in ratings:
            by_rater.setdefault(r.rater_id, []).append(r)

        consensus = reference or self._consensus(ratings)
        verdicts: dict[str, RaterVerdict] = {}
        for rater_id, rs in by_rater.items():
            reasons: list[str] = []
            weight = 1.0
            scores = np.array([r.score for r in rs], dtype=np.float32)

            if len(rs) >= self.min_ratings and float(scores.std()) < 1e-6:
                reasons.append("constant ratings")
                weight *= 0.2

            fast = [r for r in rs if r.seconds_spent is not None and r.seconds_spent < self.min_seconds]
            if len(rs) >= self.min_ratings and len(fast) / len(rs) > 0.5:
                reasons.append("many implausibly fast ratings")
                weight *= 0.3

            gaps = [
                abs(r.score - consensus[r.item_id])
                for r in rs
                if r.item_id in consensus
            ]
            if gaps and float(np.mean(gaps)) > self.max_consensus_gap:
                reasons.append("systematic disagreement with consensus")
                weight *= 0.4

            verdicts[rater_id] = RaterVerdict(
                rater_id=rater_id,
                trusted=weight >= 0.5,
                weight=float(weight),
                reasons=reasons,
            )
        return verdicts

    @staticmethod
    def _consensus(ratings: list[HumanRating]) -> dict[str, float]:
        by_item: dict[str, list[float]] = {}
        for r in ratings:
            by_item.setdefault(r.item_id, []).append(r.score)
        # Median is robust to a few troll ratings per item.
        return {item: float(np.median(vals)) for item, vals in by_item.items()}
