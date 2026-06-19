"""LLM-as-judge.

Scores a generated continuation in [0, 1]. Backends (select via ``JUDGE_BACKEND``):

- ``HeuristicJudge``: deterministic, offline proxy (default).
- ``LemonadeJudge``: local LLM via Lemonade Server's OpenAI-compatible API.
- ``OpenAIJudge``: cloud LLM via OpenAI.

The heuristic is intentionally a weak stand-in. It is good enough to drive and
test the self-improvement loop, and it is replaced by a real LLM (and humans)
when those are available.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict writing judge. Rate the continuation for coherence, "
    "relevance to the prompt, and fluency. Respond with JSON: "
    '{"score": <0..1 float>, "rationale": "<short>"}.'
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)


@dataclass
class JudgeResult:
    score: float
    rationale: str = ""


def _parse_judge_response(content: str) -> JudgeResult:
    """Parse a judge LLM reply into a clipped ``JudgeResult``."""
    text = (content or "").strip()
    if not text:
        return JudgeResult(0.0, "empty judge response")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match is None:
            return JudgeResult(0.0, f"unparseable judge response: {text[:120]}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return JudgeResult(0.0, f"unparseable judge response: {text[:120]}")

    try:
        score = float(data["score"])
    except (KeyError, TypeError, ValueError):
        return JudgeResult(0.0, f"missing score in judge response: {text[:120]}")

    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return JudgeResult(float(np.clip(score, 0.0, 1.0)), rationale)


def _judge_user_message(prompt: str, output: str) -> str:
    return f"PROMPT:\n{prompt}\n\nCONTINUATION:\n{output}"


class Judge(ABC):
    @abstractmethod
    def score(self, prompt: str, output: str, vectors: np.ndarray | None = None) -> JudgeResult:
        ...

    def score_batch(
        self, prompt: str, outputs: list[str], vectors: list[np.ndarray] | None = None
    ) -> list[JudgeResult]:
        vecs = vectors or [None] * len(outputs)
        return [self.score(prompt, o, v) for o, v in zip(outputs, vecs)]


class HeuristicJudge(Judge):
    def __init__(self, target_sentences: int = 6) -> None:
        self.target_sentences = target_sentences

    def score(self, prompt: str, output: str, vectors: np.ndarray | None = None) -> JudgeResult:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", output.strip()) if s]
        if not sentences:
            return JudgeResult(0.0, "empty output")

        words = re.findall(r"[a-z0-9']+", output.lower())
        distinct = len(set(words)) / len(words) if words else 0.0

        # Penalize verbatim sentence repetition.
        uniq_sents = len(set(s.strip().lower() for s in sentences)) / len(sentences)

        # Length sanity vs target.
        length_score = 1.0 - min(
            1.0, abs(len(sentences) - self.target_sentences) / self.target_sentences
        )

        # Semantic flow: adjacent concepts should be related but not identical.
        flow = 0.5
        if vectors is not None and len(vectors) >= 2:
            sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
            flow = float(np.clip(np.mean(1.0 - np.abs(sims - 0.5) * 2.0), 0.0, 1.0))

        score = float(
            0.30 * distinct + 0.25 * uniq_sents + 0.20 * length_score + 0.25 * flow
        )
        rationale = (
            f"distinct={distinct:.2f} uniq_sents={uniq_sents:.2f} "
            f"len={length_score:.2f} flow={flow:.2f}"
        )
        return JudgeResult(float(np.clip(score, 0.0, 1.0)), rationale)


class OpenAIJudge(Judge):
    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model or os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

    def score(self, prompt: str, output: str, vectors: np.ndarray | None = None) -> JudgeResult:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_message(prompt, output)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return _parse_judge_response(resp.choices[0].message.content or "")


class LemonadeJudge(Judge):
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url
            or os.environ.get("LEMONADE_BASE_URL", "http://localhost:13305/api/v1"),
            api_key=os.environ.get("LEMONADE_API_KEY", "lemonade"),
        )
        self._model = model or os.environ.get("LEMONADE_JUDGE_MODEL", "")
        if not self._model:
            raise ValueError("LEMONADE_JUDGE_MODEL must be set when JUDGE_BACKEND=lemonade")
        self._warned_parse_failure = False

    def score(self, prompt: str, output: str, vectors: np.ndarray | None = None) -> JudgeResult:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_message(prompt, output)},
            ],
            temperature=0.0,
            max_completion_tokens=256,
        )
        result = _parse_judge_response(resp.choices[0].message.content or "")
        if result.score == 0.0 and result.rationale.startswith("unparseable"):
            if not self._warned_parse_failure:
                logger.warning(
                    "Lemonade judge returned unparseable output; check lemonade status / "
                    "lemonade list and that the model is loaded"
                )
                self._warned_parse_failure = True
        return result


def build_judge() -> Judge:
    backend = os.environ.get("JUDGE_BACKEND", "heuristic").lower()
    if backend == "lemonade":
        return LemonadeJudge()
    if backend == "openai":
        return OpenAIJudge()
    if backend == "heuristic":
        return HeuristicJudge()
    raise ValueError(f"Unknown JUDGE_BACKEND: {backend!r} (expected heuristic, lemonade, or openai)")
