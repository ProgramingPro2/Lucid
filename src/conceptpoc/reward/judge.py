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


_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')
_RATIONALE_RE = re.compile(r'"rationale"\s*:\s*"([^"]*)')
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _normalize_score(value: float) -> float:
    """Clip to [0, 1], accepting either a 0-1 (requested) or a 0-5 rating.

    Values in (2, 5] are read as a 0-5 scale and rescaled; small overshoots
    (<= 2) are treated as 0-1 and simply clipped, so a stray 1.5 stays 1.0.
    """
    if 2.0 < value <= 5.0:
        value = value / 5.0
    return float(np.clip(value, 0.0, 1.0))


def _strip_wrappers(text: str) -> str:
    """Remove reasoning ``<think>`` blocks and a leading ```/```json fence."""
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        body = text[3:]
        nl = body.find("\n")
        if nl != -1 and body[:nl].strip().lower() in ("", "json"):
            body = body[nl + 1 :]
        text = body.split("```", 1)[0].strip()
    return text


def _parse_judge_response(content: str) -> JudgeResult:
    """Parse a judge LLM reply into a clipped ``JudgeResult``.

    Tolerant of reasoning ``<think>`` blocks, markdown code fences, a 0-5 rating
    scale, and replies truncated by the token cap before the closing brace (the
    score field is emitted before the rationale, so a regex fallback recovers it
    even from unclosed JSON).
    """
    raw = (content or "").strip()
    if not raw:
        return JudgeResult(0.0, "empty judge response")
    text = _strip_wrappers(raw)

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match is not None:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict) and "score" in data:
        try:
            score = float(data["score"])
        except (TypeError, ValueError):
            score = None
        if score is not None:
            rationale = data.get("rationale", "")
            if not isinstance(rationale, str):
                rationale = str(rationale)
            return JudgeResult(_normalize_score(score), rationale)

    score_match = _SCORE_RE.search(text)
    if score_match is not None:
        score = float(score_match.group(1))
        rationale_match = _RATIONALE_RE.search(text)
        rationale = rationale_match.group(1) if rationale_match else ""
        return JudgeResult(_normalize_score(score), rationale)

    return JudgeResult(0.0, f"unparseable judge response: {text[:120]}")


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

    def _create(self, prompt: str, output: str, *, json_mode: bool):
        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_message(prompt, output)},
            ],
            temperature=0.0,
            # Headroom so reasoning-capable judges finish any preamble AND emit
            # the closing brace; a tight cap truncates good-text replies to empty
            # content, silently scoring them 0.0.
            max_completion_tokens=512,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    def score(self, prompt: str, output: str, vectors: np.ndarray | None = None) -> JudgeResult:
        # Constrained JSON output is the most reliable (no code fences / prose);
        # fall back to a plain call once if a server rejects response_format.
        try:
            resp = self._create(prompt, output, json_mode=True)
        except Exception:  # noqa: BLE001 - server may not support response_format
            resp = self._create(prompt, output, json_mode=False)
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
