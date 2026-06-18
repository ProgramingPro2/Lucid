"""LLM-as-judge.

Scores a generated continuation in [0, 1]. Two backends:

- ``OpenAIJudge``: asks an LLM to score coherence, relevance and fluency. Used
  when ``OPENAI_API_KEY`` is set.
- ``HeuristicJudge``: a deterministic, offline proxy combining lexical
  repetition, length sanity and semantic-flow signals (cosine similarity
  between consecutive concept vectors). Lets the whole pipeline run with no
  network or API key.

The heuristic is intentionally a weak stand-in. It is good enough to drive and
test the self-improvement loop, and it is replaced by the real LLM (and humans)
when those are available.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class JudgeResult:
    score: float
    rationale: str = ""


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
        sys = (
            "You are a strict writing judge. Rate the continuation for coherence, "
            "relevance to the prompt, and fluency. Respond with JSON: "
            '{"score": <0..1 float>, "rationale": "<short>"}.'
        )
        user = f"PROMPT:\n{prompt}\n\nCONTINUATION:\n{output}"
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return JudgeResult(float(np.clip(data["score"], 0.0, 1.0)), data.get("rationale", ""))


def build_judge() -> Judge:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIJudge()
        except Exception:
            pass
    return HeuristicJudge()
