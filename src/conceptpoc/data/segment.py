"""Lightweight sentence segmentation.

A dependency-free regex segmenter that splits on sentence-final punctuation
while avoiding the most common false splits (abbreviations, decimals). This is
intentionally simple: concept vectors are robust to imperfect boundaries, and
avoiding heavy NLP dependencies keeps the project easy to install.
"""
from __future__ import annotations

import re

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "no", "vol", "p", "pp", "al",
}

_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")
_WS = re.compile(r"\s+")


def _looks_like_abbrev(chunk: str) -> bool:
    tail = chunk.rstrip().rstrip(".").split()
    if not tail:
        return False
    return tail[-1].lower() in _ABBREV


def segment_sentences(
    text: str, min_chars: int = 12, max_chars: int = 400
) -> list[str]:
    text = _WS.sub(" ", text.replace("\n", " ")).strip()
    if not text:
        return []

    raw_parts = _SPLIT.split(text)
    sentences: list[str] = []
    buffer = ""
    for part in raw_parts:
        candidate = (buffer + " " + part).strip() if buffer else part.strip()
        if _looks_like_abbrev(candidate):
            buffer = candidate
            continue
        buffer = ""
        sentences.append(candidate)
    if buffer:
        sentences.append(buffer)

    cleaned: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < min_chars:
            continue
        if len(sent) > max_chars:
            sent = sent[:max_chars].rsplit(" ", 1)[0]
        cleaned.append(sent)
    return cleaned
