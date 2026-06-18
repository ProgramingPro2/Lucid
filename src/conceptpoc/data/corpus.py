"""Corpus loading.

Documents are plain-text files. A document is split into sentences; sentence
order within a document is preserved so the predictor can learn next-concept
transitions. Document boundaries are returned so we never train across the seam
between two unrelated texts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .segment import segment_sentences


@dataclass
class Document:
    source: str
    sentences: list[str]


def load_documents(
    path: str | Path, min_chars: int = 12, max_chars: int = 400
) -> list[Document]:
    p = Path(path)
    files: list[Path]
    if p.is_dir():
        files = sorted(f for f in p.rglob("*.txt") if f.is_file())
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f"Corpus path not found: {path}")

    docs: list[Document] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        # Blank lines separate logical documents within a single file.
        for block in _split_blocks(text):
            sentences = segment_sentences(block, min_chars, max_chars)
            if len(sentences) >= 2:
                docs.append(Document(source=str(f), sentences=sentences))
    if not docs:
        raise ValueError(f"No usable documents found in {path}")
    return docs


def _split_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in text.split("\n\n")]
    blocks = [b for b in blocks if b]
    return blocks or [text]
