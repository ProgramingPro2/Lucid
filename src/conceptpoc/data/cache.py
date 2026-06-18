"""Embedding cache.

Encoding a corpus is the slow step, so we encode once and cache the results to
disk (npz). The cache stores the per-sentence concept vectors, the sentence
texts (for the retrieval decoder bank), and the document boundaries so training
never crosses unrelated texts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..codec.base import ConceptCodec
from ..progress import iter_bar, log_stage
from .corpus import Document

_ENCODE_BATCH = 256


@dataclass
class CachedCorpus:
    vectors: np.ndarray  # (n_sentences, dim) float32
    sentences: list[str]
    doc_offsets: np.ndarray  # (n_docs + 1,) int; doc i = [offsets[i], offsets[i+1])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def doc_slices(self) -> list[tuple[int, int]]:
        return [
            (int(self.doc_offsets[i]), int(self.doc_offsets[i + 1]))
            for i in range(len(self.doc_offsets) - 1)
        ]


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, docs: list[Document], codec_id: str) -> str:
        h = hashlib.sha256()
        h.update(codec_id.encode())
        for doc in docs:
            h.update(doc.source.encode())
            h.update(str(len(doc.sentences)).encode())
        return h.hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"corpus_{key}.npz"

    def build(
        self,
        docs: list[Document],
        codec: ConceptCodec,
        codec_id: str,
        progress: bool = True,
    ) -> CachedCorpus:
        key = self._key(docs, codec_id)
        path = self._path(key)
        if path.exists():
            log_stage("loaded cached embeddings", enabled=progress)
            return self._load(path)

        sentences: list[str] = []
        offsets = [0]
        for doc in docs:
            sentences.extend(doc.sentences)
            offsets.append(len(sentences))

        log_stage(
            f"encoding {len(sentences):,} sentences...",
            enabled=progress,
        )
        chunks = [
            sentences[i : i + _ENCODE_BATCH]
            for i in range(0, len(sentences), _ENCODE_BATCH)
        ]
        encoded = [
            codec.encode(batch)
            for batch in iter_bar(
                chunks,
                desc="encoding",
                total=len(chunks),
                enabled=progress,
                unit="batch",
            )
        ]
        vectors = np.concatenate(encoded, axis=0) if encoded else np.zeros(
            (0, codec.dim), dtype=np.float32
        )
        cached = CachedCorpus(
            vectors=vectors,
            sentences=sentences,
            doc_offsets=np.asarray(offsets, dtype=np.int64),
        )
        self._save(path, cached)
        return cached

    def _save(self, path: Path, cached: CachedCorpus) -> None:
        np.savez(
            path,
            vectors=cached.vectors,
            sentences=np.array(cached.sentences, dtype=object),
            doc_offsets=cached.doc_offsets,
        )

    def _load(self, path: Path) -> CachedCorpus:
        data = np.load(path, allow_pickle=True)
        return CachedCorpus(
            vectors=data["vectors"].astype(np.float32),
            sentences=list(data["sentences"]),
            doc_offsets=data["doc_offsets"].astype(np.int64),
        )
