from __future__ import annotations

from pathlib import Path

import numpy as np

from conceptpoc.codec.hashing_codec import HashingCodec
from conceptpoc.data.cache import CachedCorpus, EmbeddingCache
from conceptpoc.data.corpus import load_documents
from conceptpoc.data.dataset import ConceptSequenceDataset, collate_sequences

CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.txt"


def test_load_documents_splits_blocks():
    docs = load_documents(CORPUS)
    assert len(docs) >= 5
    assert all(len(d.sentences) >= 2 for d in docs)


def test_cache_build_and_reload(tmp_path):
    docs = load_documents(CORPUS)
    codec = HashingCodec(dim=64)
    cache = EmbeddingCache(tmp_path)
    corpus = cache.build(docs, codec, "hashing:test")
    assert corpus.vectors.shape[0] == len(corpus.sentences)
    assert corpus.dim == 64
    # Second build hits the cache and returns identical content.
    again = cache.build(docs, codec, "hashing:test")
    assert np.allclose(corpus.vectors, again.vectors)


def test_dataset_windows_respect_doc_boundaries():
    corpus = CachedCorpus(
        vectors=np.random.randn(6, 8).astype(np.float32),
        sentences=[""] * 6,
        doc_offsets=np.array([0, 3, 6], dtype=np.int64),
    )
    ds = ConceptSequenceDataset(corpus, context_len=10)
    # Doc A: targets 1,2 ; Doc B: targets 4,5 -> 4 windows, none crossing index 3.
    assert len(ds) == 4


def test_collate_right_pads():
    corpus = CachedCorpus(
        vectors=np.random.randn(5, 8).astype(np.float32),
        sentences=[""] * 5,
        doc_offsets=np.array([0, 5], dtype=np.int64),
    )
    ds = ConceptSequenceDataset(corpus, context_len=10)
    batch = [ds[0], ds[3]]  # context lengths 1 and 4
    padded, mask, target = collate_sequences(batch)
    assert padded.shape[0] == 2
    assert target.shape == (2, 8)
    # Right padding: the first column is always real; real tokens come first.
    assert (~mask[:, 0]).all()
    assert mask[0].sum() == 3  # length-1 context -> 3 pad in a width-4 batch
