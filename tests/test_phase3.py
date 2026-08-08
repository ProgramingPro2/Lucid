"""Unit tests for phase-3 paragraph writer training contract."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from conceptpoc.jepa.tokenizer import Tokenizer
from conceptpoc.writer.model import FluWriter
from conceptpoc.writer.train_phase2 import make_collate
from conceptpoc.writer.train_phase3 import (
    ParagraphBlockDataset,
    pool_paragraph_concept,
)
from experiments.sentence_pred.data import SentenceCorpus
from experiments.sentence_pred.paragraph_writer import render_paragraph


def _toy_corpus(n_docs: int = 4, sents_per_doc: int = 7, dim: int = 32) -> SentenceCorpus:
    rng = np.random.RandomState(0)
    sentences: list[str] = []
    offsets = [0]
    for d in range(n_docs):
        for i in range(sents_per_doc):
            sentences.append(f"Doc {d} sentence number {i} about topic {d}.")
        offsets.append(len(sentences))
    n = len(sentences)
    raw = rng.randn(n, dim).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-12
    doc_offsets = np.asarray(offsets, dtype=np.int64)
    train = np.arange(n_docs - 1, dtype=np.int64)
    held = np.asarray([n_docs - 1], dtype=np.int64)
    return SentenceCorpus(
        sent_vecs=raw,
        doc_offsets=doc_offsets,
        sentences=sentences,
        train_doc_ids=train,
        held_doc_ids=held,
    )


def test_pool_paragraph_concept_unit():
    v = np.eye(5, 8, dtype=np.float32)  # 5x8
    c = pool_paragraph_concept(v)
    assert c.shape == (8,)
    assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-5


def test_paragraph_dataset_shapes_and_shared_concept():
    corpus = _toy_corpus()
    tok = Tokenizer("char")
    g = 5
    ds = ParagraphBlockDataset(corpus, tok, doc_ids=corpus.train_doc_ids, g=g)
    assert len(ds) > 0
    # One block of G sentences should share the same concept across j.
    # Find first block's G items (indices 0..G-1 for first block of first eligible doc).
    concepts = [ds.items[j][2] for j in range(g)]
    for c in concepts[1:]:
        assert torch.allclose(concepts[0], c)
    assert abs(float(concepts[0].norm()) - 1.0) < 1e-4

    # Drop-arm __getitem__ returns tensors with expected ranks.
    prev, tgt, concept, has_prefix = ds[0]
    assert prev.dim() == 1 and tgt.dim() == 1
    assert concept.shape == (corpus.sent_vecs.shape[1],)
    assert isinstance(has_prefix, (bool, torch.Tensor)) or has_prefix in (True, False)


def test_collate_masking_and_concept_unit():
    corpus = _toy_corpus()
    tok = Tokenizer("char")
    ds = ParagraphBlockDataset(corpus, tok, doc_ids=corpus.train_doc_ids, g=5)
    collate = make_collate(tok, seq_len=128)
    # Prefer an item with a real prefix for masking check.
    batch_idx = None
    for i in range(min(len(ds), 50)):
        if ds.items[i][3]:  # prefix_present before drop arms
            batch_idx = i
            break
    assert batch_idx is not None
    # Build a mini-batch of 4 examples without relying on random drop wiping prefix.
    raw = []
    for i in range(batch_idx, min(batch_idx + 4, len(ds))):
        prev, tgt, concept, present = ds.items[i]
        raw.append((prev, tgt, concept, present))
    while len(raw) < 4:
        raw.append(raw[0])
    inputs, labels, concept = collate(raw)
    b = inputs.size(0)
    assert inputs.shape[0] == b and labels.shape == inputs.shape
    assert concept.shape == (b, corpus.sent_vecs.shape[1])
    # Concepts should be (approx) unit norm.
    norms = concept.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    # Labels: pad -> -100; with prefix, early positions masked.
    assert (labels == -100).any()
    # First example had a prefix: n_mask = 1+len(prev) positions of labels are -100
    prev0 = raw[0][0].view(-1).tolist()
    n_mask = 1 + len(prev0)
    assert (labels[0, :n_mask] == -100).all()
    # At least one supervised target token exists past the mask.
    assert (labels[0, n_mask:] != -100).any()


def test_render_paragraph_returns_g_strings():
    vocab = 128
    dim = 64
    writer = FluWriter(
        vocab,
        dim=dim,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        concept_dim=32,
        use_concept=True,
        use_film=True,
    )
    tok = Tokenizer("char")
    # Tiny vocab writer won't match char vocab — remap by using writer vocab size
    # via a stub tokenizer that clamps ids.
    class _StubTok:
        bos_id = 1
        eos_id = 2
        pad_id = 0
        vocab_size = vocab

        def encode(self, text: str):
            return [3 + (ord(c) % (vocab - 4)) for c in text[:20]]

        def decode(self, ids):
            return "x" * max(1, len([i for i in ids if i not in (0, 1, 2)]))

    stub = _StubTok()
    concept = torch.randn(1, 32)
    concept = concept / concept.norm(dim=-1, keepdim=True)
    g = 5
    out = render_paragraph(
        writer,
        stub,
        concept,
        first_prefix="hello",
        n_sentences=g,
        temperature=0.0,
        top_k=5,
        max_tokens=8,
    )
    assert isinstance(out, list)
    assert len(out) == g
    assert all(isinstance(s, str) for s in out)


def test_dataset_skips_short_docs():
    # One short doc (<G) and one long — only long contributes.
    sentences = ["a", "b", "c"] + [f"long {i}" for i in range(8)]
    offsets = np.asarray([0, 3, 11], dtype=np.int64)
    vecs = np.random.RandomState(1).randn(11, 16).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12
    corpus = SentenceCorpus(
        sent_vecs=vecs,
        doc_offsets=offsets,
        sentences=sentences,
        train_doc_ids=np.asarray([0, 1], dtype=np.int64),
        held_doc_ids=np.asarray([], dtype=np.int64),
    )
    tok = Tokenizer("char")
    ds = ParagraphBlockDataset(corpus, tok, doc_ids=corpus.train_doc_ids, g=5)
    # Doc 0 has 3 sents -> skipped; doc 1 has 8 -> (8-5+1)*5 = 20 examples.
    assert len(ds) == 20
