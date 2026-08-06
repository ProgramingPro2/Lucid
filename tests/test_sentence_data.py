"""Tests for per-sentence JEPA sequence data layer (Fable §4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.sentence_pred.data import (
    SentenceCorpus,
    SentenceSequenceDataset,
    build_sentence_corpus,
    collate_sentence_batch,
    corpus_cache_key,
    detok_text,
    load_sentence_corpus,
    save_sentence_corpus,
)


def test_detok_text():
    assert detok_text("well @-@ known") == "well-known"
    assert detok_text("1 @,@ 000") == "1,000"
    assert detok_text("end @.@ ") == "end."
    assert detok_text("Hello , world .") == "Hello, world."
    assert detok_text("Wow ! Really ? Yes ; ok : 'tis") == "Wow! Really? Yes; ok: 'tis"
    assert detok_text('He said " hello "') == 'He said "hello"'


def test_cache_key_sensitivity(tmp_path: Path):
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"ckpt-v1")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a. b. c. d.\n\n", encoding="utf-8")

    k128 = corpus_cache_key(ckpt, corpus, min_sents=4, max_tokens=128, detok_wikitext=True)
    k64 = corpus_cache_key(ckpt, corpus, min_sents=4, max_tokens=64, detok_wikitext=True)
    k_nodetok = corpus_cache_key(
        ckpt, corpus, min_sents=4, max_tokens=128, detok_wikitext=False
    )
    assert k128 != k64
    assert k128 != k_nodetok
    assert len(k128) == 16

    # Different keys must map to different cache dirs under out_dir.
    out_a = tmp_path / "out" / f"corpus_{k128}"
    out_b = tmp_path / "out" / f"corpus_{k64}"
    assert out_a != out_b


def test_doc_disjoint_manifest():
    n_docs = 20
    offsets = np.arange(0, n_docs + 1, dtype=np.int64) * 5
    vecs = np.zeros((n_docs * 5, 8), dtype=np.float32)
    for i in range(vecs.shape[0]):
        vecs[i, i % 8] = 1.0
    rng = np.random.RandomState(1234)
    perm = rng.permutation(n_docs)
    n_held = int(round(n_docs * 0.10))
    held = np.sort(perm[:n_held].astype(np.int64))
    train = np.sort(perm[n_held:].astype(np.int64))
    corpus = SentenceCorpus(
        sent_vecs=vecs,
        doc_offsets=offsets,
        sentences=[f"s{i}" for i in range(vecs.shape[0])],
        train_doc_ids=train,
        held_doc_ids=held,
    )
    inter = set(corpus.train_doc_ids.tolist()) & set(corpus.held_doc_ids.tolist())
    assert inter == set()
    all_ids = set(corpus.train_doc_ids.tolist()) | set(corpus.held_doc_ids.tolist())
    assert all_ids == set(range(n_docs))
    for i in all_ids:
        assert 0 <= i < corpus.n_docs


def _synthetic_unit_corpus() -> SentenceCorpus:
    """Three small docs of orthonormal-ish unit vectors (dim=4)."""
    dim = 4
    # doc0: 6 sents, doc1: 5 sents, doc2: 4 sents
    n_per = [6, 5, 4]
    vecs = []
    sents = []
    for d, n in enumerate(n_per):
        for i in range(n):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0 if (d + i) % 2 == 0 else -1.0
            if (d + i) % 3 == 0:
                v[(i + 1) % dim] = 0.5
            v = v / (np.linalg.norm(v) + 1e-12)
            vecs.append(v.astype(np.float32))
            sents.append(f"d{d}_s{i}")
    sent_vecs = np.stack(vecs, axis=0)
    offsets = np.cumsum([0] + n_per).astype(np.int64)
    train = np.array([0, 1], dtype=np.int64)
    held = np.array([2], dtype=np.int64)
    return SentenceCorpus(
        sent_vecs=sent_vecs,
        doc_offsets=offsets,
        sentences=sents,
        train_doc_ids=train,
        held_doc_ids=held,
    )


def test_shift_by_one_alignment():
    corpus = _synthetic_unit_corpus()
    ds = SentenceSequenceDataset(
        corpus, corpus.train_doc_ids, context_len=3, stride=1, eval_mode=False
    )
    assert len(ds) > 0
    for i in range(len(ds)):
        item = ds[i]
        doc_id = item["doc_id"]
        start = item["start"]
        a, _b = corpus.doc_slice(doc_id)
        T = item["context"].shape[0]
        for t in range(T):
            expected_ctx = corpus.sent_vecs[a + start + t]
            expected_tgt = corpus.sent_vecs[a + start + t + 1]
            assert torch.allclose(
                item["context"][t], torch.from_numpy(expected_ctx), atol=1e-6
            )
            assert torch.allclose(
                item["target_seq"][t], torch.from_numpy(expected_tgt), atol=1e-6
            )
            assert bool(item["valid"][t]) is True

    # Collate pads shorter windows if mixed lengths (here uniform).
    batch = [ds[i] for i in range(min(3, len(ds)))]
    context, pad_mask, target_seq, valid, doc_ids = collate_sentence_batch(batch)
    assert context.shape[0] == len(batch)
    assert pad_mask.dtype == torch.bool
    assert (~pad_mask).all()
    assert valid.all()
    assert doc_ids.shape == (len(batch),)


class _FakeCodec:
    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # Deterministic unit vector from text hash / length.
            idx = (len(t) + sum(ord(c) for c in t[:8])) % self.dim
            out[i, idx] = 1.0
            if i % 2:
                out[i, (idx + 1) % self.dim] = 0.25
            out[i] /= np.linalg.norm(out[i]) + 1e-12
        return out


def test_round_trip_build_load(tmp_path: Path):
    # Tiny synthetic corpus file: blank-line docs with enough sentences.
    corpus_path = tmp_path / "tiny.txt"
    blocks = []
    for d in range(5):
        sents = [
            f"Sentence number {d}-{i} is long enough for the segmenter filters here."
            for i in range(6)
        ]
        blocks.append(" ".join(sents))
    corpus_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")

    out_dir = tmp_path / "corpus_out"
    built = build_sentence_corpus(
        corpus_path=str(corpus_path),
        jepa_ckpt=str(ckpt),
        out_dir=str(out_dir),
        max_tokens=32,
        batch_size=8,
        min_sents=3,
        held_out_frac=0.20,
        seed=1234,
        detok_wikitext=True,
        codec=_FakeCodec(),
    )
    assert built.n_docs >= 2
    assert built.sent_vecs.shape[0] == len(built.sentences)
    assert built.doc_offsets.dtype == np.int64
    assert built.sent_vecs.dtype == np.float32

    key = corpus_cache_key(
        ckpt, corpus_path, min_sents=3, max_tokens=32, detok_wikitext=True
    )
    cache_dir = out_dir / f"corpus_{key}"
    assert cache_dir.is_dir()
    assert (cache_dir / "sent_vecs.npy").is_file()
    assert (cache_dir / "sentences.json").is_file()

    loaded = load_sentence_corpus(cache_dir)
    assert loaded.sent_vecs.shape == built.sent_vecs.shape
    assert np.allclose(loaded.doc_offsets, built.doc_offsets)
    assert loaded.sentences == built.sentences
    assert np.array_equal(loaded.train_doc_ids, built.train_doc_ids)
    assert np.array_equal(loaded.held_doc_ids, built.held_doc_ids)
    # float16 round-trip: close, not bit-identical
    assert np.allclose(loaded.sent_vecs, built.sent_vecs, atol=1e-3)

    # Second build hits cache (no codec needed).
    again = build_sentence_corpus(
        corpus_path=str(corpus_path),
        jepa_ckpt=str(ckpt),
        out_dir=str(out_dir),
        max_tokens=32,
        min_sents=3,
        detok_wikitext=True,
        codec=None,  # would fail if cache missed
    )
    assert again.sentences == built.sentences

    inter = set(built.train_doc_ids.tolist()) & set(built.held_doc_ids.tolist())
    assert inter == set()
    n_doc = built.n_docs
    for i in list(built.train_doc_ids) + list(built.held_doc_ids):
        assert 0 <= int(i) < n_doc

    # Cache key changes with max_tokens → different dir.
    key2 = corpus_cache_key(
        ckpt, corpus_path, min_sents=3, max_tokens=64, detok_wikitext=True
    )
    assert key2 != key
    build_sentence_corpus(
        corpus_path=str(corpus_path),
        jepa_ckpt=str(ckpt),
        out_dir=str(out_dir),
        max_tokens=64,
        min_sents=3,
        detok_wikitext=True,
        codec=_FakeCodec(),
    )
    assert (out_dir / f"corpus_{key2}").is_dir()


def test_save_load_helpers(tmp_path: Path):
    corpus = _synthetic_unit_corpus()
    path = tmp_path / "c"
    save_sentence_corpus(corpus, path)
    loaded = load_sentence_corpus(path)
    assert loaded.sentences == corpus.sentences
    assert np.array_equal(loaded.doc_offsets, corpus.doc_offsets)
    with open(path / "split_manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    assert "train_doc_ids" in man and "held_doc_ids" in man
