"""Tests for the E1 document-intact data module.

The properties under test are the ones the endpoint depends on:

* a document never appears in both halves of the split;
* a training window never crosses a document boundary;
* a true/shuffled eval pair shares its target tokens *exactly*;
* ``target_labels`` masks the prefix and shifts the target by one.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.e1.data import EvalItem, PackedCorpus, target_labels


def _corpus(n_docs=12, length=600, seed=0):
    rng = np.random.default_rng(seed)
    docs = [list(range(1000 + i * length, 1000 + (i + 1) * length)) for i in range(n_docs)]
    rng.shuffle(docs)
    return PackedCorpus(docs, eos_id=0)


def test_rejects_empty_corpus():
    with pytest.raises(ValueError):
        PackedCorpus([])
    with pytest.raises(ValueError):
        PackedCorpus([[], []])


def test_split_keeps_documents_disjoint():
    corpus = _corpus()
    train, test = corpus.split(test_fraction=0.25, seed=3)
    train_docs = {tuple(d) for d in train.docs}
    test_docs = {tuple(d) for d in test.docs}
    assert train_docs.isdisjoint(test_docs)
    assert train.n_tokens + test.n_tokens == corpus.n_tokens


def test_dev_split_is_disjoint_and_does_not_move_the_test_set():
    """Dev is document-disjoint, and the sealed test documents stay put.

    The convergence rule reads dev. If carving dev out reshuffled the test
    split, every arm's sealed items would change the day the split was added.
    """
    corpus = _corpus(n_docs=40, length=200)
    train, dev, test = corpus.split_with_dev(0.1, 0.05, seed=7)
    again = corpus.split_with_dev(0.1, 0.05, seed=7)
    plain_test = corpus.split(0.1, seed=7)[1].docs
    assert test.docs == plain_test
    assert train.docs == again[0].docs
    assert dev.docs == again[1].docs
    assert test.docs == again[2].docs

    groups = [
        {tuple(d) for d in train.docs},
        {tuple(d) for d in dev.docs},
        {tuple(d) for d in test.docs},
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert sum(len(g) for g in groups) == len(corpus.docs)
    assert len(dev.docs) >= 1 and len(train.docs) >= 1

    other_test = corpus.split_with_dev(0.1, 0.05, seed=8)[2].docs
    assert other_test == corpus.split(0.1, seed=8)[1].docs


def test_split_is_deterministic_and_depends_on_seed():
    corpus = _corpus()
    a = corpus.split(0.25, seed=1)[1].docs
    b = corpus.split(0.25, seed=1)[1].docs
    assert a == b
    # A different seed should usually pick different documents.
    c = corpus.split(0.25, seed=99)[1].docs
    assert set(map(tuple, a)) != set(map(tuple, c))


def test_train_batch_stays_inside_one_document():
    length = 600
    corpus = _corpus(n_docs=12, length=length)
    rng = np.random.default_rng(0)
    x, y = corpus.train_batch(rng, batch_size=8, context=64)
    assert x.shape == (8, 64) and y.shape == (8, 64)

    def doc_of(token: int) -> int:
        # Documents in ``_corpus`` are disjoint ranges of ``length`` tokens.
        return (token - 1000) // length

    for row in range(8):
        assert np.all(np.diff(x[row]) == 1)
        # The whole window, including the extra label token, lives in one doc.
        assert len({doc_of(int(t)) for t in x[row]}) == 1
        assert doc_of(int(y[row][-1])) == doc_of(int(x[row][0]))
        # Labels are the next tokens.
        assert np.all(y[row][:-1] == x[row][1:])
        assert y[row][-1] == x[row][-1] + 1


def test_train_batch_needs_a_long_document():
    corpus = PackedCorpus([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        corpus.train_batch(np.random.default_rng(0), batch_size=2, context=64)


def test_target_labels_masks_prefix_and_shifts():
    ids = [0, 1, 2, 3, 4, 5, 6, 7]
    labels = target_labels(ids, target_start=4)
    # Prefix positions 0..2 are masked. Position 3 is the last *prefix*
    # position but its successor is the first target token, so it must carry a
    # label or the first target token is never scored.
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:] == [4, 5, 6, 7, -100]
    assert len(labels) == len(ids)
    # Every token in the target span is scored exactly once.
    scored = [v for v in labels if v != -100]
    assert scored == ids[4:]


def test_target_labels_rejects_bad_start():
    with pytest.raises(ValueError):
        target_labels([1, 2, 3], 0)
    with pytest.raises(ValueError):
        target_labels([1, 2, 3], 3)


def test_eval_items_pair_true_and_shuffled_contexts():
    corpus = _corpus(n_docs=20)
    items = corpus.eval_items(10, prefix=128, target=64, seed=5)
    assert len(items) == 10
    for item in items:
        assert len(item.true_ids) == len(item.shuffled_ids) == 192
        assert item.shuffled_ids[item.target_start :] == item.true_ids[item.target_start :]
        # The contexts differ (otherwise there is nothing to contrast).
        assert item.shuffled_ids[: item.target_start] != item.true_ids[: item.target_start]


def test_eval_items_are_deterministic():
    corpus = _corpus(n_docs=20)
    a = corpus.eval_items(6, seed=11)
    b = corpus.eval_items(6, seed=11)
    assert [i.true_ids for i in a] == [i.true_ids for i in b]
    assert [i.shuffled_ids for i in a] == [i.shuffled_ids for i in b]


def test_eval_items_need_two_long_documents():
    corpus = PackedCorpus([[1] * 600])
    with pytest.raises(ValueError):
        corpus.eval_items(1, prefix=128, target=64)


def test_eval_item_validates_alignment():
    with pytest.raises(ValueError):
        EvalItem(doc_index=0, true_ids=[1, 2, 3], shuffled_ids=[1, 2], target_start=1)
    with pytest.raises(ValueError):
        EvalItem(doc_index=0, true_ids=[1, 2, 3], shuffled_ids=[1, 2, 3], target_start=0)


def test_cache_roundtrip_preserves_every_token(tmp_path):
    corpus = _corpus(n_docs=6, length=300)
    path = tmp_path / "corpus.npz"
    corpus.save(path)
    loaded = PackedCorpus.from_cache(path)
    assert loaded.docs == corpus.docs
    assert loaded.eos_id == corpus.eos_id
    assert loaded.n_tokens == corpus.n_tokens


def test_from_texts_skips_short_documents():
    class Tok:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    corpus = PackedCorpus.from_texts(["x" * 10, "y" * 100], Tok(), min_tokens=64)
    assert len(corpus.docs) == 1
    assert corpus.eos_id == 0
    assert corpus.docs[0][-1] == 0
