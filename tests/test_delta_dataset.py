"""Tests for the delta-sequence dataset."""
import numpy as np
import torch

from src.conceptpoc.data.cache import CachedCorpus
from src.conceptpoc.data.delta_dataset import DeltaSequenceDataset, collate_deltas, normalize_rows


def _fake_corpus(dim=16):
    """Two docs; first doc has a smooth oscillating trajectory (delta signal)."""
    rng = np.random.RandomState(0)
    # doc 0: oscillation along two axes
    t = np.linspace(0, 8, 20)
    z0 = np.zeros((20, dim), dtype=np.float32)
    z0[:, 0] = np.sin(t)
    z0[:, 1] = np.cos(t)
    z0 /= (np.linalg.norm(z0, axis=1, keepdims=True) + 1e-9)
    # doc 1: random noise
    z1 = rng.randn(12, dim).astype(np.float32)
    z1 /= (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-9)
    vecs = np.concatenate([z0, z1], axis=0)
    offsets = np.array([0, 20, 32], dtype=np.int64)
    return CachedCorpus(
        vectors=vecs,
        sentences=[f"s{i}" for i in range(32)],
        doc_offsets=offsets,
    )


def test_dataset_builds_and_shapes():
    corpus = _fake_corpus(dim=16)
    ds = DeltaSequenceDataset(corpus, context_len=8)
    assert len(ds) > 0
    ctx, tgt = ds[0]
    assert ctx.dim() == 2 and ctx.shape[-1] == 16
    assert tgt.shape == (16,)
    # context is 2D: (L, dim); target is normalized direction
    assert np.isclose(float(tgt.norm()), 1.0, atol=1e-3)


def test_oscillation_produces_learnable_signal():
    """The oscillation in doc0 should make deltas predictable beyond random."""
    corpus = _fake_corpus(dim=16)
    ds = DeltaSequenceDataset(corpus, context_len=8)
    lasts, tgts = [], []
    for i in range(len(ds)):
        c, t = ds[i]
        lasts.append(c[-1])   # last context delta
        tgts.append(t)
    y = torch.stack(tgts).numpy()
    last = torch.stack(lasts)
    # cosine between consecutive deltas (they anti-correlate in oscillation)
    cos = torch.nn.functional.cosine_similarity(last, torch.from_numpy(y), dim=-1).numpy()
    # random baseline
    rng = np.random.RandomState(1)
    rand = rng.randn(len(y), 16)
    rand /= (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9)
    rand_cos = (rand * y).sum(1)
    # oscillation should make |signal| stronger than random
    assert abs(cos.mean()) > 0.2, f"expected oscillation signal, got {cos.mean():.3f}"
    assert abs(cos.mean()) > abs(rand_cos.mean()), "oscillation signal should beat random"


def test_collate_right_pads():
    corpus = _fake_corpus(dim=4)
    ds = DeltaSequenceDataset(corpus, context_len=16)  # longer ctx than shortest doc
    batch = [ds[i] for i in range(4)]
    padded, mask, targets = collate_deltas(batch)
    assert padded.dim() == 3 and padded.shape[-1] == 4
    assert padded.shape[1] == mask.shape[1]
    assert mask.dtype == torch.bool
    assert targets.shape == (4, 4)
