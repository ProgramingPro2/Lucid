"""Data test: trajectory dataset on a tiny real corpus."""
import numpy as np
import torch

from src.conceptpoc.trajectory.data import TrajectoryDataset, make_trajectory_loader, collate


def _make_dummy_codec(dim=16):
    """A tiny stand-in codec: hash sentence text to a fixed vector (no model load)."""
    class _DummyCodec:
        def __init__(self):
            self.dim = dim
        def encode(self, texts):
            rng = np.random.RandomState(0)
            return rng.randn(len(texts), dim).astype(np.float32)
    return _DummyCodec()


def test_trajectory_dataset_builds():
    codec = _make_dummy_codec(16)
    ds = TrajectoryDataset("data/austen_corpus.txt", codec, C=4, K=3, max_docs=50)
    assert len(ds) > 0
    ctx, tgt = ds[0]
    assert ctx.shape == (4, 16) and tgt.shape == (3, 16)
    assert np.isfinite(ctx).all() and np.isfinite(tgt).all()


def test_trajectory_loader_packs_batches():
    codec = _make_dummy_codec(16)
    loader = make_trajectory_loader("data/austen_corpus.txt", codec, C=4, K=3,
                                    batch_size=4, streaming=False, shuffle=True,
                                    max_docs=50)
    batch = next(iter(loader))
    ctx, tgt = batch
    assert ctx.shape == (4, 4, 16) and tgt.shape == (4, 3, 16)
    assert torch.isfinite(ctx).all() and torch.isfinite(tgt).all()
