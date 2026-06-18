from __future__ import annotations

import numpy as np
import torch

from conceptpoc.config import PredictorConfig, TrainConfig
from conceptpoc.data.cache import CachedCorpus
from conceptpoc.data.dataset import ConceptSequenceDataset
from conceptpoc.predictor.model import ConceptPredictor, predictor_loss
from conceptpoc.predictor.train import train_predictor


def _tiny_cfg() -> PredictorConfig:
    return PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16)


def test_forward_shape_and_norm():
    model = ConceptPredictor(concept_dim=16, cfg=_tiny_cfg())
    ctx = torch.randn(3, 5, 16)
    out = model(ctx)
    assert out.shape == (3, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-4)


def test_loss_is_zero_for_perfect_prediction():
    target = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    loss = predictor_loss(target, target)
    assert loss.total.item() < 1e-5


def _learnable_corpus(n_docs=40, dim=16) -> CachedCorpus:
    # Build sequences with a learnable rule: next vector is a fixed rotation of
    # the current one, so a predictor should reduce loss below the random init.
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    vecs, offsets = [], [0]
    for _ in range(n_docs):
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        seq = [v]
        for _ in range(5):
            v = q @ v
            seq.append(v / np.linalg.norm(v))
        vecs.extend(seq)
        offsets.append(len(vecs))
    return CachedCorpus(
        vectors=np.asarray(vecs, dtype=np.float32),
        sentences=[""] * len(vecs),
        doc_offsets=np.asarray(offsets, dtype=np.int64),
    )


def test_training_reduces_validation_loss():
    corpus = _learnable_corpus()
    ds = ConceptSequenceDataset(corpus, context_len=6)
    cfg = _tiny_cfg()
    train_cfg = TrainConfig(batch_size=16, lr=1e-3, epochs=8, val_fraction=0.2)
    _, history = train_predictor(ds, cfg, train_cfg, device="cpu")
    assert history.val_cosine[-1] > history.val_cosine[0]
