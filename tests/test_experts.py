from __future__ import annotations

import torch

from conceptpoc.config import PredictorConfig
from conceptpoc.experts.router import GatingRouter, MixtureOfPredictors
from conceptpoc.predictor.model import ConceptPredictor


def _expert(dim=16):
    return ConceptPredictor(
        concept_dim=dim,
        cfg=PredictorConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=16),
    )


def test_router_weights_sum_to_one():
    router = GatingRouter(concept_dim=16, n_experts=3)
    ctx = torch.randn(4, 5, 16)
    w = router(ctx)
    assert w.shape == (4, 3)
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_mixture_forward_shape_and_norm():
    experts = [_expert(), _expert(), _expert()]
    moe = MixtureOfPredictors(experts, GatingRouter(16, 3))
    ctx = torch.randn(2, 4, 16)
    out = moe(ctx)
    assert out.shape == (2, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-4)
