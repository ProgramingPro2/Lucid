"""Tests for the delta predictor + its loss/decode path."""
import numpy as np
import torch

from src.conceptpoc.config import PredictorConfig
from src.conceptpoc.predictor.delta_model import (
    DeltaPredictor,
    delta_loss,
    delta_cosine,
    apply_delta,
)


def test_delta_predictor_shapes():
    cfg = PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    p = DeltaPredictor(concept_dim=16, cfg=cfg)
    ctx = torch.randn(4, 8, 16)   # window of deltas
    mask = torch.zeros(4, 8, dtype=torch.bool)
    out = p(ctx, mask)
    assert out.shape == (4, 16)
    assert torch.isclose(out.norm(dim=-1), torch.ones(4), atol=1e-3).all()


def test_delta_loss_lowers_on_aligned_directions():
    cfg = PredictorConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    p = DeltaPredictor(concept_dim=8, cfg=cfg)
    ctx = torch.randn(2, 4, 8)
    target = torch.randn(2, 8)
    out = p(ctx)
    l0 = delta_loss(out, target).total.item()
    opt = torch.optim.AdamW(p.parameters(), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        delta_loss(out, target).total.backward()
        opt.step()
        out = p(ctx)
    l1 = delta_loss(out, target).total.item()
    assert l1 <= l0 + 1e-6


def test_apply_delta_keeps_unit_norm():
    cur = torch.randn(3, 8)
    d = torch.randn(3, 8)
    nxt = apply_delta(cur, d, step=0.5)
    assert torch.isclose(nxt.norm(dim=-1), torch.ones(3), atol=1e-3).all()


def test_oscillation_predictor_reduces():
    """On a synthetic oscillating trajectory (deltas flip sign), a small
    DeltaPredictor trained on it should beat the random baseline."""
    torch.manual_seed(0)
    seq = torch.zeros(60, 16)
    seq[:, 0] = torch.linspace(-1, 1, 60)
    seq[:, 1] = torch.sin(torch.linspace(0, 8, 60))
    # deltas: consecutive change directions
    deltas = (seq[1:] - seq[:-1])
    deltas = deltas / (deltas.norm(dim=-1, keepdim=True) + 1e-9)  # (59,16)
    # build windows: predict deltas[i] from deltas[i-3:i]
    X, y = [], []
    for i in range(3, len(deltas)):
        X.append(deltas[i-3:i][:, None, :])  # keep (L, dim)
        y.append(deltas[i])
    X = torch.stack([x[:, 0] for x in X])  # (N,3,16)
    y = torch.stack(y)
    cfg = PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=8)
    p = DeltaPredictor(16, cfg)
    opt = torch.optim.AdamW(p.parameters(), lr=1e-2)
    mask = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool)
    for _ in range(50):
        opt.zero_grad()
        out = p(X, mask)
        delta_loss(out, y).total.backward()
        opt.step()
    out = p(X, mask)
    c = float(delta_cosine(out, y).item())
    rng = np.random.RandomState(0)
    rand = rng.randn(len(y), 16)
    rand /= (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9)
    c0 = float((rand * y.numpy()).sum(1).mean())
    # should at least beat (or be far from) random; oscillation is learnable
    assert c > c0 + 0.1, f"delta predictor {c:.3f} should beat random {c0:.3f}"
