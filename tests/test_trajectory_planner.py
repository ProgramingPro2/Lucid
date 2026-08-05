"""Tests for the trajectory planner + dataset."""
import numpy as np
import torch

from src.conceptpoc.trajectory.planner import TrajectoryPlanner


def test_planner_shapes():
    p = TrajectoryPlanner(dim=128, C=8, K=5, n_layers=2, n_heads=4, latent_iters=3)
    ctx = torch.randn(4, 8, 128)
    traj = p(ctx)
    assert traj.shape == (4, 5, 128), traj.shape
    assert torch.isfinite(traj).all()


def test_planner_context_sensitivity():
    torch.manual_seed(0)
    p = TrajectoryPlanner(dim=64, C=8, K=5, n_layers=2, n_heads=4)
    a = torch.randn(1, 8, 64)
    b = torch.randn(1, 8, 64)
    ta, tb = p(a), p(b)
    assert torch.isfinite(ta).all() and torch.isfinite(tb).all()
    # Zero-init head means raw trajectory is 0 pre-training; the *latent* start
    # state must still be context-sensitive (that's what feeds the plan).
    ha, hb = p.encode_context(a), p.encode_context(b)
    assert torch.dist(ha, hb) > 1e-3, "different contexts must change the latent start"
    # And each of the two contexts must produce a usable trace through blocks.
    assert torch.isfinite(p.norm_latent(ha)).all()


def test_planner_plan_matches_forward():
    torch.manual_seed(1)
    p = TrajectoryPlanner(dim=64, C=8, K=5, n_layers=2, n_heads=4)
    ctx = torch.randn(2, 8, 64)
    fwd = p(ctx)
    plan = p.plan(ctx)
    assert torch.allclose(fwd, plan, atol=1e-5)
    assert plan.shape == (2, 5, 64)


def test_planner_grad_flow():
    p = TrajectoryPlanner(dim=32, C=8, K=5, n_layers=1, n_heads=4)
    ctx = torch.randn(2, 8, 32)
    target = torch.randn(2, 5, 32)
    out = p(ctx)
    loss = torch.nn.functional.cosine_similarity(out, target, dim=-1).mean()
    loss.backward()
    grads = [p_.grad for p_ in p.parameters() if p_.grad is not None]
    assert len(grads) > 0 and all(g is not None for g in grads)
