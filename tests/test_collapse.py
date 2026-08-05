"""Tests for the collapse-vs-predictability metrics and regularizers.

The central hypothesis of the current Lucid research phase:
  anti-collapse (global spread) and forward predictability (adjacent-sentence
  correlation) are in tension WHEN the regularizer pushes globally. A
  context-relative variant should decouple them.

These tests verify the diagnostics behave correctly and that the *global*
VICReg regularizer demonstrably flattens correlation, setting up the
motivation for a context-relative design.
"""
import numpy as np
import torch

from src.conceptpoc.jepa.collapse import (
    batch_variance,
    effective_rank,
    isotropy,
    adjacent_cosine,
    vicreg_regularizer,
    sigreg_isotropic,
)


def _random(sz, seed=0, std=1.0):
    torch.manual_seed(seed)
    return torch.randn(*sz) * std


class TestMetrics:
    def test_batch_variance_collapsed_vs_spread(self):
        collapsed = torch.zeros(64, 512)
        spread = _random((64, 512), std=1.0)
        assert batch_variance(collapsed) < 1e-6
        assert batch_variance(spread) > 0.5
        assert batch_variance(spread) > batch_variance(collapsed)

    def test_effective_rank_collapsed_is_one(self):
        collapsed = torch.zeros(64, 512)
        assert effective_rank(collapsed) < 1.5, "collapsed -> rank ~1"
        # effective rank is bounded by min(batch, dim); 200x512 -> high
        spread = torch.randn(200, 512)
        assert effective_rank(spread) > 100, "spread -> high effective rank"

    def test_isotropy_spread_is_lowrank(self):
        # A spread-but-degenerate (rank-5) embedding should be anisotropic.
        low = torch.zeros(64, 512)
        low[:, :5] = torch.randn(64, 5)
        iso = isotropy(low)
        assert iso < 0.5, f"degenerate dims should be anisotropic, got {iso}"

    def test_adjacent_cosine_works(self):
        # Identical vectors -> cosine ~1; orthogonal -> ~0
        v0 = np.random.RandomState(0).randn(8, 512)
        assert adjacent_cosine(v0, v0) > 0.99
        rng = np.random.RandomState(1)
        a = rng.randn(8, 512)
        b = rng.randn(8, 512)
        # orthogonal-ish random -> near 0
        c = adjacent_cosine(a, b)
        assert abs(c) < 0.2, f"random vectors should be near-orthogonal, got {c}"


class TestRegularizers:
    def test_vicreg_penalizes_collapse(self):
        collapsed = torch.zeros(64, 512)
        spread = _random((64, 512), std=1.0)
        l_collapsed = vicreg_regularizer(collapsed)
        l_spread = vicreg_regularizer(spread)
        assert l_collapsed > l_spread, "VICReg loss must be HIGHER on collapsed batch"

    def test_sigreg_penalizes_collapse(self):
        collapsed = torch.zeros(64, 512)
        spread = _random((64, 512), std=1.0)
        l_collapsed = sigreg_isotropic(collapsed)
        l_spread = sigreg_isotropic(spread)
        assert l_collapsed > l_spread, "SIGReg loss must be HIGHER on collapsed batch"

    def test_global_spread_lowers_adjacent_correlation(self):
        """THE tension test.

        A smooth trajectory of vectors has high adjacent cosine (correlated).
        Applying ANY isotropic/global spread (scaling or whitening to equalize
        variance) pushes adjacent vectors apart and LOWERS their cosine --- the
        fundamental tradeoff. This is why a smart regularizer must spread
        LOCALLY (relative to context) rather than globally.
        """
        torch.manual_seed(0)
        T = 200
        z = torch.zeros(T, 512)
        z[:, 0] = torch.linspace(-1, 1, T)
        z[:, 1] = torch.sin(torch.linspace(0, 6, T))
        z += 0.02 * torch.randn(T, 512)
        cos_before = adjacent_cosine(z.numpy(), torch.roll(z, 1, 0).numpy())
        assert cos_before > 0.5, f"trajectory should be correlated, got {cos_before:.3f}"

        # (a) global magnitude scaling
        zc = z - z.mean(0)
        factor = torch.sqrt(2.0 / (zc.var(0) + 1e-9)).clamp(max=3.0)
        z_scale = z * factor
        cos_scale = adjacent_cosine(z_scale.numpy(), torch.roll(z_scale, 1, 0).numpy())

        # (b) whitening (isotropic shape normalization)
        cov = zc.T @ zc / (T - 1)
        e, V = torch.linalg.eigh(cov + 1e-6 * torch.eye(512))
        z_w = zc @ V @ torch.diag(1.0 / torch.sqrt(e + 1e-6)) @ V.T
        cos_w = adjacent_cosine(z_w.numpy(), torch.roll(z_w, 1, 0).numpy())

        # Honest finding: BOTH flatten correlation.
        assert cos_scale < cos_before, f"scaling must lower correlation: {cos_before:.3f}->{cos_scale:.3f}"
        assert cos_w < cos_before, f"whitening must lower correlation: {cos_before:.3f}->{cos_w:.3f}"

    def test_adjacent_vs_nonadjacent_separation(self):
        """A useful property: adjacent vectors should correlate MORE than random
        pairs (i.e. the space has *usable* forward structure to predict)."""
        torch.manual_seed(2)
        T, D = 400, 128
        z = torch.zeros(T, D)
        z[:, 0] = torch.linspace(-1, 1, T)
        z[:, 1] = torch.cos(torch.linspace(0, 5, T))
        z += 0.03 * torch.randn(T, D)
        adj = adjacent_cosine(z.numpy(), torch.roll(z, 1, 0).numpy())
        # shuffled pairs -> no structure
        rng = np.random.RandomState(0)
        idx = rng.permutation(T)
        shuf = adjacent_cosine(z.numpy(), z.numpy()[idx])
        assert adj > shuf, f"adjacent ({adj:.3f}) should exceed shuffled ({shuf:.3f})"
