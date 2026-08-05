"""Tests for the multi-space representation hypothesis.

Finding (toy experiment): a single embedding space is EITHER globally spread
(anti-collapse, high variance) OR locally structured (high adjacent cosine),
but a flat combination of two specialist spaces (concat / weighted-sum /
whitened concat) DILUTES the correlation to ~0. The correlation lives in a
few dims and gets averaged away by the spread dims.

Research implication: the way to "combine multiple spaces" is NOT a flat
merge -- it must be a routing / multi-channel design where the predictor
reads the correlation subspace and the anti-collapse term acts on the spread
subspace. These tests lock in the metric methodology and the negative result
so future experiments are measured consistently.
"""
import numpy as np
import torch

from src.conceptpoc.jepa.collapse import (
    batch_variance,
    effective_rank,
    adjacent_cosine,
)


def make_specialist_spaces(T=400, D=128, D2=128, seed=0):
    """Return (corr, spread): two spaces over the same T items.
    corr  : smooth 1-D trajectory -> adjacent correlated, LOW variance
    spread: iid random            -> HIGH variance, no adjacency
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    corr = torch.zeros(T, D)
    corr[:, 0] = torch.linspace(-1, 1, T)
    corr[:, 1] = torch.sin(torch.linspace(0, 6, T))
    corr += 0.02 * torch.randn(T, D)
    spread = torch.from_numpy(rng.randn(T, D2).astype(np.float32))
    return corr, spread


class TestSpecialistSpaces:
    def test_corr_space_is_low_variance_high_correlation(self):
        corr, _ = make_specialist_spaces()
        assert batch_variance(corr) < 0.1
        assert adjacent_cosine(corr.numpy(), torch.roll(corr, 1, 0).numpy()) > 0.7

    def test_spread_space_is_high_variance_low_correlation(self):
        _, spread = make_specialist_spaces()
        assert batch_variance(spread) > 0.5
        assert abs(adjacent_cosine(spread.numpy(), torch.roll(spread, 1, 0).numpy())) < 0.2


class TestDilutionTrap:
    def test_naive_concat_dilutes_correlation(self):
        """The core negative result: flat combination destroys adjacency."""
        corr, spread = make_specialist_spaces()
        c_corr = adjacent_cosine(corr.numpy(), torch.roll(corr, 1, 0).numpy())
        concat = torch.cat([corr, spread], dim=-1)
        c_concat = adjacent_cosine(concat.numpy(), torch.roll(concat, 1, 0).numpy())
        assert c_concat < c_corr * 0.3, (
            f"concat diluted correlation: {c_corr:.3f} -> {c_concat:.3f}"
        )

    def test_weighted_sum_dilutes_correlation(self):
        corr, spread = make_specialist_spaces()
        c_corr = adjacent_cosine(corr.numpy(), torch.roll(corr, 1, 0).numpy())
        wsum = 0.5 * corr + 0.5 * spread[:, :128]
        c_w = adjacent_cosine(wsum.numpy(), torch.roll(wsum, 1, 0).numpy())
        assert c_w < c_corr * 0.3

    def test_whitened_concat_dilutes_too(self):
        corr, spread = make_specialist_spaces()
        T, D2 = spread.shape
        c_corr = adjacent_cosine(corr.numpy(), torch.roll(corr, 1, 0).numpy())
        zc = spread - spread.mean(0)
        cov = zc.T @ zc / (T - 1)
        e, V = torch.linalg.eigh(cov + 1e-6 * torch.eye(D2))
        w = zc @ V @ torch.diag(1.0 / torch.sqrt(e + 1e-6)) @ V.T
        orth = torch.cat([corr / (corr.norm(dim=0, keepdim=True) + 1e-9), w], dim=-1)
        c_o = adjacent_cosine(orth.numpy(), torch.roll(orth, 1, 0).numpy())
        assert c_o < c_corr * 0.3, (
            f"whitened concat still diluted: {c_corr:.3f} -> {c_o:.3f}"
        )

    def test_spread_adds_variance_to_combined(self):
        """Even though correlation dies, the combined vector DOES gain the
        spread space's variance -- so multi-space helps anti-collapse, just
        not topic; the routing design is what's missing."""
        corr, spread = make_specialist_spaces()
        concat = torch.cat([corr, spread], dim=-1)
        assert batch_variance(concat) > batch_variance(corr) * 5.0
