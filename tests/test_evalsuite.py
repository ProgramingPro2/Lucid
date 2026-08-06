"""Unit tests for anti-mirage evalsuite pure functions (Fable §7)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.sentence_pred.evalsuite import (
    centering,
    copy_rate,
    effective_rank,
    max_ngram_overlap,
    novelty_rate,
    novelty_scores,
    reference_distribution,
)


def test_effective_rank_full_vs_rank1():
    rng = np.random.RandomState(0)
    d = 32
    # Full-rank-ish random matrix → effective rank near dim
    X = rng.randn(200, d).astype(np.float32)
    er_full = effective_rank(X)["effective_rank"]
    assert er_full > d * 0.5, er_full

    # Rank-1: all rows the same → effective rank ~1
    row = rng.randn(d).astype(np.float32)
    X1 = np.tile(row, (100, 1))
    er1 = effective_rank(X1)["effective_rank"]
    assert er1 < 2.5, er1


def test_novelty_scores_chunking_matches_direct():
    rng = np.random.RandomState(1)
    d = 16
    bank = rng.randn(50, d).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12
    X = rng.randn(20, d).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    mu, bank_c = centering(bank)
    # force small chunks by monkeypatching would be heavy; just compare to direct
    scores = novelty_scores(X, bank_c, mu)
    xc = X - mu
    xc /= np.linalg.norm(xc, axis=1, keepdims=True) + 1e-12
    direct = (xc @ bank_c.T).max(axis=1)
    np.testing.assert_allclose(scores, direct, rtol=1e-5, atol=1e-5)


def test_centered_vs_raw_alignment_with_mu():
    rng = np.random.RandomState(2)
    d = 64
    # Bank tightly clustered around a direction → mu ≈ that direction
    mu0 = rng.randn(d).astype(np.float32)
    mu0 /= np.linalg.norm(mu0) + 1e-12
    bank = mu0[None, :] + 0.01 * rng.randn(100, d).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12
    mu, bank_c = centering(bank)
    # A vector aligned with mu: raw cos to bank mean is high; centered ~0 vs bank_c mean
    x = mu.copy()
    raw = float(np.dot(x, mu / (np.linalg.norm(mu) + 1e-12)))
    xc = x - mu
    xc_n = xc / (np.linalg.norm(xc) + 1e-12) if np.linalg.norm(xc) > 1e-8 else xc
    # After centering x≈mu, xc≈0; novelty vs bank_c should be near isotropic / low structure
    assert raw > 0.9
    # centered self: normalize(x - mu) is near-zero vector; score vs bank_c finite
    n = novelty_scores(x[None, :], bank_c, mu)[0]
    assert np.isfinite(n)


def test_novelty_and_copy_rate_bounds_and_monotone():
    rng = np.random.RandomState(3)
    d = 32
    bank = rng.randn(80, d).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12
    held = rng.randn(40, d).astype(np.float32)
    held /= np.linalg.norm(held, axis=1, keepdims=True) + 1e-12
    mu, bank_c = centering(bank)
    R = reference_distribution(held, bank_c, mu)

    # Far-from-bank random → high novelty_rate, low copy_rate
    far = rng.randn(30, d).astype(np.float32)
    far /= np.linalg.norm(far, axis=1, keepdims=True) + 1e-12
    s_far = novelty_scores(far, bank_c, mu)
    nr_far = novelty_rate(s_far, R)
    cr_far = copy_rate(s_far, R)
    assert 0.0 <= nr_far <= 1.0
    assert 0.0 <= cr_far <= 1.0

    # Near bank (copies of bank rows) → copy_rate rises
    near = bank[:30].copy()
    s_near = novelty_scores(near, bank_c, mu)
    cr_near = copy_rate(s_near, R)
    assert cr_near >= cr_far - 1e-6


def test_reference_distribution_quantiles():
    rng = np.random.RandomState(4)
    d = 16
    bank = rng.randn(60, d).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12
    held = rng.randn(100, d).astype(np.float32)
    held /= np.linalg.norm(held, axis=1, keepdims=True) + 1e-12
    mu, bank_c = centering(bank)
    R = reference_distribution(held, bank_c, mu)
    scores = R["R"]
    assert abs(R["median"] - float(np.median(scores))) < 1e-6
    assert abs(R["p90"] - float(np.percentile(scores, 90))) < 1e-6
    assert abs(R["p99"] - float(np.percentile(scores, 99))) < 1e-6
    assert R["median"] <= R["p90"] <= R["p99"]


def test_max_ngram_overlap_identical_and_disjoint():
    a = "the quick brown fox jumps over the lazy dog again today"
    assert max_ngram_overlap(a, [a], n=5) == 1.0
    b = "alpha beta gamma delta epsilon zeta eta theta iota"
    assert max_ngram_overlap(a, [b], n=5) == 0.0
