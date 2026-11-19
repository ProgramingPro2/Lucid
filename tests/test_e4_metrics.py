"""Tests for E4 endpoints and the necessity estimator."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.e4.metrics import (
    entity_recall,
    loo_centroid_cosine,
    normalise_entity,
    retrieval_win_rate,
)
from experiments.e4.necessity import (
    absolute_gate_is_valid,
    combine_endpoints,
    evaluate_necessity,
    fidelity_gate,
)
from experiments.e4.synthetic import simulate
from experiments.protocol.stats import cluster_bootstrap_ci, mean_ci


def test_entity_recall_exact_and_case_insensitive():
    gold = ["BRCA1", "p53"]
    assert entity_recall("We assayed BRCA1 and P53 in the sample.", gold) == 1.0
    assert entity_recall("We assayed brca1 only.", gold) == 0.5


def test_entity_recall_does_not_match_substrings():
    assert entity_recall("photosynthesis was measured", ["photo"]) == 0.0


def test_entity_recall_multword_and_plural():
    assert entity_recall("the T cell receptor was cloned", ["T cell receptor"]) == 1.0
    assert entity_recall("many neutrophils", ["neutrophil"], allow_plural=True) == 1.0
    assert entity_recall("many neutrophils", ["neutrophil"], allow_plural=False) == 0.0


def test_entity_recall_rejects_empty_gold():
    with pytest.raises(ValueError):
        entity_recall("text", [])


def test_entity_recall_empty_text_is_zero():
    assert entity_recall("", ["p53"]) == 0.0


def test_normalise_entity():
    assert normalise_entity("  “BRCA1.”  ") == "brca1"


def test_loo_centroid_analytic():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    cos = loo_centroid_cosine(x, [0, 0])
    assert cos[0] == pytest.approx(0.0, abs=1e-9)
    assert cos[1] == pytest.approx(0.0, abs=1e-9)


def test_loo_centroid_identical_rows():
    x = np.array([[1.0, 0.0], [1.0, 0.0]])
    cos = loo_centroid_cosine(x, [0, 0])
    assert cos == pytest.approx(np.array([1.0, 1.0]))


def test_loo_centroid_singleton_is_nan():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    cos = loo_centroid_cosine(x, [0, 1])
    assert np.isnan(cos).all()


def test_retrieval_win_rate_planted_vs_random():
    rng = np.random.default_rng(0)
    docs = np.repeat(np.arange(12), 8)
    # Planted: each doc is a tight cluster far from the others.
    centres = rng.normal(size=(12, 16))
    x = centres[docs] + rng.normal(scale=0.01, size=(len(docs), 16))
    assert retrieval_win_rate(x, docs) > 0.95

    random = rng.normal(size=(len(docs), 16))
    wr = retrieval_win_rate(random, docs)
    assert 0.3 < wr < 0.7


def test_cluster_bootstrap_is_wider_than_iid():
    vals, clusters = [], []
    for c, centre in enumerate([0.0, 10.0, 20.0]):
        vals.extend([centre + 0.01 * i for i in range(20)])
        clusters.extend([c] * 20)
    iid = mean_ci(vals)
    clu = cluster_bootstrap_ci(vals, clusters, n_boot=2000, seed=0)
    assert clu.se > iid.se


def test_necessity_recovers_planted_effect():
    t, docs, _ = simulate("true", effect=0.12, seed=1)
    s, _, _ = simulate("shuffled", effect=0.12, seed=2)
    res = evaluate_necessity("entity_recall", t, s, docs, mpid=0.03, n_boot=4000)
    assert abs(res.difference - 0.12) < 0.03
    assert res.verdict == "PASS"


def test_necessity_with_zero_effect_never_passes():
    t, docs, _ = simulate("true", effect=0.0, seed=3)
    s, _, _ = simulate("shuffled", effect=0.0, seed=4)
    res = evaluate_necessity("entity_recall", t, s, docs, mpid=0.03, n_boot=4000)
    assert res.verdict in {"FAIL", "INCONCLUSIVE"}


def test_absolute_gate_guard_catches_f38():
    assert absolute_gate_is_valid(0.40, 0.445) is False
    assert absolute_gate_is_valid(0.30, 0.20) is True


def test_fidelity_gate():
    assert fidelity_gate(0.45) is True
    assert fidelity_gate(0.39) is False
    assert fidelity_gate(float("nan")) is False


def test_combine_endpoints_requires_all():
    t, docs, _ = simulate("true", effect=0.12, seed=5)
    s, _, _ = simulate("shuffled", effect=0.12, seed=6)
    good = evaluate_necessity("recall", t, s, docs, mpid=0.03, n_boot=2000)
    bad = evaluate_necessity("cos", t, t, docs, mpid=0.03, n_boot=2000)
    assert combine_endpoints([good, good]) == "PASS"
    assert combine_endpoints([good, bad]) in {"INCONCLUSIVE", "FAIL"}
