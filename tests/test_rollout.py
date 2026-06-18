from __future__ import annotations

import numpy as np

from conceptpoc.config import PredictorConfig
from conceptpoc.generate.rollout import generate_rollout, rollout_to_text
from conceptpoc.predictor.model import ConceptPredictor


def test_rollout_shapes_and_decode(fitted_codec, sentences):
    model = ConceptPredictor(
        concept_dim=fitted_codec.dim,
        cfg=PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16),
    )
    seed = fitted_codec.encode(sentences[:2])
    roll = generate_rollout(model, seed, steps=4, temperature=0.0)
    assert roll.generated_vectors.shape == (4, fitted_codec.dim)
    assert roll.all_vectors.shape == (6, fitted_codec.dim)

    text = rollout_to_text(roll, fitted_codec)
    assert len(text) == 4
    assert all(t in sentences for t in text)


def test_temperature_adds_diversity(fitted_codec, sentences):
    model = ConceptPredictor(
        concept_dim=fitted_codec.dim,
        cfg=PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16),
    )
    seed = fitted_codec.encode(sentences[:1])
    rng = np.random.default_rng(0)
    a = generate_rollout(model, seed, steps=3, temperature=0.5, rng=rng)
    b = generate_rollout(model, seed, steps=3, temperature=0.5, rng=rng)
    assert not np.allclose(a.generated_vectors, b.generated_vectors)
