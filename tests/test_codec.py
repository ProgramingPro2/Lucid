from __future__ import annotations

import numpy as np

from conceptpoc.codec.hashing_codec import HashingCodec


def test_encode_shape_and_norm(codec: HashingCodec, sentences):
    vecs = codec.encode(sentences)
    assert vecs.shape == (len(sentences), 64)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_is_deterministic(codec: HashingCodec, sentences):
    assert np.allclose(codec.encode(sentences), codec.encode(sentences))


def test_decode_round_trip_returns_known_sentences(fitted_codec, sentences):
    vecs = fitted_codec.encode(sentences)
    decoded = fitted_codec.decode(vecs)
    # Each sentence's own vector should retrieve itself from the bank.
    assert decoded == sentences


def test_empty_encode(codec: HashingCodec):
    assert codec.encode([]).shape == (0, 64)
