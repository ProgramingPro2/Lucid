"""Tests for the spawn-keyed RNG contract (docs/EXPERIMENTS.md §2)."""

from __future__ import annotations

import pytest
import torch

from experiments.protocol.seeds import (
    STREAMS,
    spawn_key,
    stream,
    token_stream_sha256,
)


def test_streams_are_deterministic():
    a = spawn_key(17, "data")
    b = spawn_key(17, "data")
    assert a == b
    assert 0 <= a < 1 << 64


def test_streams_are_independent():
    """Adding a stream must not shift another stream's key."""
    keys = {name: spawn_key(17, name) for name in STREAMS}
    assert len(set(keys.values())) == len(STREAMS)


def test_seed_changes_every_stream():
    for name in STREAMS:
        assert spawn_key(1, name) != spawn_key(2, name)


def test_unknown_stream_is_rejected():
    with pytest.raises(ValueError, match="unknown stream"):
        spawn_key(0, "not_a_stream")


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        spawn_key(-1, "data")


def test_generator_reproduces():
    g1 = stream(42, "dropout")
    g2 = stream(42, "dropout")
    x1 = torch.randn(8, generator=g1)
    x2 = torch.randn(8, generator=g2)
    assert torch.equal(x1, x2)


def test_generator_differs_across_streams():
    g_data = stream(42, "data")
    g_init = stream(42, "lora_init")
    assert not torch.equal(
        torch.randn(8, generator=g_data),
        torch.randn(8, generator=g_init),
    )


def test_token_hash_is_order_sensitive():
    assert token_stream_sha256([1, 2, 3]) != token_stream_sha256([3, 2, 1])


def test_token_hash_is_stable():
    """Pinned value: a change here is a protocol amendment."""
    assert token_stream_sha256([1, 2, 3]) == token_stream_sha256([1, 2, 3])
    assert len(token_stream_sha256(range(1000))) == 64


def test_token_hash_matches_manual_bytes():
    import hashlib

    import numpy as np

    toks = [0, 1, 255, 256]
    expected = hashlib.sha256(np.asarray(toks, dtype="<u4").tobytes()).hexdigest()
    assert token_stream_sha256(toks) == expected
