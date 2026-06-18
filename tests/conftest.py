"""Shared fixtures. Uses the deterministic hashing codec so tests need no
downloads and no network."""
from __future__ import annotations

import numpy as np
import pytest

from conceptpoc.codec.hashing_codec import HashingCodec


@pytest.fixture
def codec() -> HashingCodec:
    return HashingCodec(dim=64)


@pytest.fixture
def sentences() -> list[str]:
    return [
        "the river moved slowly across the valley",
        "birds rose from the reeds toward the hills",
        "a small town sat at the edge of the water",
        "markets opened early in the cool morning",
        "the old library stood quietly on the hill",
        "knowledge waited patiently among the shelves",
        "learning begins with attention and practice",
        "invention is the hardest and final step",
    ]


@pytest.fixture
def fitted_codec(codec: HashingCodec, sentences: list[str]) -> HashingCodec:
    codec.fit_decoder(sentences)
    return codec
