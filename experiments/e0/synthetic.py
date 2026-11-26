"""CPU stand-in for the G0 generator.

The generator echoes the prompt and ignores the latent. That is the
decorative channel FINDINGS F42 describes, on purpose: the dry run is
plumbing, and a decorative channel must be able to fail the dose gate.
A single seed token keeps two decode seeds from being the identical string
(N2) without giving the latent anything to do.

Embeddings are a bag of hashed words, so a text prefix of the target lands
near the target and a different document does not. The readout control can
therefore pass while the latent channel fails.
"""

from __future__ import annotations

import hashlib

import numpy as np

from experiments.e0.dose import unit


def _word_vec(word: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim)


def bow_embed(text: str, dim: int = 32) -> np.ndarray:
    words = text.split()
    if not words:
        raise ValueError("empty text")
    acc = np.zeros(dim, dtype=np.float64)
    for word in words:
        acc += _word_vec(word, dim)
    return unit(acc)


def echoing_generate(prompt: str, latent: np.ndarray | None, seed: int) -> str:
    """Return the prompt plus a seed token. ``latent`` is ignored."""
    del latent
    return f"{prompt} seed{int(seed)}"
