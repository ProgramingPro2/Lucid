"""Spawn-keyed RNG streams and token-stream hashing.

The protocol requires that no RNG stream be derived by ``seed + k``. Adding an
arm must not shift another arm's data order, and the same ``(seed, stream)``
pair must produce the same bytes on every run and every machine.

Derivation is SHA-256 over ``f"{seed}:{name}"``, truncated to 64 bits. This is
deliberately not a numpy/torch convention: it must be stable across library
versions, and it must be checkable from a shell.

See ``docs/EXPERIMENTS.md`` §2.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch

# The complete set of streams. Adding a stream is a protocol amendment because
# it changes what "the same seed" means for a comparison.
STREAMS: tuple[str, ...] = (
    "data",       # document order, window draw, packing order
    "lora_init",  # LoRA A/B matrices
    "block_init", # inserted block weights (output projection zeroed *after*)
    "head_init",  # A'/B' parameter-matched head
    "dropout",    # dropout masks
    "decode",     # eval sampling; null arms use seed + 10000 and disjoint items
)

_MASK = (1 << 64) - 1


def spawn_key(seed: int, name: str) -> int:
    """Return the 64-bit integer seed for one ``(seed, stream)`` pair."""
    if name not in STREAMS:
        raise ValueError(f"unknown stream {name!r}; expected one of {STREAMS}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & _MASK


def stream(seed: int, name: str) -> torch.Generator:
    """Return an independent ``torch.Generator`` for one ``(seed, stream)`` pair.

    Two calls with the same arguments produce generators in the same state,
    which is what makes a run reproducible from its ID alone.
    """
    gen = torch.Generator()
    gen.manual_seed(spawn_key(seed, name))
    return gen


def token_stream_sha256(tokens: Iterable[int], *, dtype: str = "<u4") -> str:
    """Hash a packed token stream.

    ``tokens`` must be the exact integer sequence fed to the model, in order,
    with no padding. The hash is asserted identical across arms; a mismatch
    voids the comparison (docs/PROTOCOL.md §10).
    """
    import numpy as np

    arr = np.asarray(list(tokens), dtype=dtype)
    return hashlib.sha256(arr.tobytes()).hexdigest()
