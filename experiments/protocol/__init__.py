"""Programme-level helpers for the 2026-11 protocol.

Nothing here trains or evaluates anything. These are the reproducibility
primitives that every manifest in ``docs/EXPERIMENTS.md`` depends on.
"""

from .seeds import STREAMS, stream, token_stream_sha256

__all__ = ["STREAMS", "stream", "token_stream_sha256"]
