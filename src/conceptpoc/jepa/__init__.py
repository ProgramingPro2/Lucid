"""Joint, self-supervised codec: train your own encoder + decoder + predictor.

Unlike the frozen codec, this learns the concept space from scratch on the
corpus. The whole design exists to defeat representation collapse:

- a reconstruction anchor (the decoder must regenerate each sentence from its
  vector, so a constant collapsed vector is impossible),
- an EMA stop-grad prediction target (the encoder cannot make targets trivial),
- VICReg-style variance/covariance regularization (a direct anti-collapse push).

See docs/ARCHITECTURE.md and docs/LIMITATIONS.md.
"""
from __future__ import annotations

from .tokenizer import Tokenizer

__all__ = ["Tokenizer"]
