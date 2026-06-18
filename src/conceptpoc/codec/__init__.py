"""Concept codecs: map text <-> concept vectors.

The codec is the frozen part of the system. ``encode`` turns sentences into
vectors (the "concept space"); ``decode`` turns vectors back into text. Three
implementations are provided:

- ``SentenceTransformerCodec``: encodes with a frozen sentence-transformer and
  decodes by snapping to the nearest sentence in a bank (retrieval decoding).
  Runs anywhere.
- ``SonarCodec``: encodes and *generatively* decodes via Meta's SONAR. Optional
  install; closest to the Large Concept Model setup.
- ``HashingCodec``: deterministic, dependency-free, for tests.
"""
from __future__ import annotations

from .base import ConceptCodec
from .factory import build_codec

__all__ = ["ConceptCodec", "build_codec"]
