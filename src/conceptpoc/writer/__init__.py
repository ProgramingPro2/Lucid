"""Hybrid writer: GPT-2-scale fluent autoregressive text generator.

Phase 1 pretrains it as a next-token LM (fluency); Phase 2 conditions it on the
JEPA reasoner's concept vector so it can be aimed at the intended next sentence.
"""
from .model import FluWriter

__all__ = ["FluWriter"]
