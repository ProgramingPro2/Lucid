"""Concept Predictor POC.

A "predict ideas, not tokens" language model: text is encoded into per-sentence
concept vectors by a frozen codec, a small transformer predicts the next concept
vector, and predicted vectors are decoded back to text. A self-improvement loop
driven by an LLM-as-judge plus human ratings optimizes the predictor.

See docs/ARCHITECTURE.md for the full design and docs/LIMITATIONS.md for an
honest account of what this does and does not achieve.
"""

__version__ = "0.1.0"
