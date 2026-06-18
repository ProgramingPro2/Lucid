"""Mixture of concept predictors with a learned gating router.

This is the prototype of the user's "experts that form bonds" idea: several
ConceptPredictor experts and a small router that, from the current context,
produces a soft weighting over experts and blends their next-vector predictions.
Training the router end-to-end (frozen or unfrozen experts) lets it learn which
expert to trust in which context.
"""
from __future__ import annotations

from .router import MixtureOfPredictors, GatingRouter

__all__ = ["MixtureOfPredictors", "GatingRouter"]
