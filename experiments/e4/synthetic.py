"""Synthetic simulators with known ground truth, used only to test estimators.

Never cite these numbers. Their job is to prove that an estimator recovers an
effect it was told about, and returns nothing when it was told nothing.
"""

from __future__ import annotations

import numpy as np


def simulate(
    condition: str,
    *,
    n_items: int = 128,
    n_docs: int = 128,
    effect: float = 0.0,
    sigma_item: float = 0.09,
    sigma_run: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scores, clusters, planted_effect) for one condition.

    ``condition`` is "true", "shuffled" or "none"; the planted effect shifts the
    mean by +effect, 0, or -effect/2 respectively.
    """
    if condition not in {"true", "shuffled", "none"}:
        raise ValueError(condition)
    rng = np.random.default_rng(seed)
    doc = np.repeat(np.arange(n_docs), int(np.ceil(n_items / n_docs)))[:n_items]
    shift = {"true": effect, "shuffled": 0.0, "none": -effect / 2.0}[condition]

    item = rng.normal(0.0, sigma_item, size=n_items)
    run = rng.normal(0.0, sigma_run, size=n_docs)[doc]
    scores = 0.5 + shift + item + run
    return scores, doc, np.array([shift])
