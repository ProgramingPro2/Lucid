"""Latent-space diagnostics for generated concept trajectories."""
from __future__ import annotations

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def latent_diagnostics(
    generated_vectors: np.ndarray, bank_vectors: np.ndarray
) -> dict:
    """Manifold proximity and autoregressive collapse metrics.

    ``mean_max_cos_to_bank``: how close each generated vector is to the nearest
    real sentence embedding (1.0 = on manifold). ``mean_consecutive_cos``:
    similarity between successive generated vectors (1.0 = fixed-point collapse).
    """
    if len(generated_vectors) == 0:
        return {
            "mean_max_cos_to_bank": float("nan"),
            "mean_consecutive_cos": float("nan"),
            "per_step_max_cos_to_bank": [],
            "per_step_consecutive_cos": [],
        }
    gen = _normalize_rows(generated_vectors)
    bank = _normalize_rows(bank_vectors)
    max_cos = (gen @ bank.T).max(axis=1)
    consecutive: list[float] = []
    for i in range(1, len(gen)):
        consecutive.append(float(np.dot(gen[i], gen[i - 1])))
    return {
        "mean_max_cos_to_bank": round(float(max_cos.mean()), 4),
        "mean_consecutive_cos": round(
            float(np.mean(consecutive)) if consecutive else float("nan"), 4
        ),
        "per_step_max_cos_to_bank": [round(float(x), 4) for x in max_cos],
        "per_step_consecutive_cos": [round(float(x), 4) for x in consecutive],
    }
