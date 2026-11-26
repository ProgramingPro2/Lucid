"""Dose of a true latent along the short arc from a shuffled draft.

``slerp(a, b, t)`` is ordinary spherical interpolation: ``t=0`` returns ``a``
and ``t=1`` returns ``b``. The rehearsal called ``slerp(z_true, z_shuf, 0.5)``.
The midpoint does not depend on argument order, so that one call is a real
50% dose. The same function at the endpoints is backwards for a dose of the
true latent: ``t=0`` would be the true draft. G0 goes through ``dose_latent``,
which takes the dose in ``[0, 1]`` and returns the shuffled draft at 0 and
the true draft at 1.
"""

from __future__ import annotations

import math

import numpy as np

_PARALLEL = 1e-8
_ANTIPODAL = 1e-6


def unit(x: np.ndarray) -> np.ndarray:
    """L2-normalise a vector. A zero or non-finite vector is an error."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        raise ValueError("non-finite vector")
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Short-arc interpolation. ``t=0`` returns ``a``, ``t=1`` returns ``b``."""
    if not np.isfinite(t) or t < 0.0 or t > 1.0:
        raise ValueError(f"slerp t must lie in [0, 1], got {t}")
    a_u = unit(a)
    b_u = unit(b)
    if t == 0.0:
        return a_u
    if t == 1.0:
        return b_u
    dot = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < _PARALLEL:
        return unit((1.0 - t) * a_u + t * b_u)
    s = math.sin(omega)
    if s < _ANTIPODAL:
        raise ValueError("antipodal vectors have no unique slerp")
    out = (math.sin((1.0 - t) * omega) / s) * a_u + (math.sin(t * omega) / s) * b_u
    return unit(out)


def dose_latent(shuffled: np.ndarray, true: np.ndarray, dose: float) -> np.ndarray:
    """Latent at ``dose`` of the true draft.

    ``dose=0`` is the shuffled draft. ``dose=1`` is the true draft. Values
    in between lie on the short arc, so cosine to the true draft is
    non-decreasing in ``dose``.
    """
    return slerp(shuffled, true, dose)
