"""E0: assay validation (G0). See docs/PROTOCOL.md §4.

The decision rule is :mod:`experiments.e0.criteria`. Thresholds are not
copied into the report by hand.
"""

from experiments.e0.criteria import (
    DOSE_100_VS_0_LB_MIN,
    DOSE_DROP_TOLERANCE,
    NULL_HALF_WIDTH_MAX,
    TEXT_PREFIX_LB_MIN,
    evaluate,
    thresholds,
)

__all__ = [
    "DOSE_100_VS_0_LB_MIN",
    "DOSE_DROP_TOLERANCE",
    "NULL_HALF_WIDTH_MAX",
    "TEXT_PREFIX_LB_MIN",
    "evaluate",
    "thresholds",
]
