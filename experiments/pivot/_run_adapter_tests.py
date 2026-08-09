#!/usr/bin/env python
"""Runner for pivot adapter tests (avoids pytest CLI quirks)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from tests.test_pivot_adapter import (  # noqa: E402
    test_adapter_forward_shape,
    test_collate_prefix_prev_target_within_ctx,
    test_fiber_corrupt_vmf_kappa700,
)


def main() -> None:
    test_adapter_forward_shape()
    print("PASS test_adapter_forward_shape")
    test_fiber_corrupt_vmf_kappa700()
    print("PASS test_fiber_corrupt_vmf_kappa700")
    test_collate_prefix_prev_target_within_ctx()
    print("PASS test_collate_prefix_prev_target_within_ctx")
    print("ALL PASS")


if __name__ == "__main__":
    main()
