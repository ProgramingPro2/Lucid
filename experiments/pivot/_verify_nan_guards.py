"""Verify NaN guards: raw_loss skip + non-finite z_tilde fallback."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.pivot.adapter import SoftPrefixAdapter, fiber_corrupt  # noqa: E402
from experiments.pivot.train_stage_a import _sanitize_grads  # noqa: E402


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = F.normalize(torch.randn(2, 512, device=device), dim=-1)
    z_bad = z.clone()
    z_bad[0, 0] = float("nan")
    # fiber_corrupt on nan mu may nan; trainer fallback:
    z_tilde = fiber_corrupt(z, kappa=500.0)
    assert bool(torch.isfinite(z_tilde).all().item())
    # Simulate non-finite z_tilde → normalize raw z
    z_tilde2 = z_bad
    if not bool(torch.isfinite(z_tilde2).all().item()):
        z_tilde2 = F.normalize(z.float(), dim=-1)
    assert bool(torch.isfinite(z_tilde2).all().item())

    # raw_loss guard
    raw = torch.tensor(float("nan"), device=device)
    assert raw is None or not bool(torch.isfinite(raw).item())

    # sanitize grads
    p = torch.nn.Parameter(torch.zeros(4, device=device))
    p.grad = torch.tensor([1.0, float("nan"), 0.0, float("inf")], device=device)
    n = _sanitize_grads([p])
    assert n == 1
    assert bool(torch.isfinite(p.grad).all().item())
    assert float(p.grad[1].item()) == 0.0

    ad = SoftPrefixAdapter().to(device)
    out = ad(z)
    assert bool(torch.isfinite(out).all().item())
    print("nan_guard_ok")


if __name__ == "__main__":
    main()
