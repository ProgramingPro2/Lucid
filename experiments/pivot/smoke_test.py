#!/usr/bin/env python
"""Smoke checks for SoftPrefixAdapter (mirrors tests/test_pivot_adapter.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.pivot.adapter import (  # noqa: E402
    SoftPrefixAdapter,
    collate_prefix_batch,
    fiber_corrupt,
)


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, **kw):
        ml = kw.get("max_length", 256)
        ids = []
        for t in texts:
            seq = [((ord(c) % 50) + 3) for c in t[:ml]]
            ids.append(seq or [3])
        return {"input_ids": ids}


def main() -> None:
    torch.manual_seed(0)
    ad = SoftPrefixAdapter(z_dim=512, d_model=2048, n_prefix=16, hidden=512)
    z = F.normalize(torch.randn(4, 512), dim=-1)
    prefix = ad(z)
    assert prefix.shape == (4, 16, 2048), prefix.shape
    n = ad.n_trainable()
    assert 15_000_000 <= n <= 20_000_000, n
    print(f"PASS adapter shape/params n={n}")

    mu = F.normalize(torch.randn(8, 512), dim=-1)
    out = fiber_corrupt(mu, kappa=700.0)
    assert out.shape == (8, 512)
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-4)
    # d=512, κ≈700 ⇒ mean resultant ≈0.70 (pooled fiber contract), not near 1
    mean_cos = float((out * mu).sum(-1).mean())
    assert 0.55 < mean_cos < 0.90, mean_cos
    print(f"PASS fiber_corrupt kappa=700 mean_cos={mean_cos:.3f}")

    batch = collate_prefix_batch(
        _FakeTok(),
        ["hello world " * 40, "short"],
        ["target paragraph one " * 30, "tgt"],
        max_prev=256,
        max_target=256,
        ctx_max=512,
        n_prefix=16,
    )
    assert batch["attention_mask"].shape[1] == 16 + batch["input_ids"].shape[1]
    assert torch.all(batch["labels"][:, :16] == -100)
    assert int(batch["attention_mask"][:, 16:].sum(1).max()) <= 512
    assert int((batch["labels"] != -100).sum()) > 0
    print("PASS collate within ctx")
    print("ALL PASS")


if __name__ == "__main__":
    main()
