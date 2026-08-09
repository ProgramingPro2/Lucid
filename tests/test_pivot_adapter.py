"""Pure-math / shape tests for the pivot Stage-A adapter (no GPU / no downloads)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from experiments.pivot.adapter import (  # noqa: E402
    SoftPrefixAdapter,
    collate_prefix_batch,
    estimate_adapter_params,
    fiber_corrupt,
)


class _FakeTok:
    """Minimal tokenizer stub for collate tests."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        truncation: bool = True,
        max_length: int = 256,
        padding: bool | str = False,
        return_attention_mask: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        ids = []
        for t in texts:
            # Deterministic pseudo-tokens from character ordinals
            seq = [((ord(c) % 50) + 3) for c in t[:max_length]]
            if not seq:
                seq = [3]
            ids.append(seq)
        return {"input_ids": ids}


def test_adapter_forward_shape():
    torch.manual_seed(0)
    ad = SoftPrefixAdapter(z_dim=512, d_model=2048, n_prefix=16, hidden=512)
    z = F.normalize(torch.randn(4, 512), dim=-1)
    prefix = ad(z)
    assert prefix.shape == (4, 16, 2048)
    n = ad.n_trainable()
    # Spec: ~17M trainable for K=16 × 2048
    assert 15_000_000 <= n <= 20_000_000, f"unexpected adapter size {n}"
    assert estimate_adapter_params() == n or abs(estimate_adapter_params() - n) < 10


def test_fiber_corrupt_vmf_kappa700():
    torch.manual_seed(0)
    mu = F.normalize(torch.randn(8, 512), dim=-1)
    out = fiber_corrupt(mu, kappa=700.0)
    assert out.shape == (8, 512)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    # d=512, κ≈700 ⇒ mean resultant ≈0.70 (pooled fiber), not near-identity
    cos = (out * mu).sum(dim=-1)
    mean_cos = float(cos.mean())
    assert 0.55 < mean_cos < 0.90, mean_cos


def test_collate_prefix_prev_target_within_ctx():
    tok = _FakeTok()
    prev = ["hello world " * 40, "short"]
    tgt = ["target paragraph one " * 30, "tgt"]
    batch = collate_prefix_batch(
        tok,
        prev,
        tgt,
        max_prev=256,
        max_target=256,
        ctx_max=512,
        n_prefix=16,
    )
    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"]
    assert ids.ndim == 2
    b, l = ids.shape
    assert attn.shape == (b, 16 + l)
    assert labels.shape == (b, 16 + l)
    # Soft-prefix positions are attended and unlabeled
    assert torch.all(attn[:, :16] == 1)
    assert torch.all(labels[:, :16] == -100)
    # Total text tokens ≤ ctx_max
    text_lens = attn[:, 16:].sum(dim=1)
    assert int(text_lens.max()) <= 512
    # Some target labels are not -100
    assert int((labels != -100).sum()) > 0
