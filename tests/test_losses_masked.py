"""Tests for neighbor-masked uniformity, adjacent InfoNCE, and anti_queue buffers."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from conceptpoc.config import JepaConfig, PredictorConfig
from conceptpoc.jepa.joint_model import JointJepa
from conceptpoc.jepa.losses import adjacent_infonce, uniformity_loss


def _neighbor_exclude_mask(rows: torch.Tensor, slots: torch.Tensor, k: int) -> torch.Tensor:
    same = rows.unsqueeze(1) == rows.unsqueeze(0)
    near = (slots.unsqueeze(1) - slots.unsqueeze(0)).abs() <= k
    return same & near


def test_masked_uniformity_excludes_neighbor_pairs():
    # Toy grid: 2 windows x 4 slots = 8 rows.
    rows = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    slots = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    n = rows.numel()
    k = 2
    excl = _neighbor_exclude_mask(rows, slots, k)

    # Count upper-triangle pairs that are excluded (same window, |d_slot|<=k).
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    n_excluded = int((triu & excl).sum().item())
    # Per window with 4 slots: pairs with |di| in {1,2} (and di=0 only on diag, not in triu).
    # Slots 0-3: (0,1),(0,2),(1,2),(1,3),(2,3) = 5 pairs with |d|<=2; (0,3) has |d|=3 > k.
    # Two windows => 10 excluded upper-tri pairs.
    assert n_excluded == 10

    z = F.normalize(torch.randn(n, 8), dim=-1)
    loss = uniformity_loss(z, exclude_mask=excl)
    assert torch.isfinite(loss)

    # Fully-masked pair set (exclude everything) returns 0.
    full_excl = torch.ones(n, n, dtype=torch.bool)
    zero = uniformity_loss(z, exclude_mask=full_excl)
    assert zero.item() == 0.0


def test_adjacent_infonce_pos_neg_structure():
    # 2 windows, 3 slots each; known successors.
    torch.manual_seed(0)
    rows = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    slots = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    z = F.normalize(torch.randn(6, 16), dim=-1)

    # Manually rebuild successor / logits logic for assertions.
    n = z.size(0)
    lookup = torch.full((2, 4), -1, dtype=torch.long)
    lookup[rows, slots] = torch.arange(n)
    pos_cols_all = lookup[rows, slots + 1]
    anchors = (pos_cols_all >= 0).nonzero(as_tuple=False).squeeze(-1)
    # Anchors: slots 0,1 in each window (4 anchors); slot 2 has no successor.
    assert anchors.tolist() == [0, 1, 3, 4]
    pos_cols = pos_cols_all[anchors]
    assert pos_cols.tolist() == [1, 2, 4, 5]  # successors

    temperature = 0.07
    logits = (z[anchors] @ z.T) / temperature
    same_window = rows[anchors].unsqueeze(1) == rows.unsqueeze(0)
    mask = same_window.clone()
    mask[torch.arange(anchors.numel()), pos_cols] = False
    logits_m = logits.masked_fill(mask, float("-inf"))

    for a_i, p in enumerate(pos_cols.tolist()):
        # Positive column finite; same-window non-positive are -inf.
        assert torch.isfinite(logits_m[a_i, p])
        same_cols = (rows == rows[anchors[a_i]]).nonzero(as_tuple=False).squeeze(-1)
        for c in same_cols.tolist():
            if c == p:
                continue
            assert logits_m[a_i, c].item() == float("-inf")
        # Cross-window negatives present (finite).
        other = (rows != rows[anchors[a_i]]).nonzero(as_tuple=False).squeeze(-1)
        assert other.numel() > 0
        assert torch.isfinite(logits_m[a_i, other]).all()

    loss = adjacent_infonce(z, rows, slots, temperature=temperature)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_anti_queue_buffer_registration():
    cfg = JepaConfig(dim=16, d_model=32, anti_queue_size=64)
    assert cfg.anticollapse_scope == "window_disjoint"
    m = JointJepa(100, 0, 1, cfg, PredictorConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=8))
    assert hasattr(m, "anti_queue")
    assert hasattr(m, "anti_queue_ptr")
    assert m.anti_queue.shape == (64, 16)
    assert m.anti_queue_ptr.shape == (2,)
    # persistent=False => not in state_dict
    sd = m.state_dict()
    assert "anti_queue" not in sd
    assert "anti_queue_ptr" not in sd
