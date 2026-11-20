"""Tests for the E1 inserted block and objectives."""

from __future__ import annotations

import pytest
import torch

from experiments.e1 import arms as arms_mod
from experiments.e1.block import EMATeacher, JEPABlock, whiten
from experiments.e1.objective import (
    combined_loss,
    context_gain,
    jepa_loss,
    vicreg_terms,
)


def test_zero_init_block_is_exactly_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    block = JEPABlock(32, n_heads=4, zero_init_output=True).double()
    assert torch.equal(block(x), torch.zeros_like(x))
    assert torch.equal(block.apply_to(x), x)


def test_nonzero_init_changes_output():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    block = JEPABlock(32, n_heads=4, zero_init_output=False)
    assert not torch.allclose(block.apply_to(x), x)


def test_shapes_and_mask_accepted():
    block = JEPABlock(48, n_heads=4, zero_init_output=False)
    x = torch.randn(3, 7, 48)
    mask = torch.ones(3, 7)
    assert block(x).shape == (3, 7, 48)
    assert block(x, attention_mask=mask).shape == (3, 7, 48)


def test_padding_does_not_change_valid_positions():
    torch.manual_seed(0)
    block = JEPABlock(32, n_heads=4, zero_init_output=False).double()
    x = torch.randn(1, 4, 32, dtype=torch.float64)
    pad_a = torch.randn(1, 3, 32, dtype=torch.float64)
    pad_b = torch.randn(1, 3, 32, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    out_a = block(torch.cat([x, pad_a], dim=1), attention_mask=mask)
    out_b = block(torch.cat([x, pad_b], dim=1), attention_mask=mask)
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-10)


def test_vicreg_mask_ignores_padding():
    torch.manual_seed(0)
    real = torch.randn(1, 5, 16, dtype=torch.float64)
    pad_a = torch.randn(1, 3, 16, dtype=torch.float64)
    pad_b = torch.randn(1, 3, 16, dtype=torch.float64) * 100
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    va, ca = vicreg_terms(torch.cat([real, pad_a], 1), mask=mask)
    vb, cb = vicreg_terms(torch.cat([real, pad_b], 1), mask=mask)
    assert torch.allclose(va, vb)
    assert torch.allclose(ca, cb)


def test_whiten_mask_zeroes_padding_and_uses_real_statistics():
    real = torch.randn(1, 4, 8)
    x = torch.cat([real, torch.full((1, 2, 8), 99.0)], dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    out = whiten(x, mask=mask)
    assert torch.allclose(out[:, 4:], torch.zeros(1, 2, 8), atol=1e-6)
    unmasked = whiten(real)
    assert torch.allclose(out[:, :4], unmasked, atol=1e-5)


def test_ema_moves_toward_student_and_never_requires_grad():
    student = torch.nn.Linear(4, 4)
    teacher = EMATeacher(student)
    assert all(not p.requires_grad for p in teacher.parameters())
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
    before = (teacher.module.weight - student.weight).abs().mean().item()
    assert before == pytest.approx(1.0)
    teacher.update(student, momentum=0.5)
    after = (teacher.module.weight - student.weight).abs().mean().item()
    assert after < before
    assert after == pytest.approx(0.5, abs=1e-6)


def test_jepa_loss_finite_and_truncates_horizon():
    torch.manual_seed(0)
    s = torch.randn(2, 16, 32)
    t = torch.randn(2, 16, 32)
    loss, diag = jepa_loss(s, t, delta=4)
    assert torch.isfinite(loss)
    assert diag["n_positions"] == 2 * 12


def test_jepa_loss_short_sequence_is_zero_not_nan():
    s = torch.randn(1, 3, 8)
    t = torch.randn(1, 3, 8)
    loss, diag = jepa_loss(s, t, delta=8)
    assert diag["n_positions"] == 0
    assert float(loss) == 0.0


def test_jepa_mask_excludes_padded_positions():
    s = torch.randn(1, 8, 8)
    t = torch.randn(1, 8, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    _, diag = jepa_loss(s, t, delta=2, mask=mask)
    assert diag["n_positions"] == 4


def test_vicreg_penalises_collapse_and_not_healthy_spread():
    collapsed = torch.ones(64, 16)
    var_bad, cov_bad = vicreg_terms(collapsed)
    assert float(var_bad) > 0.5
    assert torch.isfinite(cov_bad)

    torch.manual_seed(0)
    healthy = torch.randn(4096, 16)
    var_ok, cov_ok = vicreg_terms(healthy)
    # Per-dimension std is 1 +/- 1/sqrt(2n), so the hinge sits near 0.004.
    assert float(var_ok) < 0.01
    assert torch.isfinite(cov_ok)


def test_combined_loss_reports_components():
    torch.manual_seed(0)
    s = torch.randn(2, 10, 16)
    t = torch.randn(2, 10, 16)
    loss, diag = combined_loss(s, t, delta=2)
    assert torch.isfinite(loss)
    for k in ("pred", "var", "cov", "total"):
        assert k in diag


def test_context_gain_sign():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 2.5])
    assert torch.allclose(context_gain(a, b), torch.tensor([2.0, 0.5]))


def test_whiten_handles_constant_input():
    x = torch.ones(4, 8)
    out = whiten(x)
    assert torch.isfinite(out).all()


def test_arm_table_and_parameter_matching():
    arms = arms_mod.all_arms()
    names = {a.name for a in arms}
    assert {"A", "B", "A_prime", "B_prime", "E", "D", "N", "PC_hi", "PC_lo"} <= names
    assert arms_mod.total_training_units() == 219
    arms_mod.assert_parameter_matched(d_model=64)
    # The treatment and its isotype must differ only in the loss weight.
    a, b = arms_mod.by_name("A"), arms_mod.by_name("B")
    assert a.jepa_loss and not b.jepa_loss
    assert a.block and b.block


def test_gradient_reaches_block_output_projection():
    block = JEPABlock(16, n_heads=2, zero_init_output=False)
    x = torch.randn(2, 4, 16, requires_grad=True)
    block.apply_to(x).sum().backward()
    assert block.out.weight.grad is not None
    assert block.out.weight.grad.abs().sum() > 0
