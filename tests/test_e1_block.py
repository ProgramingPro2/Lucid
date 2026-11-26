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


def test_causal_attn_mask_is_lower_triangular():
    """0 on and below the diagonal, -inf strictly above it, dtype preserved.

    A finite penalty above the diagonal would only down-weight the future.
    The language-model loss would still move when a later token changed.
    """
    from experiments.e1.block import causal_attn_mask

    mask = causal_attn_mask(8, torch.float32, torch.device("cpu"))
    assert mask.shape == (8, 8)
    assert mask.dtype == torch.float32
    upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.isneginf(mask.masked_select(upper)).all()
    assert torch.equal(
        mask.masked_select(~upper), torch.zeros(int((~upper).sum()))
    )
    mask64 = causal_attn_mask(4, torch.float64, torch.device("cpu"))
    assert mask64.dtype == torch.float64
    assert torch.isneginf(mask64[0, 1])
    assert float(mask64[1, 0]) == 0.0
    assert float(mask64[1, 1]) == 0.0


def test_nonzero_block_does_not_leak_future_tokens_into_logits():
    """Perturbing token t+k leaves logits at positions <= t bit-identical.

    The output projection is random, so the block is not the identity. A
    zero-init block would pass a leak test without being causal. Equality is
    exact: a tolerance is how the original leak (~1e-3) was small enough to
    miss and large enough to contaminate the LM loss.
    """
    from experiments.e1.train import E1Model, TrainConfig, attach_lora, build_base

    torch.manual_seed(0)
    cfg = TrainConfig(arm="A", seed=0, device="cpu", dtype="float32")
    base = attach_lora(build_base(cfg, tiny=True), cfg, tiny=True)
    block = JEPABlock(64, n_heads=8, zero_init_output=False)
    with torch.no_grad():
        block.out.weight.normal_()
        block.out.bias.normal_()
    model = E1Model(base, block=block)
    model.eval()

    ids = torch.randint(0, 512, (2, 32))
    t, k = 16, 4
    perturbed = ids.clone()
    perturbed[:, t + k] = (perturbed[:, t + k] + 17) % 512
    assert not torch.equal(ids[:, t + k], perturbed[:, t + k])

    with torch.no_grad():
        logits = model(ids).logits
        logits_p = model(perturbed).logits
    assert torch.equal(logits[:, : t + 1], logits_p[:, : t + 1])
    assert not torch.equal(logits[:, t + k], logits_p[:, t + k])


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


def test_ablation_zeroes_the_contribution():
    torch.manual_seed(0)
    block = JEPABlock(32, n_heads=4, zero_init_output=False)
    x = torch.randn(2, 5, 32)
    assert not torch.allclose(block(x), torch.zeros_like(x))
    block.set_ablation(True)
    assert torch.equal(block(x), torch.zeros_like(x))
    assert torch.equal(block.apply_to(x), x)
    block.set_ablation(False)
    assert not torch.allclose(block(x), torch.zeros_like(x))


def test_vicreg_covariance_is_scale_invariant_and_bounded():
    """The covariance penalty must not scale with hidden-state magnitude.

    Regression for the bug where raw Qwen3 hidden states (anisotropic, max abs
    ~1e4) produced a penalty of ~1e8 and swamped both losses.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float64)
    _, cov = vicreg_terms(x)
    _, cov_big = vicreg_terms(x * 1000.0)
    assert torch.allclose(cov, cov_big, atol=1e-10)
    assert 0.0 <= float(cov) <= 1.0
    assert 0.0 <= float(cov_big) <= 1.0


def test_combined_loss_stays_small_on_anisotropic_states():
    """A few huge outlier dimensions must not explode the total loss."""
    torch.manual_seed(0)
    student = torch.randn(2, 32, 64, dtype=torch.float64)
    student[..., 0] *= 5000.0  # one dominant direction, as in a real LLM
    teacher = student + 0.01 * torch.randn_like(student)
    total, diag = combined_loss(student, teacher, delta=4)
    assert float(total) < 10.0
    assert all(
        torch.isfinite(torch.tensor(v)).all()
        for v in (diag["pred"], diag["var"], diag["cov"], diag["total"])
    )


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


def test_arm_budgets_match_the_protocol_constant():
    """A/B/A'/B'/E/D share TOKENS_PER_RUN, and the docs say the same 30M.

    The old table gave the block arms 10M and the protocol 30M. A test that
    only reads the code would not catch the docs drifting back.
    """
    from pathlib import Path

    assert arms_mod.TOKENS_PER_RUN == 30_000_000
    for name in ("A", "B", "A_prime", "B_prime", "E", "D"):
        assert arms_mod.by_name(name).token_budget == arms_mod.TOKENS_PER_RUN
    # The assay and the nuisance arm are not that budget.
    assert arms_mod.by_name("PC_hi").token_budget != arms_mod.TOKENS_PER_RUN
    assert arms_mod.by_name("N").token_budget != arms_mod.TOKENS_PER_RUN

    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
    experiments = (root / "docs" / "EXPERIMENTS.md").read_text(encoding="utf-8")
    assert "2 full epochs" not in protocol
    section = experiments.split("## 3. E1 manifest", 1)[1].split("## 4.", 1)[0]
    for row_id in (
        "`e1/A/s*`",
        "`e1/B/s*`",
        "`e1/Ap/s*`",
        "`e1/Bp/s*`",
        "`e1/E/s*`",
        "`e1/D/s*`",
        "`e1/trunk/s*`",
    ):
        line = next(line for line in section.splitlines() if row_id in line)
        assert "30M" in line, line
        assert "10M" not in line, line


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
