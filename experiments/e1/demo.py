"""CPU end-to-end plumbing check for the E1 block.

    PLUMBING ONLY — NOT A RESULT

This script does not test the hypothesis.  It runs a toy 2-layer LM on random
token ids for 30 steps and checks four mechanical facts:

1. the next-token loss is finite with and without the JEPA objective;
2. the zero-initialised block is bit-identically the identity at step 0;
3. gradients reach the block's zero-initialised output projection;
4. the EMA teacher moves toward the student.

Everything is small enough to run on CPU in a few seconds.  No downloads, no
network, no claim.

Run with::

    PYTHONPATH=src python -m experiments.e1.demo
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.e1.block import EMATeacher, JEPABlock
from experiments.e1.objective import jepa_loss

BANNER = "PLUMBING ONLY — NOT A RESULT"
VOCAB = 64
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
BATCH = 8
SEQ_LEN = 24
STEPS = 30
JEPA_WEIGHT = 0.05
EMA_MOMENTUM = 0.9


class ToyBlock(nn.Module):
    """An ordinary causal pre-norm transformer block for the toy LM.

    This is *not* the E1 block: it has no JEPA contribution and no zero-init
    output projection.  Its only job is to make a small model deep enough for
    an inserted block to have a residual stream to sit in.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=causal, need_weights=False
        )
        x = x + attn_out
        return x + self.mlp(self.ln2(x))


class ToyLM(nn.Module):
    """Embedding -> two transformer blocks (one optional inserted block) -> head.

    The inserted JEPABlock sits **between** the toy blocks, not beside them, so
    the rest of the network must consume its output.  That is the protocol's
    "insert between blocks" requirement in miniature.
    """

    def __init__(self, use_jepa: bool) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(
            [ToyBlock(D_MODEL, N_HEADS) for _ in range(N_LAYERS)]
        )
        self.jepa = (
            JEPABlock(
                d_model=D_MODEL,
                n_heads=N_HEADS,
                dropout=0.0,
                zero_init_output=True,
            )
            if use_jepa
            else None
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB)

    def forward(
        self, tokens: Tensor, return_hidden: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = self.embed(tokens)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 0 and self.jepa is not None:
                # Contribution, then explicit residual add.  At zero init
                # apply_to is exactly x.
                x = self.jepa.apply_to(x)
        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


def _next_token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, VOCAB),
        targets.reshape(-1),
    )


def _train(
    model: ToyLM,
    inputs: Tensor,
    targets: Tensor,
    *,
    use_jepa: bool,
    steps: int = STEPS,
) -> tuple[float, bool, dict[str, float]]:
    """Run ``steps`` of NTP (plus optional JEPA) and return diagnostics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    teacher = EMATeacher(model) if use_jepa else None

    block_grad_reached = False
    teacher_moved = False
    teacher_distance_ratio = float("nan")
    final_loss = float("nan")

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        if use_jepa:
            logits, hidden = model(inputs, return_hidden=True)
            with torch.no_grad():
                _, teacher_hidden = teacher(inputs, return_hidden=True)
            auxiliary, diagnostics = jepa_loss(
                hidden, teacher_hidden, delta=1
            )
            assert diagnostics["n_positions"] > 0
            loss = _next_token_loss(logits, targets) + JEPA_WEIGHT * auxiliary
        else:
            logits = model(inputs)
            loss = _next_token_loss(logits, targets)

        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        if use_jepa and step == 0:
            grad = model.jepa.out_proj.weight.grad  # type: ignore[union-attr]
            block_grad_reached = (
                grad is not None and float(grad.abs().sum()) > 0.0
            )

        optimizer.step()

        if teacher is not None:
            if step == 0:
                teacher_parameter = teacher.module.blocks[0].ln1.weight
                student_parameter = model.blocks[0].ln1.weight
                before = teacher_parameter.detach().clone()

            teacher.update(model, momentum=EMA_MOMENTUM)

            if step == 0:
                after = teacher_parameter.detach().clone()
                teacher_moved = not torch.equal(before, after)
                distance_before = float(
                    (before - student_parameter.detach()).norm()
                )
                distance_after = float(
                    (after - student_parameter.detach()).norm()
                )
                teacher_distance_ratio = distance_after / distance_before

        final_loss = float(loss.detach())

    diagnostics_out = {
        "final_loss": final_loss,
        "teacher_distance_ratio": teacher_distance_ratio,
    }
    return final_loss, block_grad_reached and teacher_moved, diagnostics_out


def main() -> int:
    print("=" * 60)
    print(BANNER)
    print("=" * 60)

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        0, VOCAB, (BATCH, SEQ_LEN + 1), generator=generator
    )
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    with_block = ToyLM(use_jepa=True)
    without_block = copy.deepcopy(with_block)
    without_block.jepa = None  # type: ignore[assignment]

    with_block.eval()
    without_block.eval()

    # ---- 1. step-0 identity ------------------------------------------------
    with torch.no_grad():
        logits_with = with_block(inputs)
        logits_without = without_block(inputs)
        loss_with = _next_token_loss(logits_with, targets)
        loss_without = _next_token_loss(logits_without, targets)

    assert torch.isfinite(loss_with), "step-0 loss with block is not finite"
    assert torch.isfinite(loss_without), "step-0 loss without block is not finite"
    assert torch.equal(
        loss_with, loss_without
    ), "zero-init block changed the step-0 loss; it is not the identity"
    print(f"step-0 loss with block:    {loss_with.item():.6f}")
    print(f"step-0 loss without block: {loss_without.item():.6f}  (bit-identical)")

    # ---- 2. train both arms ------------------------------------------------
    with_block.train()
    final_with, block_ok, diag_with = _train(
        with_block, inputs, targets, use_jepa=True
    )
    without_block.train()
    final_without, _, diag_without = _train(
        without_block, inputs, targets, use_jepa=False
    )

    assert torch.isfinite(torch.tensor(final_with)), "JEPA-arm loss is not finite"
    assert torch.isfinite(
        torch.tensor(final_without)
    ), "NTP-only-arm loss is not finite"

    # ---- 3. gradients and EMA ---------------------------------------------
    assert block_ok, "gradients did not reach the block / teacher did not move"
    grad_norm = float(with_block.jepa.out_proj.weight.grad.norm())  # type: ignore[union-attr]
    assert grad_norm > 0.0, "block output projection received zero gradient"
    assert torch.isfinite(torch.tensor(grad_norm)), "gradient norm is not finite"
    ratio = diag_with["teacher_distance_ratio"]
    assert 0.0 <= ratio < 1.0, (
        f"EMA update did not move the teacher toward the student "
        f"(distance ratio {ratio})"
    )

    print(f"30-step loss, JEPA arm:    {final_with:.6f}")
    print(f"30-step loss, NTP-only:    {final_without:.6f}")
    print(f"block out_proj grad norm:  {grad_norm:.6f}")
    print(f"EMA distance ratio step 0: {ratio:.3f} (momentum={EMA_MOMENTUM})")
    print("ALL PLUMBING CHECKS PASSED")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
