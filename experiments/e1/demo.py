"""CPU plumbing check for E1 — PLUMBING ONLY, NOT A RESULT.

This answers four questions that must be true before any GPU run is meaningful:

1. Can a JEPABlock be inserted into a small causal LM and trained end to end
   without NaNs?
2. With ``zero_init_output=True``, is the step-0 LM loss *bit-identical* with and
   without the block? (If not, A/B differ at step 0 and the contrast is
   confounded.)
3. Do gradients actually reach the inserted block?
4. Does the EMA teacher move toward the student?

It says nothing about whether the idea works. A toy model on random tokens cannot.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.e1.demo
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import EMATeacher, JEPABlock
from .objective import jepa_loss


class ToyLM(nn.Module):
    """Embedding → 2 transformer blocks → lm_head, with an optional inserted block."""

    def __init__(self, vocab: int, d_model: int, n_heads: int, use_jepa: bool) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, n_heads, dim_feedforward=4 * d_model,
                    batch_first=True, dropout=0.0,
                )
                for _ in range(2)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab)
        # Build the block *last* so the base parameters are identical whether or
        # not a block is present. Otherwise the block's own init consumes RNG
        # before lm_head and the step-0 identity check fails for the wrong reason.
        self.jepa = JEPABlock(d_model, n_heads=n_heads) if use_jepa else None

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(ids)
        h_before = self.blocks[0](x)
        if self.jepa is not None:
            h = self.jepa.apply_to(h_before)
        else:
            h = h_before
        h = self.blocks[1](h)
        return self.lm_head(h), h_before, h


def _build(seed: int, cfg: dict, use_jepa: bool) -> ToyLM:
    torch.manual_seed(seed)
    return ToyLM(cfg["vocab"], cfg["d"], cfg["heads"], use_jepa)


def _batch(cfg: dict, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq"]), generator=g)


def _lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    )


def check(args: argparse.Namespace) -> int:
    cfg = {"vocab": 256, "d": 64, "heads": 4, "batch": 8, "seq": 32}
    ids = _batch(cfg, seed=7)
    print("PLUMBING ONLY — NOT A RESULT")
    print("-" * 52)

    # ---- 2. step-0 identity -------------------------------------------------
    with_block = _build(args.seed, cfg, use_jepa=True)
    without_block = _build(args.seed, cfg, use_jepa=False)
    with torch.no_grad():
        logits_a, _, _ = with_block(ids)
        logits_b, _, _ = without_block(ids)
    loss_a = _lm_loss(logits_a, ids)
    loss_b = _lm_loss(logits_b, ids)
    identical = torch.equal(loss_a, loss_b)
    print(f"step-0 LM loss with block    : {loss_a.item():.10f}")
    print(f"step-0 LM loss without block : {loss_b.item():.10f}")
    print(f"bit-identical at step 0      : {identical}")
    if not identical:
        print("FAIL: zero-init block changed the step-0 loss", file=sys.stderr)
        return 1

    # ---- 1/3/4. short training run -----------------------------------------
    model = _build(args.seed, cfg, use_jepa=True)
    teacher = EMATeacher(model.jepa)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Snapshot the teacher at init. The claim to test is not "distance shrinks
    # from zero" (it starts at zero, so that is impossible) but "the EMA teacher
    # tracks the student better than a frozen copy would".
    frozen = [p.detach().clone() for p in teacher.module.parameters()]

    def distance_to(weights: list[torch.Tensor]) -> float:
        return sum(
            (t - s).abs().mean().item()
            for t, s in zip(weights, model.jepa.parameters())
        ) / len(weights)

    finite = True
    grad_reached = False

    for step in range(args.steps):
        logits, h_before, h = model(ids)
        lm = _lm_loss(logits, ids)
        with torch.no_grad():
            target = teacher.module.apply_to(h_before)
        jepa, diag = jepa_loss(h, target, delta=2)
        loss = lm + args.jepa_weight * jepa

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if not torch.isfinite(loss):
            finite = False
            break
        if model.jepa is not None and model.jepa.out.weight.grad is not None:
            if model.jepa.out.weight.grad.abs().sum() > 0:
                grad_reached = True

        teacher.update(model.jepa, momentum=args.momentum)

        if step in {0, args.steps - 1}:
            print(
                f"step {step:2d}  lm={lm.item():.4f}  jepa={jepa.item():.4f}  "
                f"jepa_positions={diag['n_positions']}"
            )

    live_distance = distance_to([p.detach() for p in teacher.module.parameters()])
    frozen_distance = distance_to(frozen)
    teacher_changed = any(
        not torch.equal(t.detach(), f)
        for t, f in zip(teacher.module.parameters(), frozen)
    )
    print("-" * 52)
    print(f"loss finite over {args.steps} steps : {finite}")
    print(f"gradient reached block output        : {grad_reached}")
    print(f"teacher params changed               : {teacher_changed}")
    print(f"distance to student: EMA {live_distance:.4f} vs frozen {frozen_distance:.4f}")
    print(f"EMA tracks better than frozen        : {live_distance < frozen_distance}")

    ok = (
        finite
        and grad_reached
        and teacher_changed
        and live_distance < frozen_distance
    )
    print("-" * 52)
    print("PLUMBING " + ("OK" if ok else "BROKEN") + " — still NOT a result")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jepa-weight", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.99)
    return check(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
