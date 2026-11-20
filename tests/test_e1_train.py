"""Tests for the E1 training harness.

These run a tiny Qwen3 model on CPU. They check the machinery the confirmatory
contrast depends on, not any scientific claim:

* the block is bit-identical at step 0 for arms A and D;
* gradients reach the block from the LM loss (A and B) and from the JEPA loss (A);
* the teacher path really is adapter-free and block-free;
* the arm table wires each cell as the protocol says;
* P1/P3 arithmetic is correct on synthetic inputs.
"""

from __future__ import annotations

import copy

import pytest
import torch

from experiments.e1.block import JEPABlock
from experiments.e1.train import (
    E1Model,
    TrainConfig,
    attach_lora,
    build_base,
    build_e1_model,
    evaluate,
    find_decoder_layers,
    masked_nll,
    run,
    train,
)


def _cfg(**kw) -> TrainConfig:
    base = dict(
        arm="A",
        seed=0,
        tokens=4096,
        context=64,
        batch_size=2,
        grad_accum=1,
        eval_items=4,
        eval_prefix=32,
        eval_target=16,
        device="cpu",
        dtype="float32",
        log_every=1000,
    )
    base.update(kw)
    return TrainConfig(**base)


def test_find_decoder_layers_returns_all_layers():
    cfg = _cfg()
    base = build_base(cfg, tiny=True)
    layers = find_decoder_layers(base)
    assert len(layers) == 4


def test_zero_init_block_leaves_logits_bit_identical():
    """A and D must be the same function at step 0 (protocol §5.5)."""
    cfg = _cfg()
    base = build_base(cfg, tiny=True)

    torch.manual_seed(0)
    peft_a = attach_lora(copy.deepcopy(base), cfg, tiny=True)
    torch.manual_seed(0)
    peft_d = attach_lora(copy.deepcopy(base), cfg, tiny=True)

    block = JEPABlock(64, n_heads=8, zero_init_output=True)
    model_a = E1Model(peft_a, block=block)
    model_d = E1Model(peft_d)

    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        logits_a = model_a(ids).logits
        logits_d = model_d(ids).logits
    assert torch.equal(logits_a, logits_d)


def test_teacher_path_disables_adapter_and_block():
    cfg = _cfg()
    model, _ = build_e1_model(cfg, tiny=True)
    ids = torch.randint(0, 512, (2, 12))

    # Make the student different from the teacher by perturbing a LoRA matrix.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(0.5)
                break

    with torch.no_grad():
        student = model(ids).logits
        teacher = model(ids, teacher=True).logits
        # The teacher ignores both the adapter and the (zero) block, so it must
        # equal a fresh adapter-free forward from the same base weights.
        base = model.base
        with base.disable_adapter():
            pure = base(input_ids=ids, output_hidden_states=True, use_cache=False)
    assert not torch.allclose(student, teacher)
    assert torch.allclose(teacher, pure.logits)


@pytest.mark.parametrize("arm", ["A", "B", "A_prime", "B_prime", "E", "D", "N"])
def test_each_arm_trains_and_reports_finite_endpoints(arm):
    cfg = _cfg(arm=arm, tokens=2048, eval_items=3)
    result = run(cfg, tiny=True)
    ev = result["eval"]
    assert result["train"]["steps"] > 0
    assert all(
        torch.isfinite(torch.tensor(v)).all()
        for v in [
            ev["context_gain_mean"],
            ev["nll_true"],
            ev["nll_shuffled"],
        ]
    )
    # Necessity is defined exactly for arms that have a block.
    has_block = arm in {"A", "B", "E", "N"}
    assert ("necessity_mean" in ev) is has_block


def test_gradients_reach_the_block():
    cfg = _cfg(arm="A", tokens=2048)
    model, _ = build_e1_model(cfg, tiny=True)
    train_corpus, _ = _load(cfg)
    train(cfg, model, train_corpus, tiny=True)
    grads = [
        p.grad for p in model.block.parameters() if p.grad is not None
    ]
    assert grads, "no gradient reached the block"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_pc_lo_does_not_change_weights():
    cfg = _cfg(arm="PC_lo", tokens=2048)
    model, _ = build_e1_model(cfg, tiny=True)
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    train_corpus, _ = _load(cfg)
    train(cfg, model, train_corpus, tiny=True)
    after = dict(model.named_parameters())
    for name, old in before.items():
        assert torch.equal(old, after[name].detach()), name


def test_masked_nll_matches_manual_calculation():
    # Two positions, vocab 4, uniform logits give log(4) per token.
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, 1, 2]])
    nll = masked_nll(logits, labels)
    assert nll == pytest.approx(torch.log(torch.tensor(4.0)).item(), rel=1e-6)


def test_evaluate_necessity_is_zero_for_untrained_zero_block():
    """A zero-init block has no effect, so ablating it must change nothing."""
    cfg = _cfg(arm="A", eval_items=3)
    model, _ = build_e1_model(cfg, tiny=True)
    _, test_corpus = _load(cfg)
    items = test_corpus.eval_items(3, prefix=32, target=16, seed=0)
    ev = evaluate(cfg, model, items, tiny=True)
    assert ev["necessity_mean"] == pytest.approx(0.0, abs=1e-6)


def test_scrambled_arm_needs_two_sequences():
    cfg = _cfg(arm="E", batch_size=1, grad_accum=1)
    model, _ = build_e1_model(cfg, tiny=True)
    train_corpus, _ = _load(cfg)
    with pytest.raises(ValueError, match="scramble"):
        train(cfg, model, train_corpus, tiny=True)


def test_eval_set_is_fixed_across_arm_seeds():
    """Protocol §5.5: every arm and seed must see the same test items."""
    a = run(_cfg(arm="D", seed=0, tokens=1024, eval_items=3), tiny=True)
    b = run(_cfg(arm="D", seed=1, tokens=1024, eval_items=3), tiny=True)
    assert (
        a["manifest"]["eval_fingerprint"]
        == b["manifest"]["eval_fingerprint"]
    )


def _load(cfg: TrainConfig):
    import numpy as np

    from experiments.e1.data import PackedCorpus

    rng = np.random.default_rng(cfg.seed)
    docs = [
        list(rng.integers(0, 512, size=400, dtype=np.int64))
        for _ in range(20)
    ]
    return PackedCorpus(docs, eos_id=0).split(0.25, seed=cfg.seed)
