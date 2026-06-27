"""Offline tests for E10 instruction/theme conditioning.

All deterministic, CPU-only, no network/HF/lemonade. Uses the hashing codec via
fixtures and tiny in-memory datasets.
"""
from __future__ import annotations

import numpy as np
import torch

from conceptpoc.config import PredictorConfig, TrainConfig
from conceptpoc.data.corpus import Document
from conceptpoc.data.instruction import (
    InstructionConceptDataset,
    InstructionPair,
    build_synthetic_instructions,
    collate_instructions,
    combine_instruction_datasets,
)
from conceptpoc.generate.rollout import generate_conditioned, rollout_to_text
from conceptpoc.predictor.model import ConceptPredictor
from conceptpoc.predictor.train_conditioned import train_conditioned


def _tiny_cfg(**kw) -> PredictorConfig:
    base = dict(d_model=32, n_layers=2, n_heads=4, max_seq_len=32)
    base.update(kw)
    return PredictorConfig(**base)


def _fake_pairs(n_pairs=6, dim=16, ni=2, nr=3, seed=0) -> list[InstructionPair]:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_pairs):
        instr = rng.standard_normal((ni, dim)).astype(np.float32)
        resp = rng.standard_normal((nr, dim)).astype(np.float32)
        instr /= np.linalg.norm(instr, axis=1, keepdims=True)
        resp /= np.linalg.norm(resp, axis=1, keepdims=True)
        pairs.append(InstructionPair(instruction_vecs=instr, response_vecs=resp))
    return pairs


# --- dataset shapes / segment ids -----------------------------------------


def test_dataset_item_shapes_and_segment_ids():
    ds = InstructionConceptDataset(_fake_pairs(ni=2, nr=3, dim=16))
    instr, resp, seg, theme = ds[0]
    assert instr.shape == (2, 16)
    assert resp.shape == (3, 16)
    assert seg.tolist() == [0, 0, 1, 1, 1]
    assert theme is None
    assert ds.dim == 16


def test_collate_targets_and_loss_mask():
    # Two pairs with different lengths to exercise padding.
    pairs = [
        InstructionPair(
            instruction_vecs=np.ones((2, 4), np.float32),
            response_vecs=np.ones((3, 4), np.float32),
        ),
        InstructionPair(
            instruction_vecs=np.ones((1, 4), np.float32),
            response_vecs=np.ones((2, 4), np.float32),
        ),
    ]
    ds = InstructionConceptDataset(pairs)
    seq, pad_mask, seg, targets, loss_mask, theme = collate_instructions([ds[0], ds[1]])
    assert seq.shape == (2, 5, 4)
    assert theme is None
    # Pair 0 (ni=2,nr=3,len=5): supervise t in [1, 3].
    assert loss_mask[0].tolist() == [False, True, True, True, False]
    # Pair 1 (ni=1,nr=2,len=3): supervise t in [0, 1]; padded tail is False.
    assert loss_mask[1].tolist() == [True, True, False, False, False]
    # Padding marked correctly for the shorter pair.
    assert pad_mask[1].tolist() == [False, False, False, True, True]


def test_synthetic_builder_shapes(fitted_codec, sentences):
    docs = [
        Document(source="d0", sentences=sentences[:4]),
        Document(source="d1", sentences=sentences[4:8]),
    ]
    ds = build_synthetic_instructions(
        docs,
        fitted_codec,
        templates=["Continue:", "Write about {theme}"],
        themes=["nature"],
        max_response_sentences=4,
        max_instruction_sentences=2,
    )
    assert len(ds) == 2
    assert ds.dim == fitted_codec.dim
    instr, resp, seg, theme = ds[0]
    assert instr.shape[1] == fitted_codec.dim
    assert resp.shape[0] >= 1
    assert set(seg.tolist()) <= {0, 1}


def test_combine_fills_themes_uniformly():
    # One themed dataset + one un-themed dataset -> combined set is theme-uniform.
    themed = _fake_pairs(n_pairs=3, dim=8)
    for p in themed:
        p.theme_vec = np.ones(8, dtype=np.float32)
    plain = _fake_pairs(n_pairs=2, dim=8)
    combined = combine_instruction_datasets(
        [InstructionConceptDataset(themed), InstructionConceptDataset(plain)]
    )
    assert len(combined) == 5
    assert combined.has_theme
    # The previously-themed pairs keep their theme; plain pairs get a zero theme.
    np.testing.assert_allclose(combined.pairs[0].theme_vec, np.ones(8, np.float32))
    np.testing.assert_allclose(combined.pairs[4].theme_vec, np.zeros(8, np.float32))


def test_combine_without_themes_stays_unthemed():
    combined = combine_instruction_datasets(
        [InstructionConceptDataset(_fake_pairs(2, dim=8)),
         InstructionConceptDataset(_fake_pairs(2, dim=8, seed=1))]
    )
    assert len(combined) == 4
    assert not combined.has_theme


def test_dataset_roundtrip_save_load(tmp_path):
    ds = InstructionConceptDataset(_fake_pairs())
    path = tmp_path / "pairs.npz"
    ds.save(path)
    loaded = InstructionConceptDataset.load(path)
    assert len(loaded) == len(ds)
    np.testing.assert_allclose(
        loaded.pairs[0].instruction_vecs, ds.pairs[0].instruction_vecs
    )
    np.testing.assert_allclose(loaded.pairs[2].response_vecs, ds.pairs[2].response_vecs)


# --- conditioned forward pass + backward-compat invariant ------------------


def test_conditioned_forward_output_shape():
    cfg = _tiny_cfg(use_segment_embedding=True, use_theme=True)
    model = ConceptPredictor(concept_dim=16, cfg=cfg)
    ctx = torch.randn(2, 5, 16)
    seg = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    theme = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)

    last = model(ctx, segment_ids=seg, theme_vec=theme)
    assert last.shape == (2, 16)
    assert torch.allclose(last.norm(dim=-1), torch.ones(2), atol=1e-4)

    seqout = model(ctx, segment_ids=seg, theme_vec=theme, return_sequence=True)
    assert seqout.shape == (2, 5, 16)
    assert torch.allclose(seqout.norm(dim=-1), torch.ones(2, 5), atol=1e-4)


def test_backward_compat_invariant_no_conditioning():
    # A conditioning-enabled model, called WITHOUT conditioning, must equal the
    # legacy forward exactly (the extra modules contribute nothing).
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_segment_embedding=True, use_theme=True)
    model = ConceptPredictor(concept_dim=16, cfg=cfg).eval()
    ctx = torch.randn(3, 6, 16)
    base = model(ctx)
    with_none = model(ctx, segment_ids=None, theme_vec=None)
    assert torch.equal(base, with_none)

    # And a flags-off model is bit-identical given the same shared init.
    torch.manual_seed(0)
    plain = ConceptPredictor(concept_dim=16, cfg=_tiny_cfg()).eval()
    torch.manual_seed(0)
    enabled = ConceptPredictor(
        concept_dim=16, cfg=_tiny_cfg(use_segment_embedding=True, use_theme=True)
    ).eval()
    # Copy the shared (non-conditioning) parameters so only the extra modules differ.
    shared = {k: v for k, v in plain.state_dict().items()}
    enabled.load_state_dict(shared, strict=False)
    assert torch.allclose(plain(ctx), enabled(ctx), atol=1e-6)


# --- trainer smoke test -----------------------------------------------------


def test_train_conditioned_cpu_smoke():
    ds = InstructionConceptDataset(_fake_pairs(n_pairs=8, dim=16))
    cfg = _tiny_cfg()
    train_cfg = TrainConfig(batch_size=4, lr=1e-3, epochs=2, val_fraction=0.25)
    model, history = train_conditioned(ds, cfg, train_cfg, device="cpu", progress=False)
    assert len(history.train_loss) == 2
    assert model.segment_emb is not None  # conditioning was enabled by the trainer
    # The model can run a conditioned forward after training.
    instr, resp, seg, _ = ds[0]
    out = model(
        torch.cat([instr, resp]).unsqueeze(0),
        segment_ids=seg.unsqueeze(0),
        return_sequence=True,
    )
    assert out.shape == (1, instr.shape[0] + resp.shape[0], 16)


# --- deterministic conditioned generation -----------------------------------


def test_generate_conditioned_deterministic_and_shapes():
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_segment_embedding=True)
    model = ConceptPredictor(concept_dim=16, cfg=cfg).eval()
    instr = np.random.default_rng(1).standard_normal((2, 16)).astype(np.float32)
    instr /= np.linalg.norm(instr, axis=1, keepdims=True)

    a = generate_conditioned(model, instr, steps=4, temperature=0.0)
    b = generate_conditioned(model, instr, steps=4, temperature=0.0)
    assert a.generated_vectors.shape == (4, 16)
    assert a.seed_vectors.shape == (2, 16)
    np.testing.assert_array_equal(a.generated_vectors, b.generated_vectors)


def test_generate_conditioned_decodes_with_codec(fitted_codec, sentences):
    cfg = _tiny_cfg(use_segment_embedding=True)
    model = ConceptPredictor(concept_dim=fitted_codec.dim, cfg=cfg).eval()
    instr_vecs = fitted_codec.encode(sentences[:2])
    roll = generate_conditioned(model, instr_vecs, steps=3, temperature=0.0)
    text = rollout_to_text(roll, fitted_codec)
    assert len(text) == 3
    assert all(t in sentences for t in text)
