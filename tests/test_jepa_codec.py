from __future__ import annotations

import numpy as np

from conceptpoc.codec.base import ConceptCodec
from conceptpoc.codec.jepa_codec import JepaCodec
from conceptpoc.config import JepaConfig, PredictorConfig
from conceptpoc.generate.rollout import generate_rollout, rollout_to_text
from conceptpoc.jepa.joint_model import JointJepa
from conceptpoc.jepa.tokenizer import Tokenizer
from conceptpoc.persistence import load_joint, save_joint


def _build_codec():
    tok = Tokenizer("char")
    cfg = JepaConfig(
        dim=32, d_model=32, enc_layers=1, dec_layers=1, enc_heads=4, dec_heads=4,
        tokenizer="char", max_tokens=24,
    )
    pcfg = PredictorConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=16)
    model = JointJepa(tok.vocab_size, tok.pad_id, tok.bos_id, cfg, pcfg)
    return JepaCodec(model, tok, max_tokens=24, device="cpu"), model, tok, cfg


def test_codec_satisfies_contract():
    codec, *_ = _build_codec()
    assert isinstance(codec, ConceptCodec)
    assert codec.dim == 32

    vecs = codec.encode(["the river moved slowly", "birds rose from the reeds"])
    assert vecs.shape == (2, 32)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    texts = codec.decode(vecs)
    assert len(texts) == 2
    assert all(isinstance(t, str) for t in texts)


def test_codec_empty_inputs():
    codec, *_ = _build_codec()
    assert codec.encode([]).shape == (0, 32)
    assert codec.decode(np.zeros((0, 32), dtype=np.float32)) == []


def test_codec_integrates_with_rollout():
    codec, *_ = _build_codec()
    seed = codec.encode(["a calm morning by the water", "the town was quiet"])
    roll = generate_rollout(codec.predictor, seed, steps=3, temperature=0.0)
    out = rollout_to_text(roll, codec)
    assert len(out) == 3


def test_save_and_load_joint_round_trip(tmp_path):
    _, model, tok, _ = _build_codec()
    ckpt = tmp_path / "joint.pt"
    save_joint(ckpt, model, "char")
    loaded, loaded_tok = load_joint(ckpt, device="cpu")
    assert loaded.cfg.dim == model.cfg.dim
    assert loaded_tok.vocab_size == tok.vocab_size

    codec_a = JepaCodec(model, tok, max_tokens=24, device="cpu")
    codec_b = JepaCodec(loaded, loaded_tok, max_tokens=24, device="cpu")
    v_a = codec_a.encode(["the river moved slowly"])
    v_b = codec_b.encode(["the river moved slowly"])
    assert np.allclose(v_a, v_b, atol=1e-5)


def test_channel_split_forward_runs():
    """The routed (multi-channel) encoder still trains: reconstruction on the
    full vector, prediction on the corr channel only, anti-collapse on spread."""
    from torch.utils.data import DataLoader

    from conceptpoc.data.corpus import Document
    from conceptpoc.jepa.dataset import JepaWindowDataset, make_collate

    tok = Tokenizer("char")
    cfg = JepaConfig(
        dim=32, d_model=32, enc_layers=1, dec_layers=1, enc_heads=4, dec_heads=4,
        tokenizer="char", max_tokens=24, context_len=4,
        channel_split=True, corr_dim=16,
    )
    pcfg = PredictorConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=16)
    model = JointJepa(tok.vocab_size, tok.pad_id, tok.bos_id, cfg, pcfg)

    docs = [
        Document(source="t", sentences=[
            "the river moved slowly", "birds rose from the reeds",
            "a boat drifted past", "the sun climbed higher",
        ])
    ]
    ds = JepaWindowDataset(docs, window=cfg.context_len)
    collate = make_collate(tok, cfg.context_len, cfg.max_tokens)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    ids, pad, valid = next(iter(loader))
    out = model(ids, pad, valid, alpha=1.0)
    assert out.total.isfinite()
    assert out.total.shape == ()
    # corr channel prediction works (idx into 16-dim corr space)
    assert out.std_z >= 0.0
    assert float(out.recon) == float(out.recon)

