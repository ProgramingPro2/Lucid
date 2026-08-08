"""Smoke tests for experiments.sentence_pred.train_head."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from conceptpoc.config import PredictorConfig
from experiments.sentence_pred.data import SentenceCorpus
from experiments.sentence_pred.heads import sample_vmf
from experiments.sentence_pred.train_head import train_density_head


def _vmf_chain_corpus(
    *,
    n_train_docs: int = 4,
    n_held_docs: int = 2,
    sents_per_doc: int = 20,
    dim: int = 16,
    kappa: float = 200.0,
    seed: int = 0,
) -> SentenceCorpus:
    """Docs of unit vectors drawn from a small fixed-mu vMF (learnable pattern)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mu = F.normalize(torch.randn(dim, generator=g), dim=0)
    all_vecs: list[np.ndarray] = []
    sentences: list[str] = []
    offsets = [0]
    n_docs = n_train_docs + n_held_docs
    for d in range(n_docs):
        mu_d = F.normalize(mu + 0.05 * torch.randn(dim, generator=g), dim=0)
        for i in range(sents_per_doc):
            x = sample_vmf(
                mu_d.unsqueeze(0),
                torch.tensor([kappa]),
                generator=g,
            ).squeeze(0)
            all_vecs.append(x.detach().cpu().numpy().astype(np.float32))
            sentences.append(f"d{d}_s{i}")
        offsets.append(len(all_vecs))
    sent_vecs = np.stack(all_vecs, axis=0)
    doc_offsets = np.asarray(offsets, dtype=np.int64)
    train = np.arange(n_train_docs, dtype=np.int64)
    held = np.arange(n_train_docs, n_docs, dtype=np.int64)
    return SentenceCorpus(
        sent_vecs=sent_vecs,
        doc_offsets=doc_offsets,
        sentences=sentences,
        train_doc_ids=train,
        held_doc_ids=held,
    )


def test_train_density_head_nll_does_not_increase(tmp_path: Path):
    corpus = _vmf_chain_corpus()
    pcfg = PredictorConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_len=10,
    )
    out = tmp_path / "head.pt"
    _model, history = train_density_head(
        corpus,
        predictor_cfg=pcfg,
        head="vmf",
        K=2,
        out=out,
        seed=42,
        device="cpu",
        epochs=2,
        batch_size=8,
        lr=3e-3,
        patience=5,
        context_len=8,
        log_every=0,
        progress=False,
        use_gpu_gate=False,
    )
    assert len(history["train_nll"]) == 2
    assert len(history["val_nll"]) == 2
    v0, v1 = history["val_nll"][0], history["val_nll"][-1]
    assert v1 <= v0 + 1e-5, f"val NLL rose: {v0} -> {v1}"
    assert out.is_file()


def test_checkpoint_round_trip(tmp_path: Path):
    corpus = _vmf_chain_corpus(n_train_docs=3, n_held_docs=1, sents_per_doc=16)
    pcfg = PredictorConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_len=10,
    )
    out = tmp_path / "ckpt.pt"
    model, history = train_density_head(
        corpus,
        predictor_cfg=pcfg,
        head="vmf",
        K=4,
        out=out,
        seed=7,
        device="cpu",
        epochs=1,
        batch_size=4,
        lr=3e-3,
        patience=5,
        context_len=8,
        log_every=0,
        progress=False,
        use_gpu_gate=False,
    )
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["head"] == "vmf"
    assert ckpt["concept_dim"] == 16
    assert ckpt["K"] == 4
    assert "train_nll" in ckpt["history"]
    assert "val_nll" in ckpt["history"]
    assert ckpt["history"]["train_nll"] == history["train_nll"]
    assert ckpt["history"]["val_nll"] == history["val_nll"]
    assert ckpt["epochs_trained"] == 1

    restored = type(model)(
        concept_dim=ckpt["concept_dim"],
        predictor_cfg=PredictorConfig(**ckpt["predictor_cfg"]),
        head=ckpt["head"],
        K=ckpt["K"],
    )
    restored.load_state_dict(ckpt["state_dict"])
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert k1 == k2
        assert torch.allclose(v1.cpu(), v2.cpu())
