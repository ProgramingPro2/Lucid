from __future__ import annotations

import torch

from conceptpoc.config import JepaConfig, PredictorConfig
from conceptpoc.data.corpus import load_documents
from conceptpoc.jepa.losses import covariance_loss, variance_loss
from conceptpoc.jepa.train_joint import train_joint
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.txt"


def test_variance_loss_penalizes_collapse():
    constant = torch.ones(16, 8)  # every row identical -> zero variance
    spread = torch.randn(16, 8)
    assert variance_loss(constant) > variance_loss(spread)


def test_covariance_loss_nonnegative():
    z = torch.randn(32, 8)
    assert covariance_loss(z).item() >= 0.0


def test_joint_run_stays_uncollapsed_and_learns():
    docs = load_documents(CORPUS)
    cfg = JepaConfig(
        dim=48, d_model=48, enc_layers=1, dec_layers=1, enc_heads=4, dec_heads=4,
        tokenizer="char", max_tokens=32, context_len=6, batch_size=4,
        epochs=6, recon_warmup_epochs=2, alpha_warmup_epochs=2,
        collapse_std_threshold=1e-3,
    )
    pcfg = PredictorConfig(d_model=48, n_layers=1, n_heads=4, max_seq_len=16)
    _, _, hist = train_joint(docs, cfg, pcfg, device="cpu")

    # The space does not collapse and reconstruction improves over training.
    assert not hist.collapsed
    assert hist.std_z[-1] > hist.std_z[0] * 0.5
    assert hist.recon_acc[-1] > hist.recon_acc[0]
    # Prediction loss is engaged once alpha ramps in.
    assert hist.alpha[-1] > 0
