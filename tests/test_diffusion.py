from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from conceptpoc.config import DiffusionConfig, PredictorConfig, TrainConfig
from conceptpoc.data.cache import CachedCorpus
from conceptpoc.data.dataset import ConceptSequenceDataset
from conceptpoc.eval.latent import latent_diagnostics
from conceptpoc.predictor.diffusion import DiffusionConceptPredictor, NoiseSchedule
from conceptpoc.predictor.model import ConceptPredictor, predictor_loss
from conceptpoc.predictor.train import train_predictor
from conceptpoc.predictor.train_diffusion import train_diffusion_predictor


def _tiny_predictor_cfg() -> PredictorConfig:
    return PredictorConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        head="regression",
    )


def _tiny_diffusion_cfg() -> PredictorConfig:
    return PredictorConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        head="diffusion",
        diffusion=DiffusionConfig(
            num_train_timesteps=20,
            sampling_steps=5,
            head_hidden=64,
            head_layers=2,
            time_dim=32,
        ),
    )


def test_diffusion_sample_shape_and_norm():
    cfg = _tiny_diffusion_cfg()
    model = DiffusionConceptPredictor(16, cfg)
    ctx = F.normalize(torch.randn(2, 4, 16), dim=-1)
    out = model.sample(ctx, sampling_steps=3)
    assert out.shape == (2, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-4)


def test_noise_schedule_q_sample():
    sched = NoiseSchedule(DiffusionConfig(num_train_timesteps=10))
    x0 = F.normalize(torch.randn(4, 8), dim=-1)
    t = torch.randint(0, 10, (4,))
    noise = torch.randn_like(x0)
    xt = sched.q_sample(x0, t, noise)
    assert xt.shape == x0.shape


def test_latent_diagnostics_collapse_signal():
    bank = np.eye(8, dtype=np.float32)
    repeated = np.tile(bank[0], (3, 1))
    diag = latent_diagnostics(repeated, bank)
    assert diag["mean_max_cos_to_bank"] > 0.99
    assert diag["mean_consecutive_cos"] > 0.99


def _bimodal_corpus(dim: int = 16, n_per_mode: int = 400) -> CachedCorpus:
    """Shared prefix, two possible next vectors (true multimodality)."""
    ctx = np.zeros(dim, dtype=np.float32)
    ctx[2] = 1.0
    va = np.zeros(dim, dtype=np.float32)
    vb = np.zeros(dim, dtype=np.float32)
    va[0] = 1.0
    vb[1] = 1.0
    vecs, offsets = [], [0]
    for target in (va, vb):
        for _ in range(n_per_mode):
            vecs.extend([ctx.copy(), target.copy()])
            offsets.append(len(vecs))
    return CachedCorpus(
        vectors=np.asarray(vecs, dtype=np.float32),
        sentences=[""] * len(vecs),
        doc_offsets=np.asarray(offsets, dtype=np.int64),
    )


def _bimodal_diffusion_cfg() -> PredictorConfig:
    return PredictorConfig(
        d_model=64,
        n_layers=3,
        n_heads=4,
        max_seq_len=16,
        head="diffusion",
        diffusion=DiffusionConfig(
            num_train_timesteps=50,
            sampling_steps=10,
            head_hidden=128,
            head_layers=3,
            time_dim=64,
        ),
    )


def test_bimodal_regression_averages_diffusion_recovers_modes():
    corpus = _bimodal_corpus()
    ds = ConceptSequenceDataset(corpus, context_len=1)
    train_cfg = TrainConfig(batch_size=64, lr=3e-3, epochs=40, val_fraction=0.1)

    reg_cfg = _tiny_predictor_cfg()
    reg_model, _ = train_predictor(
        ds, reg_cfg, train_cfg, device="cpu", progress=False
    )
    reg_model.eval()
    ctx = torch.zeros(1, 1, 16)
    ctx[0, 0, 2] = 1.0
    tgt_a = torch.zeros(1, 16)
    tgt_b = torch.zeros(1, 16)
    tgt_a[0, 0] = 1.0
    tgt_b[0, 1] = 1.0
    with torch.no_grad():
        pred = reg_model(ctx)
    reg_cos_a = F.cosine_similarity(pred, tgt_a, dim=-1).item()
    reg_cos_b = F.cosine_similarity(pred, tgt_b, dim=-1).item()
    # Point estimate sits between modes; neither mode is recovered well.
    assert reg_cos_a < 0.75 and reg_cos_b < 0.75

    diff_model, history = train_diffusion_predictor(
        ds, _bimodal_diffusion_cfg(), train_cfg, device="cpu", progress=False
    )
    assert history.val_sample_cosine[-1] > history.val_sample_cosine[0]
    diff_model.eval()
    hits_a, hits_b, trials = 0, 0, 48
    with torch.no_grad():
        for _ in range(trials):
            sample = diff_model.sample(ctx.squeeze(1), sampling_steps=10)
            if F.cosine_similarity(sample, tgt_a, dim=-1).item() > 0.85:
                hits_a += 1
            if F.cosine_similarity(sample, tgt_b, dim=-1).item() > 0.85:
                hits_b += 1
    # Stochastic diffusion hits each mode sometimes; regression emits one midpoint.
    assert hits_a >= 2
    assert hits_b >= 2
    assert hits_a + hits_b >= trials // 6
