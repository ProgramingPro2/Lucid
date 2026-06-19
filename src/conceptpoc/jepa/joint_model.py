"""JointJepa: encoder + EMA target encoder + decoder + reused predictor.

A training batch is a set of documents, each a fixed number ``S`` of sentence
slots (shorter documents are padded with invalid slots). For each batch the
model computes:

- reconstruction loss (decode every valid sentence from its own concept vector),
- prediction loss (predict each sentence's EMA/stop-grad vector from the
  preceding sentences' online vectors),
- variance + covariance regularization on the valid concept vectors.

Only valid sentences are passed through the encoder/decoder, which avoids
fully-padded (NaN-prone) sequences entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import JepaConfig, PredictorConfig
from ..predictor.model import ConceptPredictor
from .decoder import TextDecoder
from .ema import EmaEncoder
from .encoder import TextEncoder
from .losses import (
    covariance_loss,
    mean_dim_std,
    mean_pairwise_cosine,
    prediction_loss,
    reconstruction_loss,
    uniformity_loss,
    variance_loss,
)


@dataclass
class JepaForward:
    total: torch.Tensor
    recon: torch.Tensor
    pred: torch.Tensor
    var: torch.Tensor
    cov: torch.Tensor
    unif: torch.Tensor
    recon_acc: float
    std_z: float
    # Mean cosine between predicted and target next-vectors (a monitoring metric,
    # not part of the loss). Used for validation / best-checkpoint selection.
    pred_cos: float
    # Anisotropy monitor: mean cosine between distinct online concept vectors.
    # High values (-> 1) mean the space collapsed into a narrow directional cone.
    aniso: float


class JointJepa(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        cfg: JepaConfig,
        predictor_cfg: PredictorConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.encoder = TextEncoder(vocab_size, cfg)
        self.ema = EmaEncoder(self.encoder, momentum=cfg.ema_momentum)
        self.decoder = TextDecoder(vocab_size, cfg)
        self.predictor = ConceptPredictor(cfg.dim, predictor_cfg)

    def forward(
        self,
        sent_ids: torch.Tensor,  # (B, S, L)
        sent_pad: torch.Tensor,  # (B, S, L) True=pad token
        sent_valid: torch.Tensor,  # (B, S) True=real sentence
        alpha: float,
    ) -> JepaForward:
        b, s, length = sent_ids.shape
        dim = self.cfg.dim
        device = sent_ids.device

        flat_ids = sent_ids.reshape(b * s, length)
        flat_pad = sent_pad.reshape(b * s, length)
        flat_valid = sent_valid.reshape(b * s)
        idx = flat_valid.nonzero(as_tuple=False).squeeze(-1)

        ids_v, pad_v = flat_ids[idx], flat_pad[idx]
        z_v, raw_v = self.encoder(ids_v, pad_v, return_raw=True)

        # Reconstruction anchor (online encoder -> decoder).
        logits_v = self.decoder.teacher_forcing_logits(
            z_v, ids_v, pad_v, self.bos_id
        )
        recon = reconstruction_loss(logits_v, ids_v, self.pad_id)

        # Anti-collapse regularization on the *unnormalized* projection (keeps
        # per-dimension variance up; mean-invariant so it ignores anisotropy).
        var = variance_loss(raw_v, self.cfg.var_margin)
        cov = covariance_loss(raw_v)
        std_z = mean_dim_std(raw_v)
        # Anti-anisotropy on the *normalized* vectors (spreads directions on the
        # hypersphere so the predictor can't win by emitting the mean direction).
        unif = uniformity_loss(z_v)
        aniso = mean_pairwise_cosine(z_v)

        # Scatter online + EMA vectors back to a (B, S, dim) grid.
        z_flat = torch.zeros(b * s, dim, device=device)
        z_flat[idx] = z_v
        with torch.no_grad():
            t_v = self.ema(ids_v, pad_v)
        t_flat = torch.zeros(b * s, dim, device=device)
        t_flat[idx] = t_v
        z = z_flat.reshape(b, s, dim)
        targets = t_flat.reshape(b, s, dim)

        # Prediction: predict sentence i from sentences 0..i-1. Documents are
        # prefix-valid, so sent_valid[:, i] implies a fully-valid context.
        pred_accum = z.new_zeros(())
        cos_accum = 0.0
        count = 0
        for i in range(1, s):
            rows = sent_valid[:, i].nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            context = z[rows, :i, :]
            ctx_pad = torch.zeros(rows.numel(), i, dtype=torch.bool, device=device)
            pred = self.predictor(context, ctx_pad)
            pred_accum = pred_accum + prediction_loss(pred, targets[rows, i, :]) * rows.numel()
            with torch.no_grad():
                cos_accum += float(
                    F.cosine_similarity(pred, targets[rows, i, :], dim=-1).sum().item()
                )
            count += rows.numel()
        pred = pred_accum / max(1, count)
        pred_cos = cos_accum / max(1, count)

        total = (
            recon.loss
            + alpha * pred
            + self.cfg.var_weight * var
            + self.cfg.cov_weight * cov
            + self.cfg.uniformity_weight * unif
        )
        return JepaForward(
            total=total,
            recon=recon.loss,
            pred=pred,
            var=var,
            cov=cov,
            unif=unif,
            recon_acc=recon.accuracy,
            std_z=std_z,
            pred_cos=pred_cos,
            aniso=aniso,
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        self.ema.update(self.encoder)
