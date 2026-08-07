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
    adjacent_infonce,
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
    # Adjacent-sentence InfoNCE (0 when align_weight == 0).
    align: torch.Tensor
    # Mean cosine over valid adjacent pairs in the batch (live gate monitor).
    adj_cos: float


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
        # When channel-split, the predictor operates in the corr channel's dim.
        pred_dim = cfg.corr_dim if cfg.channel_split else cfg.dim
        self.predictor = ConceptPredictor(pred_dim, predictor_cfg)
        if cfg.anticollapse_scope == "window_disjoint" and cfg.anti_queue_size > 0:
            anti_dim = (cfg.dim - cfg.corr_dim) if cfg.channel_split else cfg.dim
            self.register_buffer(
                "anti_queue", torch.zeros(cfg.anti_queue_size, anti_dim), persistent=False
            )
            self.register_buffer(
                "anti_queue_ptr", torch.zeros(2, dtype=torch.long), persistent=False
            )
            # ptr[0] = write cursor, ptr[1] = fill count

    @staticmethod
    def _one_per_window(anti_full: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """One random valid slot per window; grads flow through selected rows."""
        device = anti_full.device
        picks: list[torch.Tensor] = []
        for r in torch.unique(rows).tolist():
            idxs = (rows == r).nonzero(as_tuple=False).squeeze(-1)
            j = idxs[torch.randint(0, idxs.numel(), (1,), device=device)]
            picks.append(j)
        return anti_full[torch.cat(picks)]

    @torch.no_grad()
    def _push_queue(self, sample: torch.Tensor) -> None:
        """FIFO push of detached anti-collapse rows into the registered queue."""
        if not hasattr(self, "anti_queue") or sample.numel() == 0:
            return
        n = int(sample.size(0))
        qsize = int(self.anti_queue.size(0))
        ptr = int(self.anti_queue_ptr[0].item())
        fill = int(self.anti_queue_ptr[1].item())
        if n >= qsize:
            self.anti_queue.copy_(sample[-qsize:].to(self.anti_queue.dtype))
            self.anti_queue_ptr[0] = 0
            self.anti_queue_ptr[1] = qsize
            return
        end = ptr + n
        sample = sample.to(self.anti_queue.dtype)
        if end <= qsize:
            self.anti_queue[ptr:end] = sample
        else:
            first = qsize - ptr
            self.anti_queue[ptr:] = sample[:first]
            self.anti_queue[: n - first] = sample[first:]
        self.anti_queue_ptr[0] = end % qsize
        self.anti_queue_ptr[1] = min(qsize, fill + n)

    @staticmethod
    def _adjacent_cosine(z: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> float:
        """Mean cosine over valid (slot, slot+1) pairs in the same window."""
        with torch.no_grad():
            n = z.size(0)
            if n < 2:
                return float("nan")
            rows = rows.long()
            slots = slots.long()
            max_row = int(rows.max().item())
            max_slot = int(slots.max().item())
            lookup = torch.full(
                (max_row + 1, max_slot + 2), -1, dtype=torch.long, device=z.device
            )
            lookup[rows, slots] = torch.arange(n, device=z.device)
            pos = lookup[rows, slots + 1]
            anchors = (pos >= 0).nonzero(as_tuple=False).squeeze(-1)
            if anchors.numel() == 0:
                return float("nan")
            a = F.normalize(z[anchors], dim=-1)
            b = F.normalize(z[pos[anchors]], dim=-1)
            return float((a * b).sum(dim=-1).mean().item())

    def forward(
        self,
        sent_ids: torch.Tensor,  # (B, S, L)
        sent_pad: torch.Tensor,  # (B, S, L) True=pad token
        sent_valid: torch.Tensor,  # (B, S) True=real sentence
        alpha: float,
    ) -> JepaForward:
        b, s, length = sent_ids.shape
        device = sent_ids.device

        flat_ids = sent_ids.reshape(b * s, length)
        flat_pad = sent_pad.reshape(b * s, length)
        flat_valid = sent_valid.reshape(b * s)
        idx = flat_valid.nonzero(as_tuple=False).squeeze(-1)

        ids_v, pad_v = flat_ids[idx], flat_pad[idx]
        z_v, raw_v = self.encoder(ids_v, pad_v, return_raw=True)

        # Reconstruction anchor (online encoder -> decoder). Uses the FULL
        # vector so collapse is still impossible regardless of channel mode.
        logits_v = self.decoder.teacher_forcing_logits(
            z_v, ids_v, pad_v, self.bos_id
        )
        recon = reconstruction_loss(logits_v, ids_v, self.pad_id)

        # --- Multi-channel (routing) mode --------------------------------
        # Split the concept space so anti-collapse doesn't flatten prediction:
        #   corr channel  : what the predictor predicts (forward-predictable)
        #   spread channel: what anti-collapse regularizes (diversity)
        # Reconstruction still uses the full vector, so no collapse is possible.
        # Keep channel_split behind the flag for compat; repair path uses full vec.
        if self.cfg.channel_split:
            cd = self.cfg.corr_dim
            pred_z = z_v[:, :cd]          # normalized corr channel
            anti_full = raw_v[:, cd:]     # unnormalized spread channel
        else:
            pred_z = z_v
            anti_full = raw_v

        rows = idx // s  # (N,) window index of each valid row
        slots = idx % s  # (N,) slot index within window

        if self.cfg.anticollapse_scope == "window_disjoint":
            # One random valid slot per window + FIFO queue for stable cov stats.
            anti_sample = self._one_per_window(anti_full, rows)
            if hasattr(self, "anti_queue"):
                fill = int(self.anti_queue_ptr[1].item())
                queue_rows = self.anti_queue[:fill]
                anti_stat = torch.cat(
                    [anti_sample, queue_rows.to(anti_sample.dtype)], dim=0
                )
            else:
                anti_stat = anti_sample
            var = variance_loss(anti_stat, self.cfg.var_margin)
            cov = covariance_loss(anti_stat)
            std_z = mean_dim_std(anti_stat)
            if hasattr(self, "anti_queue"):
                self._push_queue(anti_sample.detach())
        else:
            var = variance_loss(anti_full, self.cfg.var_margin)
            cov = covariance_loss(anti_full)
            std_z = mean_dim_std(anti_full)

        # Neighbor-masked uniformity on the full valid batch (raw; L2 inside).
        excl = None
        if self.cfg.neighbor_mask_k > 0:
            same = rows.unsqueeze(1) == rows.unsqueeze(0)
            near = (slots.unsqueeze(1) - slots.unsqueeze(0)).abs() <= self.cfg.neighbor_mask_k
            excl = same & near
        unif = uniformity_loss(anti_full, exclude_mask=excl)

        align = pred_z.new_zeros(())
        if self.cfg.align_weight > 0:
            align = adjacent_infonce(pred_z, rows, slots, self.cfg.align_temp)

        aniso = mean_pairwise_cosine(pred_z)
        adj_cos = self._adjacent_cosine(pred_z, rows, slots)

        # Scatter prediction vectors back to a (B, S, cd) grid.
        cd = pred_z.size(-1)
        z_flat = torch.zeros(b * s, cd, device=device)
        z_flat[idx] = pred_z
        with torch.no_grad():
            t_v = self.ema(ids_v, pad_v)
            if self.cfg.channel_split:
                t_v = t_v[:, :cd]
        t_flat = torch.zeros(b * s, cd, device=device)
        t_flat[idx] = t_v
        z = z_flat.reshape(b, s, cd)
        targets = t_flat.reshape(b, s, cd)

        # Prediction: predict sentence i from sentences 0..i-1. Documents are
        # prefix-valid, so sent_valid[:, i] implies a fully-valid context.
        pred_accum = z.new_zeros(())
        cos_accum = 0.0
        count = 0
        for i in range(1, s):
            win_rows = sent_valid[:, i].nonzero(as_tuple=False).squeeze(-1)
            if win_rows.numel() == 0:
                continue
            context = z[win_rows, :i, :]
            ctx_pad = torch.zeros(win_rows.numel(), i, dtype=torch.bool, device=device)
            pred = self.predictor(context, ctx_pad)
            pred_accum = pred_accum + prediction_loss(pred, targets[win_rows, i, :]) * win_rows.numel()
            with torch.no_grad():
                cos_accum += float(
                    F.cosine_similarity(pred, targets[win_rows, i, :], dim=-1).sum().item()
                )
            count += win_rows.numel()
        pred = pred_accum / max(1, count)
        pred_cos = cos_accum / max(1, count)

        total = (
            recon.loss
            + alpha * pred
            + self.cfg.var_weight * var
            + self.cfg.cov_weight * cov
            + self.cfg.uniformity_weight * unif
            + self.cfg.align_weight * align
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
            align=align,
            adj_cos=adj_cos,
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        self.ema.update(self.encoder)
