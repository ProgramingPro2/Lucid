"""Fine-tune the JEPA decoder on off-manifold / predicted concept vectors.

Closes the train/inference manifold gap by scheduled sampling: mix perturbed
and diffusion-predicted concept vectors and train the decoder to reconstruct
target sentences from whichever concept is fed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import DecoderRobustConfig, JepaConfig
from ..gpu_gate import GpuGate
from ..progress import iter_bar, log_epoch
from ..predictor.diffusion import DiffusionConceptPredictor
from .dataset import JepaWindowDataset, make_collate
from .joint_model import JointJepa
from .losses import reconstruction_loss
from .tokenizer import Tokenizer


@dataclass
class DecoderRobustHistory:
    train_loss: list[float] = field(default_factory=list)
    recon_acc: list[float] = field(default_factory=list)


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scheduled_prob(epoch: int, cfg: DecoderRobustConfig) -> float:
    if cfg.epochs <= 1:
        return cfg.scheduled_sampling_end
    frac = epoch / max(1, cfg.epochs - 1)
    return cfg.scheduled_sampling_start + frac * (
        cfg.scheduled_sampling_end - cfg.scheduled_sampling_start
    )


def train_decoder_robust(
    model: JointJepa,
    tokenizer: Tokenizer,
    docs,
    jepa_cfg: JepaConfig,
    robust_cfg: DecoderRobustConfig,
    *,
    diffusion: DiffusionConceptPredictor | None = None,
    device: str = "auto",
    progress: bool = True,
    gate: GpuGate | None = None,
) -> tuple[JointJepa, DecoderRobustHistory]:
    """Decoder-only robustness pass; encoder/EMA/predictor stay frozen."""
    dev = _device(device)
    model.to(dev)
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.ema.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True

    if diffusion is not None:
        diffusion.to(dev)
        diffusion.eval()
        for p in diffusion.parameters():
            p.requires_grad = False

    dataset = JepaWindowDataset(docs, window=jepa_cfg.context_len)
    collate = make_collate(tokenizer, jepa_cfg.context_len, jepa_cfg.max_tokens)
    dl = DataLoader(
        dataset,
        batch_size=robust_cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    opt = torch.optim.AdamW(
        model.decoder.parameters(), lr=robust_cfg.lr, weight_decay=1e-4
    )
    history = DecoderRobustHistory()

    for epoch in range(robust_cfg.epochs):
        model.decoder.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        mix_p = _scheduled_prob(epoch, robust_cfg)
        batch_iter = iter_bar(
            dl,
            desc=f"decoder-robust {epoch + 1}/{robust_cfg.epochs}",
            total=len(dl),
            enabled=progress,
            unit="batch",
        )
        for sent_ids, sent_pad, sent_valid in batch_iter:
            sent_ids = sent_ids.to(dev)
            sent_pad = sent_pad.to(dev)
            sent_valid = sent_valid.to(dev)
            b, s, length = sent_ids.shape
            flat_ids = sent_ids.reshape(b * s, length)
            flat_pad = sent_pad.reshape(b * s, length)
            flat_valid = sent_valid.reshape(b * s)
            idx = flat_valid.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            ids_v, pad_v = flat_ids[idx], flat_pad[idx]
            with torch.no_grad():
                z_real, _ = model.encoder(ids_v, pad_v, return_raw=True)
            noise = torch.randn_like(z_real) * robust_cfg.perturb_sigma
            z = F.normalize(z_real + noise, dim=-1)
            if diffusion is not None and mix_p > 0:
                z_diff = diffusion.sample(
                    z_real.unsqueeze(1),
                    sampling_steps=robust_cfg.sampling_steps,
                )
                use_pred = (torch.rand(z_real.size(0), 1, device=dev) < mix_p).float()
                z = F.normalize(
                    use_pred * z_diff + (1.0 - use_pred) * z, dim=-1
                )

            logits = model.decoder.teacher_forcing_logits(
                z, ids_v, pad_v, model.bos_id
            )
            recon = reconstruction_loss(logits, ids_v, model.pad_id)
            opt.zero_grad()
            recon.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            opt.step()
            if gate is not None:
                gate.step(model, opt)
            epoch_loss += recon.loss.item()
            epoch_acc += recon.accuracy
            n_batches += 1

        history.train_loss.append(epoch_loss / max(1, n_batches))
        history.recon_acc.append(epoch_acc / max(1, n_batches))
        log_epoch(
            "decoder-robust",
            epoch + 1,
            robust_cfg.epochs,
            {
                "train_loss": history.train_loss[-1],
                "recon_acc": history.recon_acc[-1],
                "mix_p": mix_p,
            },
            enabled=progress,
        )

    return model, history
