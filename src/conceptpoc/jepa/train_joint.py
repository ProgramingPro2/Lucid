"""Curriculum trainer for the joint JEPA codec.

Phase A (recon warmup): alpha = 0, train encoder + decoder on reconstruction so
the concept space and decoder become usable before prediction pressure is added.
Phase B: ramp alpha from 0 to alpha_max while the predictor learns to forecast
the EMA/stop-grad next-vector. Variance/covariance regularization is on
throughout, and a collapse alarm watches the per-dimension std of z.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from ..config import JepaConfig, PredictorConfig
from ..data.corpus import Document
from ..progress import iter_bar, log_epoch, log_stage
from .dataset import JepaWindowDataset, make_collate
from .joint_model import JointJepa
from .tokenizer import Tokenizer


@dataclass
class JointHistory:
    total: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    pred: list[float] = field(default_factory=list)
    recon_acc: list[float] = field(default_factory=list)
    std_z: list[float] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    collapsed: bool = False

    def latest(self) -> dict:
        if not self.total:
            return {}
        return {
            "total": self.total[-1],
            "recon": self.recon[-1],
            "pred": self.pred[-1],
            "recon_acc": self.recon_acc[-1],
            "std_z": self.std_z[-1],
            "alpha": self.alpha[-1],
            "collapsed": self.collapsed,
        }


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _alpha_for_epoch(epoch: int, cfg: JepaConfig) -> float:
    if epoch < cfg.recon_warmup_epochs:
        return 0.0
    ramp_step = epoch - cfg.recon_warmup_epochs
    if cfg.alpha_warmup_epochs <= 0:
        return cfg.alpha_max
    frac = min(1.0, (ramp_step + 1) / cfg.alpha_warmup_epochs)
    return cfg.alpha_max * frac


def train_joint(
    docs: list[Document],
    cfg: JepaConfig,
    predictor_cfg: PredictorConfig,
    device: str = "auto",
    tokenizer: Tokenizer | None = None,
    progress: bool = True,
) -> tuple[JointJepa, Tokenizer, JointHistory]:
    torch.manual_seed(cfg.seed)
    dev = _device(device)
    tokenizer = tokenizer or Tokenizer(cfg.tokenizer)

    dataset = JepaWindowDataset(docs, window=cfg.context_len)
    if len(dataset) == 0:
        raise ValueError("no training windows; need documents with >= 2 sentences")
    collate = make_collate(tokenizer, cfg.context_len, cfg.max_tokens)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )

    model = JointJepa(
        tokenizer.vocab_size, tokenizer.pad_id, tokenizer.bos_id, cfg, predictor_cfg
    ).to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = JointHistory()
    for epoch in range(cfg.epochs):
        alpha = _alpha_for_epoch(epoch, cfg)
        model.train()
        agg = {"total": 0.0, "recon": 0.0, "pred": 0.0, "acc": 0.0, "std": 0.0}
        n = 0
        batch_iter = iter_bar(
            loader,
            desc=f"epoch {epoch + 1}/{cfg.epochs}",
            total=len(loader),
            enabled=progress,
            unit="batch",
        )
        for sent_ids, sent_pad, sent_valid in batch_iter:
            sent_ids = sent_ids.to(dev)
            sent_pad = sent_pad.to(dev)
            sent_valid = sent_valid.to(dev)

            opt.zero_grad()
            out = model(sent_ids, sent_pad, sent_valid, alpha)
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            model.update_ema()

            agg["total"] += out.total.item()
            agg["recon"] += out.recon.item()
            agg["pred"] += float(out.pred.item()) if torch.is_tensor(out.pred) else float(out.pred)
            agg["acc"] += out.recon_acc
            agg["std"] += out.std_z
            n += 1

        history.total.append(agg["total"] / n)
        history.recon.append(agg["recon"] / n)
        history.pred.append(agg["pred"] / n)
        history.recon_acc.append(agg["acc"] / n)
        history.std_z.append(agg["std"] / n)
        history.alpha.append(alpha)

        log_epoch(
            "train-joint",
            epoch + 1,
            cfg.epochs,
            {
                "alpha": alpha,
                "recon_acc": history.recon_acc[-1],
                "std_z": history.std_z[-1],
                "total": history.total[-1],
            },
            enabled=progress,
        )
        if progress and history.std_z[-1] < cfg.collapse_std_threshold:
            log_stage(
                f"WARNING: std_z={history.std_z[-1]:.3f} below collapse threshold "
                f"({cfg.collapse_std_threshold}); tune var_weight, alpha_max, or "
                "recon_warmup_epochs.",
                enabled=True,
            )

    history.collapsed = bool(history.std_z[-1] < cfg.collapse_std_threshold)
    return model, tokenizer, history
