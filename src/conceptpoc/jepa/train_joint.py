"""Curriculum trainer for the joint JEPA codec.

Phase A (recon warmup): alpha = 0, train encoder + decoder on reconstruction so
the concept space and decoder become usable before prediction pressure is added.
Phase B: ramp alpha from 0 to alpha_max while the predictor learns to forecast
the EMA/stop-grad next-vector. Variance/covariance regularization is on
throughout, and a collapse alarm watches the per-dimension std of z.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, random_split

from ..config import JepaConfig, PredictorConfig
from ..data.corpus import Document
from ..gpu_gate import GpuGate
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
    # Mean pairwise cosine of concept vectors (anisotropy; -> 1 means angular
    # collapse). The uniformity regularizer should keep this low.
    aniso: list[float] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    # Held-out prediction cosine per epoch (NaN when no val split was used).
    val_pred_cos: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_cos: float = float("-inf")
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
            "aniso": self.aniso[-1] if self.aniso else float("nan"),
            "alpha": self.alpha[-1],
            "val_pred_cos": self.val_pred_cos[-1] if self.val_pred_cos else float("nan"),
            "best_epoch": self.best_epoch,
            "best_val_cos": self.best_val_cos if self.best_epoch > 0 else float("nan"),
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
    gate: GpuGate | None = None,
) -> tuple[JointJepa, Tokenizer, JointHistory]:
    torch.manual_seed(cfg.seed)
    dev = _device(device)
    tokenizer = tokenizer or Tokenizer(cfg.tokenizer)

    dataset = JepaWindowDataset(docs, window=cfg.context_len)
    if len(dataset) == 0:
        raise ValueError("no training windows; need documents with >= 2 sentences")
    collate = make_collate(tokenizer, cfg.context_len, cfg.max_tokens)

    # Hold out a validation split to monitor prediction generalization. Skip it
    # for tiny datasets where a split would leave too little to train on.
    n_total = len(dataset)
    n_val = int(round(n_total * cfg.val_fraction))
    if n_total >= 10 and n_val >= 1:
        split_gen = torch.Generator().manual_seed(cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_total - n_val, n_val], generator=split_gen)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
        )
    else:
        train_ds, val_loader = dataset, None
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
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
    best_state: dict | None = None
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        alpha = _alpha_for_epoch(epoch, cfg)
        model.train()
        agg = {"total": 0.0, "recon": 0.0, "pred": 0.0, "acc": 0.0, "std": 0.0, "aniso": 0.0}
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
            if gate is not None:
                gate.step(model, opt)

            agg["total"] += out.total.item()
            agg["recon"] += out.recon.item()
            agg["pred"] += float(out.pred.item()) if torch.is_tensor(out.pred) else float(out.pred)
            agg["acc"] += out.recon_acc
            agg["std"] += out.std_z
            agg["aniso"] += out.aniso
            n += 1

        history.total.append(agg["total"] / n)
        history.recon.append(agg["recon"] / n)
        history.pred.append(agg["pred"] / n)
        history.recon_acc.append(agg["acc"] / n)
        history.std_z.append(agg["std"] / n)
        history.aniso.append(agg["aniso"] / n)
        history.alpha.append(alpha)

        val_cos = _validate(model, val_loader, alpha, dev) if val_loader else float("nan")
        history.val_pred_cos.append(val_cos)

        # Optional best-checkpoint selection (opt-in via patience > 0). E1b showed
        # val_pred_cos plateaus early and does not track downstream eval quality
        # for this joint objective, so selecting on it can keep an under-trained
        # decoder. Default (patience == 0) trains fully and keeps the final epoch;
        # val_pred_cos is still logged as a monitor.
        if val_loader is not None and cfg.patience > 0 and alpha > 0.0:
            if val_cos > history.best_val_cos:
                history.best_val_cos = val_cos
                history.best_epoch = epoch + 1
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        log_epoch(
            "train-joint",
            epoch + 1,
            cfg.epochs,
            {
                "alpha": alpha,
                "recon_acc": history.recon_acc[-1],
                "std_z": history.std_z[-1],
                "aniso": history.aniso[-1],
                "val_pred_cos": val_cos,
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
        if cfg.patience > 0 and alpha >= cfg.alpha_max and epochs_no_improve >= cfg.patience:
            log_stage(
                f"early stop: no val improvement for {cfg.patience} epochs "
                f"(best epoch {history.best_epoch}, val_pred_cos={history.best_val_cos:.3f}).",
                enabled=progress,
            )
            break

    # Restore the best-by-validation checkpoint so the saved model isn't the
    # (potentially over-trained) final epoch.
    if best_state is not None:
        model.load_state_dict(best_state)
        log_stage(
            f"restored best checkpoint: epoch {history.best_epoch}, "
            f"val_pred_cos={history.best_val_cos:.3f}.",
            enabled=progress,
        )

    history.collapsed = bool(history.std_z[-1] < cfg.collapse_std_threshold)
    return model, tokenizer, history


@torch.no_grad()
def _validate(model: JointJepa, val_loader: DataLoader, alpha: float, dev) -> float:
    """Mean held-out prediction cosine (a generalization monitor)."""
    model.eval()
    total = 0.0
    n = 0
    for sent_ids, sent_pad, sent_valid in val_loader:
        out = model(
            sent_ids.to(dev), sent_pad.to(dev), sent_valid.to(dev), alpha
        )
        total += out.pred_cos
        n += 1
    model.train()
    return total / max(1, n)
