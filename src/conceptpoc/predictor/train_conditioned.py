"""Conditioned predictor training (E10).

Teacher-forced next-vector regression on RESPONSE positions only, over a frozen
JEPA codec (the concept pairs are encoded once and cached as an
``InstructionConceptDataset``). The same ``predictor_loss`` (MSE + cosine) used
by the continuation trainer is applied, restricted to response positions.

The trainer is GPU-gate aware (``gate.step`` per batch) like ``train_predictor``
and ``train_joint`` so long conditioned runs share the GPU politely.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, random_split

from ..config import PredictorConfig, TrainConfig
from ..data.instruction import InstructionConceptDataset, collate_instructions
from ..gpu_gate import GpuGate
from ..progress import iter_bar, log_epoch
from .model import ConceptPredictor, predictor_loss


@dataclass
class ConditionedHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_cosine: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_cosine: float = float("-inf")


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_loss(model, batch, predictor_cfg, dev):
    seq, pad_mask, segment_ids, targets, loss_mask, theme = batch
    seq, pad_mask = seq.to(dev), pad_mask.to(dev)
    segment_ids, targets, loss_mask = (
        segment_ids.to(dev),
        targets.to(dev),
        loss_mask.to(dev),
    )
    theme = theme.to(dev) if theme is not None else None
    preds = model(
        seq,
        pad_mask,
        segment_ids=segment_ids,
        theme_vec=theme,
        return_sequence=True,
    )
    sel = loss_mask.reshape(-1)
    pred_flat = preds.reshape(-1, preds.size(-1))[sel]
    tgt_flat = targets.reshape(-1, targets.size(-1))[sel]
    return predictor_loss(
        pred_flat, tgt_flat, predictor_cfg.mse_weight, predictor_cfg.cosine_weight
    )


def train_conditioned(
    dataset: InstructionConceptDataset,
    predictor_cfg: PredictorConfig,
    train_cfg: TrainConfig,
    device: str = "auto",
    model: ConceptPredictor | None = None,
    progress: bool = True,
    gate: GpuGate | None = None,
) -> tuple[ConceptPredictor, ConditionedHistory]:
    torch.manual_seed(train_cfg.seed)
    dev = _device(device)

    n_val = max(1, int(len(dataset) * train_cfg.val_fraction))
    n_val = min(n_val, max(0, len(dataset) - 1))
    gen = torch.Generator().manual_seed(train_cfg.seed)
    if n_val >= 1 and len(dataset) - n_val >= 1:
        train_ds, val_ds = random_split(
            dataset, [len(dataset) - n_val, n_val], generator=gen
        )
    else:
        train_ds, val_ds = dataset, None

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_instructions,
    )
    val_dl = (
        DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_instructions,
        )
        if val_ds is not None
        else None
    )

    if model is None:
        # Conditioning is required for this trainer; enable the segment embedding
        # (and theme projection when the data carries themes).
        predictor_cfg.use_segment_embedding = True
        predictor_cfg.use_theme = predictor_cfg.use_theme or dataset.has_theme
        model = ConceptPredictor(dataset.dim, predictor_cfg)
    model.to(dev)

    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    total_steps = max(1, train_cfg.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=train_cfg.lr,
        total_steps=total_steps,
        pct_start=min(0.3, max(1e-3, train_cfg.warmup_steps / total_steps)),
    )

    history = ConditionedHistory()
    best_state: dict | None = None
    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in iter_bar(
            train_dl,
            desc=f"epoch {epoch + 1}/{train_cfg.epochs} train",
            total=len(train_dl),
            enabled=progress,
            unit="batch",
        ):
            opt.zero_grad()
            loss = _masked_loss(model, batch, predictor_cfg, dev)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            sched.step()
            if gate is not None:
                gate.step(model, opt)
            epoch_loss += loss.total.item()
        history.train_loss.append(epoch_loss / max(1, len(train_dl)))

        val_loss, val_cos = _evaluate(model, val_dl, predictor_cfg, dev, progress)
        history.val_loss.append(val_loss)
        history.val_cosine.append(val_cos)
        if val_cos > history.best_val_cosine:
            history.best_val_cosine = val_cos
            history.best_epoch = epoch + 1
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )

        log_epoch(
            "train-conditioned",
            epoch + 1,
            train_cfg.epochs,
            {
                "train_loss": history.train_loss[-1],
                "val_loss": val_loss,
                "val_cosine": val_cos,
            },
            enabled=progress,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _evaluate(model, dl, predictor_cfg, dev, progress) -> tuple[float, float]:
    if dl is None:
        return float("nan"), float("nan")
    model.eval()
    total, cos, n = 0.0, 0.0, 0
    for batch in iter_bar(
        dl, desc="val", total=len(dl), enabled=progress and len(dl) > 1, unit="batch"
    ):
        loss = _masked_loss(model, batch, predictor_cfg, dev)
        total += loss.total.item()
        cos += 1.0 - loss.cosine.item()
        n += 1
    model.train()
    return total / max(1, n), cos / max(1, n)
