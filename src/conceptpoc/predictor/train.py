"""Training loop for the concept predictor."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, random_split

from ..config import PredictorConfig, TrainConfig
from ..data.dataset import ConceptSequenceDataset, collate_sequences
from .model import ConceptPredictor, predictor_loss


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_cosine: list[float] = field(default_factory=list)


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_predictor(
    dataset: ConceptSequenceDataset,
    predictor_cfg: PredictorConfig,
    train_cfg: TrainConfig,
    device: str = "auto",
    model: ConceptPredictor | None = None,
) -> tuple[ConceptPredictor, TrainHistory]:
    torch.manual_seed(train_cfg.seed)
    dev = _device(device)

    n_val = max(1, int(len(dataset) * train_cfg.val_fraction))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(train_cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    if model is None:
        model = ConceptPredictor(dataset.dim, predictor_cfg)
    model.to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    total_steps = max(1, train_cfg.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=train_cfg.lr, total_steps=total_steps,
        pct_start=min(0.3, train_cfg.warmup_steps / total_steps),
    )

    history = TrainHistory()
    for _ in range(train_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for context, mask, target in train_dl:
            context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
            opt.zero_grad()
            pred = model(context, mask)
            loss = predictor_loss(
                pred, target, predictor_cfg.mse_weight, predictor_cfg.cosine_weight
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            sched.step()
            epoch_loss += loss.total.item()
        history.train_loss.append(epoch_loss / max(1, len(train_dl)))

        val_loss, val_cos = _evaluate(model, val_dl, predictor_cfg, dev)
        history.val_loss.append(val_loss)
        history.val_cosine.append(val_cos)

    return model, history


@torch.no_grad()
def _evaluate(model, dl, predictor_cfg, dev) -> tuple[float, float]:
    model.eval()
    total, cos, n = 0.0, 0.0, 0
    for context, mask, target in dl:
        context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
        pred = model(context, mask)
        loss = predictor_loss(
            pred, target, predictor_cfg.mse_weight, predictor_cfg.cosine_weight
        )
        total += loss.total.item()
        cos += (1.0 - loss.cosine.item())
        n += 1
    return total / max(1, n), cos / max(1, n)
