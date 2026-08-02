"""Training loop for the diffusion concept predictor."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ..config import PredictorConfig, TrainConfig
from ..data.dataset import ConceptSequenceDataset, collate_sequences
from ..gpu_gate import GpuGate
from ..progress import iter_bar, log_epoch
from .diffusion import DiffusionConceptPredictor


@dataclass
class DiffusionTrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_sample_cosine: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_sample_cosine: float = float("-inf")


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _sample_cosine_metric(
    model: DiffusionConceptPredictor,
    dl: DataLoader,
    dev: torch.device,
    *,
    sampling_steps: int = 10,
) -> float:
    """Mean cosine between one sampled next vector and the true target."""
    model.eval()
    cos_sum, n = 0.0, 0
    for context, mask, target in dl:
        context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
        pred = model.sample(context, mask, sampling_steps=sampling_steps)
        cos_sum += F.cosine_similarity(pred, F.normalize(target, dim=-1), dim=-1).sum().item()
        n += pred.size(0)
    return cos_sum / max(1, n)


def train_diffusion_predictor(
    dataset: ConceptSequenceDataset,
    predictor_cfg: PredictorConfig,
    train_cfg: TrainConfig,
    device: str = "auto",
    model: DiffusionConceptPredictor | None = None,
    progress: bool = True,
    gate: GpuGate | None = None,
) -> tuple[DiffusionConceptPredictor, DiffusionTrainHistory]:
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
        model = DiffusionConceptPredictor(dataset.dim, predictor_cfg)
    model.to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    total_steps = max(1, train_cfg.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=train_cfg.lr,
        total_steps=total_steps,
        pct_start=min(0.3, train_cfg.warmup_steps / total_steps),
    )

    history = DiffusionTrainHistory()
    best_state: dict | None = None
    val_steps = min(10, predictor_cfg.diffusion.sampling_steps)

    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        batch_iter = iter_bar(
            train_dl,
            desc=f"epoch {epoch + 1}/{train_cfg.epochs} train",
            total=len(train_dl),
            enabled=progress,
            unit="batch",
        )
        for context, mask, target in batch_iter:
            context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
            opt.zero_grad()
            loss = model.diffusion_loss(context, target, mask)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            sched.step()
            if gate is not None:
                gate.step(model, opt)
            epoch_loss += loss.total.item()
        history.train_loss.append(epoch_loss / max(1, len(train_dl)))

        val_loss = _evaluate_loss(model, val_dl, dev, progress=progress)
        val_cos = _sample_cosine_metric(model, val_dl, dev, sampling_steps=val_steps)
        history.val_loss.append(val_loss)
        history.val_sample_cosine.append(val_cos)

        if val_cos > history.best_val_sample_cosine:
            history.best_val_sample_cosine = val_cos
            history.best_epoch = epoch + 1
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )

        log_epoch(
            "train-diffusion",
            epoch + 1,
            train_cfg.epochs,
            {
                "train_loss": history.train_loss[-1],
                "val_loss": val_loss,
                "val_sample_cosine": val_cos,
            },
            enabled=progress,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def _evaluate_loss(
    model: DiffusionConceptPredictor,
    dl: DataLoader,
    dev: torch.device,
    *,
    progress: bool = True,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for context, mask, target in iter_bar(
        dl,
        desc="val",
        total=len(dl),
        enabled=progress and len(dl) > 1,
        unit="batch",
    ):
        context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
        loss = model.diffusion_loss(context, target, mask)
        total += loss.total.item()
        n += 1
    return total / max(1, n)
