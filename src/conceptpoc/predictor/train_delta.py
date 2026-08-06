"""Training loop for the DeltaPredictor (SQLite-resumable, GPU-gated)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ..config import PredictorConfig, TrainConfig
from ..data.cache import CachedCorpus
from ..data.delta_dataset import DeltaSequenceDataset, collate_deltas
from ..gpu_gate import GpuGate
from ..progress import iter_bar, log_epoch
from .delta_model import DeltaPredictor, delta_cosine, delta_loss


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_delta_predictor(
    corpus: CachedCorpus,
    predictor_cfg: PredictorConfig,
    train_cfg: TrainConfig,
    device: str = "auto",
    progress: bool = True,
    gate: GpuGate | None = None,
    db_path: str | Path | None = None,
    run_id: int = 0,
    context_len: int = 16,
) -> tuple[DeltaPredictor, dict]:
    torch.manual_seed(train_cfg.seed)
    dev = _device(device)

    dataset = DeltaSequenceDataset(corpus, context_len=context_len)
    if len(dataset) == 0:
        raise ValueError("delta dataset empty; need docs with >=4 sentences")
    n_val = max(1, int(len(dataset) * train_cfg.val_fraction))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(train_cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                          collate_fn=collate_deltas)
    val_dl = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                        collate_fn=collate_deltas)

    model = DeltaPredictor(dataset.dim, predictor_cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    total_steps = max(1, train_cfg.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=train_cfg.lr, total_steps=total_steps,
        pct_start=min(0.3, train_cfg.warmup_steps / total_steps),
    )

    # --- SQLite resume ---
    db = None
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_cosine": []}
    if db_path is not None:
        from ..ckpt_db import CheckpointDB

        db = CheckpointDB(Path(db_path))
        db_run = run_id or db.latest_run()
        if db_run is None:
            db_run = db.start_run(
                str(corpus.sentences[0]) if corpus.sentences else "delta",
                "deltas", train_cfg.seed, dataset.dim, context_len,
                {"predictor": predictor_cfg.__dict__, "train": train_cfg.__dict__},
            )
        else:
            run_id = db_run
        loaded = db.load_latest_checkpoint(db_run, dev)
        if loaded is not None:
            try:
                model.load_state_dict(loaded[0])
                opt.load_state_dict(loaded[1])
                start_epoch = loaded[2] + 1
                print(f"[delta] resumed from epoch {start_epoch}/{train_cfg.epochs}")
            except Exception as e:  # noqa: BLE001
                print(f"[delta] checkpoint mismatch ({e}); start fresh")

    best_cos = -float("inf")
    best_state: dict | None = None
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        eloss = 0.0
        for context, mask, target in iter_bar(
            train_dl, desc=f"delta epoch {epoch+1}/{train_cfg.epochs} train",
            total=len(train_dl), enabled=progress, unit="batch",
        ):
            context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
            opt.zero_grad()
            pred = model(context, mask)
            loss = delta_loss(pred, target)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            sched.step()
            if gate is not None:
                gate.step(model, opt)
            eloss += loss.total.item()
        train_loss = eloss / max(1, len(train_dl))

        v_loss, v_cos = _eval_delta(model, val_dl, dev, progress=progress)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(v_loss)
        history["val_cosine"].append(v_cos)
        if v_cos > best_cos:
            best_cos = v_cos
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        log_epoch("delta", epoch + 1, train_cfg.epochs,
                  {"train_loss": train_loss, "val_loss": v_loss, "val_cosine": v_cos},
                  enabled=progress)
        if db is not None:
            db.record_epoch(db_run, epoch,
                            {"train_loss": train_loss, "val_loss": v_loss,
                             "val_cosine": v_cos})
            db.save_checkpoint(db_run, epoch, model, opt)

    if best_state is not None:
        model.load_state_dict(best_state)
        history["best_val_cosine"] = best_cos
    if db is not None:
        db.close()
    return model, history


@torch.no_grad()
def _eval_delta(model, dl, dev, *, progress: bool = True):
    model.eval()
    total, cos, n = 0.0, 0.0, 0
    for context, mask, target in iter_bar(
        dl, desc="val", total=len(dl), enabled=progress and len(dl) > 1, unit="batch",
    ):
        context, mask, target = context.to(dev), mask.to(dev), target.to(dev)
        pred = model(context, mask)
        total += delta_loss(pred, target).total.item()
        cos += float(delta_cosine(pred, target).item())
        n += 1
    model.train()
    return total / max(1, n), cos / max(1, n)
