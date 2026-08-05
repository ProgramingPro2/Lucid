"""Train the latent trajectory planner against real next-K concept vectors.

Loss: cosine (or MSE on normalized vectors) between predicted trajectory and
the ground-truth next-K vectors, weighted toward early steps. Frozen JEPA
encoder provides the targets; only the planner is trained, so it learns the
multi-step forward-predictable objective.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..ckpt_db import CheckpointDB
from ..codec.jepa_codec import JepaCodec
from ..config import Config
from ..jepa.tokenizer import Tokenizer
from ..persistence import load_joint
from ..progress import iter_bar, log_epoch
from .data import make_trajectory_loader
from .planner import TrajectoryPlanner


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _traj_loss(pred: torch.Tensor, target: torch.Tensor, k_weight: float = 0.6) -> torch.Tensor:
    """Cosine loss, weighting earlier trajectory steps a bit more (nearer=more predictable)."""
    # normalize per step
    p = torch.nn.functional.normalize(pred, dim=-1)
    t = torch.nn.functional.normalize(target, dim=-1)
    cos = (p * t).sum(dim=-1)  # (B, K)
    w = torch.linspace(1.0, k_weight, cos.size(1), device=cos.device)
    return (1.0 - cos.mean(dim=0)).mul(w).mean()


def train_planner(cfg: Config, progress: bool = True) -> TrajectoryPlanner:
    t = cfg.trajectory
    torch.manual_seed(t.seed)
    dev = _device()

    # Frozen JEPA reasoner -> concept targets.
    tok = Tokenizer(cfg.jepa.tokenizer)
    jepa_ckpt = cfg.writer.jepa_checkpoint or cfg.codec.jepa_checkpoint
    model, _ = load_joint(jepa_ckpt)
    codec = JepaCodec(model, tok, cfg.jepa.max_tokens, device=dev)

    C, K, dim = t.C, t.K, cfg.jepa.dim
    planner = TrajectoryPlanner(
        dim, C, K, t.n_layers, t.n_heads, t.latent_iters,
    ).to(dev)
    opt = torch.optim.AdamW(planner.parameters(), lr=t.lr, weight_decay=0.01)
    print(f"[traj] planner params: {planner.num_parameters():,}")

    db = None
    run_id = None
    start_epoch = 0
    if t.db_path:
        db = CheckpointDB(Path(t.db_path))
        run_id = t.run_id or db.latest_run()
        if run_id is None:
            run_id = db.start_run(
                str(cfg.data.corpus_path), tok.name, t.seed, cfg.jepa.max_tokens, C,
                {"trajectory": t.__dict__},
            )
        loaded = db.load_latest_checkpoint(run_id, dev)
        if loaded is not None:
            try:
                planner.load_state_dict(loaded[0])
                opt.load_state_dict(loaded[1])
                start_epoch = loaded[2] + 1
                print(f"[traj] resumed from epoch {start_epoch}/{t.epochs}")
            except Exception as e:  # noqa: BLE001 - state mismatch, start fresh
                print(f"[traj] checkpoint mismatch ({e}); starting fresh")

    loader = make_trajectory_loader(
        str(cfg.data.corpus_path), codec, C=C, K=K, batch_size=t.batch_size,
        streaming=True, max_docs=t.max_docs,
    )

    for epoch in range(start_epoch, t.epochs):
        planner.train()
        epoch_loss = 0.0
        steps = 0
        it = iter(loader)
        bar = iter_bar(
            range(2000), desc=f"traj epoch {epoch+1}/{t.epochs}",
            total=2000, enabled=progress, unit="batch",
        )
        for _ in bar:
            try:
                ctx, tgt = next(it)
            except StopIteration:
                it = iter(loader)
                ctx, tgt = next(it)
            ctx, tgt = ctx.to(dev), tgt.to(dev)
            pred = planner(ctx)
            loss = _traj_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            steps += 1
        avg = epoch_loss / max(1, steps)
        log_epoch("traj", epoch + 1, t.epochs, {"loss": avg}, enabled=progress)
        if db is not None:
            db.record_epoch(run_id, epoch, {"loss": avg})
            db.save_checkpoint(run_id, epoch, planner, opt)

    if db is not None:
        db.close()
    return planner
