"""Phase-1: pretrain FluWriter as a plain next-token LM (fluency)."""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from ..jepa.tokenizer import Tokenizer
from ..progress import iter_bar, log_epoch
from .data import make_phase1_loader
from .model import FluWriter


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_phase1(cfg: Config, progress: bool = True) -> tuple[FluWriter, Tokenizer]:
    torch.manual_seed(cfg.writer.seed if hasattr(cfg.writer, "seed") else 42)
    dev = _device()
    tok = Tokenizer(cfg.jepa.tokenizer)
    seq_len = cfg.writer.seq_len
    loader = make_phase1_loader(
        str(cfg.data.corpus_path), tok, seq_len, cfg.writer.batch_size
    )

    model = FluWriter(
        tok.vocab_size,
        dim=cfg.writer.dim,
        n_layers=cfg.writer.n_layers,
        n_heads=cfg.writer.n_heads,
        ff_mult=cfg.writer.ff_mult,
        max_seq_len=seq_len + 8,
        concept_dim=cfg.jepa.dim,
        use_concept=False,  # Phase 1: no conditioning
    )
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.writer.lr, weight_decay=0.01)

    # Accumulate gradients to hit an effective batch size without big VRAM.
    eff_batch = cfg.writer.batch_size * cfg.writer.gradient_acc

    db = None
    run_id = None
    start_epoch = 0
    if cfg.writer.db_path:
        from ..ckpt_db import CheckpointDB

        db = CheckpointDB(cfg.writer.db_path)
        run_id = cfg.writer.run_id or db.latest_run()
        if run_id is None:
            run_id = db.start_run(
                str(cfg.data.corpus_path), tok.name, 42,
                seq_len, cfg.writer.dim,
                {"writer": cfg.writer.__dict__},
            )
        loaded = db.load_latest_checkpoint(run_id, dev)
        if loaded is not None:
            model.load_state_dict(loaded[0])
            opt.load_state_dict(loaded[1])
            start_epoch = loaded[2] + 1
            print(f"[writer-p1] resumed from epoch {start_epoch}/{cfg.writer.epochs}")

    steps_per_epoch = max(1, cfg.writer.steps_per_epoch)
    for epoch in range(start_epoch, cfg.writer.epochs):
        model.train()
        opt.zero_grad()
        epoch_loss = 0.0
        it = iter(loader)
        n_steps = 0
        progress_iter = iter_bar(
            range(steps_per_epoch),
            desc=f"writer-p1 epoch {epoch+1}/{cfg.writer.epochs}",
            total=steps_per_epoch,
            enabled=progress,
            unit="batch",
        )
        for _ in progress_iter:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(dev), y.to(dev)
            loss = model(x, labels=y) / cfg.writer.gradient_acc
            loss.backward()
            if (n_steps + 1) % cfg.writer.gradient_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            epoch_loss += loss.item() * cfg.writer.gradient_acc
            n_steps += 1
        # Flush any leftover gradient accumulation.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        avg = epoch_loss / max(1, n_steps)
        log_epoch("writer-p1", epoch + 1, cfg.writer.epochs, {"loss": avg}, enabled=progress)
        if db is not None:
            db.record_epoch(run_id, epoch, {"loss": avg})
            db.save_checkpoint(run_id, epoch, model, opt)

    if db is not None:
        db.close()
    return model, tok
