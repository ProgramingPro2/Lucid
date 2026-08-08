"""Density-head trainer for per-sentence JEPA (vMF / MDN).

CLI:
  python -m experiments.sentence_pred.train_head \\
    --corpus <dir> --head {vmf,mdn} --out <path> \\
    [--epochs N] [--context-len 32] [--device auto] [--batch 256]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Respect expandable CUDA allocator segments when the caller sets the env var
# (or leave unset). Do not override an existing value.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from conceptpoc.config import PredictorConfig  # noqa: E402

from experiments.sentence_pred.data import (  # noqa: E402
    SentenceCorpus,
    SentenceSequenceDataset,
    collate_sentence_batch,
    load_sentence_corpus,
)
from experiments.sentence_pred.heads import SentenceDensityPredictor  # noqa: E402

DEFAULT_PREDICTOR_CFG = PredictorConfig(
    d_model=512,
    n_layers=6,
    n_heads=8,
    ff_mult=4,
    dropout=0.1,
    max_seq_len=34,
)


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _asdict_cfg(cfg: PredictorConfig | dict) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)


def _to_predictor_cfg(cfg: PredictorConfig | dict | None) -> PredictorConfig:
    if cfg is None:
        return PredictorConfig(**asdict(DEFAULT_PREDICTOR_CFG))
    if isinstance(cfg, PredictorConfig):
        return cfg
    return PredictorConfig(**dict(cfg))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:  # noqa: BLE001
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _maybe_gpu_gate(dev: torch.device, progress: bool = True):
    if dev.type != "cuda":
        return None
    try:
        from conceptpoc.config import GpuGateConfig
        from conceptpoc.gpu_gate import GpuGate
    except ImportError:
        return None
    gate = GpuGate(GpuGateConfig(enabled=True), dev, progress=progress)
    return gate if gate.active else None


@torch.no_grad()
def _eval_nll(
    model: SentenceDensityPredictor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for context, pad_mask, target_seq, valid, _doc_ids in loader:
        context = context.to(device)
        pad_mask = pad_mask.to(device)
        target_seq = target_seq.to(device)
        valid = valid.to(device)
        loss = model.loss(context, pad_mask, target_seq, valid)
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def train_density_head(
    corpus: SentenceCorpus,
    predictor_cfg: PredictorConfig | dict | None = None,
    head: str = "vmf",
    K: int = 8,
    out: str | Path = "artifacts/sentence_pred/density_head.pt",
    seed: int = 42,
    device: str = "auto",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    patience: int = 5,
    context_len: int = 32,
    log_every: int = 50,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    progress: bool = True,
    use_gpu_gate: bool = True,
) -> tuple[SentenceDensityPredictor, dict[str, Any]]:
    """Train a SentenceDensityPredictor with pure teacher-forced NLL.

    Returns ``(model, history)`` where history has ``train_nll`` / ``val_nll``
    lists. Best-by-val-NLL weights are restored and written to ``out``.
    """
    if head not in ("vmf", "mdn"):
        raise ValueError(f"head must be 'vmf' or 'mdn', got {head!r}")

    _seed_everything(seed)
    dev = _device(device)
    pcfg = _to_predictor_cfg(predictor_cfg)
    concept_dim = int(corpus.sent_vecs.shape[-1])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stride = max(1, context_len // 2)
    train_ds = SentenceSequenceDataset(
        corpus,
        corpus.train_doc_ids,
        context_len=context_len,
        stride=stride,
        eval_mode=False,
    )
    val_ds = SentenceSequenceDataset(
        corpus,
        corpus.held_doc_ids,
        context_len=context_len,
        stride=context_len,
        eval_mode=True,
    )
    if len(train_ds) == 0:
        raise ValueError(
            "empty train SentenceSequenceDataset; need docs with "
            f">= context_len+1 ({context_len + 1}) sentences"
        )
    if len(val_ds) == 0:
        raise ValueError(
            "empty val SentenceSequenceDataset; need held docs with "
            f">= context_len+1 ({context_len + 1}) sentences"
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sentence_batch,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sentence_batch,
    )

    model = SentenceDensityPredictor(
        concept_dim=concept_dim,
        predictor_cfg=pcfg,
        head=head,
        K=K,
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(train_dl)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    gate = _maybe_gpu_gate(dev, progress=progress) if use_gpu_gate else None

    history: dict[str, Any] = {
        "train_nll": [],
        "val_nll": [],
        "best_epoch": -1,
        "best_val_nll": float("inf"),
    }
    best_state: dict | None = None
    epochs_trained = 0
    stall = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for step, (context, pad_mask, target_seq, valid, _doc_ids) in enumerate(
            train_dl
        ):
            context = context.to(dev)
            pad_mask = pad_mask.to(dev)
            target_seq = target_seq.to(dev)
            valid = valid.to(dev)

            opt.zero_grad(set_to_none=True)
            # Pure NLL — no auxiliary cosine term.
            loss = model.loss(context, pad_mask, target_seq, valid).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            if gate is not None:
                gate.step(model, opt)

            running += float(loss.item())
            n_batches += 1
            if progress and log_every > 0 and (step + 1) % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step + 1}/{len(train_dl)} "
                    f"nll={loss.item():.4f}",
                    flush=True,
                )

        train_nll = running / max(1, n_batches)
        val_nll = _eval_nll(model, val_dl, dev)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        epochs_trained = epoch + 1

        if progress:
            print(
                f"epoch {epoch + 1}/{epochs}  train_nll={train_nll:.4f}  "
                f"val_nll={val_nll:.4f}",
                flush=True,
            )

        if val_nll < history["best_val_nll"] - 1e-8:
            history["best_val_nll"] = val_nll
            history["best_epoch"] = epochs_trained
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            stall = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "predictor_cfg": _asdict_cfg(pcfg),
                    "head": head,
                    "head_type": "density",
                    "concept_dim": concept_dim,
                    "K": int(K),
                    "history": {
                        "train_nll": list(history["train_nll"]),
                        "val_nll": list(history["val_nll"]),
                    },
                    "epochs_trained": epochs_trained,
                },
                out,
            )
        else:
            stall += 1
            if stall >= patience:
                if progress:
                    print(
                        f"early stop at epoch {epochs_trained} "
                        f"(patience={patience})",
                        flush=True,
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # No improvement path still write a checkpoint.
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "predictor_cfg": _asdict_cfg(pcfg),
                "head": head,
                "head_type": "density",
                "concept_dim": concept_dim,
                "K": int(K),
                "history": {
                    "train_nll": list(history["train_nll"]),
                    "val_nll": list(history["val_nll"]),
                },
                "epochs_trained": epochs_trained,
            },
            out,
        )

    # Ensure final checkpoint carries full history (best weights already saved).
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    ckpt["history"] = {
        "train_nll": list(history["train_nll"]),
        "val_nll": list(history["val_nll"]),
    }
    ckpt["epochs_trained"] = epochs_trained
    if best_state is not None:
        ckpt["state_dict"] = best_state
    torch.save(ckpt, out)

    return model, history


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Train per-sentence density head (vMF / MDN)"
    )
    p.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Directory from build_sentence_corpus (sent_vecs + split_manifest)",
    )
    p.add_argument("--head", choices=("vmf", "mdn"), default="vmf")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--context-len", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-gpu-gate", action="store_true")
    args = p.parse_args(argv)

    corpus = load_sentence_corpus(args.corpus)
    train_density_head(
        corpus,
        predictor_cfg=DEFAULT_PREDICTOR_CFG,
        head=args.head,
        K=args.K,
        out=args.out,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
        context_len=args.context_len,
        log_every=args.log_every,
        use_gpu_gate=not args.no_gpu_gate,
    )


if __name__ == "__main__":
    main()
