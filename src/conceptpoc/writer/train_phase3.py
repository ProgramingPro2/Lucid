"""Phase-3: paragraph-committed writer — pooled G-block concept + prefix chaining.

Fine-tunes the two-channel FiLM writer so a pooled paragraph centroid is
in-distribution for the FiLM channel. For each G=5 sentence block:

  paragraph_concept = L2-normalize(mean of the G sentence vectors)
  for j in 0..G-1:
      prefix = tokens of sentence j-1 (or doc-opening if none)
      concept = paragraph_concept   # frozen for the whole paragraph
      target  = tokens of sentence j

Drop arms match phase-2.5 (p=0.10 prefix drop; p=0.10 zero concept; never both).
Trainable recipe matches phase-2.5: concept_proj + all FiLM + top-3 blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Allow `python -m conceptpoc.writer.train_phase3` and direct script runs.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.ckpt_db import CheckpointDB  # noqa: E402
from conceptpoc.config import Config  # noqa: E402
from conceptpoc.gpu_gate import GpuGate  # noqa: E402
from conceptpoc.jepa.tokenizer import Tokenizer  # noqa: E402
from conceptpoc.persistence import load_writer, save_writer  # noqa: E402
from conceptpoc.progress import iter_bar, log_epoch  # noqa: E402
from conceptpoc.writer.model import FluWriter  # noqa: E402
from conceptpoc.writer.train_phase2 import (  # noqa: E402
    DROP_P,
    PREFIX_MAX,
    SEQ_CAP,
    TARGET_MAX,
    apply_drop_arms,
    make_collate,
)

G_DEFAULT = 5
DEFAULT_CORPUS = "artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1"
DEFAULT_P25 = "artifacts/writer_p25/writer_two_channel.pt"
DEFAULT_OUT = "artifacts/writer_p3/writer_paragraph.pt"


def pool_paragraph_concept(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize(mean of G sentence vectors) -> (512,) float32."""
    x = np.asarray(vecs, dtype=np.float32)
    if x.ndim == 1:
        c = x
    else:
        c = x.mean(axis=0)
    n = float(np.linalg.norm(c)) + eps
    return (c / n).astype(np.float32)


class ParagraphBlockDataset(Dataset):
    """Within-doc G-sentence blocks -> (prefix, target, paragraph_concept, has_prefix).

    Yields one training example per (doc, block_start, j) with j in 0..G-1.
    ``paragraph_concept`` is the pooled centroid of the whole block (same for all j).
    Paragraphs never cross document boundaries. Iterates ``doc_ids`` only.
    """

    def __init__(
        self,
        corpus,  # SentenceCorpus
        tokenizer: Tokenizer,
        doc_ids: np.ndarray | None = None,
        g: int = G_DEFAULT,
        max_docs: int | None = None,
        stride: int = 1,
    ) -> None:
        self.tok = tokenizer
        self.g = int(g)
        self.stride = max(1, int(stride))
        ids = (
            np.asarray(doc_ids, dtype=np.int64)
            if doc_ids is not None
            else np.asarray(corpus.train_doc_ids, dtype=np.int64)
        )
        if max_docs is not None:
            ids = ids[: int(max_docs)]

        # Pre-tokenize all sentences once (shared across blocks).
        sent_ids: list[torch.Tensor] = []
        for s in corpus.sentences:
            ids_tok = tokenizer.encode(s)[:TARGET_MAX]
            sent_ids.append(torch.tensor(ids_tok, dtype=torch.long))

        # items: (prefix_ids, target_ids, concept, prefix_present)
        self.items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
        for doc_id in ids.tolist():
            a, b = corpus.doc_slice(int(doc_id))
            n = b - a
            if n < self.g:
                continue
            for block_start in range(0, n - self.g + 1, self.stride):
                lo = a + block_start
                hi = lo + self.g
                concept = torch.from_numpy(
                    pool_paragraph_concept(corpus.sent_vecs[lo:hi])
                )
                for j in range(self.g):
                    gi = lo + j
                    local_i = block_start + j  # position within the doc
                    target_ids = sent_ids[gi][:TARGET_MAX]
                    if local_i == 0:
                        prev_ids = torch.zeros(0, dtype=torch.long)
                        prefix_present = False
                    else:
                        prev_ids = sent_ids[gi - 1][:PREFIX_MAX]
                        prefix_present = True
                    self.items.append(
                        (prev_ids, target_ids, concept.clone(), prefix_present)
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        prev_ids, target_ids, concept, prefix_present = self.items[idx]
        prev_ids, concept, prefix_present = apply_drop_arms(
            idx, prev_ids, concept, prefix_present, drop_p=DROP_P
        )
        return prev_ids, target_ids, concept, prefix_present


def _set_trainable(writer: FluWriter, n_layers: int) -> list[str]:
    """concept_proj + all FiLM + top-3 blocks (same recipe as phase-2.5)."""
    trainable = ["concept_proj"]
    trainable += [f"blocks.{i}." for i in range(n_layers - 3, n_layers)]
    trainable += [f"blocks.{i}.film" for i in range(n_layers)]
    for name, p in writer.named_parameters():
        p.requires_grad = any(t in name for t in trainable)
    return trainable


@torch.no_grad()
def _eval_loss(
    writer: FluWriter,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    writer.eval()
    total = 0.0
    n = 0
    for i, (inputs, labels, concept) in enumerate(loader):
        if i >= max_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        concept = concept.to(device)
        loss = writer(inputs, concept, labels=labels)
        total += float(loss.item())
        n += 1
    writer.train()
    return total / max(1, n)


def train_phase3(
    cfg: Config,
    *,
    corpus_dir: str = DEFAULT_CORPUS,
    phase25_checkpoint: str = DEFAULT_P25,
    out_path: str = DEFAULT_OUT,
    g: int = G_DEFAULT,
    max_docs: int | None = None,
    val_batches: int = 50,
    progress: bool = True,
) -> tuple[FluWriter, Tokenizer]:
    from experiments.sentence_pred.data import load_sentence_corpus

    torch.manual_seed(cfg.writer.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.writer.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(cfg.jepa.tokenizer)

    corpus = load_sentence_corpus(corpus_dir)
    print(
        f"[writer-p3] corpus {corpus_dir}: "
        f"{corpus.n_docs} docs / {corpus.n_sentences} sents; G={g}",
        flush=True,
    )

    train_ds = ParagraphBlockDataset(
        corpus, tok, doc_ids=corpus.train_doc_ids, g=g, max_docs=max_docs
    )
    val_ds = ParagraphBlockDataset(
        corpus, tok, doc_ids=corpus.held_doc_ids, g=g, max_docs=None
    )
    print(
        f"[writer-p3] train examples={len(train_ds)} val examples={len(val_ds)}",
        flush=True,
    )
    if len(train_ds) == 0:
        raise RuntimeError("ParagraphBlockDataset is empty — check corpus / G")

    collate = make_collate(tok, cfg.writer.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.writer.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.writer.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # Load phase-2.5 two-channel writer as init.
    writer = load_writer(phase25_checkpoint, device="cpu")
    writer.to(dev)
    _set_trainable(writer, cfg.writer.n_layers)
    n_train = sum(p.numel() for p in writer.parameters() if p.requires_grad)
    print(
        f"[writer-p3] loaded {phase25_checkpoint}; trainable params={n_train:,}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in writer.parameters() if p.requires_grad],
        lr=cfg.writer.lr2,
        weight_decay=0.01,
    )
    grad_acc = max(1, int(cfg.writer.gradient_acc))

    gate = None
    if cfg.gpu_gate.enabled and dev.type == "cuda":
        gate = GpuGate(cfg.gpu_gate, dev, progress=progress)

    db = None
    run_id = None
    start_epoch = 0
    best_val = float("inf")
    best_state: dict | None = None
    if cfg.writer.db_path:
        db = CheckpointDB(cfg.writer.db_path)
        run_id = cfg.writer.run_id or db.latest_run()
        if run_id is None:
            run_id = db.start_run(
                corpus_dir,
                tok.name,
                cfg.writer.seed,
                cfg.writer.seq_len,
                cfg.jepa.dim,
                {
                    "writer": cfg.writer.__dict__,
                    "phase3": True,
                    "g": g,
                    "phase25_checkpoint": phase25_checkpoint,
                },
            )
        loaded = db.load_latest_checkpoint(run_id, str(dev))
        if loaded is not None:
            writer.load_state_dict(loaded[0])
            if loaded[1]:
                try:
                    opt.load_state_dict(loaded[1])
                except Exception as exc:  # noqa: BLE001
                    print(f"[writer-p3] opt state skipped: {exc}", flush=True)
            start_epoch = loaded[2] + 1
            print(
                f"[writer-p3] resumed from epoch {start_epoch}/{cfg.writer.epochs}",
                flush=True,
            )

    steps_per_epoch = max(1, cfg.writer.steps_per_epoch)
    history: list[dict] = []

    for epoch in range(start_epoch, cfg.writer.epochs):
        writer.train()
        opt.zero_grad()
        epoch_loss = 0.0
        n_steps = 0
        it = iter(train_loader)
        progress_iter = iter_bar(
            range(steps_per_epoch),
            desc=f"writer-p3 epoch {epoch + 1}/{cfg.writer.epochs}",
            total=steps_per_epoch,
            enabled=progress,
            unit="batch",
        )
        for _ in progress_iter:
            if gate is not None:
                gate.step(writer, opt)
            try:
                inputs, labels, concept = next(it)
            except StopIteration:
                it = iter(train_loader)
                inputs, labels, concept = next(it)
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            concept = concept.to(dev)
            loss = writer(inputs, concept, labels=labels) / grad_acc
            loss.backward()
            if (n_steps + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in writer.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()
            epoch_loss += float(loss.item()) * grad_acc
            n_steps += 1

        # Flush leftover microbatches.
        if n_steps % grad_acc != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in writer.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad()

        avg = epoch_loss / max(1, n_steps)
        val = _eval_loss(writer, val_loader, dev, max_batches=val_batches)
        metrics = {"loss": avg, "val_loss": val}
        history.append({"epoch": epoch, **metrics})
        log_epoch(
            "writer-p3",
            epoch + 1,
            cfg.writer.epochs,
            metrics,
            enabled=progress,
        )
        print(
            f"[writer-p3] epoch {epoch + 1}/{cfg.writer.epochs} "
            f"loss={avg:.4f} val_loss={val:.4f}",
            flush=True,
        )

        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in writer.state_dict().items()}
            print(f"[writer-p3] new best val_loss={best_val:.4f}", flush=True)

        if db is not None:
            db.record_epoch(run_id, epoch, metrics)
            db.save_checkpoint(run_id, epoch, writer, opt)

    if best_state is not None:
        writer.load_state_dict(best_state)
        print(f"[writer-p3] restored best val_loss={best_val:.4f}", flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_writer(
        out,
        writer,
        {
            "vocab_size": tok.vocab_size,
            "dim": cfg.writer.dim,
            "n_layers": cfg.writer.n_layers,
            "n_heads": cfg.writer.n_heads,
            "ff_mult": cfg.writer.ff_mult,
            "max_seq_len": cfg.writer.seq_len + 8,
            "concept_dim": cfg.jepa.dim,
            "use_concept": True,
            "use_film": True,
            "two_channel": True,
            "phase3": True,
            "paragraph_committed": True,
            "G": g,
            "phase25_checkpoint": phase25_checkpoint,
            "corpus_dir": corpus_dir,
            "history": history,
            "best_val_loss": best_val if best_val < float("inf") else None,
        },
    )
    print(f"[writer-p3] saved {out}", flush=True)

    if db is not None:
        db.close()
    return writer, tok


def _load_config(path: str | None) -> tuple[Config, dict]:
    """Load Config + optional phase3 extras (popped before Config.merge)."""
    if path is None:
        return Config(), {}
    raw = json.loads(Path(path).read_text())
    extras = dict(raw.pop("phase3", {}) or {})
    # Also allow top-level aliases for convenience.
    for key in (
        "corpus_dir",
        "phase25_checkpoint",
        "out_path",
        "g",
        "max_docs",
        "val_batches",
    ):
        if key in raw:
            extras[key] = raw.pop(key)
    cfg = Config().merged(raw)
    return cfg, extras


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Phase-3 paragraph-committed writer fine-tune")
    p.add_argument("--config", type=str, default="configs/writer_phase3.json")
    p.add_argument("--corpus-dir", type=str, default=None)
    p.add_argument("--phase25-checkpoint", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--g", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr2", type=float, default=None)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--gradient-acc", type=int, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--no-gpu-gate", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    cfg, extras = _load_config(args.config)
    if args.epochs is not None:
        cfg.writer.epochs = args.epochs
    if args.batch_size is not None:
        cfg.writer.batch_size = args.batch_size
    if args.lr2 is not None:
        cfg.writer.lr2 = args.lr2
    if args.steps_per_epoch is not None:
        cfg.writer.steps_per_epoch = args.steps_per_epoch
    if args.gradient_acc is not None:
        cfg.writer.gradient_acc = args.gradient_acc
    if args.seed is not None:
        cfg.writer.seed = args.seed
    if args.no_gpu_gate:
        cfg.gpu_gate.enabled = False

    corpus_dir = args.corpus_dir or extras.get("corpus_dir", DEFAULT_CORPUS)
    phase25 = args.phase25_checkpoint or extras.get(
        "phase25_checkpoint", DEFAULT_P25
    )
    out_path = args.out or extras.get("out_path", DEFAULT_OUT)
    g = int(args.g if args.g is not None else extras.get("g", G_DEFAULT))
    max_docs = args.max_docs if args.max_docs is not None else extras.get("max_docs")
    val_batches = int(
        args.val_batches
        if args.val_batches is not None
        else extras.get("val_batches", 50)
    )

    cfg.ensure_dirs()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    train_phase3(
        cfg,
        corpus_dir=str(corpus_dir),
        phase25_checkpoint=str(phase25),
        out_path=str(out_path),
        g=g,
        max_docs=int(max_docs) if max_docs is not None else None,
        val_batches=val_batches,
        progress=True,
    )


if __name__ == "__main__":
    main()
