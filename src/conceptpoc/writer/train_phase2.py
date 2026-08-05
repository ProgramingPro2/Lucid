"""Phase-2: condition the pretrained writer on JEPA concept vectors.

The JEPA reasoner (encoder only, frozen) turns a context sentence into a
concept vector. The writer is fine-tuned to produce the *next* sentence's
tokens conditioned on that vector — so at inference, the JEPA diffusion
predictor supplies the concept and the writer renders fluent text.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ..codec.jepa_codec import JepaCodec
from ..config import Config
from ..data.corpus import load_documents
from ..persistence import load_joint
from ..jepa.tokenizer import Tokenizer
from ..progress import iter_bar, log_epoch
from .model import FluWriter


class ConceptSentenceDataset(Dataset):
    """Pairs (context_sentence_concept_vector, current_sentence_tokens).

    Concept = encoded vector of the preceding sentence (window of 2). The writer
    is asked to generate the current sentence given that vector.
    """

    def __init__(self, corpus_path: str, codec: JepaCodec, tokenizer: Tokenizer, seq_len: int) -> None:
        docs = load_documents(corpus_path)
        sents = [s for d in docs for s in d.sentences if s.strip()]
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Pre-encode every sentence once (codec.encode batched, but handle large
        # corpora in chunks to limit peak memory).
        chunk = 512
        vecs: list = []
        for i in range(0, len(sents), chunk):
            v = codec.encode(sents[i : i + chunk])
            vecs.append(torch.from_numpy(v))
        all_vecs = torch.cat(vecs, dim=0)
        for i in range(1, len(sents)):
            ctx = all_vecs[i - 1]  # context sentence's concept
            target = sents[i]
            ids, _ = tokenizer.encode_batch([target], seq_len)
            self.items.append((ctx.unsqueeze(0), ids[0]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ctx, ids = self.items[idx]
        return ctx, ids


def _collate(batch):
    ctxs = torch.cat([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    maxlen = max(x.size(0) for x in ids)
    pad_id = 0
    padded = torch.full((len(ids), maxlen), pad_id, dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, : x.size(0)] = x
    return ctxs, padded


def train_phase2(
    cfg: Config,
    progress: bool = True,
    phase1_path: str | None = None,
) -> tuple[FluWriter, Tokenizer]:
    torch.manual_seed(cfg.writer.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(cfg.jepa.tokenizer)

    # Load the frozen JEPA reasoner (encoder for concepts).
    jepa_ckpt = (cfg.writer.jepa_checkpoint or cfg.codec.jepa_checkpoint or
                 str(cfg.artifacts_dir / "jepa" / "joint_robust.pt"))
    model, _ = load_joint(jepa_ckpt)
    codec = JepaCodec(model, tok, cfg.jepa.max_tokens, device=dev)
    for p in model.parameters():
        p.requires_grad = False

    # Load Phase-1 pretrained writer, enable concept conditioning.
    writer = FluWriter(
        tok.vocab_size,
        dim=cfg.writer.dim,
        n_layers=cfg.writer.n_layers,
        n_heads=cfg.writer.n_heads,
        ff_mult=cfg.writer.ff_mult,
        max_seq_len=cfg.writer.seq_len + 8,
        concept_dim=cfg.jepa.dim,
        use_concept=cfg.writer.use_concept,
        use_film=cfg.writer.use_film,
    )
    phase1_path = phase1_path or cfg.writer.phase1_checkpoint
    if phase1_path:
        blob = torch.load(phase1_path, map_location="cpu", weights_only=False)
        # Phase-1 saved a use_concept=False model; map weights by ignoring the
        # extra concept_proj (freshly initialized).
        state = {k: v for k, v in blob["state_dict"].items() if "concept_proj" not in k}
        writer.load_state_dict(state, strict=False)
        print(f"[writer-p2] loaded Phase-1 weights from {phase1_path}")
    writer.to(dev)

    # Freeze most of the writer; tune only concept projection + top few blocks so
    # we don't catastrophically forget Phase-1 fluency.
    trainable = ["concept_proj"]
    trainable += [f"blocks.{i}." for i in range(cfg.writer.n_layers - 3, cfg.writer.n_layers)]
    for name, p in writer.named_parameters():
        p.requires_grad = any(t in name for t in trainable)
    opt = torch.optim.AdamW(
        [p for p in writer.parameters() if p.requires_grad], lr=cfg.writer.lr2
    )

    dataset = ConceptSentenceDataset(str(cfg.data.corpus_path), codec, tok, cfg.writer.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.writer.batch_size, shuffle=True, collate_fn=_collate)

    db = None
    run_id = None
    start_epoch = 0
    if cfg.writer.db_path:
        from ..ckpt_db import CheckpointDB

        db = CheckpointDB(cfg.writer.db_path)
        run_id = cfg.writer.run_id or db.latest_run()
        if run_id is None:
            run_id = db.start_run(
                str(cfg.data.corpus_path), tok.name, cfg.writer.seed,
                cfg.writer.seq_len, cfg.jepa.dim, {"writer": cfg.writer.__dict__},
            )
        loaded = db.load_latest_checkpoint(run_id, dev)
        if loaded is not None:
            writer.load_state_dict(loaded[0])
            opt.load_state_dict(loaded[1])
            start_epoch = loaded[2] + 1
            print(f"[writer-p2] resumed from epoch {start_epoch}/{cfg.writer.epochs}")

    steps_per_epoch = max(1, cfg.writer.steps_per_epoch)
    for epoch in range(start_epoch, cfg.writer.epochs):
        writer.train()
        epoch_loss = 0.0
        n_steps = 0
        progress_iter = iter_bar(
            range(steps_per_epoch),
            desc=f"writer-p2 epoch {epoch+1}/{cfg.writer.epochs}",
            total=steps_per_epoch,
            enabled=progress,
            unit="batch",
        )
        it = iter(loader)
        for _ in progress_iter:
            try:
                ctx, tgt = next(it)
            except StopIteration:
                it = iter(loader)
                ctx, tgt = next(it)
            ctx, tgt = ctx.to(dev), tgt.to(dev)
            # inputs = target[:-1]; labels = target[1:], masking pad (-100)
            labels = tgt[:, 1:]
            labels[labels == 0] = -100  # pad_id masked out
            opt.zero_grad()
            loss = writer(tgt[:, :-1], ctx, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in writer.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1
        avg = epoch_loss / max(1, n_steps)
        log_epoch("writer-p2", epoch + 1, cfg.writer.epochs, {"loss": avg}, enabled=progress)
        if db is not None:
            db.record_epoch(run_id, epoch, {"loss": avg})
            db.save_checkpoint(run_id, epoch, writer, opt)

    if db is not None:
        db.close()
    return writer, tok
