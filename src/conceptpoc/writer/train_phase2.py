"""Phase-2.5: two-channel writer — previous-sentence tokens + same-sentence concept.

Channel 1: previous sentence tokens as a prefix (local coherence).
Channel 2: sentence i's concept vector via concept token + all-block FiLM (steering).
Same-sentence pairing (concept_i -> tokens_i) fixes the Phase-2 D1 off-by-one.
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

PREFIX_MAX = 96
TARGET_MAX = 128
SEQ_CAP = 256
DROP_P = 0.10


class TwoChannelSentenceDataset(Dataset):
    """Within-document (prev_tokens, target_tokens, concept_of_target) triples.

    - pairs NEVER cross document boundaries (iterate docs, then i in range(len(sents)))
    - concept = all_vecs[global_index(i)]   # SAME sentence as target (fixes D1 off-by-one)
    - i == 0 items use the prefix-drop form (no prefix)
    - pre-encodes all sentences once, chunked 512, as before
    """

    def __init__(
        self,
        corpus_path: str,
        codec: JepaCodec,
        tokenizer: Tokenizer,
        seq_len: int = SEQ_CAP,
        max_docs: int | None = None,
    ) -> None:
        self.tok = tokenizer
        self.seq_len = seq_len
        docs = load_documents(corpus_path)
        if max_docs is not None:
            docs = docs[:max_docs]
        # Flat sentence list + per-doc ranges for within-doc pairing.
        sents: list[str] = []
        doc_ranges: list[tuple[int, int]] = []
        for d in docs:
            clean = [s for s in d.sentences if s.strip()]
            if not clean:
                continue
            start = len(sents)
            sents.extend(clean)
            doc_ranges.append((start, len(sents)))

        chunk = 512
        vecs: list[torch.Tensor] = []
        for i in range(0, len(sents), chunk):
            v = codec.encode(sents[i : i + chunk])
            vecs.append(torch.from_numpy(v))
        all_vecs = torch.cat(vecs, dim=0) if vecs else torch.zeros(0, 1)

        # Pre-tokenize once.
        sent_ids: list[torch.Tensor] = []
        for s in sents:
            ids = tokenizer.encode(s)
            sent_ids.append(torch.tensor(ids, dtype=torch.long))

        self.items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
        for start, end in doc_ranges:
            for i in range(start, end):
                local_i = i - start
                target_ids = sent_ids[i][:TARGET_MAX]
                concept = all_vecs[i]
                if local_i == 0:
                    prev_ids = torch.zeros(0, dtype=torch.long)
                    prefix_present = False
                else:
                    prev_ids = sent_ids[i - 1][:PREFIX_MAX]
                    prefix_present = True
                self.items.append((prev_ids, target_ids, concept, prefix_present))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        prev_ids, target_ids, concept, prefix_present = self.items[idx]
        prev_ids, concept, prefix_present = apply_drop_arms(
            idx, prev_ids, concept, prefix_present
        )
        return prev_ids, target_ids, concept, prefix_present


def apply_drop_arms(
    idx: int,
    prev_ids: torch.Tensor,
    concept: torch.Tensor,
    prefix_present: bool,
    drop_p: float = DROP_P,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """p=drop_p drop prefix; p=drop_p zero concept; never both. Seed 42+idx."""
    g = torch.Generator()
    g.manual_seed(42 + idx)
    r = float(torch.rand(1, generator=g).item())
    if prefix_present and r < drop_p:
        return prev_ids, concept, False
    if drop_p <= r < 2 * drop_p:
        return prev_ids, torch.zeros_like(concept), prefix_present
    return prev_ids, concept, prefix_present


def make_collate(tok: Tokenizer, seq_len: int = SEQ_CAP):
    """Build §5.1 sequences: [BOS]+prev+[EOS]+tgt+[EOS] with correct label masks."""

    def _collate(batch):
        full_seqs: list[list[int]] = []
        n_prefix_masks: list[int] = []
        concepts: list[torch.Tensor] = []
        for prev_ids, target_ids, concept, prefix_present in batch:
            prev = prev_ids.view(-1).tolist()[:PREFIX_MAX]
            tgt = target_ids.view(-1).tolist()[:TARGET_MAX]
            if prefix_present and prev:
                ids = [tok.bos_id] + prev + [tok.eos_id] + tgt + [tok.eos_id]
                # Mask labels predicting inside the prefix (BOS..last prev → EOS sep).
                # Separator position (index 1+len(prev)) supervises the first target token.
                n_mask = 1 + len(prev)
            else:
                ids = [tok.bos_id] + tgt + [tok.eos_id]
                n_mask = 0
            ids = ids[:seq_len]
            full_seqs.append(ids)
            n_prefix_masks.append(n_mask)
            concepts.append(concept.view(-1).float())

        max_full = min(seq_len, max(len(s) for s in full_seqs))
        b = len(full_seqs)
        padded = torch.full((b, max_full), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(full_seqs):
            s = s[:max_full]
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)

        # writer(input[:, :-1], concept, labels=input[:, 1:]) with masks
        inputs = padded[:, :-1].contiguous()
        labels = padded[:, 1:].clone()
        labels[labels == tok.pad_id] = -100
        for i, n_mask in enumerate(n_prefix_masks):
            if n_mask > 0:
                labels[i, :n_mask] = -100
        concept_batch = torch.stack(concepts, dim=0)
        return inputs, labels, concept_batch

    return _collate


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

    # Freeze most of the writer; tune concept_proj + top blocks + all-block FiLM (D4).
    trainable = ["concept_proj"]
    trainable += [f"blocks.{i}." for i in range(cfg.writer.n_layers - 3, cfg.writer.n_layers)]
    trainable += [f"blocks.{i}.film" for i in range(cfg.writer.n_layers)]
    for name, p in writer.named_parameters():
        p.requires_grad = any(t in name for t in trainable)
    opt = torch.optim.AdamW(
        [p for p in writer.parameters() if p.requires_grad], lr=cfg.writer.lr2
    )

    dataset = TwoChannelSentenceDataset(
        str(cfg.data.corpus_path), codec, tok, cfg.writer.seq_len
    )
    collate = make_collate(tok, cfg.writer.seq_len)
    loader = DataLoader(
        dataset, batch_size=cfg.writer.batch_size, shuffle=True, collate_fn=collate
    )

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
                inputs, labels, concept = next(it)
            except StopIteration:
                it = iter(loader)
                inputs, labels, concept = next(it)
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            concept = concept.to(dev)
            opt.zero_grad()
            loss = writer(inputs, concept, labels=labels)
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
