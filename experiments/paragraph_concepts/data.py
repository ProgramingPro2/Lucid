"""Paragraph-scale concept corpus: group sentence JEPA vectors into G-blocks.

Non-overlapping groups of G sentences are mean-pooled and L2-normalized to form
paragraph concepts. A sliding window over paragraph concepts yields
(context -> next) training examples for the ranking reasoner.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Repo bootstrap so `python -m experiments.paragraph_concepts` works from root.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.data.corpus import Document, load_documents  # noqa: E402
from conceptpoc.progress import iter_bar, log_stage  # noqa: E402

_ENCODE_BATCH = 128


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x)) + eps
        return x / n
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / denom


@dataclass
class Paragraph:
    """One paragraph concept (mean-pooled G sentences)."""

    vector: np.ndarray  # (dim,) float32, L2-normalized
    sentences: list[str]
    doc_id: int
    position: int  # paragraph index within document


@dataclass
class CachedParagraphCorpus:
    """Paragraph concepts + document boundaries (npz-cacheable)."""

    vectors: np.ndarray  # (n_paragraphs, dim) float32, L2-normalized
    sentences: list[list[str]]  # per-paragraph source sentences
    doc_offsets: np.ndarray  # (n_docs + 1,) int; doc i = [offsets[i], offsets[i+1])
    group_size: int
    sent_vectors: np.ndarray | None = None  # optional raw sentence vecs

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def n_paragraphs(self) -> int:
        return int(self.vectors.shape[0])

    def doc_slices(self) -> list[tuple[int, int]]:
        return [
            (int(self.doc_offsets[i]), int(self.doc_offsets[i + 1]))
            for i in range(len(self.doc_offsets) - 1)
        ]

    def paragraphs_for_doc(self, doc_id: int) -> np.ndarray:
        a, b = self.doc_slices()[doc_id]
        return self.vectors[a:b]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Flatten sentences with lengths for object-array-free roundtrip.
        flat_sents: list[str] = []
        sent_counts = np.zeros(self.n_paragraphs, dtype=np.int32)
        for i, sents in enumerate(self.sentences):
            sent_counts[i] = len(sents)
            flat_sents.extend(sents)
        payload = {
            "vectors": self.vectors.astype(np.float32),
            "doc_offsets": self.doc_offsets.astype(np.int64),
            "group_size": np.asarray([self.group_size], dtype=np.int32),
            "sent_counts": sent_counts,
            "flat_sents": np.array(flat_sents, dtype=object),
        }
        if self.sent_vectors is not None:
            payload["sent_vectors"] = self.sent_vectors.astype(np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "CachedParagraphCorpus":
        data = np.load(path, allow_pickle=True)
        sent_counts = data["sent_counts"].astype(np.int32)
        flat = list(data["flat_sents"])
        sentences: list[list[str]] = []
        cursor = 0
        for n in sent_counts:
            n = int(n)
            sentences.append([str(s) for s in flat[cursor : cursor + n]])
            cursor += n
        sent_vecs = None
        if "sent_vectors" in data.files:
            sent_vecs = data["sent_vectors"].astype(np.float32)
        return cls(
            vectors=data["vectors"].astype(np.float32),
            sentences=sentences,
            doc_offsets=data["doc_offsets"].astype(np.int64),
            group_size=int(data["group_size"][0]),
            sent_vectors=sent_vecs,
        )


def _cache_key(docs: list[Document], group_size: int, codec_id: str) -> str:
    h = hashlib.sha256()
    h.update(codec_id.encode())
    h.update(f"G={group_size}".encode())
    h.update(str(len(docs)).encode())
    for doc in docs:
        h.update(doc.source.encode())
        h.update(str(len(doc.sentences)).encode())
        if doc.sentences:
            h.update(doc.sentences[0][:64].encode())
            h.update(doc.sentences[-1][:64].encode())
    return h.hexdigest()[:16]


def _group_doc_sentences(
    sent_vecs: np.ndarray,
    sents: list[str],
    group_size: int,
    doc_id: int,
) -> list[Paragraph]:
    """Non-overlapping mean-pool of consecutive sentence vectors."""
    n, d = sent_vecs.shape
    g = max(1, int(group_size))
    n_groups = n // g
    out: list[Paragraph] = []
    for p in range(n_groups):
        a, b = p * g, (p + 1) * g
        pooled = sent_vecs[a:b].mean(axis=0)
        vec = _l2_normalize(pooled)
        out.append(
            Paragraph(
                vector=vec,
                sentences=list(sents[a:b]),
                doc_id=doc_id,
                position=p,
            )
        )
    return out


def encode_documents(
    docs: list[Document],
    codec,
    *,
    batch_size: int = _ENCODE_BATCH,
    progress: bool = True,
) -> tuple[list[np.ndarray], list[list[str]]]:
    """Encode every sentence; return per-doc (vectors, sentences)."""
    flat: list[str] = []
    offsets = [0]
    for doc in docs:
        flat.extend(doc.sentences)
        offsets.append(len(flat))
    log_stage(f"encoding {len(flat):,} sentences across {len(docs)} docs...", enabled=progress)
    chunks = [flat[i : i + batch_size] for i in range(0, len(flat), batch_size)]
    encoded = [
        codec.encode(batch)
        for batch in iter_bar(
            chunks, desc="encode", total=len(chunks), enabled=progress, unit="batch"
        )
    ]
    all_vecs = (
        np.concatenate(encoded, axis=0)
        if encoded
        else np.zeros((0, codec.dim), dtype=np.float32)
    )
    all_vecs = _l2_normalize(all_vecs)
    per_doc_vecs: list[np.ndarray] = []
    per_doc_sents: list[list[str]] = []
    for i in range(len(docs)):
        a, b = offsets[i], offsets[i + 1]
        per_doc_vecs.append(np.ascontiguousarray(all_vecs[a:b], dtype=np.float32))
        per_doc_sents.append(list(docs[i].sentences))
    return per_doc_vecs, per_doc_sents


def build_paragraph_corpus(
    docs: list[Document],
    codec,
    group_size: int = 5,
    *,
    cache_dir: str | Path | None = None,
    codec_id: str = "jepa_v3_128t",
    batch_size: int = _ENCODE_BATCH,
    progress: bool = True,
    keep_sent_vectors: bool = False,
) -> CachedParagraphCorpus:
    """Encode sentences, group into paragraph concepts, optionally cache to disk."""
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(docs, group_size, codec_id)
        cache_path = cache_dir / f"paragraphs_G{group_size}_{key}.npz"
        if cache_path.exists():
            log_stage(f"loaded cached paragraph corpus {cache_path.name}", enabled=progress)
            return CachedParagraphCorpus.load(cache_path)

    per_doc_vecs, per_doc_sents = encode_documents(
        docs, codec, batch_size=batch_size, progress=progress
    )

    paragraphs: list[Paragraph] = []
    offsets = [0]
    sent_vecs_all: list[np.ndarray] = []
    for doc_id, (sv, sents) in enumerate(zip(per_doc_vecs, per_doc_sents)):
        paras = _group_doc_sentences(sv, sents, group_size, doc_id)
        paragraphs.extend(paras)
        offsets.append(len(paragraphs))
        if keep_sent_vectors:
            sent_vecs_all.append(sv)

    if not paragraphs:
        raise ValueError(
            f"No paragraph concepts built (need docs with >= {group_size} sentences)"
        )

    vectors = np.stack([p.vector for p in paragraphs]).astype(np.float32)
    sentences = [p.sentences for p in paragraphs]
    corpus = CachedParagraphCorpus(
        vectors=vectors,
        sentences=sentences,
        doc_offsets=np.asarray(offsets, dtype=np.int64),
        group_size=group_size,
        sent_vectors=np.concatenate(sent_vecs_all, axis=0) if sent_vecs_all else None,
    )
    if cache_path is not None:
        corpus.save(cache_path)
        log_stage(f"saved paragraph cache -> {cache_path}", enabled=progress)
    return corpus


def load_and_build(
    corpus_path: str | Path = "data/corpus_1m.txt",
    jepa_ckpt: str | Path = "artifacts/jepa_v3_128t/jepa/joint_robust.pt",
    group_size: int = 5,
    max_docs: int | None = 2000,
    min_sents: int | None = None,
    device: str = "auto",
    cache_dir: str | Path | None = "artifacts/paragraph_concepts_cache",
    seed: int = 42,
    progress: bool = True,
) -> CachedParagraphCorpus:
    """End-to-end: load docs, JEPA-encode, build paragraph corpus."""
    import torch
    from conceptpoc.codec.jepa_codec import JepaCodec
    from conceptpoc.persistence import load_joint

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    docs = load_documents(corpus_path)
    g = max(1, int(group_size))
    need = min_sents if min_sents is not None else g * 2  # >=2 paragraphs
    eligible = [d for d in docs if len(d.sentences) >= need]
    if not eligible:
        eligible = [d for d in docs if len(d.sentences) >= g]
    if max_docs is not None and len(eligible) > max_docs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eligible), size=max_docs, replace=False)
        eligible = [eligible[int(i)] for i in sorted(idx.tolist())]
    log_stage(
        f"using {len(eligible)} docs (G={g}, min_sents>={need})",
        enabled=progress,
    )

    model, tok = load_joint(str(jepa_ckpt), device=device)
    codec = JepaCodec(model, tok, 128, device=device)
    return build_paragraph_corpus(
        eligible,
        codec,
        group_size=g,
        cache_dir=cache_dir,
        codec_id=f"jepa_v3_128t@{Path(jepa_ckpt).name}",
        progress=progress,
    )


class ParagraphSequenceDataset(Dataset):
    """(context paragraph window -> next paragraph) with in-doc ranking candidates.

    Each item returns:
      context: (T, dim) float tensor  — up to ``window`` prior paragraphs
      target:  (dim,) float tensor    — true next paragraph
      candidates: (n_cand, dim)       — [true_next, following..., distractors...]
      n_positives: int                — how many leading candidates are positives
      doc_id, para_idx: ints for eval / debugging
    """

    def __init__(
        self,
        corpus: CachedParagraphCorpus,
        window: int = 8,
        n_candidates: int = 100,
        n_following_pos: int = 2,
        seed: int = 42,
    ) -> None:
        self.corpus = corpus
        self.vectors = torch.from_numpy(np.ascontiguousarray(corpus.vectors))
        self.window = max(1, int(window))
        self.n_candidates = max(2, int(n_candidates))
        self.n_following_pos = max(0, int(n_following_pos))
        self.rng = np.random.default_rng(seed)
        # (doc_id, local_para_idx_of_last_context) such that next exists
        self._index: list[tuple[int, int]] = []
        for doc_id, (a, b) in enumerate(corpus.doc_slices()):
            n = b - a
            if n < 2:
                continue
            # Predict local positions 1..n-1 from a prefix ending at pos-1
            for local_next in range(1, n):
                self._index.append((doc_id, local_next))

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self._index)

    def _global(self, doc_id: int, local: int) -> int:
        a, _ = self.corpus.doc_slices()[doc_id]
        return a + local

    def __getitem__(self, idx: int):
        doc_id, local_next = self._index[idx]
        a, b = self.corpus.doc_slices()[doc_id]
        n = b - a
        # Context: up to `window` paragraphs immediately before local_next
        ctx_start_local = max(0, local_next - self.window)
        ctx_g = [self._global(doc_id, i) for i in range(ctx_start_local, local_next)]
        context = self.vectors[ctx_g]  # (T, dim)
        target = self.vectors[self._global(doc_id, local_next)]

        # Positives: true next + following paragraphs in-doc
        pos_locals = [local_next]
        for k in range(1, self.n_following_pos + 1):
            if local_next + k < n:
                pos_locals.append(local_next + k)
        pos_set = set(pos_locals)
        # Also exclude context positions from distractors
        banned = set(range(ctx_start_local, local_next)) | pos_set
        pool = [i for i in range(n) if i not in banned]
        need_neg = self.n_candidates - len(pos_locals)
        if need_neg < 0:
            pos_locals = pos_locals[: self.n_candidates]
            need_neg = 0
        if need_neg > 0:
            if len(pool) >= need_neg:
                neg_locals = self.rng.choice(pool, size=need_neg, replace=False).tolist()
            elif pool:
                neg_locals = self.rng.choice(pool, size=need_neg, replace=True).tolist()
            else:
                # Degenerate short doc: repeat true next as filler (rare)
                neg_locals = [local_next] * need_neg
        else:
            neg_locals = []

        cand_locals = pos_locals + [int(x) for x in neg_locals]
        cand_g = [self._global(doc_id, i) for i in cand_locals]
        candidates = self.vectors[cand_g]
        return {
            "context": context,
            "target": target,
            "candidates": candidates,
            "n_positives": len(pos_locals),
            "doc_id": doc_id,
            "para_idx": local_next,
        }


def collate_paragraph_batch(batch: list[dict]):
    """Right-pad contexts; stack fixed-size candidate sets."""
    dim = batch[0]["context"].shape[-1]
    max_t = max(b["context"].shape[0] for b in batch)
    n_cand = batch[0]["candidates"].shape[0]
    B = len(batch)
    context = torch.zeros(B, max_t, dim, dtype=torch.float32)
    pad_mask = torch.ones(B, max_t, dtype=torch.bool)  # True = pad
    targets = torch.zeros(B, dim, dtype=torch.float32)
    candidates = torch.zeros(B, n_cand, dim, dtype=torch.float32)
    n_positives = torch.zeros(B, dtype=torch.long)
    doc_ids = torch.zeros(B, dtype=torch.long)
    para_idxs = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["context"].shape[0]
        context[i, :t] = b["context"]
        pad_mask[i, :t] = False
        targets[i] = b["target"]
        # Candidate count may vary if last docs are short — pad/truncate
        c = b["candidates"]
        k = min(n_cand, c.shape[0])
        candidates[i, :k] = c[:k]
        if c.shape[0] < n_cand:
            candidates[i, c.shape[0] :] = c[-1:].expand(n_cand - c.shape[0], -1)
        n_positives[i] = min(int(b["n_positives"]), k)
        doc_ids[i] = b["doc_id"]
        para_idxs[i] = b["para_idx"]
    return context, pad_mask, targets, candidates, n_positives, doc_ids, para_idxs
