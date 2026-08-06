"""Sentence-sequence corpus and teacher-forced windows for per-sentence JEPA.

See Fable §4 (per-sentence prediction spec): document-disjoint split, detokenized
WikiText text matching encoded vectors, and cache-keyed persistence.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402
from conceptpoc.data.corpus import load_documents  # noqa: E402
from conceptpoc.persistence import load_joint  # noqa: E402


def detok_text(s: str) -> str:
    """WikiText-style detokenization so stored text matches encoded text.

    Rules:
      - unescape `` @-@ `` / `` @,@ `` / `` @.@ `` (also @ / @)
      - collapse space before closing punctuation
      - collapse space *inside* quotes: after an opening quote and before a
        closing quote, but keep the space before an opening quote ('said "hi"')
    """
    s = re.sub(r"\s*@-@\s*", "-", s)
    s = re.sub(r"\s*@,@\s*", ",", s)
    s = re.sub(r"\s*@\.@\s*", ".", s)
    for punct in ",.!?;:":
        s = re.sub(rf"\s+{re.escape(punct)}", punct, s)
    # collapse space inside quotes:
    #   after an opening quote  ->  " hi    =>  "hi
    s = re.sub(r'"\s+', '"', s)
    #   before a closing quote (quote followed by non-word or end) -> hi " => hi"
    s = re.sub(r'\s+"(?=\W|$)', '"', s)
    s = s.rstrip()
    return s


def corpus_cache_key(
    jepa_ckpt: str | Path,
    corpus_path: str | Path,
    min_sents: int,
    max_tokens: int,
    detok_wikitext: bool,
) -> str:
    """sha256(ckpt path + size + mtime + corpus + min_sents + max_tokens + detok)[:16]."""
    ckpt = Path(jepa_ckpt)
    if ckpt.is_file():
        st = ckpt.stat()
        size_mtime = f"{st.st_size}:{st.st_mtime_ns}"
    else:
        size_mtime = "missing:0"
    payload = (
        f"{ckpt.resolve() if ckpt.exists() else ckpt}|"
        f"{size_mtime}|"
        f"{Path(corpus_path)}|"
        f"{int(min_sents)}|"
        f"{int(max_tokens)}|"
        f"{bool(detok_wikitext)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class SentenceCorpus:
    sent_vecs: np.ndarray  # (N_sent, 512) float32, L2-normalized
    doc_offsets: np.ndarray  # (N_doc + 1,) int64
    sentences: list[str]  # flat, aligned with sent_vecs
    train_doc_ids: np.ndarray  # document-disjoint split, seed 1234, 90/10
    held_doc_ids: np.ndarray

    def doc_slice(self, doc_id: int) -> tuple[int, int]:
        return (int(self.doc_offsets[doc_id]), int(self.doc_offsets[doc_id + 1]))

    @property
    def n_docs(self) -> int:
        return int(self.doc_offsets.shape[0] - 1)

    @property
    def n_sentences(self) -> int:
        return int(self.sent_vecs.shape[0])


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x)) + eps
        return x / n
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / denom).astype(np.float32)


def _make_codec(jepa_ckpt: str | Path, max_tokens: int, device: str = "cpu") -> JepaCodec:
    model, tok = load_joint(jepa_ckpt, device=device)
    return JepaCodec(model, tok, max_tokens, device=device)


def _encode_sentences(codec, texts: list[str], batch_size: int = 256) -> np.ndarray:
    if not texts:
        dim = int(getattr(codec, "dim", 512))
        return np.zeros((0, dim), dtype=np.float32)
    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    parts = [np.asarray(codec.encode(batch), dtype=np.float32) for batch in chunks]
    return _l2_normalize(np.concatenate(parts, axis=0))


def _split_doc_ids(
    n_docs: int, held_out_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_docs)
    n_held = int(round(n_docs * held_out_frac))
    held = np.sort(perm[:n_held].astype(np.int64))
    train = np.sort(perm[n_held:].astype(np.int64))
    assert len(np.intersect1d(train, held)) == 0
    return train, held


def save_sentence_corpus(corpus: SentenceCorpus, cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "sent_vecs.npy", corpus.sent_vecs.astype(np.float16))
    np.save(cache_dir / "doc_offsets.npy", corpus.doc_offsets.astype(np.int64))
    with open(cache_dir / "sentences.json", "w", encoding="utf-8") as f:
        json.dump(corpus.sentences, f, ensure_ascii=False)
    manifest = {
        "train_doc_ids": [int(x) for x in corpus.train_doc_ids.tolist()],
        "held_doc_ids": [int(x) for x in corpus.held_doc_ids.tolist()],
    }
    with open(cache_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return cache_dir


def load_sentence_corpus(cache_dir: str | Path) -> SentenceCorpus:
    cache_dir = Path(cache_dir)
    sent_vecs = np.load(cache_dir / "sent_vecs.npy").astype(np.float32)
    sent_vecs = _l2_normalize(sent_vecs)
    doc_offsets = np.load(cache_dir / "doc_offsets.npy").astype(np.int64)
    with open(cache_dir / "sentences.json", encoding="utf-8") as f:
        sentences = list(json.load(f))
    with open(cache_dir / "split_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    train = np.asarray(manifest["train_doc_ids"], dtype=np.int64)
    held = np.asarray(manifest["held_doc_ids"], dtype=np.int64)
    assert len(np.intersect1d(train, held)) == 0
    return SentenceCorpus(
        sent_vecs=sent_vecs,
        doc_offsets=doc_offsets,
        sentences=sentences,
        train_doc_ids=train,
        held_doc_ids=held,
    )


def build_sentence_corpus(
    corpus_path: str = "data/corpus_1m.txt",
    jepa_ckpt: str = "artifacts/jepa_v3_128t/jepa/joint_robust.pt",
    out_dir: str = "artifacts/sentence_pred/corpus",
    max_tokens: int = 128,
    batch_size: int = 256,
    min_sents: int = 4,
    held_out_frac: float = 0.10,
    seed: int = 1234,
    detok_wikitext: bool = True,
    max_docs: int | None = None,
    device: str = "cpu",
    codec=None,
) -> SentenceCorpus:
    """Build or load a cached per-sentence corpus (document-disjoint 90/10 split)."""
    key = corpus_cache_key(
        jepa_ckpt, corpus_path, min_sents, max_tokens, detok_wikitext
    )
    cache_dir = Path(out_dir) / f"corpus_{key}"
    if (
        (cache_dir / "sent_vecs.npy").is_file()
        and (cache_dir / "doc_offsets.npy").is_file()
        and (cache_dir / "sentences.json").is_file()
        and (cache_dir / "split_manifest.json").is_file()
    ):
        return load_sentence_corpus(cache_dir)

    docs = load_documents(corpus_path)
    if max_docs is not None:
        docs = docs[: int(max_docs)]
    eligible = [d for d in docs if len(d.sentences) >= int(min_sents)]
    if not eligible:
        raise ValueError(
            f"No documents with >= {min_sents} sentences in {corpus_path}"
        )

    flat_sents: list[str] = []
    offsets = [0]
    for doc in eligible:
        sents = list(doc.sentences)
        if detok_wikitext:
            sents = [detok_text(s) for s in sents]
        flat_sents.extend(sents)
        offsets.append(len(flat_sents))
    doc_offsets = np.asarray(offsets, dtype=np.int64)

    if codec is None:
        codec = _make_codec(jepa_ckpt, max_tokens, device=device)
    sent_vecs = _encode_sentences(codec, flat_sents, batch_size=batch_size)
    if sent_vecs.shape[0] != len(flat_sents):
        raise RuntimeError(
            f"encode length mismatch: {sent_vecs.shape[0]} vs {len(flat_sents)}"
        )

    n_docs = len(eligible)
    train_doc_ids, held_doc_ids = _split_doc_ids(n_docs, held_out_frac, seed)
    assert len(np.intersect1d(train_doc_ids, held_doc_ids)) == 0

    corpus = SentenceCorpus(
        sent_vecs=np.ascontiguousarray(sent_vecs, dtype=np.float32),
        doc_offsets=doc_offsets,
        sentences=flat_sents,
        train_doc_ids=train_doc_ids,
        held_doc_ids=held_doc_ids,
    )
    save_sentence_corpus(corpus, cache_dir)
    return corpus


class SentenceSequenceDataset(Dataset):
    """Teacher-forced windows over one document's sentence vectors.

    Items: dict with
      context: (T,512) f32, target_seq: (T,512), valid: (T,) bool,
      doc_id: int, start: int
    context[t] = v[start+t]; target_seq[t] = v[start+t+1]; T = context_len.
    Windows: stride = context_len // 2 (train); eval mode uses non-overlapping anchors.
    """

    def __init__(
        self,
        corpus: SentenceCorpus,
        doc_ids: np.ndarray,
        context_len: int = 32,
        stride: int = 16,
        eval_mode: bool = False,
    ) -> None:
        self.corpus = corpus
        self.context_len = int(context_len)
        self.stride = int(self.context_len if eval_mode else stride)
        self.eval_mode = bool(eval_mode)
        self._index: list[tuple[int, int]] = []  # (doc_id, start)
        for doc_id in np.asarray(doc_ids, dtype=np.int64).tolist():
            a, b = corpus.doc_slice(int(doc_id))
            n = b - a
            # Need context_len context vectors + one target after the last.
            if n < self.context_len + 1:
                continue
            for start in range(0, n - self.context_len, self.stride):
                self._index.append((int(doc_id), int(start)))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        doc_id, start = self._index[idx]
        a, _b = self.corpus.doc_slice(doc_id)
        T = self.context_len
        lo = a + start
        context = self.corpus.sent_vecs[lo : lo + T].astype(np.float32)
        target_seq = self.corpus.sent_vecs[lo + 1 : lo + T + 1].astype(np.float32)
        valid = np.ones((T,), dtype=bool)
        return {
            "context": torch.from_numpy(np.ascontiguousarray(context)),
            "target_seq": torch.from_numpy(np.ascontiguousarray(target_seq)),
            "valid": torch.from_numpy(valid),
            "doc_id": int(doc_id),
            "start": int(start),
        }


def collate_sentence_batch(
    batch: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad variable-length windows.

    Returns:
      context (B,T,512), pad_mask (B,T) True=pad,
      target_seq (B,T,512), valid (B,T), doc_ids (B,)
    """
    if not batch:
        raise ValueError("empty batch")
    dim = int(batch[0]["context"].shape[-1])
    lengths = [int(item["context"].shape[0]) for item in batch]
    T = max(lengths)
    B = len(batch)
    context = torch.zeros(B, T, dim, dtype=torch.float32)
    target_seq = torch.zeros(B, T, dim, dtype=torch.float32)
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    valid = torch.zeros(B, T, dtype=torch.bool)
    doc_ids = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        L = lengths[i]
        context[i, :L] = item["context"]
        target_seq[i, :L] = item["target_seq"]
        pad_mask[i, :L] = False
        valid[i, :L] = item["valid"][:L]
        doc_ids[i] = int(item["doc_id"])
    return context, pad_mask, target_seq, valid, doc_ids
