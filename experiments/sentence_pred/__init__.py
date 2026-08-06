"""Per-sentence density prediction (vMF / MDN heads)."""

from .data import (
    SentenceCorpus,
    SentenceSequenceDataset,
    build_sentence_corpus,
    collate_sentence_batch,
    corpus_cache_key,
    detok_text,
    load_sentence_corpus,
)
from .heads import (
    MixtureVmfHead,
    SentenceDensityPredictor,
    VmfHead,
    log_iv_uae,
    sample_vmf,
    vmf_log_normalizer,
    vmf_nll,
)

__all__ = [
    "SentenceCorpus",
    "SentenceSequenceDataset",
    "build_sentence_corpus",
    "collate_sentence_batch",
    "corpus_cache_key",
    "detok_text",
    "load_sentence_corpus",
    "MixtureVmfHead",
    "SentenceDensityPredictor",
    "VmfHead",
    "log_iv_uae",
    "sample_vmf",
    "vmf_log_normalizer",
    "vmf_nll",
]
