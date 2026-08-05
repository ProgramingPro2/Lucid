"""Debug: find out-of-range token ids in Phase-2 dataset."""
import sys
sys.path.insert(0, "src")
import torch
from src.conceptpoc.data.corpus import load_documents
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.train_phase2 import ConceptSentenceDataset, _collate

tok = Tokenizer("gpt2")
print("vocab_size:", tok.vocab_size)
print("pad_id:", tok.pad_id, "eos_id:", tok.eos_id, "bos_id:", tok.bos_id)

docs = load_documents("data/austen_corpus.txt")
sents = [s for d in docs for s in d.sentences if s.strip()]
print("num sentences:", len(sents))

# Reproduce the encode path without a codec (just tokenize)
for i in range(min(20, len(sents))):
    ids, _ = tok.encode_batch([sents[i]], 256)
    row = ids[0]
    bad = (row >= tok.vocab_size).nonzero()
    if bad.numel():
        print(f" sentence {i} ({sents[i][:40]!r}) has bad idx {row[bad].tolist()}")
print("checked", min(20, len(sents)), "sentences - done")
