"""Reproduce Phase-2 GPU batch to find out-of-range token."""
import sys
sys.path.insert(0, "src")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from src.conceptpoc.data.corpus import load_documents
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.model import FluWriter

dev = torch.device("cuda")
tok = Tokenizer("gpt2")
print("vocab:", tok.vocab_size, "pad:", tok.pad_id, "eos:", tok.eos_id)

# Build a realistic batch like the dataset + collate would
docs = load_documents("data/austen_corpus.txt")
sents = [s for d in docs for s in d.sentences if s.strip()]
# take a few, tokenize individually like encode_batch([target], 256) does
items = []
for s in sents[:8]:
    ids, _ = tok.encode_batch([s], 256)
    items.append(ids[0])

# _collate
maxlen = max(x.size(0) for x in items)
padded = torch.full((len(items), maxlen), 0, dtype=torch.long)
for i, x in enumerate(items):
    padded[i, : x.size(0)] = x
tgt = padded.to(dev)
print("tgt max id:", tgt.max().item(), ">=vocab?", (tgt >= tok.vocab_size).sum().item())

ctx = torch.randn(len(items), 512, device=dev)
w = FluWriter(tok.vocab_size, dim=768, n_layers=12, n_heads=12, max_seq_len=264, concept_dim=512)
w.to(dev)
labels = tgt[:, 1:]
labels[labels == 0] = -100
try:
    loss = w(tgt[:, :-1], ctx, labels=labels)
    print("GPU forward OK, loss:", loss.item())
except Exception as e:
    print("GPU ERROR:", repr(e))
    bad = torch.nonzero((tgt[:, :-1] >= tok.vocab_size), as_tuple=False)
    print("bad coords count:", bad.size(0), "values:", tgt[:, :-1][bad[:,0], bad[:,1]].tolist()[:10])
