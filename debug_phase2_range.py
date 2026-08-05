"""Pin down exact embedding input range."""
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
docs = load_documents("data/austen_corpus.txt")
sents = [s for d in docs for s in d.sentences if s.strip()]
# Longer realistic batch to hit the failure
items = []
for s in sents[:64]:
    ids, _ = tok.encode_batch([s], 256)
    items.append(ids[0])
maxlen = max(x.size(0) for x in items)
padded = torch.full((len(items), maxlen), 0, dtype=torch.long)
for i, x in enumerate(items):
    padded[i, : x.size(0)] = x
tgt = padded.to(dev)
inp = tgt[:, :-1]
print("input shape:", inp.shape)
print("input min:", inp.min().item(), "max:", inp.max().item())
print("input count >= vocab:", (inp >= tok.vocab_size).sum().item())
print("input count < 0:", (inp < 0).sum().item())

w = FluWriter(tok.vocab_size, dim=768, n_layers=12, n_heads=12, max_seq_len=264, concept_dim=512)
w.to(dev)
print("embedding weight shape:", w.tok_embed.weight.shape)
ctx = torch.randn(len(items), 512, device=dev)
labels = tgt[:, 1:]
labels[labels == 0] = -100
# Embed directly to isolate the failure
print("embedding input max vs weight rows:", inp.max().item(), "vs", w.tok_embed.weight.shape[0])
