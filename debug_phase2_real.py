"""Reproduce exact train_phase2 forward with Phase-1 weight load, CUDA blocking."""
import sys
sys.path.insert(0, "src")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.model import FluWriter
from src.conceptpoc.data.corpus import load_documents

dev = torch.device("cuda")
tok = Tokenizer("gpt2")
w = FluWriter(tok.vocab_size, dim=768, n_layers=12, n_heads=12, ff_mult=4, max_seq_len=264, concept_dim=512, use_concept=True)
# Load phase1 weights (skipping concept_proj) like train_phase2 does
blob = torch.load("artifacts/writer_p1/writer_p1.pt", map_location="cpu", weights_only=False)
state = {k: v for k, v in blob["state_dict"].items() if "concept_proj" not in k}
missing, unexpected = w.load_state_dict(state, strict=False)
print("missing:", list(missing)[:5], "unexpected:", list(unexpected)[:5])
w.to(dev)

docs = load_documents("data/austen_corpus.txt")
sents = [s for d in docs for s in d.sentences if s.strip()]
items = []
for s in sents[:32]:
    ids, _ = tok.encode_batch([s], 256)
    items.append(ids[0])
maxlen = max(x.size(0) for x in items)
padded = torch.full((len(items), maxlen), 0, dtype=torch.long)
for i, x in enumerate(items):
    padded[i, : x.size(0)] = x
tgt = padded.to(dev)
ctx = torch.randn(len(items), 512, device=dev)
labels = tgt[:, 1:]
labels[labels == 0] = -100
print("labels range:", labels.min().item(), labels.max().item())
try:
    loss = w(tgt[:, :-1], ctx, labels=labels)
    print("PH2 forward OK, loss:", loss.item())
except Exception as e:
    print("PH2 ERROR:", repr(e)[:200])
