"""Isolate GPU embedding assert: test weight-tying vs concept-prepend."""
import sys
sys.path.insert(0, "src")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from src.conceptpoc.jepa.tokenizer import Tokenizer

dev = torch.device("cuda")
tok = Tokenizer("gpt2")

B, T = 4, 64
ids = torch.randint(0, 1000, (B, T), device=dev)  # valid small ids

# Test 1: plain nn.Embedding on GPU
e = torch.nn.Embedding(2000, 128).to(dev)
out = e(ids)
print("TEST1 plain embedding OK:", out.shape)

# Test 2: weight-tied embedding+linear
emb = torch.nn.Embedding(2000, 128).to(dev)
lin = torch.nn.Linear(128, 2000, bias=False).to(dev)
emb.weight = lin.weight  # tie
out2 = emb(ids)
print("TEST2 tied embedding OK:", out2.shape)

# Test 3: embed then cat concept (like FluWriter)
h = emb(ids)
c = torch.randn(B, 1, 128, device=dev)
h2 = torch.cat([c, h], dim=1)
print("TEST3 concat concept OK:", h2.shape)

# Test 4: full tiny FluWriter on GPU
from src.conceptpoc.writer.model import FluWriter
w = FluWriter(tok.vocab_size, dim=256, n_layers=4, n_heads=4, max_seq_len=272, concept_dim=512, use_concept=True).to(dev)
ids2 = torch.randint(0, 1000, (B, 64), device=dev)
ctx = torch.randn(B, 512, device=dev)
try:
    loss = w(ids2, ctx, labels=ids2[:, 1:].clone())
    print("TEST4 tiny FluWriter (full vocab, fresh) OK:", loss.item())
except Exception as ex:
    print("TEST4 ERROR:", repr(ex)[:120])
