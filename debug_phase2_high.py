"""Test if high token ids (pad/eos near vocab) trigger embedding assert."""
import sys
sys.path.insert(0, "src")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.model import FluWriter

dev = torch.device("cuda")
tok = Tokenizer("gpt2")
V = tok.vocab_size
print("vocab:", V, "pad:", tok.pad_id, "eos:", tok.eos_id)
w = FluWriter(V, dim=768, n_layers=12, n_heads=12, max_seq_len=272, concept_dim=512, use_concept=True).to(dev)
B = 4
ctx = torch.randn(B, 512, device=dev)

def run(name, ids):
    ids = ids.to(dev)
    try:
        loss = w(ids[:, :-1], ctx, labels=ids[:, 1:].clone())
        print(f"{name}: OK loss={loss.item():.3f}")
    except Exception as e:
        print(f"{name}: ERROR {repr(e)[:80]}")

ids = torch.randint(0, V, (B, 64), device=dev)
run("randint full vocab", ids)
ids2 = torch.full((B, 64), tok.eos_id, device=dev)
run("all eos", ids2)
ids3 = torch.full((B, 64), tok.pad_id, device=dev)
run("all pad", ids3)
ids4 = torch.randint(0, V-3, (B, 64), device=dev)  # exclude pad
run("randint excl pad/eos", ids4)
