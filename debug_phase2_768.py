"""Reproduce: fresh 768/12 FluWriter on GPU with full-vocab ids, no weight load."""
import sys
sys.path.insert(0, "src")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.model import FluWriter

dev = torch.device("cuda")
tok = Tokenizer("gpt2")
B, T = 4, 64
w = FluWriter(tok.vocab_size, dim=768, n_layers=12, n_heads=12, max_seq_len=272, concept_dim=512, use_concept=True).to(dev)
ids = torch.randint(0, tok.vocab_size-2, (B, T), device=dev)
labels = ids[:, 1:].clone()
ctx = torch.randn(B, 512, device=dev)
try:
    loss = w(ids[:, :-1], ctx, labels=labels)
    print("FRESH 768/12 OK:", loss.item())
except Exception as ex:
    print("FRESH 768/12 ERROR:", repr(ex)[:150])
