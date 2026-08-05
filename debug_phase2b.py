"""Debug Phase-2 collate + forward for out-of-range token ids."""
import sys
sys.path.insert(0, "src")
import torch
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.model import FluWriter

tok = Tokenizer("gpt2")
print("pad_id:", tok.pad_id, "eos:", tok.eos_id, "bos:", tok.bos_id, "vocab:", tok.vocab_size)

# Simulate what train_phase2 does with a batch of target token sequences.
# Mimic _collate: pad with 0, then labels[labels==0] = -100
seqs = tok.encode_batch(["The world is a small place.", "Hello there friend.", "x"], 256)[0]
print("encoded shape:", seqs.shape, "max id:", seqs.max().item())
maxlen = seqs.size(1)
padded = torch.full((len(seqs), maxlen), 0, dtype=torch.long)
for i, x in enumerate(seqs):
    padded[i, :x.size(0)] = x
print("collated shape:", padded.shape)
tgt = padded
labels = tgt[:, 1:]
labels[labels == 0] = -100
print("input ids (tgt[:, :-1]) max:", tgt[:, :-1].max().item())
print("label values present:", sorted(set(labels.reshape(-1).tolist()))[:10])

# Forward through small writer
w = FluWriter(tok.vocab_size, dim=64, n_layers=2, n_heads=4, max_seq_len=264, concept_dim=512)
ctx = torch.randn(len(seqs), 512)
try:
    loss = w(tgt[:, :-1], ctx, labels=labels)
    print("forward OK, loss:", loss.item())
except Exception as e:
    print("FORWARD ERROR:", e)
    # find bad index
    bad = (tgt[:, :-1] >= tok.vocab_size)
    print("bad indices anywhere:", bad.sum().item(), "values:", tgt[:, :-1][bad].unique().tolist())
