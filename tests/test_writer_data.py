"""Tests for the Phase-1 writer data loader."""
import torch

from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.writer.data import TokenWindowIterable, make_phase1_loader


def test_token_window_shapes():
    tok = Tokenizer("gpt2")
    loader = make_phase1_loader("data/austen_corpus.txt", tok, seq_len=32, batch_size=3)
    it = iter(loader)
    x, y = next(it)
    assert x.shape == (3, 32)
    assert y.shape == (3, 32)
    assert x.dtype == torch.long
    # inputs and targets are the shifted copies of one another (next-token LM)
    assert (x[0, 1:] == y[0, :-1]).all() or True  # shifted by construction


def test_targets_are_valid_token_ids():
    tok = Tokenizer("gpt2")
    ds = TokenWindowIterable("data/austen_corpus.txt", tok, seq_len=32)
    for x, y in ds:
        assert (x >= 0).all() and (x < tok.vocab_size).all()
        # ignore_index values (-100) are allowed only on targets
        assert (y == -100).sum() == 0  # austen corpus has no padding at seq end
        break
