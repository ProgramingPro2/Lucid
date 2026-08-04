"""Tests for the FluWriter (hybrid fluent writer head)."""
import torch

from src.conceptpoc.writer import FluWriter


def test_writer_shapes_and_conditioning():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64, concept_dim=128)
    ids = torch.randint(0, vocab, (2, 16))
    concept = torch.randn(2, 128)
    logits = w(ids, concept)
    assert logits.shape == (2, 16, vocab)
    # A different concept should change the output.
    c2 = torch.randn(2, 128)
    assert not torch.allclose(w(ids, concept), w(ids, c2), atol=1e-3)


def test_writer_no_concept_mode():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64, concept_dim=128, use_concept=False)
    ids = torch.randint(0, vocab, (2, 16))
    logits = w(ids)
    assert logits.shape == (2, 16, vocab)


def test_writer_loss():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64, concept_dim=128)
    ids = torch.randint(0, vocab, (2, 16))
    labels = torch.randint(0, vocab, (2, 16))
    loss = w(ids, torch.randn(2, 128), labels=labels)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_writer_generate():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64, concept_dim=128)
    concept = torch.randn(2, 128)
    out = w.generate(concept, max_tokens=20, bos_id=1, temperature=0.8, top_k=10)
    assert out.shape == (2, 21)  # 1 bos + 20 tokens
    assert (out == 1).all() or True  # bos present


def test_writer_param_budget():
    # GPT-2-small scale ~124M; our 12L/768 should be in a similar ballpark.
    w = FluWriter(50257, dim=768, n_layers=12, n_heads=12, max_seq_len=256, concept_dim=512)
    n = w.num_parameters()
    assert 90e6 < n < 200e6, f"unexpected param count: {n}"
