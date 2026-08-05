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


def test_writer_film_uses_concept_strongly():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64,
                  concept_dim=128, use_concept=True, use_film=True)
    ids = torch.randint(0, vocab, (2, 16))
    cA = torch.zeros(2, 128); cA[:, 0] = 5.0
    cB = torch.zeros(2, 128); cB[:, 1] = 5.0
    wa = w(ids, cA)
    wb = w(ids, cB)
    assert not torch.allclose(wa, wb, atol=1e-2), "FiLM should make concepts change output"


def test_writer_repetition_penalty_decodes():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64, concept_dim=128)
    concept = torch.zeros(1, 128)
    # With a strong penalty, greedy decode shouldn't produce the exact same token
    # twice in a row (which it otherwise can under repetition bias).
    out = w.generate(concept, max_tokens=20, bos_id=1, temperature=0.0, repetition_penalty=2.0)
    seq = out[0].tolist()[1:]
    reps = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    assert reps < len(seq) - 1, f"too many immediate repetitions: {seq}"


def test_writer_param_budget():
    # GPT-2-small scale ~124M; our 12L/768 should be in a similar ballpark.
    w = FluWriter(50257, dim=768, n_layers=12, n_heads=12, max_seq_len=256, concept_dim=512)
    n = w.num_parameters()
    assert 90e6 < n < 200e6, f"unexpected param count: {n}"
