"""Phase-2.5 two-channel writer: label masks, prefix generate, drop arms, per-row penalty."""
from __future__ import annotations

import torch

from conceptpoc.jepa.tokenizer import Tokenizer
from conceptpoc.writer.model import FluWriter
from conceptpoc.writer.train_phase2 import (
    apply_drop_arms,
    make_collate,
)


def test_label_masking_prefix_and_separator():
    """Prefix positions -100; separator supervises first target; target+final EOS kept."""
    tok = Tokenizer("char")  # bos/eos/pad unambiguous; no HF download
    collate = make_collate(tok, seq_len=256)
    # prev = [10, 11, 12], tgt = [20, 21]
    # (char vocab is small — use ids in range)
    prev = torch.tensor([10, 11, 12], dtype=torch.long)
    tgt = torch.tensor([20, 21], dtype=torch.long)
    concept = torch.randn(8)
    inputs, labels, concepts = collate([(prev, tgt, concept, True)])
    assert inputs.shape == labels.shape
    assert concepts.shape[0] == 1

    # Full sequence: [BOS] + prev + [EOS] + tgt + [EOS]
    full = [tok.bos_id] + prev.tolist() + [tok.eos_id] + tgt.tolist() + [tok.eos_id]
    # inputs = full[:-1], labels align to full[1:] with masks
    assert inputs[0].tolist() == full[:-1]

    n_mask = 1 + len(prev)  # BOS..last-prev predictions masked
    assert (labels[0, :n_mask] == -100).all()
    # Separator position (= index n_mask) supervises first target token
    assert labels[0, n_mask].item() == tgt[0].item()
    # Remaining target tokens + final EOS supervised
    assert labels[0, n_mask + 1].item() == tgt[1].item()
    assert labels[0, n_mask + 2].item() == tok.eos_id
    # No pad_id left unmasked as a real label
    assert (labels[0] != tok.pad_id).all()


def test_label_masking_opening_form_all_supervised():
    tok = Tokenizer("char")
    collate = make_collate(tok, seq_len=256)
    tgt = torch.tensor([20, 21, 22], dtype=torch.long)
    concept = torch.randn(8)
    inputs, labels, _ = collate([(torch.zeros(0, dtype=torch.long), tgt, concept, False)])
    full = [tok.bos_id] + tgt.tolist() + [tok.eos_id]
    assert inputs[0].tolist() == full[:-1]
    # Opening form: every non-pad label supervised (BOS predicts first target)
    assert (labels[0] != -100).all()
    assert labels[0].tolist() == full[1:]


def test_per_row_repetition_penalty():
    """Two rows with different seen tokens must not share a batch-global seen list."""
    vocab = 50
    w = FluWriter(
        vocab, dim=32, n_layers=2, n_heads=2, max_seq_len=32, concept_dim=8
    )
    concept = torch.zeros(2, 8)

    def controlled_forward(input_ids, concept=None, labels=None):
        b, t = input_ids.shape
        out = torch.zeros(b, t, vocab)
        if t == 1:
            # First new token: row0 -> 5, row1 -> 9
            out[0, -1, 5] = 10.0
            out[1, -1, 9] = 10.0
        else:
            # Second step: without per-row fix, global seen={5,9} would push both
            # to token 3. Per-row: row0 keeps 9 unpenalized, row1 keeps 5.
            out[:, -1, :] = 1.0
            out[0, -1, 5] = 8.0
            out[0, -1, 9] = 8.0
            out[0, -1, 3] = 7.0
            out[1, -1, 5] = 8.0
            out[1, -1, 9] = 8.0
            out[1, -1, 3] = 7.0
        return out

    w.forward = controlled_forward  # type: ignore[method-assign]
    out = w.generate(
        concept, max_tokens=2, bos_id=1, temperature=0.0, repetition_penalty=2.0
    )
    assert out.shape == (2, 2)
    # Per-row: second tokens are 9 and 5 (not both 3 from a global seen union)
    assert out[0, 0].item() == 5
    assert out[1, 0].item() == 9
    assert out[0, 1].item() == 9, f"row0 second={out[0, 1].item()} (global-seen bug => 3)"
    assert out[1, 1].item() == 5, f"row1 second={out[1, 1].item()} (global-seen bug => 3)"


def test_prefix_generate_round_trip():
    vocab = 128
    w = FluWriter(
        vocab, dim=32, n_layers=2, n_heads=2, max_seq_len=64, concept_dim=8
    )
    concept = torch.randn(2, 8)
    torch.manual_seed(0)
    # No prefix: returns only NEW tokens (not the BOS prime)
    out_none = w.generate(concept, max_tokens=5, bos_id=1, temperature=0.0)
    assert out_none.shape == (2, 5)

    prefix = torch.tensor([[1, 10, 11, 2], [1, 20, 21, 2]], dtype=torch.long)
    torch.manual_seed(0)
    out_pref = w.generate(
        concept, max_tokens=4, bos_id=1, temperature=0.0, prefix_ids=prefix
    )
    assert out_pref.shape == (2, 4)
    # New tokens only — must not echo the prefix
    assert not torch.equal(out_pref[:, :4], prefix[:, :4])


def test_drop_arm_logic():
    """prefix_present alternation via drop arms; never both arms; seed 42+idx."""
    concept = torch.ones(4)
    prev = torch.tensor([1, 2, 3], dtype=torch.long)

    # Collect outcomes over many indices
    n_drop_prefix = 0
    n_zero_concept = 0
    n_both = 0
    n = 200
    for idx in range(n):
        p, c, present = apply_drop_arms(idx, prev, concept.clone(), True)
        dropped = not present
        zeroed = bool(torch.all(c == 0))
        if dropped:
            n_drop_prefix += 1
        if zeroed:
            n_zero_concept += 1
        if dropped and zeroed:
            n_both += 1
    assert n_both == 0, "drop arms must never both fire"
    # Roughly 10% each
    assert 5 < n_drop_prefix < 40, f"drop_prefix count={n_drop_prefix}"
    assert 5 < n_zero_concept < 40, f"zero_concept count={n_zero_concept}"

    # Deterministic per index
    a = apply_drop_arms(7, prev, concept.clone(), True)
    b = apply_drop_arms(7, prev, concept.clone(), True)
    assert a[2] == b[2]
    assert torch.equal(a[1], b[1])

    # i==0 style (already no prefix): still never invents a prefix
    _, _, present = apply_drop_arms(0, torch.zeros(0, dtype=torch.long), concept, False)
    assert present is False
