from __future__ import annotations

from conceptpoc.jepa.tokenizer import Tokenizer


def test_char_round_trip():
    tok = Tokenizer("char")
    text = "The river moved slowly."
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_special_ids_distinct():
    tok = Tokenizer("char")
    assert len({tok.pad_id, tok.bos_id, tok.eos_id}) == 3
    assert tok.vocab_size > 10


def test_encode_batch_shapes_and_eos():
    tok = Tokenizer("char")
    ids, pad = tok.encode_batch(["short.", "a much longer sentence here."], 32)
    assert ids.shape == pad.shape
    assert ids.shape[0] == 2
    # First row is shorter, so it must contain padding; second row drives length.
    assert pad[0].sum() > pad[1].sum()
    # EOS appears in each row (appended before padding).
    assert (ids == tok.eos_id).any(dim=1).all()


def test_decode_strips_specials():
    tok = Tokenizer("char")
    ids = [tok.bos_id] + tok.encode("hello") + [tok.eos_id, tok.pad_id]
    assert tok.decode(ids) == "hello"
