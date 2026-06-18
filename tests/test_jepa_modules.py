from __future__ import annotations

import torch

from conceptpoc.config import JepaConfig
from conceptpoc.jepa.decoder import TextDecoder
from conceptpoc.jepa.ema import EmaEncoder
from conceptpoc.jepa.encoder import TextEncoder
from conceptpoc.jepa.tokenizer import Tokenizer


def _cfg() -> JepaConfig:
    return JepaConfig(
        dim=32, d_model=32, enc_layers=1, dec_layers=1, enc_heads=4, dec_heads=4,
        tokenizer="char", max_tokens=20,
    )


def test_encoder_output_normalized():
    tok = Tokenizer("char")
    enc = TextEncoder(tok.vocab_size, _cfg())
    ids, pad = tok.encode_batch(["the river moved", "birds rose up high"], 20)
    z = enc(ids, pad)
    assert z.shape == (2, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-4)


def test_encoder_returns_raw():
    tok = Tokenizer("char")
    enc = TextEncoder(tok.vocab_size, _cfg())
    ids, pad = tok.encode_batch(["hello there friend"], 20)
    z, raw = enc(ids, pad, return_raw=True)
    assert z.shape == raw.shape == (1, 32)
    # raw is unnormalized; z is its normalization.
    assert torch.allclose(torch.nn.functional.normalize(raw, dim=-1), z, atol=1e-5)


def test_decoder_logits_and_generate():
    tok = Tokenizer("char")
    dec = TextDecoder(tok.vocab_size, _cfg())
    z = torch.randn(2, 32)
    ids, pad = tok.encode_batch(["abc def", "ghi jkl mno"], 20)
    logits = dec.teacher_forcing_logits(z, ids, pad, tok.bos_id)
    assert logits.shape == (2, ids.size(1), tok.vocab_size)

    out = dec.generate(z, tok.bos_id, tok.eos_id, max_tokens=10)
    assert len(out) == 2
    assert all(isinstance(seq, list) for seq in out)


def test_ema_tracks_online():
    tok = Tokenizer("char")
    enc = TextEncoder(tok.vocab_size, _cfg())
    ema = EmaEncoder(enc, momentum=0.5)
    # Perturb the online encoder, then confirm EMA moves toward it.
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    before = next(iter(ema.target.parameters())).clone()
    ema.update(enc)
    after = next(iter(ema.target.parameters()))
    assert not torch.allclose(before, after)
