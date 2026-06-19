"""Autoregressive text decoder conditioned on a concept vector.

The concept vector ``z`` is projected to a single memory token that the decoder
cross-attends to. Training uses teacher forcing (predict each token from the
prefix); inference decodes greedily or with temperature sampling until EOS.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import JepaConfig
from .modules import PositionalEncoding


class TextDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: JepaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_tokens + 2)
        self.from_concept = nn.Linear(cfg.dim, cfg.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.dec_heads,
            dim_feedforward=cfg.d_model * cfg.dec_ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.dec_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size)

    def _memory(self, z: torch.Tensor) -> torch.Tensor:
        return self.from_concept(z).unsqueeze(1)  # (B, 1, d_model)

    def forward(
        self, z: torch.Tensor, tgt_in: torch.Tensor, tgt_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """z: (B, dim); tgt_in: (B, L); returns logits (B, L, vocab)."""
        seq_len = tgt_in.size(1)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt_in.device),
            diagonal=1,
        )
        h = self.pos(self.embed(tgt_in))
        h = self.decoder(
            h,
            self._memory(z),
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.lm_head(self.norm(h))

    def teacher_forcing_logits(
        self, z: torch.Tensor, ids: torch.Tensor, pad_mask: torch.Tensor, bos_id: int
    ) -> torch.Tensor:
        """Build a bos-prefixed input and return logits aligned to predict ``ids``."""
        bos = torch.full((ids.size(0), 1), bos_id, dtype=torch.long, device=ids.device)
        tgt_in = torch.cat([bos, ids[:, :-1]], dim=1)
        false_col = torch.zeros(ids.size(0), 1, dtype=torch.bool, device=ids.device)
        tgt_in_pad = torch.cat([false_col, pad_mask[:, :-1]], dim=1)
        return self.forward(z, tgt_in, tgt_in_pad)

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> list[list[int]]:
        """Autoregressively decode token ids for each row of ``z``.

        ``temperature == 0`` (or no sampling knobs) decodes greedily. Otherwise
        logits are temperature-scaled and optionally restricted to the top-``k``
        tokens and/or the smallest set whose cumulative probability exceeds
        ``top_p`` (nucleus) before sampling.
        """
        self.eval()
        device = z.device
        batch = z.size(0)
        tokens = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        outputs: list[list[int]] = [[] for _ in range(batch)]

        for _ in range(max_tokens):
            pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
            logits = self.forward(z, tokens, pad_mask)[:, -1]
            if temperature and temperature > 0:
                logits = _filter_logits(logits / temperature, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(1)
            else:
                nxt = logits.argmax(dim=-1)
            for b in range(batch):
                if not done[b]:
                    if int(nxt[b]) == eos_id:
                        done[b] = True
                    else:
                        outputs[b].append(int(nxt[b]))
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            if bool(done.all()):
                break
        return outputs


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask logits (B, V) to the top-k and/or nucleus (top-p) set with -inf."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Keep tokens up to and including the one that crosses top_p.
        remove = cum - F.softmax(sorted_logits, dim=-1) >= top_p
        remove_orig = remove.gather(-1, sorted_idx.argsort(dim=-1))
        logits = logits.masked_fill(remove_orig, float("-inf"))
    return logits
