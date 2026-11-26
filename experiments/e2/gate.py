"""Top-1 emission gate: halt, one latent code, or one text token.

The hard choice is argmax (the manifest's top-1 gate), not a sample. The
straight-through path for *content* runs through the codebook and the
up-projection. The choice itself is trained by the log-probability of the
argmax action (see the trainer): that path reads a detached hidden state, so
the policy gradient stops at the gate and does not invent a second content loss.

Selecting halt appends nothing. That is the no-latent path, and it is a legal
output of the same gate. The capacity cap forces halt once ``capacity`` latent
tokens have already been emitted, even if the logits still prefer a code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import Quantized, VectorQuantizer

HALT = 0
LATENT = 1
TEXT = 2
N_ACTIONS = 3


@dataclass
class GateDecision:
    """One greedy decision for a batch of frontier states."""

    choice: torch.Tensor          # (batch,) long
    log_prob: torch.Tensor        # (batch,) log-prob of the greedy action
    injected: torch.Tensor        # (batch, d_model); zero on halt
    append: torch.Tensor          # (batch,) bool, False on halt
    indices: torch.Tensor         # (batch,) code index, -1 unless latent
    commitment: torch.Tensor      # scalar geometry term, 0 if no latent row
    n_latent: int
    n_text: int

    @property
    def halted_all(self) -> bool:
        return bool((~self.append).all().item()) if self.append.numel() else True


class EmissionGate(nn.Module):
    """Discrete gate on top of a backbone hidden state.

    The action bias starts at ``(-1, +2, -1)`` so an untrained gate emits a
    latent rather than already taking the no-latent path. That prior is fixed;
    it is not fit to a result. The capacity cap and a halt action can still
    select zero latent tokens.
    """

    def __init__(self, d_model: int, code_dim: int) -> None:
        super().__init__()
        if d_model < 1 or code_dim < 1:
            raise ValueError("d_model and code_dim must be positive")
        self.d_model = int(d_model)
        self.code_dim = int(code_dim)
        self.to_code = nn.Linear(self.d_model, self.code_dim)
        self.up = nn.Linear(self.code_dim, self.d_model)
        self.action = nn.Linear(self.d_model, N_ACTIONS)
        nn.init.normal_(self.to_code.weight, std=0.02)
        nn.init.zeros_(self.to_code.bias)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.action.weight)
        self.action.bias.data.copy_(torch.tensor([-1.0, 2.0, -1.0]))

    def forward(
        self,
        hidden: torch.Tensor,
        codebook: VectorQuantizer,
        *,
        emitted: torch.Tensor,
        capacity: int,
        text_embed: torch.Tensor | None = None,
        allow_text: bool = True,
    ) -> GateDecision:
        """Decide one step.

        ``hidden`` is ``(batch, d_model)``, the frontier state. ``emitted`` is
        a long tensor ``(batch,)`` counting latent tokens already kept.
        ``capacity`` is the hard cap. Rows at the cap halt.
        """
        if hidden.dim() != 2 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must be (batch, {self.d_model})")
        if capacity < 1:
            raise ValueError("capacity must be at least 1 so the cap is real")
        batch = hidden.shape[0]
        if emitted.shape != (batch,):
            raise ValueError("emitted must be (batch,)")

        # Policy logits see a detached state: the choice is not a content loss.
        logits = self.action(hidden.detach())
        over = emitted.to(device=hidden.device) >= int(capacity)
        if bool(over.any()):
            blocked = torch.zeros_like(logits)
            blocked[:, LATENT] = -1e9
            blocked[:, TEXT] = -1e9
            logits = logits + blocked * over.to(dtype=logits.dtype).unsqueeze(-1)
        if not allow_text:
            logits = logits.clone()
            logits[:, TEXT] = -1e9

        log_probs = F.log_softmax(logits, dim=-1)
        choice = logits.argmax(dim=-1)
        chosen_lp = log_probs.gather(1, choice.unsqueeze(-1)).squeeze(-1)

        # Content path. Gradients reach ``to_code`` and, through straight-through,
        # the backbone that produced ``hidden``. They do not reach the table.
        quant: Quantized = codebook(self.to_code(hidden), update=False)
        latent_embed = self.up(quant.z)
        if text_embed is None:
            text_embed = torch.zeros_like(latent_embed)
        elif text_embed.shape != latent_embed.shape:
            raise ValueError("text_embed must match the latent embedding shape")

        use_latent = (choice == LATENT).to(dtype=latent_embed.dtype).unsqueeze(-1)
        use_text = (choice == TEXT).to(dtype=latent_embed.dtype).unsqueeze(-1)
        injected = use_latent * latent_embed + use_text * text_embed.detach()
        append = choice != HALT

        latent_rows = choice == LATENT
        if bool(latent_rows.any()):
            commitment = quant.commitment
        else:
            commitment = hidden.new_zeros(())
        indices = torch.where(latent_rows, quant.indices, torch.full_like(quant.indices, -1))
        return GateDecision(
            choice=choice,
            log_prob=chosen_lp,
            injected=injected,
            append=append,
            indices=indices,
            commitment=commitment,
            n_latent=int(latent_rows.sum().item()),
            n_text=int((choice == TEXT).sum().item()),
        )
