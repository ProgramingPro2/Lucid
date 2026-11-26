"""Latent span, depth span, and the answer loss.

The supervised target is the answer tokens. The reasoning span is not teacher
forced: a latent step contributes a code embedding, a text step contributes the
model's own greedy token (detached), and halt contributes nothing. No tensor in
the loss is a target code index.

Training refowards the growing prefix so the gradient is defined. That refoward
is more work than a KV-cache decode of the same number of tokens. Inference
accounting therefore measures the cache path separately (see the trainer) and
does not treat the training refoward as the matched-FLOP number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import VectorQuantizer
from .gate import LATENT, TEXT, EmissionGate, GateDecision


@dataclass
class Span:
    """Prompt plus whatever the gate appended. Batch size is 1."""

    embeds: torch.Tensor
    mask: torch.Tensor
    n_latent: int
    n_text: int
    n_forwards: int
    log_probs: list[torch.Tensor] = field(default_factory=list)
    commitment: torch.Tensor | None = None
    code_indices: list[int] = field(default_factory=list)
    prompt_len: int = 0
    capacity: int = 1

    @property
    def emission_rate(self) -> float:
        """Latent tokens divided by the capacity cap.

        The cap is the position budget the gate was given. Halting after one
        token of a 16-token cap is an emission rate of 1/16, not 100%. A gate
        that halts immediately has rate 0, which is the decorative-path case.
        """
        return float(self.n_latent) / float(self.capacity)


@dataclass
class LossParts:
    loss: torch.Tensor
    ce: torch.Tensor
    charge: torch.Tensor
    commitment: torch.Tensor
    policy: torch.Tensor
    reward: torch.Tensor
    n_latent: int
    n_text: int
    code_indices: list[int]


class TinyBackbone(nn.Module):
    """A causal-shaped toy used by the unit tests. Not a Qwen model."""

    def __init__(self, vocab: int = 48, d_model: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.embed = nn.Embedding(self.vocab, self.d_model)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Tanh())
            for _ in range(self.n_layers)
        )
        self.head = nn.Linear(self.d_model, self.vocab)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids)

    def forward_hidden(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = inputs_embeds
        if attention_mask is not None:
            h = h * attention_mask.to(dtype=h.dtype).unsqueeze(-1)
        for layer in self.layers:
            h = layer(h)
        return h

    def reapply(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One extra application of the last layer. The depth control on the toy."""
        h = hidden
        if attention_mask is not None:
            h = h * attention_mask.to(dtype=h.dtype).unsqueeze(-1)
        return self.layers[-1](h)

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


def _ones_mask(embeds: torch.Tensor) -> torch.Tensor:
    return torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)


def _frontier(backbone, embeds: torch.Tensor, mask: torch.Tensor, depth_repeats: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    hidden = backbone.forward_hidden(embeds, mask)
    forwards = 1
    for _ in range(int(depth_repeats) - 1):
        hidden = backbone.reapply(hidden, mask)
        forwards += 1
    return hidden, hidden[:, -1, :], forwards


def emit(
    backbone,
    gate: EmissionGate,
    codebook: VectorQuantizer,
    prompt_embeds: torch.Tensor,
    *,
    capacity: int,
    depth_repeats: int = 1,
    allow_text: bool = True,
    force_halt: bool = False,
    ablate: bool = False,
) -> Span:
    """Run the gate until halt or the capacity cap.

    ``ablate`` still counts a latent choice, then writes zeros at that position.
    That is "zeroing latents at emission sites": the site is the gate's choice,
    the content is removed. ``force_halt`` never offers a latent and is the
    no-latent path.
    """
    if prompt_embeds.dim() != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError("emit expects a batch of 1")
    if depth_repeats < 1:
        raise ValueError("depth_repeats must be >= 1")
    seq = prompt_embeds
    mask = _ones_mask(seq)
    n_latent = 0
    n_text = 0
    forwards = 0
    log_probs: list[torch.Tensor] = []
    indices: list[int] = []
    commitment: torch.Tensor | None = None
    for _step in range(int(capacity)):
        _hidden, frontier, n_fwd = _frontier(backbone, seq, mask, depth_repeats)
        forwards += n_fwd
        text_embed = None
        if allow_text:
            token = backbone.logits_from_hidden(frontier).argmax(dim=-1)
            text_embed = backbone.embed_tokens(token.view(1, 1)).squeeze(1).detach()
        emitted = torch.tensor(
            [int(capacity) if force_halt else n_latent],
            device=seq.device,
            dtype=torch.long,
        )
        decision: GateDecision = gate(
            frontier,
            codebook,
            emitted=emitted,
            capacity=int(capacity),
            text_embed=text_embed,
            allow_text=allow_text and not force_halt,
        )
        log_probs.append(decision.log_prob[0])
        choice = int(decision.choice[0].item())
        if choice not in (LATENT, TEXT) or not bool(decision.append[0].item()):
            break
        injected = decision.injected
        if ablate and choice == LATENT:
            injected = torch.zeros_like(injected)
        if choice == LATENT:
            n_latent += 1
            indices.append(int(decision.indices[0].item()))
            commitment = decision.commitment if commitment is None else commitment + decision.commitment
            if codebook.training:
                codebook.ema_update(
                    gate.to_code(frontier).detach().reshape(1, -1),
                    decision.indices[:1].detach(),
                )
        else:
            n_text += 1
        seq = torch.cat([seq, injected.unsqueeze(1)], dim=1)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype, device=mask.device)], dim=1)
    if commitment is None:
        commitment = prompt_embeds.new_zeros(())
    elif n_latent > 0:
        commitment = commitment / float(n_latent)
    return Span(
        embeds=seq,
        mask=mask,
        n_latent=n_latent,
        n_text=n_text,
        n_forwards=forwards,
        log_probs=log_probs,
        commitment=commitment,
        code_indices=indices,
        prompt_len=int(prompt_embeds.shape[1]),
        capacity=int(capacity),
    )


def emit_depth(
    backbone,
    depth_token: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    capacity: int,
    depth_repeats: int = 1,
) -> Span:
    """Append the same learned vector ``capacity`` times. No codebook, no halt.

    The positions and the extra block applications match arm A. The content is
    one vector, not a discrete code, which is the engineering control for
    "extra depth did the work".
    """
    if prompt_embeds.shape[0] != 1:
        raise ValueError("emit_depth expects a batch of 1")
    if depth_token.shape != (prompt_embeds.shape[-1],):
        raise ValueError("depth token must be (d_model,)")
    seq = prompt_embeds
    mask = _ones_mask(seq)
    forwards = 0
    for _ in range(int(capacity)):
        _hidden, _frontier, n_fwd = _frontier(backbone, seq, mask, depth_repeats)
        forwards += n_fwd
        seq = torch.cat([seq, depth_token.view(1, 1, -1)], dim=1)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype, device=mask.device)], dim=1)
    return Span(
        embeds=seq,
        mask=mask,
        n_latent=0,
        n_text=0,
        n_forwards=forwards,
        commitment=prompt_embeds.new_zeros(()),
        prompt_len=int(prompt_embeds.shape[1]),
        capacity=int(capacity),
    )


def answer_nll(backbone, prefix: torch.Tensor, mask: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of ``answer_ids`` given ``prefix``. The only content target."""
    if answer_ids.dim() == 1:
        answer_ids = answer_ids.unsqueeze(0)
    if answer_ids.shape[0] != 1 or answer_ids.numel() == 0:
        raise ValueError("answer_ids must be a non-empty batch of 1")
    answer_embeds = backbone.embed_tokens(answer_ids)
    full = torch.cat([prefix, answer_embeds], dim=1)
    full_mask = torch.cat(
        [mask, torch.ones(1, answer_ids.shape[1], dtype=mask.dtype, device=mask.device)],
        dim=1,
    )
    hidden = backbone.forward_hidden(full, full_mask)
    logits = backbone.logits_from_hidden(hidden)
    start = prefix.shape[1] - 1
    pred = logits[:, start : start + answer_ids.shape[1], :]
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), answer_ids.reshape(-1))


def span_loss(
    backbone,
    span: Span,
    answer_ids: torch.Tensor,
    *,
    lambda_token: float,
    commitment_beta: float,
    baseline: torch.Tensor | None,
    policy_coef: float = 1.0,
) -> tuple[LossParts, torch.Tensor]:
    """Answer cross-entropy, the λ charge, the geometry anchor, and the gate.

    The charge and the policy term are how λ enters the objective. The charge
    is added as a number so the reported loss contains it. Its gradient reaches
    the gate only through the advantage, because the token count is discrete.
    ``baseline`` is an exponential moving average of the reward; the first step
    has no baseline and therefore no policy gradient.
    """
    ce = answer_nll(backbone, span.embeds, span.mask, answer_ids)
    charge = ce.new_tensor(float(span.n_text) * float(lambda_token))
    commit = span.commitment if span.commitment is not None else ce.new_zeros(())
    commit = commit.to(device=ce.device, dtype=ce.dtype)
    reward = -(ce.detach() + charge.detach())
    if baseline is None or not span.log_probs:
        policy = ce.new_zeros(())
        new_baseline = reward.detach()
    else:
        advantage = reward.detach() - baseline.detach()
        log_prob = torch.stack([lp.to(device=ce.device, dtype=ce.dtype) for lp in span.log_probs]).sum()
        policy = -float(policy_coef) * log_prob * advantage.detach()
        new_baseline = (0.9 * baseline.detach() + 0.1 * reward.detach()).detach()
    loss = ce + float(commitment_beta) * commit + charge + policy
    parts = LossParts(
        loss=loss,
        ce=ce,
        charge=charge,
        commitment=commit.detach(),
        policy=policy.detach(),
        reward=reward.detach(),
        n_latent=span.n_latent,
        n_text=span.n_text,
        code_indices=list(span.code_indices),
    )
    return parts, new_baseline


def text_target_nll(backbone, prompt_embeds: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of an explicit text trace given the prompt.

    Used by B_text. The trace is the generator's solution plus the answer, so
    this arm is supervised on the text the latent arm is deliberately denied.
    """
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)
    target_embeds = backbone.embed_tokens(target_ids)
    full = torch.cat([prompt_embeds, target_embeds], dim=1)
    mask = _ones_mask(full)
    hidden = backbone.forward_hidden(full, mask)
    logits = backbone.logits_from_hidden(hidden)
    start = prompt_embeds.shape[1] - 1
    pred = logits[:, start : start + target_ids.shape[1], :]
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target_ids.reshape(-1))
