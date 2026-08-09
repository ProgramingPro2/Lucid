"""z → soft-prefix adapter + fiber (vMF) corruption for Stage-A RC training.

Adapter: z(512) → LayerNorm → Linear(512,512) → GELU → Linear(512, K*d_model)
→ reshape (B, K, d_model). ~17M params for K=16, d_model=2048.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the in-repo Wood/Ulrich sampler (fiber contract).
from experiments.sentence_pred.heads import sample_vmf


class SoftPrefixAdapter(nn.Module):
    """Map a unit topic vector z ∈ R^{z_dim} to K soft prefix tokens."""

    def __init__(
        self,
        z_dim: int = 512,
        d_model: int = 2048,
        n_prefix: int = 16,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.n_prefix = int(n_prefix)
        self.ln = nn.LayerNorm(self.z_dim)
        self.fc1 = nn.Linear(self.z_dim, int(hidden))
        self.fc2 = nn.Linear(int(hidden), self.n_prefix * self.d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Small init so prefixes start near zero (CFG-friendly / bf16-safe).
        nn.init.normal_(self.fc2.weight, std=1e-4)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, z_dim) → soft prefix (B, K, d_model). Always computes in fp32."""
        if z.ndim != 2 or z.size(-1) != self.z_dim:
            raise ValueError(f"expected z (B,{self.z_dim}), got {tuple(z.shape)}")
        z32 = z.float()
        h = F.gelu(self.fc1(self.ln(z32)))
        out = self.fc2(h)
        return out.view(z.size(0), self.n_prefix, self.d_model)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def fiber_corrupt(
    z: torch.Tensor,
    kappa: float = 700.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample z̃ ~ vMF(z, κ). z must be (B, D) L2-normalized (or will be)."""
    if z.ndim != 2:
        raise ValueError(f"fiber_corrupt expects (B,D), got {tuple(z.shape)}")
    mu = F.normalize(z.float(), dim=-1)
    k = torch.full((mu.size(0),), float(kappa), device=mu.device, dtype=mu.dtype)
    return sample_vmf(mu, k, generator=generator).to(dtype=z.dtype)


def collate_prefix_batch(
    tokenizer,
    prev_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    max_prev: int = 256,
    max_target: int = 256,
    ctx_max: int = 512,
    n_prefix: int = 16,
    pad_to_multiple: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize [prev; target] for soft-prefix causal LM training.

    Returns:
      input_ids: (B, L)  — prev + target tokens (no soft prefix; those are embeds)
      attention_mask: (B, n_prefix + L)
      labels: (B, n_prefix + L) with -100 on prefix + prev; target ids on target
      prev_lens / target_lens: (B,)
    Soft-prefix slots are reserved at the front of attention_mask/labels; the
    trainer splices adapter outputs into inputs_embeds.
    """
    if len(prev_texts) != len(target_texts):
        raise ValueError("prev_texts and target_texts length mismatch")
    b = len(prev_texts)
    # Encode separately so we know the prev/target boundary.
    prev_enc = tokenizer(
        list(prev_texts),
        add_special_tokens=False,
        truncation=True,
        max_length=max_prev,
        padding=False,
        return_attention_mask=False,
    )
    tgt_enc = tokenizer(
        list(target_texts),
        add_special_tokens=False,
        truncation=True,
        max_length=max_target,
        padding=False,
        return_attention_mask=False,
    )

    eos_id = tokenizer.eos_token_id
    rows: list[list[int]] = []
    prev_lens: list[int] = []
    tgt_lens: list[int] = []
    for i in range(b):
        prev_ids = list(prev_enc["input_ids"][i])
        tgt_ids = list(tgt_enc["input_ids"][i])
        # Budget: total token context ≤ ctx_max (soft prefix is embed-only, not in this budget
        # for the text half — but plan says ctx ≤ 512 for prev+target).
        budget = int(ctx_max)
        if len(prev_ids) + len(tgt_ids) > budget:
            # Prefer keeping the target; truncate prev from the left.
            keep_prev = max(0, budget - len(tgt_ids))
            prev_ids = prev_ids[-keep_prev:] if keep_prev else []
            if len(prev_ids) + len(tgt_ids) > budget:
                tgt_ids = tgt_ids[: max(1, budget - len(prev_ids))]
        if eos_id is not None and (not tgt_ids or tgt_ids[-1] != eos_id):
            if len(prev_ids) + len(tgt_ids) + 1 <= budget:
                tgt_ids = tgt_ids + [eos_id]
        rows.append(prev_ids + tgt_ids)
        prev_lens.append(len(prev_ids))
        tgt_lens.append(len(tgt_ids))

    max_l = max((len(r) for r in rows), default=0)
    if pad_to_multiple and max_l > 0:
        rem = max_l % pad_to_multiple
        if rem:
            max_l += pad_to_multiple - rem
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((b, max_l), pad_id, dtype=torch.long)
    text_attn = torch.zeros((b, max_l), dtype=torch.long)
    for i, r in enumerate(rows):
        if not r:
            continue
        input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        text_attn[i, : len(r)] = 1

    # Prepend n_prefix slots for the soft tokens.
    k = int(n_prefix)
    attn = torch.cat([torch.ones((b, k), dtype=torch.long), text_attn], dim=1)
    labels = torch.full((b, k + max_l), -100, dtype=torch.long)
    for i in range(b):
        pl, tl = prev_lens[i], tgt_lens[i]
        if tl <= 0:
            continue
        # Target tokens sit at positions [k+pl : k+pl+tl] in the full sequence.
        start = k + pl
        end = start + tl
        labels[i, start:end] = input_ids[i, pl : pl + tl]

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "prev_lens": torch.tensor(prev_lens, dtype=torch.long),
        "target_lens": torch.tensor(tgt_lens, dtype=torch.long),
        "n_prefix": torch.tensor(k, dtype=torch.long),
    }


def build_inputs_embeds(
    backbone: nn.Module,
    adapter: SoftPrefixAdapter,
    z: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    drop_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate soft-prefix embeds with token embeds.

    drop_mask: (B,) bool — True ⇒ zero the prefix (CFG / prefix-drop).
    """
    embed = backbone.get_input_embeddings()
    tok_emb = embed(input_ids)  # (B, L, D)
    # Keep adapter MLP in fp32 for stable grads under bf16/4-bit.
    prefix = adapter(z.float())
    prefix = prefix.to(dtype=tok_emb.dtype)
    # Bound soft-prefix magnitude so CE stays finite under bf16.
    prefix = torch.clamp(prefix, -10.0, 10.0)
    if drop_mask is not None:
        prefix = prefix * (~drop_mask).to(dtype=prefix.dtype).view(-1, 1, 1)
    # Ensure grad reaches the adapter through the concat (kbit training).
    # Do not requires_grad_(True) on token embeds — that retains a huge graph.
    if not tok_emb.requires_grad:
        tok_emb = tok_emb.detach()
    out = torch.cat([prefix, tok_emb], dim=1)
    # Guard against rare non-finite embeds before the backbone sees them.
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
    return out


def estimate_adapter_params(
    z_dim: int = 512, d_model: int = 2048, n_prefix: int = 16, hidden: int = 512
) -> int:
    # LN: 2*z_dim; fc1: z_dim*hidden + hidden; fc2: hidden*(K*d) + K*d
    ln = 2 * z_dim
    fc1 = z_dim * hidden + hidden
    fc2 = hidden * (n_prefix * d_model) + n_prefix * d_model
    return ln + fc1 + fc2


def lora_target_modules_for(model_type: str) -> list[str]:
    """Attention + MLP projection names for common 1–2B decoder LMs."""
    mt = (model_type or "").lower()
    if "qwen" in mt or "olmo" in mt or "llama" in mt or "mistral" in mt:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    # Fallback: catch-all common names
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def human_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


# Silence unused import warning for math (kept for potential κ helpers).
_ = math
