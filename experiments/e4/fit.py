"""Fit SoftPrefixAdapter on a frozen refiner.

E4-S1 does not train the refiner (``docs/EXPERIMENTS.md`` §5). The adapter is
the interface: z goes in, K soft-prefix tokens come out, and the frozen LM's
token loss is the only training signal. Three fits are the manifest, because
one fit drops the projection-training component out of the interval
(``docs/PROTOCOL.md`` §7.1).

Fit seeds are experimental units (0, 1, 2 by default), not ``seed + k`` offsets
of a single run. Initialisation is the ``head_init`` stream. Example order is
the ``dropout`` stream, which the frozen refiner does not otherwise use, so
the draw cannot move the ``data`` stream that built the items.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from experiments.pivot.adapter import SoftPrefixAdapter, build_inputs_embeds, collate_prefix_batch
from experiments.protocol.seeds import spawn_key

# The confirmatory fits. Each one is its own seed.
DEFAULT_FIT_SEEDS: tuple[int, ...] = (0, 1, 2)


def fit_adapter(
    backbone: torch.nn.Module,
    tokenizer,
    examples: Sequence,
    *,
    steps: int,
    seed: int,
    device: str | torch.device = "cpu",
    lr: float = 5e-4,
    n_prefix: int = 16,
    hidden: int = 512,
    z_dim: int = 512,
    batch_size: int = 1,
    max_prev: int = 128,
    max_target: int = 64,
    prefix_drop_p: float = 0.0,
    grad_clip: float = 1.0,
) -> tuple[SoftPrefixAdapter, dict]:
    """Train one adapter. The backbone's weights are not stepped.

    ``examples`` items need ``prompt``, ``target`` and ``z`` (a 1-D array).
    Returns the adapter and a small record (loss, skipped non-finite steps).
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if len(examples) < 1:
        raise ValueError("no fit examples")
    device = torch.device(device)
    d_model = int(backbone.config.hidden_size)

    # nn.init reads the global RNG. Seed it from the head_init stream.
    torch.manual_seed(spawn_key(seed, "head_init") % (2**63 - 1))
    adapter = SoftPrefixAdapter(
        z_dim=z_dim, d_model=d_model, n_prefix=n_prefix, hidden=hidden
    ).to(device)

    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False
    if hasattr(backbone, "gradient_checkpointing_enable") and device.type == "cuda":
        backbone.gradient_checkpointing_enable()

    adapter.train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    order = np.random.default_rng(spawn_key(seed, "dropout"))
    embed_ref = backbone.get_input_embeddings().weight.detach().clone()

    losses: list[float] = []
    nan_skips = 0
    cursor = 0
    indices = order.permutation(len(examples))

    for _step in range(steps):
        if cursor + batch_size > len(indices):
            indices = order.permutation(len(examples))
            cursor = 0
        batch_idx = indices[cursor : cursor + batch_size]
        cursor += batch_size
        batch = [examples[int(i)] for i in batch_idx]
        prev = [ex.prompt for ex in batch]
        tgt = [ex.target for ex in batch]
        z = torch.stack(
            [torch.as_tensor(np.asarray(ex.z), dtype=torch.float32) for ex in batch]
        ).to(device)

        collated = collate_prefix_batch(
            tokenizer,
            prev,
            tgt,
            max_prev=max_prev,
            max_target=max_target,
            ctx_max=max_prev + max_target,
            n_prefix=n_prefix,
        )
        input_ids = collated["input_ids"].to(device)
        attn = collated["attention_mask"].to(device)
        labels = collated["labels"].to(device)
        drop = None
        if prefix_drop_p > 0:
            drop = torch.rand(z.size(0), device=device) < prefix_drop_p

        embeds = build_inputs_embeds(backbone, adapter, z, input_ids, drop_mask=drop)
        out = backbone(inputs_embeds=embeds, attention_mask=attn, labels=labels)
        loss = out.loss
        if loss is None or not bool(torch.isfinite(loss).detach()):
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach().float()))

    if not torch.equal(embed_ref, backbone.get_input_embeddings().weight.detach()):
        raise RuntimeError("frozen refiner embedding changed during the adapter fit")

    adapter.eval()
    # The next caller decodes. Checkpointing and a disabled cache are training
    # settings; leaving them on makes generation wrong or very slow.
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    if hasattr(backbone, "config"):
        backbone.config.use_cache = True
    if not losses:
        raise RuntimeError(f"every fit step was non-finite ({nan_skips} skips)")
    return adapter, {
        "seed": int(seed),
        "steps": int(steps),
        "steps_finite": len(losses),
        "nan_skips": int(nan_skips),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "n_trainable": int(adapter.n_trainable()),
        "d_model": d_model,
        "n_prefix": int(n_prefix),
        "z_dim": int(z_dim),
    }


def fit_replicates(
    backbone,
    tokenizer,
    examples: Sequence,
    seeds: Sequence[int] = DEFAULT_FIT_SEEDS,
    **kwargs,
) -> list[tuple[SoftPrefixAdapter, dict]]:
    """One adapter per seed. The backbone stays frozen across all of them."""
    return [
        fit_adapter(backbone, tokenizer, examples, seed=int(s), **kwargs) for s in seeds
    ]
