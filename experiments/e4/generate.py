"""Decode the six E4-S1 arms.

Latent arms share the prompt. Only the soft prefix changes, which is what
makes the difference a use-of-draft measure. The text-prefix arms put the
draft in the text and zero the prefix, so a failure there is a readout
failure. The null pair samples, with the second arm keyed off ``seed + 10000``
as ``docs/EXPERIMENTS.md`` §2 requires, because a greedy pair is a point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.e4.data import E4Item, NullItem
from experiments.pivot.adapter import SoftPrefixAdapter
from experiments.protocol.seeds import spawn_key


@dataclass(frozen=True)
class DecodeConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    batch_size: int = 1
    max_prompt_tokens: int = 256


def draft_inputs(item: E4Item, arm: str) -> tuple[np.ndarray, bool, str]:
    """Return ``(z, drop_prefix, prompt)`` for one latent or text-prefix arm."""
    if arm == "true":
        return item.z_true, False, item.prompt
    if arm == "shuf":
        return item.z_shuffled, False, item.prompt
    if arm == "none":
        return item.z_none, True, item.prompt
    if arm == "dose50":
        return item.z_dose, False, item.prompt
    if arm == "textpc_true":
        return item.z_none, True, item.text_prefix_true
    if arm == "textpc_rand":
        return item.z_none, True, item.text_prefix_random
    raise ValueError(f"unknown arm {arm}")


def null_seed(seed: int, arm: str) -> int:
    """Decode-stream seed. Arm ``b`` uses ``seed + 10000`` before the hash.

    That offset is the one exception ``docs/EXPERIMENTS.md`` §2 writes down.
    The hash still runs, so the generator is not ``seed + 10000`` raw.
    """
    if arm == "a":
        base = seed
    elif arm == "b":
        base = seed + 10000
    else:
        raise ValueError(arm)
    return spawn_key(base, "decode") % (2**63 - 1)


@torch.no_grad()
def generate_prefixed(
    model: torch.nn.Module,
    tokenizer,
    adapter: SoftPrefixAdapter,
    prompts: list[str],
    zs: np.ndarray,
    drop: np.ndarray,
    cfg: DecodeConfig,
    *,
    seed: int | None = None,
) -> list[str]:
    """Soft-prefix generation. ``drop[i]`` zeros the prefix (the no-draft arm)."""
    if len(prompts) == 0:
        return []
    if seed is not None:
        torch.manual_seed(int(seed))

    device = next(model.parameters()).device
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        out: list[str] = []
        for start in range(0, len(prompts), cfg.batch_size):
            batch_p = prompts[start : start + cfg.batch_size]
            enc = tokenizer(
                batch_p,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_tokens,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            tok_mask = enc["attention_mask"].to(device)
            z = torch.as_tensor(
                zs[start : start + len(batch_p)], dtype=torch.float32, device=device
            )
            embed = model.get_input_embeddings()
            tok_emb = embed(input_ids)
            prefix = adapter(z).to(dtype=tok_emb.dtype)
            prefix = torch.clamp(prefix, -10.0, 10.0)
            drop_b = torch.as_tensor(
                drop[start : start + len(batch_p)], dtype=torch.bool, device=device
            )
            prefix = prefix * (~drop_b).to(dtype=prefix.dtype).view(-1, 1, 1)
            inputs = torch.cat([prefix, tok_emb], dim=1)
            attn = torch.cat(
                [
                    torch.ones(prefix.shape[:2], dtype=torch.long, device=device),
                    tok_mask,
                ],
                dim=1,
            )
            gen_kwargs = dict(
                inputs_embeds=inputs,
                attention_mask=attn,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if cfg.do_sample:
                gen_kwargs["temperature"] = cfg.temperature
            gen = model.generate(**gen_kwargs)
            # inputs_embeds generation returns new tokens in current transformers,
            # and a longer sequence in some older ones. Keep only the new part
            # when the sequence grew by more than the prompt.
            prompt_len = input_ids.shape[1]
            if gen.shape[1] > prompt_len:
                new_ids = gen[:, prompt_len:]
            else:
                new_ids = gen
            out.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return [t.strip() for t in out]
    finally:
        tokenizer.padding_side = padding_side


def generate_arm(
    model,
    tokenizer,
    adapter: SoftPrefixAdapter,
    items: list[E4Item],
    arm: str,
    cfg: DecodeConfig,
    *,
    seed: int | None = None,
) -> list[str]:
    zs = []
    drops = []
    prompts = []
    for item in items:
        z, drop, prompt = draft_inputs(item, arm)
        zs.append(z)
        drops.append(drop)
        prompts.append(prompt)
    return generate_prefixed(
        model,
        tokenizer,
        adapter,
        prompts,
        np.stack(zs),
        np.asarray(drops, dtype=bool),
        cfg,
        seed=seed,
    )


def generate_null(
    model,
    tokenizer,
    adapter: SoftPrefixAdapter,
    items: list[NullItem],
    cfg: DecodeConfig,
    *,
    seed: int,
    arm: str,
) -> list[str]:
    """Sampled decode, zero prefix. The seed differs between arms a and b."""
    sampled = DecodeConfig(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
        batch_size=cfg.batch_size,
        max_prompt_tokens=cfg.max_prompt_tokens,
    )
    zdim = adapter.z_dim
    zs = np.zeros((len(items), zdim), dtype=np.float32)
    drops = np.ones(len(items), dtype=bool)
    prompts = [item.prompt for item in items]
    return generate_prefixed(
        model,
        tokenizer,
        adapter,
        prompts,
        zs,
        drops,
        sampled,
        seed=null_seed(seed, arm),
    )
