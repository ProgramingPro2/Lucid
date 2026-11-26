"""Turn items into the margins the G0 rule reads.

``generate(prompt, latent, seed) -> text`` and ``embed(text) -> vector`` are
injected. The GPU run and the CPU dry run share this function. The latent
passed to ``generate`` is None for the zero-prefix and text-prefix arms.

The endpoint is the protocol margin ``cos(gen, z_true) - cos(gen, z_shuf)``.
Absolute cosine is not a gate (FINDINGS F38).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from experiments.e0.criteria import AssayScores
from experiments.e0.dose import dose_latent, unit
from experiments.e0.items import Item

Generate = Callable[[str, np.ndarray | None, int], str]
Embed = Callable[[str], np.ndarray]


def text_prefix(text: str, fraction: float = 0.7) -> str:
    """Opening of ``text``, cut on a word boundary. Same length rule for both arms."""
    body = " ".join(text.split())
    if not body:
        raise ValueError("empty text-prefix source")
    cut = max(80, int(len(body) * fraction))
    cut = min(cut, len(body))
    if " " in body[:cut] and cut < len(body):
        cut = body.rfind(" ", 0, cut) or cut
    out = body[:cut].strip()
    if len(out) < 40:
        raise ValueError("text prefix shorter than 40 characters")
    return out


def margin(generated: np.ndarray, z_true: np.ndarray, z_shuf: np.ndarray) -> float:
    g = unit(generated)
    t = unit(z_true)
    s = unit(z_shuf)
    return float(g @ t - g @ s)


def _mean_margin(
    generate: Generate,
    embed: Embed,
    prompt: str,
    latent: np.ndarray | None,
    seeds: Sequence[int],
    z_true: np.ndarray,
    z_shuf: np.ndarray,
) -> float:
    scores = []
    for seed in seeds:
        text = generate(prompt, latent, int(seed))
        if not str(text).strip():
            raise ValueError("empty generation")
        scores.append(margin(embed(text), z_true, z_shuf))
    return float(np.mean(scores))


def collect_scores(
    items: Sequence[Item],
    generate: Generate,
    embed: Embed,
    *,
    dose_seeds: Sequence[int],
    zero_seed: int,
    text_seed: int,
    null_seed_a: int,
    null_seed_b: int,
) -> AssayScores:
    """Score every arm. Raises on an empty generation or a zero embedding."""
    if len(items) < 2:
        raise ValueError("need at least 2 items")
    if not dose_seeds:
        raise ValueError("need at least one dose seed")

    z_true = [unit(embed(item.target)) for item in items]
    z_shuf = [z_true[item.partner_index] for item in items]
    targets = [item.target for item in items]

    dose0 = np.empty(len(items))
    dose50 = np.empty(len(items))
    dose100 = np.empty(len(items))
    zero = np.empty(len(items))
    text_true = np.empty(len(items))
    text_rand = np.empty(len(items))

    for i, item in enumerate(items):
        prompt = item.prompt
        zt, zs = z_true[i], z_shuf[i]
        for dest, dose in ((dose0, 0.0), (dose50, 0.5), (dose100, 1.0)):
            latent = dose_latent(zs, zt, dose)
            dest[i] = _mean_margin(generate, embed, prompt, latent, dose_seeds, zt, zs)
        zero[i] = _mean_margin(generate, embed, prompt, None, (zero_seed,), zt, zs)
        own = text_prefix(targets[i])
        other = text_prefix(targets[item.partner_index])
        # Match the random arm's length to the true arm so the control is not
        # a length effect.
        other = other[: len(own)].rsplit(" ", 1)[0].strip() or other[: len(own)]
        text_true[i] = _mean_margin(generate, embed, own, None, (text_seed,), zt, zs)
        text_rand[i] = _mean_margin(generate, embed, other, None, (text_seed,), zt, zs)

    null_a_idx = [i for i, item in enumerate(items) if item.null_split == "a"]
    null_b_idx = [i for i, item in enumerate(items) if item.null_split == "b"]
    n_pairs = min(len(null_a_idx), len(null_b_idx))
    if n_pairs < 2:
        raise ValueError("null split produced fewer than 2 pairs")
    null_a = np.empty(n_pairs)
    null_b = np.empty(n_pairs)
    for k in range(n_pairs):
        ia, ib = null_a_idx[k], null_b_idx[k]
        # Fresh decode of the shuffled latent (dose 0), disjoint items, two seeds.
        null_a[k] = _mean_margin(
            generate,
            embed,
            items[ia].prompt,
            z_shuf[ia],
            (null_seed_a,),
            z_true[ia],
            z_shuf[ia],
        )
        null_b[k] = _mean_margin(
            generate,
            embed,
            items[ib].prompt,
            z_shuf[ib],
            (null_seed_b,),
            z_true[ib],
            z_shuf[ib],
        )
    return AssayScores(
        dose0=dose0,
        dose50=dose50,
        dose100=dose100,
        text_true=text_true,
        text_rand=text_rand,
        zero=zero,
        null_a=null_a,
        null_b=null_b,
    )
