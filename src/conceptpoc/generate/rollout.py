"""Autoregressive latent rollout.

Starting from a seed of concept vectors (a prompt encoded sentence-by-sentence),
the predictor proposes the next concept vector; we append it and repeat. The
resulting trajectory of vectors is then decoded to text by the codec.

Diversity for sampling N candidates (needed by the self-improvement loop) comes
from Gaussian noise injected in concept space before re-normalizing, controlled
by ``temperature``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..codec.base import ConceptCodec
from ..predictor.model import ConceptPredictor


@dataclass
class Rollout:
    seed_vectors: np.ndarray  # (s, dim)
    generated_vectors: np.ndarray  # (steps, dim)

    @property
    def all_vectors(self) -> np.ndarray:
        return np.concatenate([self.seed_vectors, self.generated_vectors], axis=0)


@torch.no_grad()
def generate_rollout(
    model: ConceptPredictor,
    seed_vectors: np.ndarray,
    steps: int,
    temperature: float = 0.0,
    max_context: int | None = None,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
) -> Rollout:
    model.eval()
    dev = torch.device(device) if isinstance(device, str) else device
    model.to(dev)
    rng = rng or np.random.default_rng()
    max_context = max_context or model.cfg.max_seq_len

    seed = np.asarray(seed_vectors, dtype=np.float32)
    if seed.ndim == 1:
        seed = seed[None, :]
    context = [v for v in seed]
    generated: list[np.ndarray] = []

    for _ in range(steps):
        window = np.stack(context[-max_context:])
        ctx = torch.from_numpy(window).unsqueeze(0).to(dev)
        pred = model(ctx).squeeze(0).cpu().numpy()
        if temperature > 0:
            pred = pred + rng.normal(0.0, temperature, size=pred.shape).astype(
                np.float32
            )
            pred = ConceptCodec.normalize(pred)
        context.append(pred)
        generated.append(pred)

    return Rollout(
        seed_vectors=seed,
        generated_vectors=np.stack(generated) if generated else np.zeros((0, seed.shape[1])),
    )


def rollout_to_text(rollout: Rollout, codec: ConceptCodec, include_seed: bool = False) -> list[str]:
    vectors = rollout.all_vectors if include_seed else rollout.generated_vectors
    if len(vectors) == 0:
        return []
    return codec.decode(vectors)
