"""Autoregressive latent rollout.

Starting from a seed of concept vectors (a prompt encoded sentence-by-sentence),
the predictor proposes the next concept vector; we append it and repeat. The
resulting trajectory of vectors is then decoded to text by the codec.

Diversity for sampling N candidates (needed by the self-improvement loop) comes
from Gaussian noise injected in concept space before re-normalizing, controlled
by ``temperature``. Diffusion predictors sample stochastically via reverse
diffusion; ``temperature`` adds extra jitter after sampling.
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


def _is_diffusion_model(model) -> bool:
    return getattr(model.cfg, "head", "regression") == "diffusion" or hasattr(
        model, "diff_cfg"
    )


def _predict_next(
    model,
    ctx: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    theme_vec: torch.Tensor | None = None,
    temperature: float = 0.0,
) -> torch.Tensor:
    """One next-vector step; dispatches regression vs diffusion sampling."""
    if _is_diffusion_model(model):
        return model.sample(
            ctx,
            segment_ids=segment_ids,
            theme_vec=theme_vec,
            temperature=temperature,
        )
    pred = model(ctx, segment_ids=segment_ids, theme_vec=theme_vec)
    if temperature > 0:
        pred = pred + torch.randn_like(pred) * temperature
        pred = torch.nn.functional.normalize(pred, dim=-1)
    return pred


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
        pred = _predict_next(model, ctx, temperature=temperature).squeeze(0).cpu().numpy()
        context.append(pred)
        generated.append(pred)

    return Rollout(
        seed_vectors=seed,
        generated_vectors=np.stack(generated) if generated else np.zeros((0, seed.shape[1])),
    )


@torch.no_grad()
def generate_conditioned(
    model: ConceptPredictor,
    instruction_vecs: np.ndarray,
    steps: int,
    theme_vec: np.ndarray | None = None,
    temperature: float = 0.0,
    max_context: int | None = None,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
) -> Rollout:
    """Roll out response vectors conditioned on an instruction (+ optional theme).

    The instruction vectors form a segment-0 prefix; each generated response
    vector is appended as segment 1 and fed back in. Greedy (``temperature=0``)
    generation is deterministic for regression; diffusion models remain
    stochastic unless temperature is 0 and sampling is deterministic DDIM.
    Returns a ``Rollout`` whose ``seed_vectors`` are the instruction and
    ``generated_vectors`` are the response.
    """
    model.eval()
    dev = torch.device(device) if isinstance(device, str) else device
    model.to(dev)
    rng = rng or np.random.default_rng()
    max_context = max_context or model.cfg.max_seq_len

    instr = np.asarray(instruction_vecs, dtype=np.float32)
    if instr.ndim == 1:
        instr = instr[None, :]
    n_instr = instr.shape[0]
    context = [v for v in instr]
    generated: list[np.ndarray] = []

    theme_t = None
    if theme_vec is not None:
        theme_t = torch.from_numpy(
            np.asarray(theme_vec, dtype=np.float32).reshape(1, -1)
        ).to(dev)

    for _ in range(steps):
        window = np.stack(context[-max_context:])
        ctx = torch.from_numpy(window).unsqueeze(0).to(dev)
        n_resp_in_window = len(generated[-max_context:]) if generated else 0
        n_instr_in_window = window.shape[0] - n_resp_in_window
        seg = torch.cat(
            [
                torch.zeros(n_instr_in_window, dtype=torch.long),
                torch.ones(n_resp_in_window, dtype=torch.long),
            ]
        ).unsqueeze(0).to(dev)
        pred = (
            _predict_next(
                model,
                ctx,
                segment_ids=seg,
                theme_vec=theme_t,
                temperature=temperature,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        context.append(pred)
        generated.append(pred)

    return Rollout(
        seed_vectors=instr,
        generated_vectors=np.stack(generated)
        if generated
        else np.zeros((0, instr.shape[1]), dtype=np.float32),
    )


def rollout_to_text(rollout: Rollout, codec: ConceptCodec, include_seed: bool = False) -> list[str]:
    vectors = rollout.all_vectors if include_seed else rollout.generated_vectors
    if len(vectors) == 0:
        return []
    return codec.decode(vectors)
