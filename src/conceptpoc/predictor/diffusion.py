"""Concept-space conditional diffusion predictor.

Models the full conditional distribution over the next concept vector given a
prefix, replacing the mean-seeking point-regression head. Training uses standard
DDPM denoising; inference uses DDIM (or full DDPM) reverse sampling followed by
L2 normalization so outputs live on the same unit sphere as codec encodings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DiffusionConfig, PredictorConfig
from .model import ContextTransformer


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class DenoiseHead(nn.Module):
    """MLP that predicts diffusion target from (x_noisy, context, timestep)."""

    def __init__(
        self,
        concept_dim: int,
        context_dim: int,
        cfg: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.time_emb = SinusoidalTimestepEmbedding(cfg.time_dim)
        in_dim = concept_dim + context_dim + cfg.time_dim
        layers: list[nn.Module] = []
        hidden = cfg.head_hidden
        d = in_dim
        for _ in range(cfg.head_layers):
            layers.extend([nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)])
            d = hidden
        layers.append(nn.Linear(d, concept_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x_noisy: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        te = self.time_emb(t)
        h = torch.cat([x_noisy, context, te], dim=-1)
        return self.net(h)


class NoiseSchedule(nn.Module):
    """Linear beta schedule with precomputed DDPM/DDIM coefficients."""

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.num_train_timesteps = cfg.num_train_timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        # For reverse DDPM step.
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(1e-12),
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sa = self.sqrt_alpha_bar[t].unsqueeze(-1)
        so = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sa * x0 + so * noise

    def predict_x0_from_output(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_out: torch.Tensor,
        predict_type: str,
    ) -> torch.Tensor:
        if predict_type == "x0":
            return model_out
        sa = self.sqrt_alpha_bar[t].unsqueeze(-1)
        so = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return (x_t - so * model_out) / sa.clamp_min(1e-8)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        model_out: torch.Tensor,
        predict_type: str,
    ) -> torch.Tensor:
        x0 = self.predict_x0_from_output(x_t, t, model_out, predict_type)
        ab_t = self.alpha_bar[t].unsqueeze(-1)
        ab_prev = self.alpha_bar[t_prev].unsqueeze(-1)
        eps = (x_t - torch.sqrt(ab_t) * x0) / torch.sqrt(1.0 - ab_t).clamp_min(1e-8)
        return torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps


@dataclass
class DiffusionLossOutput:
    total: torch.Tensor
    mse: torch.Tensor


class DiffusionConceptPredictor(nn.Module):
    """Conditional diffusion over the next concept vector."""

    def __init__(
        self,
        concept_dim: int,
        predictor_cfg: PredictorConfig,
        diffusion_cfg: DiffusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.cfg = predictor_cfg
        self.diff_cfg = diffusion_cfg or predictor_cfg.diffusion
        self.trunk = ContextTransformer(concept_dim, predictor_cfg)
        self.head = DenoiseHead(concept_dim, predictor_cfg.d_model, self.diff_cfg)
        self.schedule = NoiseSchedule(self.diff_cfg)

    def encode_context(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.trunk(
            context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec
        )
        return ContextTransformer.last_real_token(h, pad_mask)

    def _model_output(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        context_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(x_noisy, t, context_emb)

    def diffusion_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
    ) -> DiffusionLossOutput:
        """Denoising MSE conditioned on the prefix's last real position."""
        c = self.encode_context(
            context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec
        )
        x0 = F.normalize(target, dim=-1)
        b = x0.size(0)
        t = torch.randint(
            0, self.schedule.num_train_timesteps, (b,), device=x0.device, dtype=torch.long
        )
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        pred = self._model_output(x_t, t, c)
        if self.diff_cfg.predict_type == "x0":
            target_out = x0
        else:
            target_out = noise
        mse = F.mse_loss(pred, target_out)
        return DiffusionLossOutput(total=mse, mse=mse)

    def forward(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
        sampling_steps: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Sample one next concept vector (B, dim), L2-normalized."""
        return self.sample(
            context,
            pad_mask,
            segment_ids=segment_ids,
            theme_vec=theme_vec,
            sampling_steps=sampling_steps,
            temperature=temperature,
        )

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        theme_vec: torch.Tensor | None = None,
        sampling_steps: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Reverse diffusion from Gaussian noise, conditioned on context."""
        c = self.encode_context(
            context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec
        )
        b, device = c.size(0), c.device
        steps = sampling_steps or self.diff_cfg.sampling_steps
        x = torch.randn(b, self.concept_dim, device=device)

        if self.diff_cfg.sampler == "ddim":
            timesteps = torch.linspace(
                self.schedule.num_train_timesteps - 1,
                0,
                steps,
                device=device,
            ).long()
            for i, t_val in enumerate(timesteps):
                t = torch.full((b,), t_val.item(), device=device, dtype=torch.long)
                t_prev = (
                    timesteps[i + 1]
                    if i + 1 < len(timesteps)
                    else torch.zeros((), device=device, dtype=torch.long)
                )
                t_prev_b = torch.full((b,), t_prev.item(), device=device, dtype=torch.long)
                out = self._model_output(x, t, c)
                x = self.schedule.ddim_step(x, t, t_prev_b, out, self.diff_cfg.predict_type)
        else:
            for t_val in reversed(range(self.schedule.num_train_timesteps)):
                t = torch.full((b,), t_val, device=device, dtype=torch.long)
                out = self._model_output(x, t, c)
                x0 = self.schedule.predict_x0_from_output(
                    x, t, out, self.diff_cfg.predict_type
                )
                if t_val > 0:
                    noise = torch.randn_like(x)
                    var = self.schedule.posterior_variance[t]
                    x = x0 + torch.sqrt(var) * noise
                else:
                    x = x0

        jitter = temperature if temperature > 0 else self.diff_cfg.sample_jitter
        if jitter > 0:
            x = x + torch.randn_like(x) * jitter
        return F.normalize(x, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
