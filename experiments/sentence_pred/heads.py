"""Conditional von Mises–Fisher and mixture density heads for sentence prediction.

Exact math from Fable per-sentence spec §3.1–§3.3.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.config import PredictorConfig  # noqa: E402
from conceptpoc.predictor.model import ContextTransformer  # noqa: E402


def log_iv_uae(nu: float, kappa: torch.Tensor) -> torch.Tensor:
    """log I_nu(kappa), uniform asymptotic expansion. Valid for nu >> 1.

    t = kappa / nu;  s = sqrt(1 + t^2)
    eta = s + log(t) - log(1 + s)
    return nu * eta - 0.5*log(2*pi*nu) - 0.25*log(1 + t*t)
    """
    t = (kappa / nu).clamp(1e-6, 20.0)
    s = torch.sqrt(1.0 + t * t)
    eta = s + torch.log(t) - torch.log(1.0 + s)
    return (
        nu * eta
        - 0.5 * math.log(2.0 * math.pi * nu)
        - 0.25 * torch.log(1.0 + t * t)
    )


def vmf_log_normalizer(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """log C_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2*pi) - log_iv_uae(d/2-1, kappa)."""
    nu = d / 2.0 - 1.0
    return (
        nu * torch.log(kappa)
        - (d / 2.0) * math.log(2.0 * math.pi)
        - log_iv_uae(nu, kappa)
    )


def vmf_nll(mu: torch.Tensor, kappa: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """(..., D) inputs -> (...,) per-element NLL."""
    d = mu.size(-1)
    kappa = kappa.squeeze(-1) if kappa.shape == mu.shape[:-1] + (1,) else kappa
    log_c = vmf_log_normalizer(d, kappa)
    return -(log_c + kappa * (mu * x).sum(dim=-1))


def sample_vmf(
    mu: torch.Tensor,
    kappa: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Wood/Ulrich rejection sampler, batched. Returns (N, D) unit vectors.

    Exact procedure from Fable spec §3.3.
    """
    n, d = mu.shape
    device = mu.device
    dtype = mu.dtype
    k = kappa.reshape(n).to(dtype=dtype)
    dm1 = float(d - 1)
    b = (-2.0 * k + torch.sqrt(4.0 * k * k + dm1 * dm1)) / dm1
    x0 = (1.0 - b) / (1.0 + b)
    c = k * x0 + dm1 * torch.log(1.0 - x0 * x0)

    w = torch.zeros(n, device=device, dtype=dtype)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    a_beta = dm1 / 2.0
    beta = torch.distributions.Beta(
        torch.tensor(a_beta, device=device, dtype=dtype),
        torch.tensor(a_beta, device=device, dtype=dtype),
    )
    for _ in range(64):
        if not pending.any():
            break
        idx = pending.nonzero(as_tuple=False).squeeze(-1)
        m = idx.numel()
        if generator is None:
            z = beta.sample((m,))
            u = torch.rand(m, device=device, dtype=dtype)
        else:
            z = _beta_aa_sample(m, a_beta, device, dtype, generator)
            u = torch.rand(m, device=device, dtype=dtype, generator=generator)
        b_p, k_p, x0_p, c_p = b[idx], k[idx], x0[idx], c[idx]
        w_cand = (1.0 - (1.0 + b_p) * z) / (1.0 - (1.0 - b_p) * z)
        accept = (
            k_p * w_cand + dm1 * torch.log(1.0 - x0_p * w_cand) - c_p
            >= torch.log(u)
        )
        w[idx[accept]] = w_cand[accept]
        pending[idx[accept]] = False

    assert not pending.any(), "sample_vmf: not all samples accepted within 64 rounds"

    g = torch.randn(n, d, device=device, dtype=dtype, generator=generator)
    proj = (g * mu).sum(dim=-1, keepdim=True)
    v = F.normalize(g - proj * mu, dim=-1)
    w = w.unsqueeze(-1)
    return w * mu + torch.sqrt((1.0 - w * w).clamp_min(0.0)) * v


def _beta_aa_sample(
    m: int,
    a: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Beta(a, a) via two Gamma(a, 1) draws (generator-aware)."""
    g1 = _gamma_sample(m, a, device, dtype, generator)
    g2 = _gamma_sample(m, a, device, dtype, generator)
    return g1 / (g1 + g2)


def _gamma_sample(
    m: int,
    a: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Gamma(a, 1) via Marsaglia-Tsang (generator-aware)."""
    if a < 1.0:
        g = _gamma_sample(m, a + 1.0, device, dtype, generator)
        u = torch.rand(m, device=device, dtype=dtype, generator=generator)
        return g * u.pow(1.0 / a)
    d_mt = a - 1.0 / 3.0
    c_mt = 1.0 / math.sqrt(9.0 * d_mt)
    out = torch.empty(m, device=device, dtype=dtype)
    filled = 0
    while filled < m:
        need = m - filled
        x = torch.randn(need, device=device, dtype=dtype, generator=generator)
        v = (1.0 + c_mt * x).pow(3)
        u = torch.rand(need, device=device, dtype=dtype, generator=generator)
        accept = (v > 0) & (
            (u < 1.0 - 0.0331 * x.pow(4))
            | (torch.log(u) < 0.5 * x * x + d_mt * (1.0 - v + torch.log(v)))
        )
        n_acc = int(accept.sum().item())
        if n_acc:
            out[filled : filled + n_acc] = d_mt * v[accept]
            filled += n_acc
    return out



def _softplus_inv(y: float) -> float:
    # softplus^{-1}(y) = log(exp(y) - 1); for large y ≈ y
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


# Stretch the softplus pre-activation so Adam (lr ~ 1e-3) can traverse κ ∈ [1, 1000]
# in a few dozen steps. Equivalent to softplus(Linear(h)) with a fixed gain on the
# linear map; bias is initialized so softplus(scale * bias) = kappa_init.
_KAPPA_PRE_SCALE = 40.0


class VmfHead(nn.Module):
    """h (B, T, d_model) -> mu (B, T, D) unit-norm, kappa (B, T, 1)."""

    def __init__(
        self,
        d_model: int = 512,
        concept_dim: int = 512,
        kappa_min: float = 1.0,
        kappa_max: float = 1000.0,
        kappa_init: float = 60.0,
    ) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.mu_proj = nn.Linear(d_model, concept_dim)
        self.kappa_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.kappa_proj.weight)
        nn.init.constant_(
            self.kappa_proj.bias, _softplus_inv(kappa_init) / _KAPPA_PRE_SCALE
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = F.normalize(self.mu_proj(h), dim=-1)
        kappa = torch.clamp(
            F.softplus(_KAPPA_PRE_SCALE * self.kappa_proj(h)),
            self.kappa_min,
            self.kappa_max,
        )
        return mu, kappa


class MixtureVmfHead(nn.Module):
    """K-component conditional vMF mixture."""

    def __init__(
        self,
        d_model: int = 512,
        concept_dim: int = 512,
        K: int = 8,
        kappa_min: float = 1.0,
        kappa_max: float = 1000.0,
        kappa_init: float = 60.0,
    ) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.K = K
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.pi_proj = nn.Linear(d_model, K)
        self.mu_proj = nn.Linear(d_model, K * concept_dim)
        self.kappa_proj = nn.Linear(d_model, K)
        nn.init.zeros_(self.kappa_proj.weight)
        nn.init.constant_(
            self.kappa_proj.bias, _softplus_inv(kappa_init) / _KAPPA_PRE_SCALE
        )

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = h.shape
        pi_logits = self.pi_proj(h)  # (B, T, K)
        mu = self.mu_proj(h).view(b, t, self.K, self.concept_dim)
        mu = F.normalize(mu, dim=-1)
        kappa = torch.clamp(
            F.softplus(_KAPPA_PRE_SCALE * self.kappa_proj(h)),
            self.kappa_min,
            self.kappa_max,
        )  # (B, T, K)
        return pi_logits, mu, kappa
    @staticmethod
    def nll(
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position mixture NLL (...,)."""
        d = mu.size(-1)
        log_pi = F.log_softmax(pi_logits, dim=-1)  # (..., K)
        # x: (..., D) -> broadcast against (..., K, D)
        log_c = vmf_log_normalizer(d, kappa)  # (..., K)
        dot = (mu * x.unsqueeze(-2)).sum(dim=-1)  # (..., K)
        log_comp = log_pi + log_c + kappa * dot
        return -torch.logsumexp(log_comp, dim=-1)

    @staticmethod
    def weight_entropy(pi_logits: torch.Tensor) -> torch.Tensor:
        """Entropy of mixture weights in nats, per position."""
        p = F.softmax(pi_logits, dim=-1)
        return -(p * (p + 1e-12).log()).sum(dim=-1)


class SentenceDensityPredictor(nn.Module):
    """ContextTransformer trunk + vMF or MDN head."""

    def __init__(
        self,
        concept_dim: int,
        predictor_cfg: PredictorConfig,
        head: str = "vmf",
        K: int = 8,
    ) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.head_type = head
        self.trunk = ContextTransformer(concept_dim, predictor_cfg)
        d_model = predictor_cfg.d_model
        if head == "vmf":
            self.head: nn.Module = VmfHead(d_model, concept_dim)
        elif head == "mdn":
            self.head = MixtureVmfHead(d_model, concept_dim, K=K)
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        return_sequence: bool = True,
    ):
        h = self.trunk(context, pad_mask)
        if not return_sequence:
            h = ContextTransformer.last_real_token(h, pad_mask).unsqueeze(1)
        return self.head(h)

    def loss(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None,
        target_seq: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced masked mean NLL over valid positions."""
        params = self.forward(context, pad_mask, return_sequence=True)
        if self.head_type == "vmf":
            mu, kappa = params
            nll = vmf_nll(mu, kappa, target_seq)
        else:
            pi_logits, mu, kappa = params
            nll = MixtureVmfHead.nll(pi_logits, mu, kappa, target_seq)
        mask = valid.to(dtype=nll.dtype)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        n_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """(B, n, D). kappa_eff = kappa / temperature."""
        h = self.trunk(context, pad_mask)
        h_last = ContextTransformer.last_real_token(h, pad_mask)  # (B, d_model)
        b = h_last.size(0)
        d = self.concept_dim
        if self.head_type == "vmf":
            mu, kappa = self.head(h_last.unsqueeze(1))
            mu = mu.squeeze(1)  # (B, D)
            kappa = kappa.squeeze(1).squeeze(-1) / temperature  # (B,)
            mu_rep = mu.unsqueeze(1).expand(b, n_samples, d).reshape(b * n_samples, d)
            k_rep = kappa.unsqueeze(1).expand(b, n_samples).reshape(b * n_samples)
            out = sample_vmf(mu_rep, k_rep)
            return out.view(b, n_samples, d)
        pi_logits, mu, kappa = self.head(h_last.unsqueeze(1))
        pi_logits = pi_logits.squeeze(1)  # (B, K)
        mu = mu.squeeze(1)  # (B, K, D)
        kappa = kappa.squeeze(1) / temperature  # (B, K)
        pi = F.softmax(pi_logits, dim=-1)
        # Component per sample
        comps = torch.multinomial(pi, n_samples, replacement=True)  # (B, n)
        # Gather mu, kappa
        mu_g = torch.gather(
            mu.unsqueeze(1).expand(b, n_samples, pi.size(-1), d),
            2,
            comps.unsqueeze(-1).unsqueeze(-1).expand(b, n_samples, 1, d),
        ).squeeze(2)  # (B, n, D)
        k_g = torch.gather(kappa.unsqueeze(1).expand(b, n_samples, pi.size(-1)), 2, comps.unsqueeze(-1)).squeeze(-1)
        out = sample_vmf(mu_g.reshape(b * n_samples, d), k_g.reshape(b * n_samples))
        return out.view(b, n_samples, d)

    @torch.no_grad()
    def predict_mu(
        self,
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu_last, kappa_last) at the last real position."""
        h = self.trunk(context, pad_mask)
        h_last = ContextTransformer.last_real_token(h, pad_mask).unsqueeze(1)
        if self.head_type == "vmf":
            mu, kappa = self.head(h_last)
            return mu.squeeze(1), kappa.squeeze(1)
        pi_logits, mu, kappa = self.head(h_last)
        pi_logits = pi_logits.squeeze(1)
        mu = mu.squeeze(1)
        kappa = kappa.squeeze(1)
        best = pi_logits.argmax(dim=-1)  # (B,)
        b = mu.size(0)
        mu_best = mu[torch.arange(b, device=mu.device), best]
        k_best = kappa[torch.arange(b, device=mu.device), best].unsqueeze(-1)
        return mu_best, k_best
