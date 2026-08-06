"""Tests for vMF math, Wood/Ulrich sampler, and two-mode anti-mean-seeking gate."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from conceptpoc.config import PredictorConfig
from conceptpoc.predictor.model import ContextTransformer, predictor_loss
from experiments.sentence_pred.heads import (
    MixtureVmfHead,
    SentenceDensityPredictor,
    VmfHead,
    log_iv_uae,
    sample_vmf,
    vmf_log_normalizer,
    vmf_nll,
)


def test_log_iv_uae_vs_scipy():
    from scipy.special import ive

    def _ref_log_iv(nu: float, kappa: float) -> float:
        # log I_nu(kappa) = log(ive(nu, kappa)) + kappa
        ive_val = float(ive(nu, kappa))
        if ive_val > 0.0 and math.isfinite(ive_val):
            return math.log(ive_val) + kappa
        # float64 underflow: fall back to mpmath high-precision I_nu
        from mpmath import besseli, log, mpf

        return float(log(besseli(mpf(nu), mpf(kappa))))

    for nu in (191.0, 255.0):
        for kappa in (1.0, 10.0, 60.0, 137.0, 500.0, 1000.0):
            k = torch.tensor(kappa, dtype=torch.float64)
            approx = float(log_iv_uae(nu, k).item())
            ref = _ref_log_iv(nu, kappa)
            rel = abs(approx - ref) / max(abs(ref), 1e-12)
            assert rel <= 1e-3, f"nu={nu} kappa={kappa}: rel={rel} approx={approx} ref={ref}"


def _mean_resultant_ad(d: int, kappa: float) -> float:
    """A_d(kappa) via autograd of -log C_d wrt kappa."""
    k = torch.tensor(kappa, dtype=torch.float64, requires_grad=True)
    nll_norm = -vmf_log_normalizer(d, k)
    (g,) = torch.autograd.grad(nll_norm, k)
    return float(g.item())


def test_sample_vmf_mean_resultant_and_unit_norm():
    torch.manual_seed(0)
    d = 512
    n = 4096
    mu = F.normalize(torch.randn(1, d), dim=-1).expand(n, d).contiguous()
    means = []
    for kappa in (50.0, 137.0, 500.0):
        k = torch.full((n,), kappa)
        x = sample_vmf(mu, k)
        norms = x.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        mean_res = float((mu * x).sum(dim=-1).mean().item())
        ad = _mean_resultant_ad(d, kappa)
        assert abs(mean_res - ad) <= 0.02, f"kappa={kappa}: mean={mean_res} A_d={ad}"
        means.append(mean_res)
    # E[mu . x] monotone in kappa
    assert means[0] < means[1] < means[2]


def test_two_mode_recovery():
    """Anti-mean-seeking gate (spec §3.4).

    Prints measured numbers for the deliverable report.
    """
    torch.manual_seed(42)
    d = 512
    d_model = 16
    device = torch.device("cpu")

    # Two well-separated modes on S^{511}.
    mu_a = F.normalize(torch.randn(d), dim=0)
    mu_b = F.normalize(torch.randn(d), dim=0)
    # Orthogonalize B against A for clearer separation.
    mu_b = F.normalize(mu_b - (mu_b @ mu_a) * mu_a, dim=0)
    assert float((mu_a @ mu_b).abs()) < 0.05

    # Context codebook: single-token contexts on the sphere.
    c_a = F.normalize(torch.randn(d), dim=0)
    c_b = F.normalize(torch.randn(d), dim=0)
    c_b = F.normalize(c_b - (c_b @ c_a) * c_a, dim=0)
    c_amb = F.normalize(c_a + c_b, dim=0)

    kappa_true = 150.0

    def make_batch(labels: list[str], n_per: int = 64):
        ctxs, tgts, valids = [], [], []
        for lab in labels:
            for _ in range(n_per):
                if lab == "A":
                    ctx = c_a
                    tgt = sample_vmf(mu_a.unsqueeze(0), torch.tensor([kappa_true])).squeeze(0)
                elif lab == "B":
                    ctx = c_b
                    tgt = sample_vmf(mu_b.unsqueeze(0), torch.tensor([kappa_true])).squeeze(0)
                else:
                    ctx = c_amb
                    if torch.rand(()) < 0.5:
                        tgt = sample_vmf(mu_a.unsqueeze(0), torch.tensor([kappa_true])).squeeze(0)
                    else:
                        tgt = sample_vmf(mu_b.unsqueeze(0), torch.tensor([kappa_true])).squeeze(0)
                ctxs.append(ctx)
                tgts.append(tgt)
                valids.append(True)
        context = torch.stack(ctxs).unsqueeze(1)  # (B, 1, D)
        target = torch.stack(tgts).unsqueeze(1)
        valid = torch.ones(len(ctxs), 1, dtype=torch.bool)
        return context, target, valid

    pcfg = PredictorConfig(
        d_model=d_model, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0, max_seq_len=4
    )

    # --- (i) single VmfHead on unambiguous A/B ---
    vmf = SentenceDensityPredictor(d, pcfg, head="vmf").to(device)
    opt = torch.optim.AdamW(vmf.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(80):
        ctx, tgt, valid = make_batch(["A", "B"], n_per=32)
        loss = vmf.loss(ctx, None, tgt, valid)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu_hat_a, k_hat_a = vmf.predict_mu(c_a.view(1, 1, d))
        mu_hat_b, k_hat_b = vmf.predict_mu(c_b.view(1, 1, d))
    cos_a = float((mu_hat_a.squeeze() @ mu_a).item())
    cos_b = float((mu_hat_b.squeeze() @ mu_b).item())
    kap_a = float(k_hat_a.squeeze().item())
    kap_b = float(k_hat_b.squeeze().item())
    print(
        f"two_mode_vmf: cos_A={cos_a:.4f} cos_B={cos_b:.4f} "
        f"kappa_A={kap_a:.2f} kappa_B={kap_b:.2f}"
    )
    assert cos_a >= 0.95, f"cos_A={cos_a}"
    assert cos_b >= 0.95, f"cos_B={cos_b}"
    assert kappa_true / 1.5 <= kap_a <= kappa_true * 1.5, f"kappa_A={kap_a}"
    assert kappa_true / 1.5 <= kap_b <= kappa_true * 1.5, f"kappa_B={kap_b}"

    # --- (ii) MixtureVmfHead on ambiguous 50/50 ---
    mdn = SentenceDensityPredictor(d, pcfg, head="mdn", K=2).to(device)
    opt_m = torch.optim.AdamW(mdn.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(120):
        ctx, tgt, valid = make_batch(["A", "B", "AMB"], n_per=24)
        loss = mdn.loss(ctx, None, tgt, valid)
        opt_m.zero_grad()
        loss.backward()
        opt_m.step()

    with torch.no_grad():
        pi_logits, mu_mix, kappa_mix = mdn.forward(c_amb.view(1, 1, d), return_sequence=True)
        pi = F.softmax(pi_logits.squeeze(), dim=-1)  # (K,)
        mus = mu_mix.squeeze(0).squeeze(0)  # (K, D)
        # Match components to mu_A / mu_B by cosine
        cos_to_a = (mus @ mu_a)
        cos_to_b = (mus @ mu_b)
        # Component closest to A and to B
        i_a = int(cos_to_a.argmax().item())
        i_b = int(cos_to_b.argmax().item())
        # Prefer distinct indices
        if i_a == i_b:
            # Assign greedily
            if float(cos_to_a[i_a]) >= float(cos_to_b[i_b]):
                i_b = 1 - i_a
            else:
                i_a = 1 - i_b
        w_a = float(pi[i_a].item())
        w_b = float(pi[i_b].item())
        cos_comp_a = float(cos_to_a[i_a].item())
        cos_comp_b = float(cos_to_b[i_b].item())
    print(
        f"two_mode_mdn: w_A={w_a:.4f} w_B={w_b:.4f} "
        f"cos_comp_A={cos_comp_a:.4f} cos_comp_B={cos_comp_b:.4f}"
    )
    assert abs(w_a - 0.5) <= 0.1, f"w_A={w_a}"
    assert abs(w_b - 0.5) <= 0.1, f"w_B={w_b}"
    assert cos_comp_a >= 0.9, f"cos_comp_A={cos_comp_a}"
    assert cos_comp_b >= 0.9, f"cos_comp_B={cos_comp_b}"

    # --- point-regression control: mean-seeking ---
    class PointReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = ContextTransformer(d, pcfg)
            self.out = nn.Linear(d_model, d)

        def forward(self, context):
            h = self.trunk(context, None)
            return F.normalize(self.out(h[:, -1]), dim=-1)

    point = PointReg().to(device)
    opt_p = torch.optim.AdamW(point.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(120):
        ctx, tgt, _ = make_batch(["AMB"], n_per=48)
        pred = point(ctx)
        target = tgt.squeeze(1)
        loss = predictor_loss(pred, target).total
        opt_p.zero_grad()
        loss.backward()
        opt_p.step()

    with torch.no_grad():
        pred_amb = point(c_amb.view(1, 1, d)).squeeze(0)
        cos_pa = float((pred_amb @ mu_a).item())
        cos_pb = float((pred_amb @ mu_b).item())
        mid = F.normalize(mu_a + mu_b, dim=0)
        cos_mid = float((pred_amb @ mid).item())
    print(
        f"two_mode_point: cos_A={cos_pa:.4f} cos_B={cos_pb:.4f} cos_mid={cos_mid:.4f}"
    )
    assert cos_pa <= 0.75, f"point cos_A={cos_pa}"
    assert cos_pb <= 0.75, f"point cos_B={cos_pb}"
