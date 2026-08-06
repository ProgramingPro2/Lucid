# Compulsory-Corr-Channel JEPA: Forcing the Decoder to Depend on the Predictable Subspace

## Diagnosis (why the current `channel_split` doesn't work)

The routing design already exists in [`src/conceptpoc/jepa/joint_model.py`](src/conceptpoc/jepa/joint_model.py) (`JointJepa.forward`, lines ~94-137): the encoder's 512-dim output is split into `pred_z = z_v[:, :corr_dim]` (normalized, fed to the predictor) and `anti = raw_v[:, corr_dim:]` (unnormalized, fed to variance/covariance/uniformity). This part is correct and should **not** change — anti-collapse is already scoped away from `corr`.

The bug is exactly what the prompt states:

```94:101:src/conceptpoc/jepa/joint_model.py
z_v, raw_v = self.encoder(ids_v, pad_v, return_raw=True)

# Reconstruction anchor (online encoder -> decoder). Uses the FULL
# vector so collapse is still impossible regardless of channel mode.
logits_v = self.decoder.teacher_forcing_logits(
    z_v, ids_v, pad_v, self.bos_id
)
recon = reconstruction_loss(logits_v, ids_v, self.pad_id)
```

Reconstruction always feeds the **full** 512-dim `z_v` into `TextDecoder.from_concept` (a single dense `nn.Linear(cfg.dim, cfg.d_model)` in [`decoder.py`](src/conceptpoc/jepa/decoder.py) line 24). The decoder is free to route around `corr` and read everything it needs out of `spread`. Reconstruction loss stays low and "proves" corr is safe to collapse — this is precisely the dilution trap already documented and unit-tested in [`tests/test_multispace.py`](tests/test_multispace.py) (`TestDilutionTrap`): a flat concatenation lets one subspace's signal be optimized-away by the other. Experiment 2 (corr variance 0.0007, effective rank 91/256) is this failure mode observed for real.

The fix must make the decoder **provably need** `corr`, not just permit it.

---

## 1. Loss-objective design: making `corr` predictable + required + non-collapsing

### 1a. Add a spread-ablated reconstruction pass (the core fix)

Add a **second** decode of the *same* tokens, from a concept vector where the spread channel has been destroyed. This is a direct, differentiable "can you still reconstruct?" test that only passes if the encoder has actually put reconstruction-relevant information into `corr`.

```python
# in JointJepa.forward, after computing z_v, raw_v, cd = corr_dim
if self.cfg.channel_split and self.cfg.corr_forcing_weight > 0:
    corr_part = z_v[:, :cd]
    if self.cfg.corr_forcing_mode == "shuffle":
        # Permute the spread sub-vector across the batch: destroys any
        # (sentence, spread) pairing without changing its marginal
        # statistics, so the decoder can't learn a "detect zero" shortcut.
        perm = torch.randperm(z_v.size(0), device=z_v.device)
        spread_part = raw_v[perm, cd:]
    else:  # "zero"
        spread_part = torch.zeros_like(raw_v[:, cd:])
    z_ablated = F.normalize(torch.cat([corr_part, spread_part], dim=-1), dim=-1)
    logits_ablated = self.decoder.teacher_forcing_logits(
        z_ablated, ids_v, pad_v, self.bos_id
    )
    recon_ablated = reconstruction_loss(logits_ablated, ids_v, self.pad_id)
else:
    recon_ablated = None
```

Use **shuffle** (permute spread across the batch), not zeroing, as the default (`corr_forcing_mode="shuffle"`). Zeroing lets the decoder special-case an out-of-distribution "all-zero" pattern rather than genuinely learning to route information through `corr`; shuffling keeps spread's marginal distribution realistic while destroying its correlation with the target sentence, which is the actual property we want to test.

Total loss becomes:

```python
total = (
    recon.loss
    + self.cfg.corr_forcing_weight * recon_ablated.loss   # NEW — forces corr to be sufficient
    + alpha * pred
    + self.cfg.var_weight * var
    + self.cfg.cov_weight * cov
    + self.cfg.uniformity_weight * unif
)
```

This is the "compulsory" mechanism: gradients from `recon_ablated.loss` flow only through `corr_part` (spread is either detached-and-permuted or zero, contributing no useful gradient), so the encoder is pushed to make `corr` alone carry sentence-identifying content — directly counteracting the collapse mode from Experiment 2.

### 1b. Cross-channel decorrelation (anti-leakage / anti-redundancy)

A subtler failure: `corr` and `spread` could become **correlated copies** of each other (spread "reflects" corr), which would let 1a pass its ablation test via redundancy rather than 1a forcing an actually distinct, purposeful corr channel, and would waste spread's capacity. Add a cross-covariance penalty in [`jepa/losses.py`](src/conceptpoc/jepa/losses.py):

```python
def cross_covariance_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Penalize correlation BETWEEN two channels (not within either)."""
    if a.size(0) < 2:
        return a.new_zeros(())
    ac = a - a.mean(dim=0, keepdim=True)
    bc = b - b.mean(dim=0, keepdim=True)
    cross = (ac.T @ bc) / (a.size(0) - 1)
    return cross.pow(2).sum() / (a.size(1) * b.size(1))
```

Wire it in with a small weight (`cross_cov_weight`, default e.g. `0.02`, same order as `cov_weight`) between `pred_z` and `anti`. This is a leakage/bottleneck term in the sense requested: it stops `spread` from being usable as a proxy for `corr`'s information, which keeps the ablation test (1a) honest.

### 1c. Norm-balance term (prevents spread from "starving" corr's share of the unit sphere)

Because `z_v = normalize(concat(corr_raw, spread_raw))`, if `spread_raw`'s norm grows much larger than `corr_raw`'s (which variance_loss with `var_margin` actively encourages, since it pushes spread's per-dim std toward 1.0 regardless of corr's scale), `corr`'s direction becomes a vanishingly small, noise-dominated fraction of the normalized vector even though its raw values look fine in isolation. Add a soft balance hinge:

```python
def norm_balance_loss(corr_raw: torch.Tensor, spread_raw: torch.Tensor, target_share: float) -> torch.Tensor:
    """Keep corr's share of total raw energy from collapsing toward 0."""
    c2 = corr_raw.pow(2).sum(dim=-1)
    s2 = spread_raw.pow(2).sum(dim=-1)
    share = c2 / (c2 + s2 + 1e-6)
    return F.relu(target_share - share).mean()
```

`target_share` defaults to `corr_dim / dim` (i.e. don't ask for more than an equal split by default). Weight small (`balance_weight`, e.g. `0.05`) — this is a guardrail, not the primary mechanism.

### 1d. Recommended `corr_dim`

Current default is `corr_dim=256` (50/50 split). Recommend dropping to **`corr_dim=128`** (25% of 512) as the new default for this fix, sweeping `{128, 192, 256}`:

- Smaller `corr` concentrates the reconstruction-forcing pressure (1a) into fewer dimensions, which empirically makes ablation-reconstruction harder to fake and easier to reason about (matches the `test_multispace.py` toy setup where a 2-of-128 dim trajectory carries the correlated signal — correlation lives in a *few* dims, not many).
- Smaller `corr` also matches the predictor's proven working regime: the delta-predictor experiment reaching 0.643 val_cosine operated on a similarly modest-width signal, not a full 512-dim target — narrower targets are easier for `ConceptPredictor` to fit without overfitting noise.
- Leaves 384 dims (75%) for `spread`, which comfortably absorbs the variance/covariance/uniformity budget without needing to shrink `var_margin`.

Sweep order: start `corr_dim=128`, and only grow it if the metric gates in Section 4 show `corr` is *information-starved* (ablated recon accuracy far below full-vector accuracy even after the forcing curriculum ramps to full weight).

---

## 2. Spread-channel anti-collapse without touching `corr`

No structural change needed here — this is already correctly scoped in the existing code and must stay that way:

```120:124:src/conceptpoc/jepa/joint_model.py
var = variance_loss(anti, self.cfg.var_margin)
cov = covariance_loss(anti)
std_z = mean_dim_std(anti)
unif = uniformity_loss(anti)
aniso = mean_pairwise_cosine(pred_z)
```

`variance_loss`, `covariance_loss`, and `uniformity_loss` all take `anti = raw_v[:, cd:]` only. The new terms in Section 1 must preserve this scoping:

- `cross_covariance_loss(pred_z, anti)` — by construction touches both, but its gradient on `corr` only pushes it to be **decorrelated from spread**, not to spread out internally. It does not add variance/uniformity pressure onto `corr`, so it cannot flatten adjacency (which is what killed the single-space design).
- `norm_balance_loss` gradients: the `c2` term pulls `corr_raw`'s magnitude up (not its per-dim spread or direction), and only when spread already dominates share. This is a much weaker, non-directional pressure than VICReg's per-dim margin hinge — it cannot flatten adjacency the way `variance_loss` did in Experiment 2.
- `recon_ablated.loss` gradients on `corr_part` push it toward *carrying reconstruction-useful, sentence-specific information*, which is the same direction that keeps adjacent sentences distinguishable-yet-correlated (structure, not noise) — this is the opposite pressure from VICReg's "push every pair apart" collapse-avoidance, so it should reinforce rather than fight adjacency.

**Verification step (cheap, before any real training):** re-run the existing `tests/test_multispace.py::TestDilutionTrap` methodology but with the ablation-reconstruction proxy standing in for a decoder — confirm that forcing reconstruction through the "corr-only" projection (analogous to `corr` in the toy test) does not require touching `spread`'s construction at all.

---

## 3. Training schedule / curriculum

Reuse the existing alpha-ramp pattern in [`jepa/train_joint.py`](src/conceptpoc/jepa/train_joint.py) (`_alpha_for_epoch`) and add a parallel ramp for the new forcing term, but **offset it earlier than prediction**, so `corr` already has forced content before the predictor starts trying to predict it:

```
epoch:        0 ---- recon_warmup_epochs ---- +corr_warmup_epochs ---- +alpha_warmup_epochs ---- done
recon.loss:   on throughout
recon_ablated:        off  ->  ramp to corr_forcing_weight_max  (during corr_warmup_epochs)
pred (alpha):                              0  ->  ramp to alpha_max  (during alpha_warmup_epochs, AFTER corr ramp starts)
var/cov/unif: on throughout (scoped to spread only, per Section 2)
```

Concretely, add:

```python
# JepaConfig
corr_warmup_epochs: int = 3          # epochs to ramp recon_ablated weight, starts right after recon_warmup
corr_forcing_weight_max: float = 1.0 # final weight on recon_ablated.loss (matches recon.loss scale)
corr_forcing_mode: str = "shuffle"   # "shuffle" | "zero"
cross_cov_weight: float = 0.02
balance_weight: float = 0.05
balance_target_share: float | None = None  # None -> corr_dim/dim
```

```python
def _corr_forcing_weight_for_epoch(epoch: int, cfg: JepaConfig) -> float:
    start = cfg.recon_warmup_epochs
    if epoch < start:
        return 0.0
    ramp = epoch - start
    if cfg.corr_warmup_epochs <= 0:
        return cfg.corr_forcing_weight_max
    frac = min(1.0, (ramp + 1) / cfg.corr_warmup_epochs)
    return cfg.corr_forcing_weight_max * frac
```

And shift `_alpha_for_epoch`'s start point to `recon_warmup_epochs + corr_warmup_epochs` (prediction should not compete for gradient with an as-yet-uninformative `corr`). Rationale: if prediction pressure ramps concurrently with forcing pressure, the predictor spends early epochs chasing a still-shifting, still-mostly-empty `corr` target — wasted compute and a noisier `val_pred_cos` signal. Sequencing (recon -> corr-forcing -> prediction) mirrors the existing curriculum philosophy already in this file's docstring.

No change needed to `alpha_warmup_epochs`/`alpha_max` semantics — just their epoch offset.

---

## 4. Metric gates (concrete pass/fail criteria)

All of the following must hold on the **held-out validation split** (reuse `_validate` / `val_loader` machinery in `train_joint.py`), computed on `pred_z` (the corr channel), using the diagnostics already implemented in [`jepa/collapse.py`](src/conceptpoc/jepa/collapse.py) and [`eval/latent.py`](src/conceptpoc/eval/latent.py):

| Gate | Metric | Threshold | Tool |
|---|---|---|---|
| G1 — corr predictable | `adjacent_cosine(corr[t], corr[t+1])` on real (in-order) held-out sentences | > 0.15, **and** > 2x the shuffled-pair control `adjacent_cosine(corr[t], corr[shuffled])` | `jepa/collapse.py::adjacent_cosine`, extend `eval/ceiling.py` to run on `corr` slice specifically |
| G2 — corr not collapsed | `batch_variance(corr)` | mean per-dim std within reach of `var_margin`-comparable health, e.g. > 0.3 (not the 0.0007 collapse observed in Experiment 2) | `collapse.py::batch_variance` |
| G3 — corr not rank-degenerate | `effective_rank(corr)` | > 0.5 * corr_dim (e.g. > 64 for corr_dim=128; contrast with the observed 91/256 ≈ 0.35 failure) | `collapse.py::effective_rank` |
| G4 — corr is *required*, not just present | `recon_acc` (full vector) vs `recon_acc_ablated` (corr-only, spread shuffled) | ablated accuracy ≥ 0.6x full accuracy, **and** clearly above a random-token-accuracy floor | `losses.py::reconstruction_loss` (already returns `.accuracy`) — log both in `JointHistory` |
| G5 — spread still does its job | `std_z` (mean dim-std of spread) | stays ≥ `collapse_std_threshold` (existing alarm, unchanged) | already logged |
| G6 — end-to-end topic coherence | seq-cosine of a generated multi-sentence continuation vs. its own trajectory (topic-tracking, not step-to-step) | > 0.09 (current hybrid) with a required delta improvement > 0.037 over the current pipeline on the same eval prompts | new: see below |

**G6 implementation note:** no committed "seq-cosine" function currently exists in the repo (the `~0.09` / `~0.037` numbers are from prior manual measurement); the closest existing building block is `eval/latent.py::latent_diagnostics` (`mean_consecutive_cos`), which measures step-to-step drift, not topic-tracking-to-seed. Add a `topic_coherence(seed_vec, generated_vecs) -> mean cosine of each generated vector to the seed vector` function alongside it, and report it on **both** the full concept vector (apples-to-apples with the ~0.09 baseline) and the corr channel alone (to attribute any gain specifically to this fix). Run it through the existing `generate-hybrid` + `eval-harness` CLI path referenced in `.hermes/plans/2026-08-04_203000-sharpen-reasoner-fix-topic-drift.md`.

**All of G1-G5 must pass before spending GPU time on G6** (downstream writer/predictor retraining) — they're cheap (CPU, minutes) and directly diagnose whether the core bug is fixed; G6 is the expensive end-to-end confirmation.

---

## 5. Exact implementation points

- **`src/conceptpoc/config.py`** (`JepaConfig`, after existing `channel_split`/`corr_dim` block, line ~173): add `corr_forcing_weight_max`, `corr_forcing_mode`, `corr_warmup_epochs`, `cross_cov_weight`, `balance_weight`, `balance_target_share`.
- **`src/conceptpoc/jepa/losses.py`**: add `cross_covariance_loss(a, b)` and `norm_balance_loss(corr_raw, spread_raw, target_share)`, next to the existing `variance_loss`/`covariance_loss` (keep the module's existing docstring convention of explaining *why* each term exists).
- **`src/conceptpoc/jepa/joint_model.py`** (`JointJepa.forward`, the `channel_split` block at lines 108-124):
  - Add the spread-ablation decode + `recon_ablated` (Section 1a), gated by `cfg.corr_forcing_weight` (passed in from the epoch-level scheduler, analogous to how `alpha` is already passed in as a forward argument — extend the `forward(..., alpha)` signature to `forward(..., alpha, corr_forcing_weight)`).
  - Add `cross_covariance_loss(pred_z, anti)` and `norm_balance_loss(z_v[:, :cd], raw_v[:, cd:], ...)` calls, summed into `total` with their configured weights.
  - Extend `JepaForward` dataclass with `recon_ablated: torch.Tensor`, `recon_ablated_acc: float`, `cross_cov: torch.Tensor`, `balance: torch.Tensor` for logging/gating.
- **`src/conceptpoc/jepa/train_joint.py`**:
  - Add `_corr_forcing_weight_for_epoch` (Section 3) alongside `_alpha_for_epoch`; shift `_alpha_for_epoch`'s effective start by `+ cfg.corr_warmup_epochs`.
  - Pass the new weight into `model(...)` calls (both the training loop and `_validate`).
  - Extend `JointHistory` with `recon_ablated`, `recon_ablated_acc`, `cross_cov` lists, and log them per-epoch (reuse `log_epoch`).
  - Extend the collapse-alarm block (currently checks `std_z < collapse_std_threshold`) with a parallel check on `corr` variance/effective-rank (Gates G2/G3) so a corr-collapse is caught during training, not just post-hoc.
- **`src/conceptpoc/jepa/collapse.py`**: no changes needed — `batch_variance`, `effective_rank`, `adjacent_cosine` are already generic and reusable directly on the `corr` slice for gating.
- **`src/conceptpoc/eval/latent.py`**: add `topic_coherence(seed_vec, generated_vecs)` (Section 4, G6).
- **`src/conceptpoc/eval/ceiling.py`**: extend to optionally slice to `corr_dim` when `channel_split` is on, so the ceiling probe measures the channel that actually matters.
- **Tests**: extend `tests/test_multispace.py` / add `tests/test_corr_forcing.py` with a synthetic case mirroring `TestDilutionTrap` — construct a toy `(corr, spread)` pair, run a tiny decoder-like linear probe through the ablation procedure, and assert that without forcing, a corr-uninformative encoding still "reconstructs" (dilution reproduced), and with the ablation loss backpropagated for a few steps, corr becomes required.
- **Config file**: add `configs/jepa_route_v2_forced.json` (copy of `jepa_route_v1.json` referenced in `.hermes/plans/2026-08-05_030000-routing-jepa-encoder.md`, with `channel_split: true`, `corr_dim: 128`, and the new forcing fields set) as an isolated A/B config — do not overwrite `jepa_route_v1.json`.

---

## 6. Risks and honest failure modes

- **Ablation loss destabilizes early training.** Ramping `corr_forcing_weight` from 0 (Section 3) mitigates this, but if `recon_ablated.loss` starts very high (corr has near-zero information at epoch 0), its gradient magnitude could dominate `recon.loss` and distort the encoder before it has a stable full-vector representation. Mitigation: keep `corr_warmup_epochs` >= `recon_warmup_epochs` in length, and consider gradient-scaling `recon_ablated` by a running estimate of `recon.loss` if instability appears (log both losses per epoch to catch this early).
- **Shrinking `corr_dim` to 128 may starve capacity for genuinely multi-topic or long sentences.** If G4 shows ablated accuracy persistently far below 0.6x full accuracy even at max forcing weight, the honest conclusion is `corr_dim=128` is too narrow for this corpus/tokenizer — grow to 192/256 rather than over-tuning loss weights indefinitely.
- **The forcing mechanism can pass G4 by memorizing per-sentence identity in `corr` without that identity being *forward-predictable*** (i.e., corr becomes a reconstruction-only hash, not a semantic direction adjacent sentences share). This is the actual risk that G1 (adjacent-cosine on real order vs. shuffled control) is designed to catch — G4 passing alone is **not sufficient**; G1 must also pass. If G4 passes but G1 doesn't improve over the current ~0.05 wall, the honest conclusion (per the project's own `docs/LIMITATIONS.md`-style rigor and the sharpen-reasoner plan's Task-1 gate) is that **adjacent-sentence forward structure is genuinely weak in this corpus**, and the fix should pivot toward the change-direction (delta) framing that already reached 0.643 val_cosine, applied on top of the now-forced `corr` channel rather than the full vector.
- **`cross_covariance_loss` and `norm_balance_loss` are new, untested regularizers** — they are the least risky additions (small weights, guardrails) but could still interact unexpectedly with `uniformity_loss` on `spread`. Verify by ablating each new term independently (four runs: forcing-only, +cross-cov, +balance, all three) before combining in the final config, so a regression is attributable to a specific term.
- **G6 (topic coherence) is the true test but is expensive and confounded** by the writer and predictor, which are trained downstream of the encoder. A pass on G1-G5 (encoder-only, cheap) does **not guarantee** G6 improves — the predictor and writer must be retrained on the new `corr`-only target (`pred_dim = cfg.corr_dim`, already correctly wired in `JointJepa.__init__` line 74) before G6 is measurable. Budget for a full retrain pass (predictor + decoder-robust + writer phase-2), not just the encoder change, before declaring victory.
- **Shuffle-based ablation (`corr_forcing_mode="shuffle"`) uses batch-internal permutation**, so its statistical power depends on batch size and diversity; very small batches (e.g. `batch_size=16`, the current default) give a weak decorrelation signal. If G4 is noisy/inconsistent across epochs, consider increasing `batch_size` for this run or drawing the permutation from a larger pool (e.g. a replay buffer of recent spread vectors) rather than just the current batch.
