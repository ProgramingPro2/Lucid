# Lucid — Per-Sentence Prediction: Complete Architecture + Implementation Spec

Status: build-ready. Written 2026-08-06 against repo state `f7c8871` (`/home/jpearce/Documents/Lucid`).
This spec supersedes the paragraph-first ordering of `.hermes/plans/2026-08-06_120000-opus5-fixes-plan.md` for the per-sentence goal, and implements the encoder-repair-on-critical-path + vMF→MDN ladder + two-channel writer decisions of `.hermes/plans/2026-08-06_130000-fable5-review.md` (S1/S2/S3, M3, §3.2, §3.3).

---

## 0. Scope, measured facts, and the one-paragraph design

**Goal.** A single per-sentence prediction method that, given the sentence history of a document, emits a distribution over the next sentence's 512-d concept vector, samples from it, and renders the sample as novel, coherent text through the writer. Not retrieval. Not paragraph-level.

**The design in one paragraph.** (1) Repair the encoder so the anti-collapse machinery stops actively repelling adjacent sentences: compute variance/covariance on a window-disjoint subsample + FIFO queue, mask in-window neighbors out of the uniformity kernel, and add an adjacent-sentence InfoNCE alignment term with cross-window negatives. (2) Predict directly in the repaired 512-d sentence-concept space — no subspace, no hierarchy — with a conditional von Mises–Fisher head (mean direction μ(c), concentration κ(c); exact spherical log-likelihood; κ is the learned coherence/novelty dial), upgraded to a K-component vMF mixture only if held-out NLL says the conditional is genuinely multimodal. (3) The predictor trunk consumes the *sequence* of sentence vectors (T≤32), not a pooled vector. (4) Retrain the writer Phase-2.5 as a two-channel decoder: previous-sentence token prefix (local coherence, entity carry-over) + same-sentence concept vector through *all-block* FiLM (steering), with correct pad id, trained BOS, within-document pairs. (5) Prove it with a document-disjoint, fixed-seed, distinct-candidate eval suite with identity/kNN/shuffled-context/paraphrase controls and an external MiniLM co-measure, ending in one command that produces `artifacts/sentence_pred/RESULTS.md`.

**Measured facts this spec is built on (trust these; sources cited):**

| Fact | Value | Source |
|---|---|---|
| Isotropic cosine floor, d=512 | 1/√512 = 0.0442 | arithmetic |
| JEPA adjacent vs within-doc-shuffled (Austen) | 0.037 vs 0.146 — **active repulsion** | `experiments/test_targets2.py` docstring; brainstorm §0.1 |
| Stage-0 on corpus_1m | JEPA and MiniLM both show adjacent ≈ within-doc shuffled; cross-doc is the only strong contrast | commit `78149de` |
| Granularity probe (M3 top-1, in-doc distractors) | G=1→10: MiniLM 11%→47%, JEPA 5%→40% | commit `866e4d1` |
| In-doc distractor pool at G=1 | blocks average ~8–14 sentences ⇒ ~7 distinct distractors ⇒ chance ≈ 12.5%; MiniLM's 11% is *at chance*, JEPA's 5% is *below chance* (repulsion) | `data/corpus_1m.txt` structure + probe method |
| Writer is conditioned | concept-ablation overlap 0.06–0.09 (output changes when concept changes) | commit `866e4d1` |
| Paragraph reasoner "36%" | n=200, 95% CI ±6.7 pts, best-epoch-of-10 winner's curse, degenerate in-doc negatives | `artifacts/paragraph_reasoner.metrics.json`; fixes-plan Findings 2–3 |
| Writer Phase-2 defects | trained `writer(concept(sent i−1)) → tokens(sent i)`; cross-doc pairs; `pad_id=0` (= `!`); BOS never in inputs; FiLM frozen in blocks 0–8 | `src/conceptpoc/writer/train_phase2.py:41-45,59,107-110,160`; review D1–D4 |
| Point regression | conditional-mean/mean-seeking failure, three times | review §0, strategy doc §1 |
| Current v3 encoder recipe | 6 enc / 6 dec layers, d_model 512, dim 512, max_tokens 128, window S=8, batch 12, var 1.0 / cov 0.04 / uniformity 1.0, 15 epochs on corpus_1m | `configs/jepa_1m_128t.json` |

**Interpretation locked in:** the sentence-level wall is *manufactured*. All three anti-collapse terms take their statistics from batches whose rows are windows of 8 consecutive sentences of one document (`JepaWindowDataset`, `src/conceptpoc/jepa/dataset.py:20-29`; flattened at `src/conceptpoc/jepa/joint_model.py:88-91`): the uniformity gradient (softmax over −t·d²) concentrates on the *closest* pairs — the adjacent sentences (`losses.py:80-83`); covariance decorrelation deletes the leading PCs — the topic subspace (`losses.py:56-64`). The stage-0 corpus_1m result (MiniLM also ≈) does not contradict this: MiniLM's adjacent ≈ within-doc-shuffled at a *high* absolute level (topically-uniform paragraphs), while JEPA's adjacent sits *below* its own within-doc control near the isotropic floor. The G=1→10 M3 curve says predictability rises with scale, but the G=1 in-doc denominator (~8) means the sentence-level number was never honestly measured. This spec both repairs the encoder and installs the honest sentence-level measurement.

**Deliberately out of scope (decided, not forgotten):** corpus rebuild (M4 of the review) — per-sentence adjacency survives inside the existing blank-line blocks, each of which is a real paragraph; rebuilding changes every artifact and is only load-bearing for *cross-paragraph discourse*, which is not this deliverable. WikiText ` @-@ ` detokenization is included (cheap, affects text metrics). Diffusion (`src/conceptpoc/predictor/diffusion.py`) is not trained in this program; it is the third ladder rung, gated on MDN results, and its C1/C2 defects (norm-scale, terminal SNR) must be fixed before any future run. The paragraph reasoner, trajectory planner, and RAFT loop are untouched.

**Do-not-touch list (controls):** `artifacts/jepa_v3_128t/jepa/joint_robust.pt`, `artifacts/writer_p2_film/writer_cond.pt`, `artifacts/paragraph_concepts_cache/*`, `artifacts/paragraph_reasoner.pt`. All new artifacts land in new directories (`artifacts/jepa_repair*/`, `artifacts/writer_p25/`, `artifacts/sentence_pred/`).

---

## 1. Encoder repair (critical path)

### 1.1 What changes, mechanically

Three coordinated changes, all in `src/conceptpoc/jepa/losses.py` and `src/conceptpoc/jepa/joint_model.py`, controlled by new `JepaConfig` fields. No dataset changes: window row-index `b` and slot index `s` are recoverable inside `forward` from the flat valid indices (`row = idx // S`, `slot = idx % S`), and each window is consecutive sentences of a single document, so "same window" ⇒ "same document, nearby" — exactly the pairs to protect.

**(A) Window-disjoint variance/covariance with a FIFO queue** (brainstorm A1; removes Mechanism 1 for var/cov).
Today: `var = variance_loss(anti, ...)`, `cov = covariance_loss(anti)` over all B·S valid rows (`joint_model.py:120-121`). New: sample **one random valid slot per window** → `anti_sample` (B, dim) with gradients; maintain a registered buffer queue of the last `anti_queue_size=4096` such rows (detached); compute `variance_loss` and `covariance_loss` over `cat([anti_sample, queue])`. The queue supplies stable 512×512 covariance statistics that a 12-row batch cannot (the known A1 failure mode); only current-batch rows receive gradients. Queue entries are pre-normalization `raw` vectors, matching current behavior.

**(B) Neighbor-masked uniformity** (brainstorm A2; disarms Mechanism 2).
`uniformity_loss` gets an optional pair-exclusion mask. Pairs `(i, j)` with `row_i == row_j` and `|slot_i − slot_j| ≤ neighbor_mask_k` (default **k=2**) are removed from the logsumexp. Within-window pairs at distance > k and all cross-window pairs remain — so within-doc-distant repulsion (the healthy "documents have internal diversity" pressure) is preserved, and the adjacent-vs-shuffled gate below stays meaningful. Applied to the full B·S batch as today.

**(C) Adjacent-sentence InfoNCE alignment** (brainstorm A3; pays for adjacency instead of merely not punishing it).
New loss: for every valid anchor `(b, s)` whose successor `(b, s+1)` is valid, positive = `z_{b,s+1}`, negatives = all valid `z` rows with a different window index `b' ≠ b`. Temperature `align_temp=0.07`, weight `align_weight=0.25`. Computed on the *normalized online* vectors `z` (not raw), both sides receiving gradients (SimCSE-style; the cross-window negatives prevent the everything-similar cheat, and the reconstruction anchor prevents within-window collapse). Windows from the same document colliding in a batch is possible but negligible (~140k windows, batch 12) — accepted and documented.

Everything else in the objective is unchanged: reconstruction anchor on the full vector, EMA/stop-grad prediction term with `alpha` curriculum, `channel_split` stays `False` (deprecated path, untouched).

### 1.2 Exact code changes

**`src/conceptpoc/config.py` — add to `JepaConfig` (after `corr_dim`):**

```python
# --- Encoder repair (neighbor-safe anti-collapse) ---
# "window_disjoint": var/cov on one-sentence-per-window + FIFO queue;
# "full": legacy behavior (all B*S rows).
anticollapse_scope: str = "window_disjoint"
anti_queue_size: int = 4096
# Exclude same-window pairs with |slot_i - slot_j| <= k from uniformity_loss.
neighbor_mask_k: int = 2
# Adjacent-sentence InfoNCE (positive = next sentence, negatives = other windows).
align_weight: float = 0.25
align_temp: float = 0.07
```

Defaults are the repair values; legacy runs are reproduced with `{"anticollapse_scope": "full", "neighbor_mask_k": 0, "align_weight": 0.0}`. Old checkpoints load fine (`load_joint` builds `JepaConfig(**blob["jepa_cfg"])`; missing new keys take defaults — and the buffers below must be registered with `persistent=False` so old state_dicts load without them).

**`src/conceptpoc/jepa/losses.py` — two changes:**

```python
def uniformity_loss(
    z: torch.Tensor, t: float = 2.0, exclude_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """exclude_mask: (N, N) bool, True = drop this pair from the repulsion sum.
    Replaces torch.pdist with an explicit masked pairwise matrix."""
    z = F.normalize(z, dim=-1)
    n = z.size(0)
    if n < 2:
        return z.new_zeros(())
    sq = torch.cdist(z, z, p=2).pow(2)                     # (N, N)
    keep = torch.triu(torch.ones(n, n, dtype=torch.bool, device=z.device), diagonal=1)
    if exclude_mask is not None:
        keep &= ~exclude_mask
    vals = sq[keep]
    if vals.numel() == 0:
        return z.new_zeros(())
    return torch.logsumexp(-t * vals, dim=0) - torch.log(
        vals.new_tensor(float(vals.numel()))
    )


def adjacent_infonce(
    z: torch.Tensor,            # (N, dim) normalized online vectors (valid rows)
    rows: torch.Tensor,         # (N,) long: window index of each valid row
    slots: torch.Tensor,        # (N,) long: slot index within window
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE with positive = the same window's next slot, negatives =
    all rows from other windows. Anchors without a valid successor are skipped.
    Same-window non-positive columns are masked to -inf (never negatives)."""
    # Build successor index: pos_index[i] = j with rows[j]==rows[i], slots[j]==slots[i]+1
    # (vectorized via a (max_row+1, max_slot+2) lookup grid; -1 = no successor)
    ...
    logits = (z[anchors] @ z.T) / temperature       # (A, N)
    same_window = rows[anchors].unsqueeze(1) == rows.unsqueeze(0)
    mask = same_window.clone()
    mask[torch.arange(anchors.numel()), pos_cols] = False   # keep the positive
    logits = logits.masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, pos_cols)
```

(≈45 lines total including the lookup-grid successor construction. `mean_pairwise_cosine`, `variance_loss`, `covariance_loss`, `prediction_loss`, `reconstruction_loss` unchanged.)

**`src/conceptpoc/jepa/joint_model.py` — in `__init__`:**

```python
if cfg.anticollapse_scope == "window_disjoint" and cfg.anti_queue_size > 0:
    self.register_buffer(
        "anti_queue", torch.zeros(cfg.anti_queue_size, cfg.dim), persistent=False
    )
    self.register_buffer("anti_queue_ptr", torch.zeros(2, dtype=torch.long), persistent=False)
    # ptr[0] = write cursor, ptr[1] = fill count
```

**in `forward`, replacing lines 108–124 (`channel_split` block through `aniso = ...`):**

```python
pred_z, anti_full = z_v, raw_v            # channel_split path removed from the hot path;
                                          # keep the old branch behind cfg.channel_split for compat.
rows = idx // s                            # (N,) window index of each valid row
slots = idx % s                            # (N,) slot index

if self.cfg.anticollapse_scope == "window_disjoint":
    # one random valid slot per window (rows are grouped; pick via randint per window)
    anti_sample = _one_per_window(anti_full, rows)          # (B_eff, dim), grads flow
    queue_rows = self.anti_queue[: int(self.anti_queue_ptr[1])]
    anti_stat = torch.cat([anti_sample, queue_rows.to(anti_sample.dtype)], dim=0)
    var = variance_loss(anti_stat, self.cfg.var_margin)
    cov = covariance_loss(anti_stat)
    std_z = mean_dim_std(anti_stat)
    self._push_queue(anti_sample.detach())
else:
    var = variance_loss(anti_full, self.cfg.var_margin)
    cov = covariance_loss(anti_full)
    std_z = mean_dim_std(anti_full)

# Neighbor-masked uniformity on the full valid batch (raw vectors, normalized inside).
excl = None
if self.cfg.neighbor_mask_k > 0:
    same = rows.unsqueeze(1) == rows.unsqueeze(0)
    near = (slots.unsqueeze(1) - slots.unsqueeze(0)).abs() <= self.cfg.neighbor_mask_k
    excl = same & near                       # symmetric; diagonal handled by triu
unif = uniformity_loss(anti_full, exclude_mask=excl)

align = z_v.new_zeros(())
if self.cfg.align_weight > 0:
    align = adjacent_infonce(pred_z, rows, slots, self.cfg.align_temp)

aniso = mean_pairwise_cosine(pred_z)
```

**total (line 160):** `total = recon.loss + alpha * pred + var_w*var + cov_w*cov + unif_w*unif + self.cfg.align_weight * align`. Add `align: torch.Tensor` and `adj_cos: float` (mean cosine over the batch's valid adjacent pairs, no_grad — the live gate monitor) to `JepaForward`; log both in `train_joint.py`'s epoch metrics and CheckpointDB rows.

**`src/conceptpoc/jepa/train_joint.py` (~25 lines):** log `align`/`adj_cos`; extend `_validate` to also return mean val `adj_cos` and mean val within-doc-shuffled cosine (shuffle targets within each window — 6 lines) so the gate ratio is visible every epoch without a separate eval pass.

### 1.3 Runs, configs, gates

**Run R1 — pilot A/B on corpus_100k** (`configs/jepa_repair_100k.json` = copy of `configs/jepa_100k_unif.json` with `max_tokens: 128`, `epochs: 12`, `artifacts_dir: "artifacts/jepa_repair_100k"`, repair fields at defaults). Control arm = existing `artifacts/jepa_100k_unif` checkpoint (re-measure only). Cost: ~45–90 min on the 3090.

**Run R2 — full retrain on corpus_1m** (`configs/jepa_repair_1m.json` = copy of `configs/jepa_1m_128t.json` + repair fields, `artifacts_dir: "artifacts/jepa_repair_1m"`, `seed: 42`, epochs 15 with the per-epoch gate monitor; stop early if the gate is stably met ≥3 consecutive epochs after alpha ramp). Cost: ~10–16 h (GpuGate-shared 3090; the v3 run at this exact scale completed before). Output: `artifacts/jepa_repair_1m/jepa/joint.pt`. Optionally run `train_decoder_robust` afterward exactly as v3 did (same config section) → `joint_robust.pt`; the heads and writer consume whichever passes G1 (default: `joint.pt`; robust variant only if renderability in §7 falls short).

**Gate G1 (encoder repaired)** — measured by `evalsuite.m1_table` (§7) on ≥5,000 held-out-document adjacent pairs, fixed seed 1234, distinct candidates, identical method to `scripts/stage0_diagnostics.py`:

| Metric | Threshold | Rationale |
|---|---|---|
| adjacent cosine | ≥ 0.15 | > 3× the 0.0442 isotropic floor (user gate) |
| adjacent − within-doc shuffled | ≥ +0.05 | inverts the measured −0.11 repulsion; "identifiable neighbor" |
| adjacent cosine | ≤ 0.60 | over-alignment guard (MiniLM healthy band is 0.3–0.5) |
| cross-doc shuffled | ≤ 0.10 | global spread preserved |
| M3-identity, in-doc (distinct-count reported) | ≥ chance + 5 pts | ends below-chance repulsion (JEPA was 5% vs ~12.5% chance) |
| effective rank of 4,096 held-out sentence vectors | ≥ 100 / 512 | no quiet collapse (`collapse.py::effective_rank`) |
| std_z (train monitor) | ≥ 0.1 | existing alarm |
| reconstruction accuracy | ≥ control − 0.05 | space stays decodable |

**Pre-decided fallbacks if G1 fails on the pilot:** (i) `align_weight 0.25 → 0.5` and `neighbor_mask_k 2 → 3`, rerun pilot; (ii) additionally `cov_weight 0.04 → 0.0` (reconstruction + variance + masked uniformity carry anti-collapse alone); (iii) if adjacent still ≤ within-doc shuffled after both: the manufactured-repulsion thesis is wrong at this scale — freeze encoder work, and run the entire remaining spec (§3–§8) on **frozen MiniLM vectors** (`src/conceptpoc/codec/sentence_transformer_codec.py`, dim 384) as the E4 reference arm, delivering per-sentence prediction in a known-good space while the encoder question escalates separately. Every downstream component in this spec is encoder-agnostic by construction (they consume cached vectors), so this pivot costs one config change.

---

## 2. Representation for prediction: decision and rationale

**Decision: (a) predict in the repaired 512-d sentence-concept space, with a density head.** Options (b) low-dim slow/direction subspace and (c) hierarchical sentence-from-paragraph-with-residual are rejected as *primary* targets, with one diagnostic kept from each.

Rationale, against the "predictability lives at coarser scale" finding:

1. **The coarse-scale signal lives inside the repaired sentence space.** Covariance decorrelation was deleting the leading PCs — the topic subspace — which is precisely the coarse-scale component (brainstorm Mechanism 2b). The G=1→10 M3 climb was measured *through a broken G=1 space with a fake in-doc denominator*; pooling at G=5–10 partially reconstructed the topic signal that the loss had deleted per-sentence. After repair, that component exists per-sentence again, and a causal trunk over the sentence-vector *sequence* (§4) can compute any running mean it wants — option (c)'s "paragraph anchor + residual" is a strict special case of what a 6-layer transformer over 32 sentence vectors can represent internally. We keep a **doc-centered diagnostic** (report centered and uncentered prediction cosine; `experiments/test_targets2.py::centered_adj` method) to verify where the signal lives, not as an architecture.
2. **(b) would bake a probe result into the architecture.** The slow-subspace dimension is unknown (the B1 CCA probe was never run); choosing k now is a guess, the writer consumes 512-d vectors so a subspace target needs an extra decoder mapping, and the vMF/MDN μ-heads already learn whatever low-rank predictable structure exists (μ(c) spans a learned subspace automatically). The eval reports the effective rank of predicted μ across contexts, which *is* the measurement (b) wanted, for free.
3. **(c) adds a second prediction problem and a second error source** (paragraph-level target + residual), inherits the stride/near-copy hazards of windowed paragraph targets (review B1), and its anchor (a paragraph centroid) is exactly the OOD-for-the-writer object the review flagged (§2.2 of the strategy doc). Per-sentence with sequence context gets the same information without the plumbing.
4. **Interface economy.** The writer's conditioning channel is a single sentence-scale 512-d vector (Phase-2.5, §5). Predicting exactly that object means zero adapters between predictor and renderer, and renderability (§7) directly measures the whole path.

**Concrete tensor shapes (the contract for §3–§4):**

- Sentence vector: `float32 (512,)`, L2-normalized, produced by `JepaCodec.encode` (`src/conceptpoc/codec/jepa_codec.py:47-53`) with `max_tokens=128` from `artifacts/jepa_repair_1m/jepa/joint.pt`.
- Training item: `context (T, 512)` with `T ≤ 32` previous sentence vectors; `target_seq (T, 512)` = the same window shifted by one (teacher forcing over every position); `valid (T,)` bool.
- Trunk hidden: `(B, T, 512) → ContextTransformer(d_model=512, n_layers=6, n_heads=8) → (B, T, 512)`.
- vMF head output: `mu (B, T, 512)` unit-norm, `kappa (B, T, 1)` in [1, 1000].
- MDN head output: `pi (B, T, K)` simplex, `mu (B, T, K, 512)`, `kappa (B, T, K, 1)`, K=8.
- Sample: `(B, n_samples, 512)` unit-norm.

---

## 3. The density head (centerpiece): conditional vMF, then vMF mixture

New file `experiments/sentence_pred/heads.py`. Everything below is exact.

### 3.1 vMF math (d = 512, ν = d/2 − 1 = 255)

Density on the unit sphere: `f(x; μ, κ) = C_d(κ) · exp(κ · μᵀx)`, ‖μ‖=1, κ ≥ 0.

```
log C_d(κ) = (d/2 − 1)·log κ − (d/2)·log(2π) − log I_{d/2−1}(κ)
NLL(x; μ, κ) = −log C_d(κ) − κ·μᵀx
```

`log I_ν(κ)` via the uniform asymptotic expansion (Abramowitz & Stegun 9.7.7) — fully differentiable in torch, no scipy at runtime, excellent at ν = 255:

```python
def log_iv_uae(nu: float, kappa: torch.Tensor) -> torch.Tensor
    """log I_nu(kappa), uniform asymptotic expansion. Valid for nu >> 1.
    t = kappa / nu;  s = sqrt(1 + t^2)
    eta = s + log(t) - log(1 + s)            # = s + log(t / (1 + s))
    return nu * eta - 0.5*log(2*pi*nu) - 0.25*log(1 + t*t)"""
```

Unit test (`tests/test_vmf.py`): compare against `scipy.special.ive` (`log I = log(ive) + κ`) at κ ∈ {1, 10, 60, 137, 500, 1000}, ν ∈ {191, 255} (384-d MiniLM fallback and 512-d): require relative error ≤ 1e−3. scipy is a test-only dependency (already transitively present; if not, add to a dev extras list, not runtime).

Gradient identity used implicitly by autograd: `∂/∂κ [−log C_d(κ)] = A_d(κ) = I_{d/2}(κ)/I_{d/2−1}(κ)` (the mean resultant length). Two closed forms used in code:

- **Banerjee inversion** (for the fallback calibration and for initialization): `κ̂(ρ) = ρ·(d − ρ²)/(1 − ρ²)` where ρ = mean realized cosine. At d=512: ρ=0.10 → κ̂≈52; ρ=0.25 → κ̂≈137; ρ=0.40 → κ̂≈243.
- **A_d(κ) approximation** (for the reliability curve): solve `A_d(κ) ≈ ρ` by inverting Banerjee, or compute `A_d = autograd.grad(−log C_d(κ), κ)`.

### 3.2 Heads

```python
class VmfHead(nn.Module):
    """h (B, T, d_model) -> mu (B, T, D) unit-norm, kappa (B, T, 1) in [kappa_min, kappa_max]."""
    def __init__(self, d_model: int = 512, concept_dim: int = 512,
                 kappa_min: float = 1.0, kappa_max: float = 1000.0,
                 kappa_init: float = 60.0) -> None:
        # mu_proj: nn.Linear(d_model, concept_dim)         (mu = F.normalize(mu_proj(h)))
        # kappa_proj: nn.Linear(d_model, 1); bias initialized to softplus^-1(kappa_init)
        # kappa = clamp(softplus(kappa_proj(h)), kappa_min, kappa_max)
    def forward(self, h) -> tuple[Tensor, Tensor]: ...

def vmf_nll(mu: Tensor, kappa: Tensor, x: Tensor) -> Tensor:
    """(..., D) inputs -> (...,) per-element NLL:
    -( (D/2-1)*log(kappa) - (D/2)*log(2*pi) - log_iv_uae(D/2-1, kappa) + kappa * (mu*x).sum(-1) )"""

class MixtureVmfHead(nn.Module):
    """K-component conditional vMF mixture. K=8 default.
    pi_proj: Linear(d_model, K); mu_proj: Linear(d_model, K*concept_dim);
    kappa_proj: Linear(d_model, K), softplus + clamp, bias init softplus^-1(60).
    nll = -logsumexp_k( log_softmax(pi_logits)_k + log C_D(kappa_k) + kappa_k * mu_k . x )"""
    def forward(self, h) -> tuple[Tensor, Tensor, Tensor]: ...   # pi_logits, mu, kappa
    @staticmethod
    def nll(pi_logits, mu, kappa, x) -> Tensor: ...
    @staticmethod
    def weight_entropy(pi_logits) -> Tensor: ...                  # nats, per position

class SentenceDensityPredictor(nn.Module):
    """ContextTransformer trunk (reuses src/conceptpoc/predictor/model.py verbatim)
    + one of the heads. This is the deployable per-sentence predictor."""
    def __init__(self, concept_dim: int, predictor_cfg: PredictorConfig,
                 head: str = "vmf", K: int = 8) -> None: ...
    def forward(self, context, pad_mask=None, return_sequence=True): ...
    def loss(self, context, pad_mask, target_seq, valid) -> Tensor:
        """Teacher-forced mean NLL over valid positions.
        h = trunk(context, pad_mask)          # (B, T, d_model)
        params = head(h); nll over target_seq; masked mean."""
    @torch.no_grad()
    def sample(self, context, pad_mask=None, n_samples: int = 1,
               temperature: float = 1.0) -> Tensor:
        """(B, n, D). Effective concentration kappa_eff = kappa / temperature.
        Mixture: component ~ Categorical(pi) per sample, then vMF sample."""
    @torch.no_grad()
    def predict_mu(self, context, pad_mask=None) -> tuple[Tensor, Tensor]:
        """(mu_last, kappa_last) at the last real position — the point-estimate view
        (mixture: the highest-weight component's mu, plus full params)."""
```

Trunk config (`configs/sentence_pred.json → predictor`): `d_model=512, n_layers=6, n_heads=8, ff_mult=4, dropout=0.1, max_seq_len=34, head="regression"` (the `head` field is unused by this experiment; the density head lives in `heads.py`). ~19M params + 0.3M (vMF) / 2.4M (MDN K=8) head.

### 3.3 Exact sampling procedure (Wood/Ulrich, batched)

```
sample_vmf(mu (N, D), kappa (N,), generator) -> (N, D):
  b  = (-2k + sqrt(4k^2 + (D-1)^2)) / (D-1)
  x0 = (1 - b) / (1 + b)
  c  = k*x0 + (D-1)*log(1 - x0^2)
  repeat (vectorized, mask-accept, cap 64 rounds, assert all accepted):
      Z ~ Beta((D-1)/2, (D-1)/2)        # torch.distributions.Beta, per row
      w = (1 - (1+b)Z) / (1 - (1-b)Z)
      U ~ Uniform(0,1)
      accept where  k*w + (D-1)*log(1 - x0*w) - c >= log(U)
  v = normalize(g - (g . mu) mu)  where g ~ N(0, I_D)   # tangent direction
  return w*mu + sqrt(1 - w^2)*v
```

Acceptance rate at d=512, κ ≤ 1000 is ≥ ~0.66 per round; 64 rounds is astronomically safe. Unit tests: (i) mean resultant of 4,096 samples at κ ∈ {50, 137, 500} matches `A_d(κ)` within ±0.02; (ii) `E[μᵀx]` monotone in κ; (iii) all outputs unit-norm.

**κ is the coherence/novelty dial:** sampling uses `κ_eff = κ(c)/τ`. τ=1 reproduces the learned conditional; τ>1 broadens (novelty), τ<1 sharpens (coherence). τ ∈ {0.5, 1.0, 2.0} is swept in §7 and written to the frontier file. Learned κ(c) itself is the model's *context-dependent* confidence — report its distribution and its calibration (§7 G4).

### 3.4 Training recipe (both heads)

- Data: teacher-forced windows from `SentenceSequenceDataset` (§4), train docs only.
- Optimizer: AdamW, lr 3e−4, weight_decay 0.01, batch 256 windows, OneCycleLR pct_start 0.1, grad-clip 1.0, 30 epochs, early stop on val NLL (patience 5). Seed 42. GpuGate enabled.
- Loss: pure NLL. **No auxiliary cosine term** — that is the mean-seeking objective this replaces.
- Checkpointing: `torch.save({"state_dict", "predictor_cfg": asdict, "head": "vmf"|"mdn", "concept_dim", "history"}, path)`; save best-by-val-NLL. Format mirrors `save_predictor` (`src/conceptpoc/persistence.py:28-39`) with a `head_type: "density"` key so `load_any_predictor` can be extended in one `elif`.
- Cost: ~140k windows × 31 positions; ≈ 1–2 min/epoch on the 3090 → 30–60 min per head.

**Anti-mean-seeking synthetic test (must pass before real training)** — `tests/test_vmf.py::test_two_mode_recovery`: build contexts labeled A/B on S^511 with next-vector ~ vMF(μ_A, 150) or vMF(μ_B, 150), plus an ambiguous context whose successor is 50/50 A/B. Require: (i) single vMF on unambiguous contexts recovers μ within cosine ≥ 0.95 and κ within ×/÷1.5; (ii) on the ambiguous context, the **MDN** places two components with weights 0.5±0.1 on μ_A, μ_B, while a **point-regression control** (same trunk, cosine+MSE loss) lands near the midpoint with cosine ≤ 0.75 to either mode — the measured demonstration that the ladder fixes the documented failure.

### 3.5 Instability fallbacks (pre-decided)

κ pathologies and responses, in order:
1. **κ pinned at `kappa_max`** (overconfidence, NLL exploding on val): lower `kappa_max` to 500; if persists, apply per-sample gradient scaling `(κ/κ₀)^{−β}` with β=0.5 (β-NLL style) on the κ branch only.
2. **κ pinned at `kappa_min` / NLL not beating marginal:** freeze κ globally — **post-hoc calibrated vMF**: train μ with pure cosine loss (the trunk's `out_proj` view), then set a single global κ̂ by Banerjee inversion of the realized val cosine, and a per-quintile κ̂ table keyed on a small confidence MLP if the reliability curve (§7 G4) is non-flat. This is the "fixed-κ vMF = temperature-calibrated cosine regression" rung: strictly weaker but unbreakable.
3. **`log_iv_uae` numerical issues** (won't happen at ν=255 for κ∈[1,1000], but): clamp `t = κ/ν` to [1e−6, 20] and add the scipy-checked unit test to CI so regressions are caught.

**MDN promotion rule (decided now):** MDN replaces single vMF as the deployed head iff held-out NLL improves by ≥ 1.0 nat/sentence AND mean mixture-weight entropy > 0.5 nats on ≥ 25% of held-out contexts (genuine multimodality). Otherwise vMF (K=1) ships — Occam. Diffusion is only considered if the MDN wins by ≥ 3 nats and its best-of-8 vs mean-of-8 gap (§7) stays large — and then only after `diffusion.py` C1/C2 repairs (√d target scaling, zero-terminal-SNR) land, per the review.

---

## 4. Predictor input: the sequence of sentence vectors

**Decision:** the trunk consumes the raw sequence of per-sentence vectors — never a pooled context vector. This is review S3 applied at sentence scale: pooling the prefix is pure information loss when the vectors already exist.

New file `experiments/sentence_pred/data.py`:

```python
@dataclass
class SentenceCorpus:
    sent_vecs: np.ndarray          # (N_sent, 512) float32, L2-normalized
    doc_offsets: np.ndarray        # (N_doc + 1,) int64
    sentences: list[str]           # flat, aligned with sent_vecs
    train_doc_ids: np.ndarray      # document-disjoint split, seed 1234, 90/10
    held_doc_ids: np.ndarray
    def doc_slice(self, doc_id) -> tuple[int, int]

def build_sentence_corpus(
    corpus_path: str = "data/corpus_1m.txt",
    jepa_ckpt: str = "artifacts/jepa_repair_1m/jepa/joint.pt",
    out_dir: str = "artifacts/sentence_pred/corpus",
    max_tokens: int = 128, batch_size: int = 256,
    min_sents: int = 4, held_out_frac: float = 0.10, seed: int = 1234,
    detok_wikitext: bool = True,
) -> SentenceCorpus
```

- Loads via `conceptpoc.data.corpus.load_documents`; drops docs with < `min_sents` sentences; applies WikiText detokenization to the *stored text* (` @-@ `→`-`, ` @,@ `→`,`, ` @.@ `→`.`, collapse space-before-punctuation; review A2) — **encoding uses the same detokenized text** so vectors and text agree.
- Encodes every sentence once with `JepaCodec` (§2 contract); saves `sent_vecs` float16 on disk (`.npy` memmap, cast to float32 on load), `doc_offsets`, flat sentences, and `split_manifest.json` (document-level ids; assert zero overlap).
- **Cache key = sha256(checkpoint path + file size + mtime + corpus path + min_sents + max_tokens + detok flag)[:16]**, embedded in the filename — closes the F4-class silent-stale-cache trap explicitly.
- Cost: ~1M sentences through the 6-layer encoder at 128 tokens, batch 256 ≈ 5–15 min GPU.

```python
class SentenceSequenceDataset(Dataset):
    """Teacher-forced windows over one document's sentence vectors.
    Items: {"context": (T,512) f32, "target_seq": (T,512), "valid": (T,) bool,
            "doc_id": int, "start": int}
    context[t] = v[start+t], target_seq[t] = v[start+t+1]; T = context_len.
    Windows: stride = context_len // 2 (train), non-overlapping anchors for eval."""
    def __init__(self, corpus: SentenceCorpus, doc_ids: np.ndarray,
                 context_len: int = 32, stride: int = 16) -> None
def collate_sentence_batch(batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    # context (B,T,512), pad_mask (B,T) True=pad, target_seq (B,T,512),
    # valid (B,T), doc_ids (B,)
```

No positional-margin machinery is needed at G=1 (review M1 concerned overlapping *pooled windows*; sentences are atomic — a target sentence never appears inside its own context, and eval distractor exclusion is by sentence index). `max_seq_len` for the trunk: **34** (32 + margin), set in `configs/sentence_pred.json`; the `ContextTransformer` positional buffer sizes from it.

---

## 5. Writer conditioning fix: two-channel Phase-2.5

**Design (review §3.2 + M3, exact):** the writer generates sentence *i* from (channel 1) the previous sentence's **tokens** as a prefix — local coherence, entity carry-over, style — and (channel 2) sentence *i*'s **concept vector** through the concept token + **all-block FiLM** — steering, and the port where sampled novelty enters. Same-sentence pairing kills the D1 off-by-one; the prefix channel is what makes same-sentence conditioning unambiguous.

### 5.1 Sequence format (fixes D3's BOS/first-token hole with zero model changes)

```
input_ids = [BOS] + prev_ids[:96] + [EOS] + tgt_ids[:128] + [EOS]        (≤ 227 ≤ seq_len 256)
labels    = [-100]*(1 + len(prev)+1 skipped... ) aligned as below
```

`FluWriter.forward` prepends the concept embedding at position 0 and emits `logits = lm_head(h[:, 1:])` (`src/conceptpoc/writer/model.py:162-172`), i.e. logits position *j* predicts `labels[j]` from tokens ≤ *j*. With `labels = input_ids[1:] + [pad]` semantics used by the trainer (`loss = writer(input_ids[:, :-1], concept, labels=input_ids[:, 1:])`):

- position of `[BOS]` predicts the first prefix token (masked, −100),
- prefix positions predict prefix (masked),
- position of the `[EOS]` separator predicts **the first target token** — trained, from a real token the writer saw throughout Phase-1 (EOS separates sentences in Phase-1 data, `writer/data.py:31-35`),
- target positions predict target + final EOS (supervised).

Because gpt2's added specials give `bos_id = eos_id = 50256` and `pad_id = 50257` (`src/conceptpoc/jepa/tokenizer.py:38-56`), the BOS embedding is *already trained* by Phase-1 (it is the separator token); Phase-2.5 additionally trains the first-token prediction that Phase-2 never did. Inference primes with exactly the training prefix `[BOS] + prev_ids + [EOS]` — no distribution gap.

**Drop arms (trained in, for controls and robustness):** with p=0.10 drop the prefix (`input = [BOS] + tgt + [EOS]`, all supervised — trains the document-opening case); with p=0.10 zero the concept (prefix-only arm — enables the concept-contribution ablation and makes the writer robust to weak vectors); never both. RNG per-index, seed 42.

### 5.2 Exact `src/conceptpoc/writer/train_phase2.py` changes (~90 lines net; file stays < 300)

Replace `ConceptSentenceDataset` (lines 22–52) with:

```python
class TwoChannelSentenceDataset(Dataset):
    """Within-document (prev_sentence_tokens, target_sentence_tokens, enc(target)) triples.
    - pairs NEVER cross document boundaries (iterate docs, then i in range(len(sents)))
    - concept = all_vecs[global_index(i)]        # SAME sentence  (was: i-1  <- the D1 bug)
    - i == 0 items use the prefix-drop form.
    Pre-encodes all sentences once, chunked 512, exactly as before."""
```

Replace `_collate` (lines 55–63): pad with `tok.pad_id` (**not 0**); build `labels` from the padded ids with `labels[labels == tok.pad_id] = -100` **and** prefix positions set to −100 via the per-item prefix length (returned by the dataset). Delete the `labels[labels == 0] = -100` line (`:160`) — token 0 (`!`) is a real token again.

Trainable set (lines 107–110) becomes:

```python
trainable = ["concept_proj"]
trainable += [f"blocks.{i}." for i in range(cfg.writer.n_layers - 3, cfg.writer.n_layers)]
trainable += [f"blocks.{i}.film" for i in range(cfg.writer.n_layers)]     # D4: all-block FiLM
```

(FiLM linears are zero-init (`writer/model.py:100-104`) so un-freezing them is gradient-safe; 12 × 512×1536 ≈ 9.4M new trainable params.)

Training math unchanged otherwise (AdamW lr2=1e−4, grad-clip 1.0, CheckpointDB resume). Config `configs/writer_phase25.json` = copy of `configs/writer_phase2.json` with `artifacts_dir: "artifacts/writer_p25"`, `codec.jepa_checkpoint`/`writer.jepa_checkpoint`: `artifacts/jepa_repair_1m/jepa/joint.pt`, `writer.phase1_checkpoint` unchanged (`artifacts/writer_p1/writer_p1.pt`), `epochs: 6`, `steps_per_epoch: 2000`, `batch_size: 8` (fallback 4 + grad_acc 2 on OOM; seq is now ~227), `db_path: "artifacts/writer_p25/train.db"`. Save via `save_writer(path, model, config | {"two_channel": True})` → `artifacts/writer_p25/writer_two_channel.pt`. **`artifacts/writer_p2_film/writer_cond.pt` is never touched — it is the A/B control.**

### 5.3 `src/conceptpoc/writer/model.py::generate` changes (~35 lines)

```python
def generate(self, concept, max_tokens, bos_id, eos_id=None, temperature=0.8,
             top_k=50, top_p=0.0, repetition_penalty=1.0,
             prefix_ids: torch.Tensor | None = None) -> torch.Tensor:
    """prefix_ids (B, P): pre-tokenized `[BOS] prev [EOS]` prompt; when None,
    primes with [BOS] as before. max_tokens counts NEW tokens only.
    Returns only the newly generated ids (B, <=max_tokens)."""
```

Plus the D5 correctness fix required for batched eval: replace the batch-global `seen` list (`model.py:204-214`) with a per-row `(B, vocab_size)` bool mask updated per step; penalty applied row-wise. KV-cache (review S5) is explicitly deferred — sentence-length renders (≤ 64 new tokens) at batch 32 keep the §7 eval under ~15 min without it; add only if measured wall-clock says otherwise.

**Gate G5 (writer repaired)** — via §7's harness, 1,000 held-out sentences, fixed seed:
- Oracle renderability `cos(enc(render(v_true, prefix=true_prev)), v_true)` (centered, anisotropy floor reported): **new writer ≥ old writer + 0.10** (review M3's gate; the old writer renders the *successor* of its conditioning, so this should be decisive).
- Concept-shuffle control: rendering with a shuffled concept (same prefix) must diverge — `MiniLM cos(render_true, render_shuffled) ≤ 0.80` mean (writer-ablation probe method, commit `866e4d1`).
- Prefix contribution: prefix+concept beats concept-only on oracle renderability by ≥ 0.05 (else the prefix channel is dead weight — investigate before proceeding).

---

## 6. Training order, parallelism, GPU budget (one RTX 3090, GpuGate on)

| # | Step | Depends on | GPU time | Wall notes |
|---|---|---|---|---|
| 0 | Losses + heads + tests (CPU) | — | 0 | 1 day of code; all unit gates run on CPU |
| 1 | R1 pilot encoder A/B (100k) | 0 | 0.75–1.5 h | gate G1-pilot; fallbacks §1.3 |
| 2 | R2 full encoder retrain (1m) | 1 | 10–16 h | overnight; per-epoch adj/shuf monitor; early-stop on stable gate |
| 3 | `build_sentence_corpus` + M1 table (G1 formal) | 2 | 0.3 h | plus MiniLM co-table (CPU ok) |
| 4 | vMF head train + eval (G3, G4) | 3 | 0.5–1 h | |
| 5 | MDN head train + promotion decision | 4 | 0.5–1 h | parallelizable with 6 (short jobs; run first) |
| 6 | Writer Phase-2.5 retrain (G5) | 3 | 5–9 h | second overnight; independent of 4–5 |
| 7 | Full evalsuite + rollouts + RESULTS.md (G6) | 4–6 | 1–2 h | single command, §8 |

Total ≈ 18–30 GPU-hours ≈ 2–3 calendar days on the shared card. True parallelism on one GPU is limited; the real parallelism is: **all of step 0, the evalsuite, and the tests are CPU work done while R2 runs**, and steps 4–5 (minutes-scale) slot into the same day as step 6's launch. If G1-pilot fails twice, the MiniLM pivot (§1.3-iii) removes steps 2–3-GPU entirely (MiniLM encoding of 1M sentences on CPU ≈ 1–2 h, or ~10 min on GPU).

---

## 7. Evaluation that cannot be a mirage

New file `experiments/sentence_pred/evalsuite.py`. Global rules: **every** split is document-disjoint from `split_manifest.json`; **every** RNG is seeded 1234 and independent of epoch (review E1); every cosine is reported raw *and* centered (`x_c = normalize(x − μ_bank)`) with the anisotropy floor from `src/conceptpoc/eval/metrics.py::_anisotropy` printed alongside; every retrieval reports its **distinct-candidate count** (review E2); every row is appended to `artifacts/sentence_pred/frontier.jsonl` with one schema.

### 7.1 Metric functions (signatures)

```python
def m1_table(encode_fn, corpus, n_pairs=5000, seed=1234) -> dict
    # adjacent / within-doc-shuffled / cross-doc-shuffled / 1/sqrt(d) floor
    # + M3-identity(indoc) + effective_rank; same method as scripts/stage0_diagnostics.py.
    # Run for the repaired JEPA AND frozen MiniLM (the external reference column).

def m3_eval(query_fn, dataset, corpus, mode: str, n_eval=5000, n_candidates=1000,
            seed=1234) -> dict
    # mode="global": distractors sampled corpus-wide from OTHER held-out documents
    #   (999 + true next; chance = 0.1%) -- PRIMARY.
    # mode="indoc": distractors = same-doc sentences excluding context & target;
    #   report top1/top5/top10/MRR + mean distinct candidates (~8) -- SECONDARY,
    #   comparable to the historical probes.
    # query_fn(context, pad_mask) -> (B, 512): plugs mu, a sample, or a baseline.

def baselines(dataset, corpus, seed=1234) -> dict
    # identity      : query = z_t                     (the killer control)
    # prefix_mean   : query = normalize(mean(context))
    # knn_transition: nearest train-bank z to z_t -> that sentence's true successor vector
    # random        : random unit vector (must land at 1/n_candidates)

def shuffled_context_control(query_fn, dataset, corpus, seed=1234) -> dict
    # contexts re-drawn from a different held-out document; conditioning-reality check.

def nll_report(model, dataset, corpus) -> dict
    # held-out mean NLL vs (a) unconditional vMF fit on the train marginal (single mu,kappa
    # by Banerjee), (b) per-document-mean vMF. Delta in nats/sentence.

def kappa_reliability(model, dataset, n=5000) -> dict
    # bin predicted kappa into quintiles; realized cos(mu, x_true) per bin;
    # Spearman(kappa_hat, realized cos).

def novelty_suite(gen_vecs, gen_texts, corpus, writer, minilm, seed=1234) -> dict
    # n(x) = max_b cos(x_c, b_c) over the TRAIN bank (chunked matmul, 4096-row blocks)
    # R = {n(h) : h in held-out sentence vecs}  -> median/p90/p99 (frozen once, JSON)
    # novelty_rate = frac(n(gen) <= median(R)); copy_rate = frac(n(gen) >= p99(R))
    # PARAPHRASE BAND: n(enc(render(v_true, prefix=true_prev))) over 1000 held-out
    #   sentences -- "same thought, new words" band; generated samples must not be
    #   novel ONLY at the paraphrase level.
    # EXTERNAL CO-MEASURE: same construction in MiniLM space over rendered TEXT
    #   (n_ext, R_ext) -- kills circularity of measuring novelty in the model's own space.
    # TEXT overlap: max 5-gram overlap vs the 50 nearest train sentences (detokenized).

def renderability(vecs, prefixes, writer, tokenizer, codec) -> dict
    # cos(enc(render(v, prefix)), v), centered; plus the ORACLE arm on true vectors
    # (the ceiling) and the OLD-writer arm (the A/B control).

def multimodality_report(model, dataset, n_ctx=1000, n_samples=8) -> dict
    # mean-of-8 vs best-of-8 M3-global gap; effective_rank of 1000 one-sample-per-context
    # draws; per-context spread = mean pairwise cos of the 8; MDN weight entropy if MDN.

def write_row(path, **metrics) -> None
```

### 7.2 Controls and what each one catches

| Control | Catches |
|---|---|
| identity baseline (query = z_t) | "prediction" that is really topic persistence — the model must beat it |
| μ-vs-z_t copy check: mean cos(μ(c), z_t) ≤ mean cos(z_t, z_{t+1}) + 0.10 | predictor collapsing to a copier of its last input |
| shuffled-context | conditioning is decorative; model learned the marginal |
| random-query row | harness self-test (must equal 1/n_candidates ± noise) |
| distinct-candidate counts | the fake-denominator mirage (G=1 in-doc pools are ~8, not 100) |
| paraphrase band | novelty_rate inflated by re-wording rather than new thoughts |
| MiniLM co-measure | novelty/coherence artifacts of the model's own geometry |
| effective rank + per-context spread | mode collapse behind a good mean |
| mean-of-8 vs best-of-8 | multimodality actually captured (this gap *is* the signal) |
| anisotropy floor on every cosine | the original 0.05-style mirage |

### 7.3 SUCCESS = "per-sentence prediction now works", as numbers

All on held-out documents, n ≥ 5,000 (CI ≤ ±1.3 pts), seed 1234. **G1 and G5 are prerequisites** (§1.3, §5.3). The per-sentence verdict is G3+G4+G6 jointly:

**G3 — the predictor has real forward skill:**
- Held-out NLL(conditional vMF) ≤ NLL(best unconditional vMF) − **5.0 nats/sentence** (≈ the gain from lifting mean directional accuracy from the marginal ~0.10 to ≥ ~0.22 at d=512).
- Centered cos(μ(c), x_true) ≥ **0.15** and ≥ identity baseline + **0.05**.
- M3-global-1000 (query = μ): top-1 ≥ **max(5%, 2 × identity)** and top-10 ≥ **20%**; M3-indoc (distinct ≈ 8): top-1 ≥ identity + **5 pts** (identity ≈ chance ≈ 12–13% — the model must demonstrate *ordering* skill cosine retrieval provably lacks).
- Shuffled-context M3-global top-1 ≤ **max(0.3%, 3 × random)**; μ-copy check passes.

**G4 — the distribution (not just the point) is right:**
- κ reliability: Spearman(κ̂ quintile, realized cosine) ≥ **0.5** — κ is a working confidence dial.
- Sampled arm (τ=1, mean-of-8 M3-global) retains ≥ **70%** of the μ-arm's top-1; effective rank of 1,000 single samples ≥ **100/512**; per-context 8-sample spread within the p10–p90 band of real next-sentence spread around μ.
- Multimodality verdict reported: best-of-8 − mean-of-8 gap, MDN ΔNLL and weight entropy; MDN promoted only per §3.5's rule.

**G6 — the full path produces novel, coherent text:**
- Renderability: cos(enc(render(x̂, prefix)), x̂) ≥ **0.9 × oracle** renderability (oracle = same writer on true vectors; absolute values calibrated in-run, both printed).
- Novelty (vector space): novelty_rate(τ=1 samples) ≥ **0.35** and copy_rate ≤ **0.02**; median n(x̂) ≤ median of the paraphrase band (more novel than re-wording).
- Novelty (external): MiniLM-space novelty_rate ≥ **0.25**; max 5-gram overlap ≤ real-held-out p90.
- Closed-loop (100 seeds × 8 sentences, free-running §8 rollout): MiniLM cos(sentence_k, seed) at k=8 ≥ shuffled-seed control + **0.10**; no fixed point (consecutive-sentence MiniLM cos p90 ≤ 0.90); distinct-2 ≥ 0.3.
- The frontier file contains the τ-sweep rows; the headline plot is novelty_rate (x) vs M3-global top-1 (y) with the identity and kNN-transition baselines marked. **Success = at least one τ strictly above-and-right of both baselines, with all controls passing.**

Every gate that fails is reported in `RESULTS.md` with its measured value — a failed gate with an honest number is a deliverable; a passed gate without its controls is not.

---

## 8. Implementation map and build order

### 8.1 Files

**New package `experiments/sentence_pred/`** (each file < 300 lines per repo guidelines):

| File | Contents | ~LOC |
|---|---|---|
| `__init__.py` | exports | 10 |
| `data.py` | `SentenceCorpus`, `build_sentence_corpus`, `SentenceSequenceDataset`, `collate_sentence_batch`, wikitext detok helper | 260 |
| `heads.py` | `log_iv_uae`, `vmf_log_normalizer`, `vmf_nll`, `sample_vmf`, `VmfHead`, `MixtureVmfHead`, `SentenceDensityPredictor` | 290 |
| `train_head.py` | `train_density_head(corpus, cfg, head="vmf"|"mdn", out, seed=42, device="auto") -> (model, history)`; CLI `--head --epochs --out`; GpuGate + CheckpointDB reuse | 200 |
| `baselines.py` | identity / prefix-mean / kNN-transition / random query_fns | 110 |
| `evalsuite.py` | §7.1 functions + frontier writer | 340 |
| `rollout.py` | `generate_document(seed_text, predictor, writer, tokenizer, codec, steps=8, temperature=1.0, feedback="free"|"latent") -> list[str]`; per-step instrumentation (MiniLM retention, n(x), consecutive cos) | 200 |
| `run_proof.py` | stage orchestrator (below) | 170 |

**Modified files:**

| File | Change | Section |
|---|---|---|
| `src/conceptpoc/config.py` | +5 `JepaConfig` fields | §1.2 |
| `src/conceptpoc/jepa/losses.py` | masked `uniformity_loss`, new `adjacent_infonce` | §1.2 |
| `src/conceptpoc/jepa/joint_model.py` | window-disjoint var/cov + queue, masked unif, align term, `adj_cos` monitor | §1.2 |
| `src/conceptpoc/jepa/train_joint.py` | log align/adj_cos; val adjacent-vs-shuffled monitor | §1.2 |
| `src/conceptpoc/writer/train_phase2.py` | `TwoChannelSentenceDataset`, pad/labels fix, all-block FiLM trainable set | §5.2 |
| `src/conceptpoc/writer/model.py` | `generate(prefix_ids=...)`, per-row repetition penalty | §5.3 |
| `src/conceptpoc/persistence.py` | `head_type: "density"` load/save branch (~15 lines) | §3.4 |

**New configs:** `configs/jepa_repair_100k.json`, `configs/jepa_repair_1m.json` (§1.3), `configs/writer_phase25.json` (§5.2), and:

```json
// configs/sentence_pred.json
{
  "corpus_path": "data/corpus_1m.txt",
  "jepa_checkpoint": "artifacts/jepa_repair_1m/jepa/joint.pt",
  "writer_checkpoint": "artifacts/writer_p25/writer_two_channel.pt",
  "old_writer_checkpoint": "artifacts/writer_p2_film/writer_cond.pt",
  "out_dir": "artifacts/sentence_pred",
  "min_sents": 4, "context_len": 32, "held_out_frac": 0.10,
  "predictor": {"d_model": 512, "n_layers": 6, "n_heads": 8, "ff_mult": 4,
                 "dropout": 0.1, "max_seq_len": 34},
  "vmf": {"kappa_min": 1.0, "kappa_max": 1000.0, "kappa_init": 60.0},
  "mdn": {"K": 8},
  "train": {"batch_size": 256, "lr": 3e-4, "epochs": 30, "patience": 5, "seed": 42},
  "eval": {"seed": 1234, "n_eval": 5000, "n_candidates_global": 1000,
            "n_samples": 8, "temperatures": [0.5, 1.0, 2.0],
            "n_render": 1000, "n_rollout_seeds": 100, "rollout_steps": 8}
}
```

**New tests (`tests/`):** `test_vmf.py` (normalizer vs scipy; sampler mean-resultant; two-mode recovery incl. point-regression control — §3.4), `test_losses_masked.py` (masked uniformity excludes exactly the |Δs|≤k same-window pairs; adjacent_infonce positive/negative bookkeeping on a toy grid; queue push/stat shapes), `test_sentence_data.py` (doc-disjoint manifest, shift-by-one alignment, cache-key sensitivity), `test_writer_prefix.py` (label masking: prefix −100, first target token supervised from the separator position; per-row repetition penalty; prefix generate round-trip). Temporary probe scripts are deleted after their artifact is written, per repo policy.

### 8.2 Build order (every step ends verifiable)

1. **Losses + config + tests** → `pytest tests/test_losses_masked.py` green. (CPU)
2. **Heads + tests** → `pytest tests/test_vmf.py` green, incl. the two-mode anti-mean-seeking demonstration. (CPU)
3. **R1 pilot** → G1-pilot table printed and stored `artifacts/jepa_repair_100k/m1_pilot.json`; fallback ladder §1.3 if needed.
4. **R2 full encoder** → `artifacts/jepa_repair_1m/jepa/joint.pt` + per-epoch adj/shuf curve in the DB.
5. **Corpus + formal G1** → `run_proof.py --stage data` writes `corpus/` cache, `split_manifest.json`, `m1_table.json` (JEPA-new + MiniLM columns), `reference_distribution.json` (R; frozen from here on).
6. **Writer Phase-2.5** (launch overnight) → `writer_two_channel.pt`; `--stage writer-check` computes G5.
7. **vMF + MDN heads** → `--stage train-vmf`, `--stage train-mdn`; `vmf.pt`, `mdn.pt`, per-epoch val NLL; promotion decision recorded.
8. **Full proof** → `--stage eval --stage rollout --stage report`.

### 8.3 The single command

```bash
.venv/bin/python -m experiments.sentence_pred.run_proof \
    --config configs/sentence_pred.json --all
```

`run_proof.py` executes stages `data → check-encoder(G1) → train-vmf → train-mdn(+promotion) → baselines → writer-check(G5) → eval(G3,G4) → rollout → report(G6)`, skipping any stage whose artifact already exists (`--force` to redo), and finishes by writing **`artifacts/sentence_pred/RESULTS.md`**: the G1–G6 pass/fail table with measured values, the frontier rows, the baseline/control table, 20 rendered rollout excerpts, and the one-line verdict *"per-sentence prediction: PASS/FAIL"* determined strictly by §7.3. Encoder and writer training runs (steps R2, Phase-2.5) are launched by their existing CLIs (`train-joint` path with `configs/jepa_repair_1m.json`; `train_phase2` with `configs/writer_phase25.json`) and are prerequisites the orchestrator checks for, not jobs it hides.

---

## 9. Risk register (pre-decided responses)

| Risk | Detector | Response |
|---|---|---|
| Repair overshoots → local collapse (whole paragraph → one point) | adjacent > 0.60, effrank falling, recon_acc drop | halve `align_weight`, raise `align_temp` to 0.1; recon anchor makes full collapse impossible |
| Repair undershoots (adjacent still ≤ shuffled) | G1-pilot | §1.3 ladder → MiniLM pivot (spec unchanged downstream) |
| κ instability | val NLL spikes; κ pinned | §3.5 ladder → post-hoc calibrated vMF |
| MDN component collapse | usage histogram; weight entropy ≈ 0 | keep vMF (promotion rule already requires entropy) |
| Identity baseline beats the head | G3 comparison | the head adds nothing at G=1: report honestly; the sequence-trunk + in-doc ordering gate (which identity provably fails) is the discriminator; if the head can't beat identity even there, per-sentence prediction fails its honest test and RESULTS.md says so |
| Writer ignores concept after prefix is added | G5 concept-shuffle + prefix-only arms | raise concept_drop to 0.2 (forces reliance mix), lengthen FiLM training; the ablation arms were trained in specifically to measure this |
| Eval wall-clock (no KV cache) | render stage > 30 min | implement KV cache (S5, ~80 lines) — deferred, not forgotten |
| Stale caches after encoder swap | cache key includes ckpt size+mtime | rebuild is automatic; old caches never match |

---

## Appendix A — formulas cheat-sheet (for the coding agent)

```
d = 512, nu = 255
floor            = 1/sqrt(d) = 0.0442
log I_nu(k)      ~ nu*eta - 0.5*log(2*pi*nu) - 0.25*log(1+t^2),  t=k/nu,
                   s=sqrt(1+t^2), eta = s + log(t) - log(1+s)
log C_d(k)       = (d/2-1)*log(k) - (d/2)*log(2*pi) - log I_{d/2-1}(k)
vMF NLL          = -(log C_d(k) + k * mu.x)
Banerjee         : kappa_hat(rho) = rho*(d - rho^2)/(1 - rho^2)
mixture NLL      = -logsumexp_k( log pi_k + log C_d(k_k) + k_k * mu_k.x )
uniformity       = logsumexp(-2 * d_ij^2 over kept pairs) - log(n_kept)
adjacent InfoNCE = CE( (z_a @ z_all^T)/0.07 with same-window cols masked, pos = successor )
sampling temp    : kappa_eff = kappa / tau
```

## Appendix B — historical numbers to print in RESULTS.md for context

Old world: adjacent 0.037 < within-doc-shuffled 0.146 (Austen, v3 encoder); sentence M3-indoc ≈ 5% at ~12.5% chance; paragraph reasoner "36%" at n=200 ±6.7; writer renders the successor of its conditioning; `bank=0.365` uncentered (unanchored). New world is measured by §7 and stands or falls on G1–G6.
