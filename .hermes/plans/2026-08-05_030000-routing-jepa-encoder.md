# Routing JEPA Encoder: Split Concept Space into Correlation + Anti-Collapse Channels

> **For Hermes:** Work task-by-task. Reuse resumable SQLite checkpoints. Each change measured. Do NOT delete old artifacts.

**Goal:** Retrain the JEPA encoder so its concept space is BOTH forward-predictable (adjacent sentences correlate) AND non-collapsed (high variance). Proven bottleneck: the single VICReg pushes ALL dims apart globally → adjacent-sentence cosine collapses to ~0.05 no matter the predictor.

**The fix (multi-channel / routing):** instead of one concept vector fighting both jobs, split the encoder's output into two *channels*:
- **corr channel** (first half): carries the forward-predictable structure — predictor operates HERE.
- **spread channel** (second half): carries diversity — anti-collapse (variance/covariance/uniformity) operates HERE.
- Reconstruction uses the **full** vector (so collapse is still impossible).

This decouples the two losses so anti-collapse stops destroying prediction. Validated as the only mechanism that works by our toy + real-sentence dilution-trap tests.

**Status vs prior work:** The anti-collapse-vs-predictability tension is a known open problem (VICReg 2105.04906, LeJEPA 2511.08544, JEPA-in-language paradox 2607.23531). Our contribution: **channel/routing split of the concept space for text-JEPA**, tested empirically. Novel-enough to pursue.

**Tech Stack:** Lucid, RTX 3090, PyTorch 2.6, existing JointJepa encoder + SQLite checkpoints.

---

## Config: add `channel_split` + `corr_dim` to JepaConfig

```python
# in JepaConfig
# Multi-channel (routing) mode: when True, split the encoder output into a
# correlation channel (predictor) and a spread channel (anti-collapse) so
# anti-collapse no longer flattens forward predictability.
channel_split: bool = False
corr_dim: int = 256   # dims of the correlation channel when split
```

**Files:** `src/conceptpoc/config.py`

**Step 1:** add the two fields after `uniformity_weight`.
**Verify:** `Config().jepa.channel_split is False`, `corr_dim == 256`.

---

## Task 1: implement multi-channel forward in JointJepa

**Files:** `src/conceptpoc/jepa/joint_model.py`

**Step 1:** after computing `z_v` (normalized) and `raw_v` (unnormalized), if `channel_split`:
- `corr_v = z_v[:, :corr_dim]`  (predictor channel — normalized)
- `spread_raw = raw_v[:, corr_dim:]` (anti-collapse channel — unnormalized)
- reconstruction still uses full `z_v`
- prediction: all context/target vectors use ONLY their `corr` halves
- variance+bvariance+uniformity: computed on `spread_raw`

**Step 2:** `_validate` must use the split too (predict on corr halves).

**Code sketch:**

```python
# in forward, split before prediction
if self.cfg.channel_split:
    corr_z = z_v[:, :self.cfg.corr_dim]
    spread_raw = raw_v[:, self.cfg.corr_dim:]
    anti = spread_raw
else:
    corr_z = z_v
    anti = raw_v

# anti-collapse on the SPREAD channel only
var = variance_loss(anti, self.cfg.var_margin)
cov = covariance_loss(anti)
unif = uniformity_loss(anti)            # anti-anisotropy on spread channel
std_z = mean_dim_std(anti)
aniso = mean_pairwise_cosine(corr_z)    # monitor corr channel

# scatter corr vector for prediction
... use corr_z instead of z_v in the z grid for prediction ...
# targets: EMA corr halves
... t_v[:, :corr_dim] ...
```

**Step 3:** commit.

---

## Task 2: train the routed encoder vs baseline (A/B)

**Files:** `configs/jepa_route_v1.json` (channel_split: true, corr_dim 256, fresh artifacts dir)

Run on 1M corpus with DB checkpointing + GPU gate:

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli --config configs/jepa_route_v1.json train-joint
```

**Measurement to watch (the gate):**
- `aniso` (monitor corr channel) — should stay low (directions spread)
- `std_z` (spread channel) — should stay healthy (~margin)
- `val_pred_cos` — should be HIGHER than baseline's 0.127 because prediction no longer fights anti-collapse
- **The real gate:** re-run `ceiling.py` skill on the new encoder — adjacent-sentence cosine should EXCEED 0.054.

**Acceptance:** corr-channel prediction cosine meaningfully above the one-step 0.048/0.054 wall, AND std_z healthy (no collapse). If both — the routing design is validated on real text and we rebuild the diffusion predictor + writer on top.

---

## Task 3: (post-gate) rebuild downstream on the routed encoder

Only if Task 2 gate passes:
- retrain diffusion predictor on new encoder (`train-diffusion`)
- decoder-robust
- hybrid writer Phase-2
- hybrid generate → confirm topic now sticks

## Decision gates

- Task 2: if `val_pred_cos` / adjacent-cosine rises well above 0.054 while `std_z` stays healthy → success, proceed Task 3.
- If encoder collapses more (std_z low) → `corr_dim` too small or spread channel too weak; tune corr_dim / var_weight.
- If no improvement → the single-space tension isn't the whole story; the distributional-target idea (Stage 3) becomes the next experiment.

## Risks

| Risk | Mitigation |
|------|-----------|
| corr channel collapses (anti-collapse not applied to it) | reconstruction anchor still uses full vector, so collapse still impossible; monitor corr aniso |
| Prediction uses 256 dims (less capacity) | keep dim=512 total; corr can be tuned |
| Matching baseline isn't apples-to-apples | run same config with channel_split=False as control |

## Files changed

| File | Change |
|------|--------|
| `src/conceptpoc/config.py` | `channel_split`, `corr_dim` |
| `src/conceptpoc/jepa/joint_model.py` | routed forward + validation |
| `configs/jepa_route_v1.json` | new routed config |

## Execution order

```
config fields -> Task 1 (routed forward) -> Task 2 (train A/B + gate agent)
             -> [gate pass] -> Task 3 (downstream rebuild)
```
