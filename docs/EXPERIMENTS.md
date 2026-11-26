# Experiments — run manifests

**Status: specified, not yet launched.** Companion to [PROTOCOL.md](PROTOCOL.md)
(the pre-registration) and [CORPUS.md](CORPUS.md) (the data stage).

Every run has an ID, a seed, a frozen spawn-key set, and a token budget. If a run
cannot be reproduced from its ID and its seed, it does not count.

---

## 1. Run ID scheme

```
<experiment>/<cell>/s<seed>[.r<repeat>]

examples
  e1/A/s17              E1 cell A, seed 17
  e1/trunk/s17          E1 shared LoRA trunk, seed 17
  e1/A/s17.joint        E1 cell A, joint-training confirmation, seed 17
  e4s2/true/s04.i0231   E4-S2, true-draft condition, refiner seed 4, item 231
  e2/F2/A/s09           E2, family F2, cell A, seed 9
```

The seed is the **experimental unit** for every training-time contrast. Items are
the unit for inference-time contrasts and are always clustered by document.

---

## 2. Spawn-keyed RNG

No stream is derived by `seed + k`. Every stream is derived independently, so adding
an arm cannot shift another arm's data order.

```python
import hashlib, torch

def stream(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8]
    g = torch.Generator()
    g.manual_seed(int.from_bytes(digest, "big"))
    return g
```

| Stream | Governs |
|---|---|
| `data` | document order, window draw, packing order |
| `lora_init` | LoRA A/B matrices |
| `block_init` | inserted block weights (output projection zeroed **after** init) |
| `head_init` | A′/B′ parameter-matched head |
| `dropout` | dropout masks |
| `decode` | eval sampling. Greedy arms have none; the **null arm** uses `seed + 10000` and a disjoint item set |

`decode` is separate on purpose: the null must resample, or its variance is zero and
the CI is a point (N2 in [PROTOCOL.md](PROTOCOL.md) §1).

---

## 3. E1 manifest

Four cells at *n* = 32 (pilot-set, floor 24, cap 64), shared-trunk design.

| ID pattern | Count | Unit | Params trained | Tokens/run |
|---|---|---|---|---|
| `e1/trunk/s*` | 32 | run | LoRA only | 30M |
| `e1/A/s*` | 32 | run | block | 30M |
| `e1/B/s*` | 32 | run | block (loss weight 0) | 30M |
| `e1/Ap/s*` | 32 | run | head | 30M |
| `e1/Bp/s*` | 32 | run | head (loss weight 0) | 30M |
| `e1/E/s*` | 12 | run | block, permuted targets | 30M |
| `e1/D/s*` | 6 | run | none (vehicle) | 30M |
| `e1/N/s*` | 3 | run | none (frozen random block) | — |
| `e1/PChi/s*`, `e1/PClo/s*` | 3 + 3 | run | LoRA, 2M tokens | 2M |
| `e1/*/s*.joint` | 8 × 4 | run | trunk + block jointly | 30M |
| `e1/A/s*.dose*` | 48 | run | block | 30M |

**Total E1:** 219 training units. Every confirmatory training run is 30M tokens
(`experiments.e1.arms.TOKENS_PER_RUN`, `docs/PROTOCOL.md` §5.6), including the
dose rows. The GPU-hour totals in §7 and in PROTOCOL §11 were estimated under
an earlier mix of 10M and 30M and are not re-derived here. Throughput the
estimates used: 4,029 tok/s (bf16, ctx 512, micro-batch 4).

**Analysis.** Per-seed paired difference A − B, then mean across seeds. A′ − B′ the
same. The joint-confirmation arm is analysed separately and never pooled with the
shared-trunk arm.

---

## 4. E2 manifest

| ID pattern | Count | Unit | Notes |
|---|---|---|---|
| `e2/<F>/A/s*` | 12 × 3 | run | latent emission, top-1 gate, capacity cap |
| `e2/<F>/Btext/s*` | 12 × 3 | run | matched-FLOP text CoT, verified within 5 % |
| `e2/<F>/Bdepth/s*` | 12 × 3 | run | matched-FLOP extra depth |
| `e2/<F>/E/s*` | 12 × 3 | run | vehicle |
| `e2/F1/C/s*` | 3 × 3 | run | λ = 0 fault detection |
| `e2/F1/dose-*/s*` | 36 | run | λ ∈ {low, mid, high}; VQ ∈ {1, 8, 16} bits |

**Total E2:** 189 runs ≈ **290 GPU-hours** at 1.7B instruct. An 8B replication of
the full manifest is ≈ 4.7× (≈ 1,360 h); if run, it is reported separately and
never pooled.

**Backbone precondition.** `Qwen/Qwen3-1.7B` (instruct), not `-Base`. Before any
run, `experiments/e2/evaluate.py` must clear the §6.3.1 capability screen
(unweighted mean CoT−direct ≥ 0.10). Measured: 1.7B Base +0.067 **VOID**,
1.7B instruct +0.35 **PASS**, 8B instruct +0.63 **PASS**. The CoT budget is
**1024 new tokens for every family** (F48: effects saturate by 1024 and 2048 adds
nothing; at 512 all F3 traces truncate, and at 256 8B truncates F1). Thinking is
disabled so `B_text` is explicit text CoT, not hidden reasoning.

**Analysis.** Unweighted mean of exact match across F1/F2/F3. Two-level bootstrap:
items within run, then runs. Family-level results are reported, never gating.
Responder proportion uses an exact binomial CI.

**Checkpoint.** `e2/capability/<model>/<family>.json` (screen; n = 20, not pooled)
and `e2/<F>/<arm>/s*` (trained runs). The screen's rows are reported alongside
the trained arm so the comparator's ceiling is visible.

---

## 5. E4 manifests

### E4-S1 — interface (no training of the refiner)

| ID pattern | Count | Unit |
|---|---|---|
| `e4s1/adapter/f*` | 3 | adapter fit |
| `e4s1/{true,shuf,none,dose50}/f*/i*` | 4 × 3 × 192 | generation |
| `e4s1/textpc/{true,rand}/f*/i*` | 2 × 3 × 96 | generation |
| `e4s1/null/{a,b}/i*` | 2 × 192 | generation |

≈ 3,300 generations, ≈ **16 GPU-hours**.

### E4-S2 — refiner

| ID pattern | Count | Unit |
|---|---|---|
| `e4s2/{true,nodraft}/s*` | 10 + 10 | training run |
| `e4s2/{true,shuf,none}/s*/i*` | 3 × 10 × 128 | generation |

**Total:** 20 training runs, ≈ 3,840 generations, ≈ **190 GPU-hours**.

---

## 6. G5 — OLMo-2-1B

| Stage | Runs |
|---|---|
| E0 repeat | 3 adapter fits × 128 items |
| E1 | 16 seeds × 4 cells = 64 |
| E4-S1 | 3 adapter fits × 128 items |
| E4-S2 | 5 seeds × 2 conditions = 10 |

≈ **270 GPU-hours**. Decision: same sign and overlapping CI with Qwen; never pooled.

---

## 7. Run order

Serial on one card. Ordered so the cheapest decisive information arrives first.

| # | Stage | Hours | Cumulative |
|---|---|---|---|
| 0 | Corpus build (G-1) | 0 | — |
| 1 | G0 assay | 11 | 11 |
| 2 | G3 E4-S1 | 16 | 27 |
| 3 | G1 E1 pilot (20 runs) | 100 | 127 |
| 4 | G2 E1 main | 600 | 727 |
| 5 | G4 E4-S2 (conditional on G2) | 190 | 917 |
| 6 | G2b E2 | 290 | 1,207 |
| 7 | G5 OLMo | 270 | 1,477 |
| 8 | G6 dose curves (conditional) | 240 | 1,717 |

**Stop points.** After stage 2: if E4-S1 necessity ≈ 0, the latent line is dead —
skip 5 and 8, and re-scope 6 (E2 has its own endpoint and can still run). After
stage 4: if G2 fails, skip 5, 8 and halt the programme per
[PROTOCOL.md](PROTOCOL.md) §3.

---

## 8. Pipeline qualification

Run before stage 1 and re-run after any harness change. Any failure = VOID.

**Tooling status.** Two checks have tooling today; the other five must be
instrumented before G1. This is the first engineering ticket, not a note.

| Check | How | Status |
|---|---|---|
| Guard logic is correct (NaN/Inf stripped from grads) | `python -m experiments.pivot._verify_nan_guards` | **exists** — unit test of the guard, prints `nan_guard_ok` |
| Throughput and peak VRAM at ctx 512, micro-batch 4 | `python -m experiments.pivot._profile_hotspot --n 200` | **exists** — profiler, no ctx/micro-batch flags yet |
| Sanitised microbatches over 500 steps = 0 | needs a counter in the trainer, logged per arm | **must build** |
| Step-0 loss bit-identical across A / B / E | needs a step-0 probe run for each cell after zero-init | **must build** |
| Token-stream SHA-256 identical across arms | hash the packed token stream per run; all arms must report one value | **must emit** the file |
| Eval NaN/Inf = 0 | assertion in every eval path; F27 shows a gate can pass on `Infinity` | **must build** |
| Arm sanitise-rate spread ≤ 2 pp | same counter as row 3, aggregated per arm | **must build** |

Why each threshold exists: the sanitise counter prevents an arm-dependent
intervention (measured 10.6 % skip rate in `train2.log`); throughput and
micro-batch are *power* requirements, not speed ones (`n ∝ σ²`); the step-0 loss
identity is the only proof that the inserted block is the sole difference between
arms; the token hash prevents data-order drift; and the NaN assertion exists
because F27 shows a gate that passed with `Infinity`.

---

## 9. Checkpoint and analysis policy

- Checkpoint every 500 steps; keep all. Only the **final** checkpoint is analysed.
- **No best-checkpoint selection.** Choosing a checkpoint by its score is optional
  stopping and is the mechanism behind the 8-seed → 20-seed reversal.
- The analysis script is frozen and hashed into the manifest **before** unblinding.
- The sealed test split is read once per claim. The read is logged with a timestamp.
- Arm labels are stripped for any human check; the judge sees `output_a` /
  `output_b` in randomised order.

---

## 10. Which ledger row each run would repeat

The rule from [PLAN.md](PLAN.md): *if the wall is the same and the mechanism is the
same, don't run it.* Each experiment names the rows it must not repeat.

| Experiment | Would repeat | Why it does not |
|---|---|---|
| E1 | F24 (steered arm loses to identity), F26 (`fed_vs_concept` flat), F28/F29 (judge up, held-out flat) | First test of necessity with an ablation *and* an isotype at matched parameters |
| E2 | F23/F24 (auxiliary channel ignored) | First test with an independent exact-match endpoint and a charged token budget |
| E4-S1 | F18 (oracle render 0.185), F20 (undertrained writer) | First interface test with a *fitted* projector rather than an untrained one, and a necessity contrast rather than an absolute floor |
| E4-S2 | F17 (retain Δ +0.034), F19 (negative token-prefix) | First joint refiner with a draft-blind training-time control |
| OLMo | F33 (81 % Wikipedia) | First measurement of contamination on a backbone with open training data |

---

## 11. Reproducibility record

Per run, persist: run ID, git commit, config hash, corpus token-stream SHA-256, all
six spawn keys, token budget, final checkpoint path, wall-clock, peak VRAM, and the
sanitised-microbatch count. A run missing any of these is excluded from analysis and
reported as excluded.
