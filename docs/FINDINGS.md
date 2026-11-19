# Findings ledger

Every claim this project has tested, with the number that decided it. Read
[WALLS.md](WALLS.md) first; the walls are the constraints extracted from these
rows.

**How to use this file**

- Append, don't rewrite. A correction gets a new row with the old row's ID.
- Every row names the wall it tested (or `—` if it measured something usable).
- A `FAIL` is a result, not a setback. It is the reason the project knows what it
  knows.
- Before a new run: find the row it would repeat. If the wall is the same and
  the mechanism is the same, don't run it.

Verdicts: `PASS` (survives its gate), `FAIL` (failed its gate, for a known
reason), `INVALID` (the measurement could not have failed), `MIRAGE` (a gain
that dissolved under a corrected baseline), `PARKED` (not disproven, not
funded).

---

## A. Representation and encoder

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F1 | Codec round-trips text | `reconstruction_cosine = 1.000` | **INVALID** — retrieval decoder returns a bank sentence, so its own embedding is the target | W5 | `artifacts/metrics.json` |
| F2 | From-scratch JEPA decoder can read real concept vectors | round-trip re-encode **0.982** on fed vectors | **PASS** — the decoder was fine | — | `artifacts/diagnostics/decode_coherence.md` |
| F3 | The predictor can predict the next sentence vector | cosine **0.148** (diagnostics run); top-1 retrieval **0.05** over 512 candidates (metrics run) | **FAIL** | W1 | `artifacts/metrics.json`, diagnostics |
| F4 | Encoder repair creates adjacency signal | `val_adj − val_shuf` stuck at **+0.005..+0.009** across 15 epochs | **FAIL** — adjacency was paid for explicitly and still did not appear | W1 | `artifacts/jepa_repair_1m/train.db` |
| F5 | The space preserves order at all | delta autocorrelation **−0.4865** vs bound −0.5 ⇒ ρ ≈ 0.04 | **FAIL** — the space is a bag of sentences | W1 | reassessment §1d, Part 2 |
| F6 | `adj − within` is a valid gate | −0.017 official; **+0.016** self-pair-corrected; healthy spaces fail it too | **INVALID GATE** | W5 | reassessment §1b |
| F7 | Something in the space is predictable | `prefix_mean` centered cos **0.173** > identity 0.102 > vMF 0.120 > MDN 0.040 | **PASS (narrow)** — the topic centroid is the only signal | W1 | `artifacts/sentence_pred/RESULTS.md` G3 |
| F8 | Paragraph-scale topic persists | G=5 centroids: **0.7037** consecutive vs 0.5728 cross-doc, Δ **+0.131–0.138** | **PASS** — the salvageable signal | — | reassessment §1d |
| F9 | A sentence vector retains its sentence | teacher-forced reconstruction accuracy **44%** | Limitation — it is a topic code, not a content code | W3 | `artifacts/jepa_repair_1m/jepa/metrics.json` |
| F10 | `val_pred_cos` measures predictive quality | **0.80** (anisotropy 0.61) vs **0.10** (anisotropy 0.003) across the regularizer ablation | **INVALID** — collapse inflates it 8×. ⚠ arms ambiguously named; confirm before citing | W5 | `artifacts/ablate_{none,vicreg}.log` |

## B. Predictor and density heads

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F11 | Point regression predicts the next concept | converges to the conditional mean of a multimodal, near-unconditional target | **FAIL** | W1 | reassessment §1b |
| F12 | Delta predictor gains "+0.39" | optimal answer for an exchangeable sequence is **−0.5·Δt** | **MIRAGE** — mean reversion, not modelling | W5 | reassessment §1d |
| F13 | MDN captures a conditional next-sentence distribution | **50.4 nats** mixture capacity, **0.4 nats** conditioning; component eff-rank 9.3/512; all 400 docs share an argmax; π std ≤ 0.023 | **INVALID as a transition model** — it is a good unconditional prior | W5 | reassessment §1d |
| F14 | MDN should be promoted on +45 nats | the comparison was K=8 mixture vs K=1 vMF | **INVALID** — strawman baseline | W5 | `artifacts/sentence_pred/RESULTS.md` §3.5 |
| F15 | Diffusion head fixes the mean-seeking | Lemonade judge **0.0**; max cos to bank **0.34**; consecutive cos **0.02** | **FAIL** | W1 | `artifacts/jepa_1m_diffusion/eval_lemonade_diffusion/report.json` |
| F16 | Per-sentence sampling from the MDN composes | **82%** component-switch rate (Σπ² ≈ 0.184) | **FAIL by arithmetic** — genre roulette | W1/W5 | reassessment §1d |

## C. Writer and rendering

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F17 | Two-channel writer keeps a document coherent | retain Δ **+0.034** vs human **+0.251** | **FAIL** | W3 | `artifacts/sentence_pred/forward_gen/RESULTS_forward.md` |
| F18 | FiLM can carry content from a concept vector | oracle same-z render **0.185** MiniLM vs human **0.650** (chance 0.053) | **FAIL** — a position-uniform tint cannot place content | W3 | reassessment §1c |
| F19 | The token-prefix channel helps | **−0.041** measured, yet chained at inference | **FAIL** — and load-bearing with no evidence | W3 | reassessment §1c |
| F20 | Phase-3 fine-tune renders pooled topics | **0.221** vs 0.25 gate; 80k examples ≈ 35% of one epoch; val still falling 3.659 → 3.449 | **FAIL, and undertrained** — the run was stopped mid-descent | W3 | `artifacts/writer_p3/train3.log`, `p3_probe.json` |
| F21 | Scratch writer follows the seed | Gutenberg characters appear regardless of seed | **FAIL (qualitative)** — corpus prior dominates | W3 | reassessment §1c |

## D. Sampling and control

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F22 | Anchored sampler holds a paragraph topic | **0.876** cos@k8 vs 0.246 flat; concept space **0.355** vs human **0.336** | **PASS** — the one mechanism that cleared its gate | — | `artifacts/sentence_pred/coarse_steer/RESULTS_coarse.md` |
| F23 | Coarse steer improves text coherence | Δ **+0.0134** vs ≥ +0.10; identity 0.112 and zero 0.124 both ≥ coarse 0.105 | **FAIL** | W3 + W2 | `RESULTS_coarse.md` |
| F24 | The steered arm beats an identity control | retain@k8 **0.118** (steered) vs **0.171** (identity) | **FAIL** — the concept channel was ignorable | W2 | `RESULTS_forward.md` |
| F25 | Lucid's generated text is novel | re-encoded novelty **0.750** vs human median **0.745** | **FAIL** — novelty was never the scarce resource | W5 | reassessment §1f |

## E. Pretrained-LM pivot

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F26 | A JEPA latent prefix steers a pretrained LM | `fed_vs_concept` **0.663** vs **0.667** unsteered; MiniLM 0.454; 2-AFC **0.755** vs 0.85 | **FAIL_BUT_NOT_KILL** — renderability floor passed; controllability never moved | W2 | `artifacts/pivot_stageA/gate_g1_epoch01.json` |
| F27 | The fluency guard is a guard | "PASS" at win 0.469 ≥ 0.45 with `mean_nll = Infinity`, `ppl_ratio = NaN` | **INVALID** | W5 | same file |
| F28 | Stage-B RAFT improves on Phase A | overall_pass false: 2-AFC **0.805** vs 0.88; heldout judge **+0.006** vs +0.08 | **FAIL** | W2 | `artifacts/pivot_stageB/gate_b0.json` |
| F29 | Stage-B GRPO improves on Phase A | 2-AFC **0.797**; heldout **+0.008**; train judge 0.235 → 0.333 | **FAIL** — train judge up, heldout flat = reward overfit | W2/W5 | `artifacts/pivot_stageB/gate_grpo_step500.json` |
| F30 | "Moon is closer than the Sun" demo | n=1, single seed | **NOT EVIDENCE** | W5 | git `f48e7c4` |

## F. Data

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F31 | The corpus contains documents | 113,216 single-paragraph blocks; shuffle confirmed (0.3714 vs 0.3715) | **FAIL** | W4 | reassessment §1a |
| F32 | The corpus supports sequence training | **1,274 / 31,000 docs (4.1%)** yield a ctx-32 window → **1,677 windows**, 14.7% of sentences | **FAIL — starved** | W4 | `experiments/sentence_pred/data.py:266` |
| F33 | The corpus is register-neutral | 81% Wikipedia by sentence; filter keeps 93.8% of wiki but **40.9% of fiction** | **FAIL** | W4 | `experiments/sentence_pred/data.py:199` |

## G. Design and inference (2026-11 programme review)

These rows are not results about the model. They are results about how to measure
the model, and they are the reason [PROTOCOL.md](PROTOCOL.md) has the shape it has.

| ID | Claim tested | Key measurement | Verdict | Wall | Evidence |
|---|---|---|---|---|---|
| F34 | Item-level replication is enough for training-time contrasts | σ_run is 0.5–1.5 % of NLL (0.011–0.034 nats); σ_item is 0.147 cos. A 100-decode sweep is 100 technical replicates of one checkpoint | **INVALID** — decode count is not run count; `n ∝ δ⁻²` and δ is set by the run variance | W5 | [PROTOCOL.md](PROTOCOL.md) §2, §5.4 |
| F35 | The shared-trunk design cuts the required *n* | Fitting A/B/A′/B′ on one frozen trunk per seed makes the contrast within-trunk. If σ_diff falls 0.020 → 0.008, *n* falls 32 → ~12 | **PASS (design)** — variance reduction before replication; pre-registered at [EXPERIMENTS.md](EXPERIMENTS.md) §3 | — | [PROTOCOL.md](PROTOCOL.md) §5.2 |
| F36 | A fixed-sequence gate costs α | E1 → E4-S2 as a chain, plus an intersection–union test inside E1, controls FWER ≤ 0.025 with no per-test penalty. Three independent tests would need α = 0.0167 each | **PASS (design)** — the chain buys back ~20 % of *n* | — | [PROTOCOL.md](PROTOCOL.md) §2 |
| F37 | Micro-batch 1 is a speed problem | `grad_nan_skips = 102` after 15 steps (10.6 %); 7.17 / 24 GB used; `tgt_tok` varies 161–257 at ctx 512 | **INVALID HARNESS** — gradient variance inflates σ, and `n ∝ σ²`. A 3× σ reduction is a 9× *n* reduction | W5 | `train2.log`, `_profile_hotspot.py` |
| F38 | MiniLM cosine can be an absolute gate | unsteered null **0.445** vs gate **0.40**; human ceiling 0.650 | **INVALID GATE** — 0.445 > 0.40, so the gate cannot fail. Valid only as a **paired difference**, where 0.445 cancels | W5 | reassessment §1c; F18 |
| F39 | Dose curves strengthen a null | at *n* = 8 per level, a flat curve cannot distinguish "no effect" from "small effect" | **PARKED** — mechanism stage only, run after a positive (G6), never used as evidence for absence | W5 | [PROTOCOL.md](PROTOCOL.md) §9 |
| F40 | The throughput gate is achievable on this card | Qwen3-1.7B-Base, ctx 512, micro-batch 4, LoRA r=16, bf16: **4,029 tok/s**, 7.59 GB peak, 0 non-finite gradients. Greedy decode: **397 tok/s**, 3.26 GB | **PASS** — 2.0× the gate. The old 810–857 tok/s planning figure was ~4.8× pessimistic, so the calendar falls from 9–11 to ~2–3 weeks serial | — | `experiments/runner/measure_train.py`, `measure.py`; [PROTOCOL.md](PROTOCOL.md) §11 |
| F41 | QLoRA is needed at 1.7B | 4-bit NF4: **2,589 tok/s**, 7.02 GB — *slower* than bf16 (4,029 tok/s, 7.59 GB) and only 0.57 GB smaller | **FAIL (as a strategy)** — quantisation overhead exceeds its benefit below ~4B. bf16 is the primary training path; 4-bit is reserved for ≥4B if needed | — | `measure_train.py --load-in-4bit` |
| F42 | The E0 assay can detect a real effect before G0 | Legacy corpus, n = 14, 48 new tokens, real Phase-A adapter: positive control (true text-prefix − random text-prefix) **+0.186 cos, CI [0.109, 0.263]**, six times its 0.03 gate. But true-latent **0.574** is beaten by zero-latent **0.657**; shuffled 0.549; dose ladder [0.549, 0.577, 0.574] is **non-monotone**; necessity true−shuffled +0.121, CI [−0.029, 0.272] | **G0 NO-GO SIGNAL (endpoint, not channel)** — the readout sees content, but the latent channel is decorative on this corpus/adapter. This is exactly what G0 exists to catch; do not commit GPU-weeks until G0 runs on the new post-cutoff corpus | — | `artifacts/e0/rehearsal.json`, `experiments/e0/rehearsal.py` |

---

## What survives (the salvageable set)

| Asset | Why it survives | Number |
|---|---|---|
| Frozen JEPA encoder as a **paragraph topic codec** | Paragraph-scale persistence is real and human-level usable | Δ +0.131–0.138 |
| `anchored_sample` / slerp leash | Cleared its gate; correct in concept space | 0.876 cos@k8; 0.355 vs human 0.336 |
| vMF/MDN machinery as an **explicit prior** over paragraph centroids | Good unconditional density; wrong job title | 45 nats of capacity |
| Anti-mirage eval protocol: doc-disjoint splits, shuffled controls, centering/anisotropy floors, distinct-candidate counts, 20-seed confirms, blind sheets | It reliably killed mirages, including its own author's | killed 6 architectures |
| The negative measurements themselves | Genuinely new quantified anatomy | see W1–W5 |
| Soft-prefix adapter plumbing (`adapter.pt`, `encode_bank`, Phase-A trainer) | Parked, not disproven — it was never run with a content-capable channel | F26 |

## Open questions (the live edge)

| ID | Question | Why it is open | Cheapest test |
|---|---|---|---|
| Q1 | Can a latent channel ever become **necessary** rather than decorative? | Never measured; identity/zero arms matching steered arms is consistent with necessity ≈ 0 | ablate the channel at matched compute; measure the drop |
| Q2 | Can a pretrained backbone render a pooled centroid at ≥ 0.40 MiniLM? | The scratch writer could not (0.185–0.221); no pretrained renderer was tried with a real channel | one QLoRA run, ≥ 2 epochs, K=16–32 soft prefix |
| Q3 | Do paragraph centroids beat **text plans** as an intermediate? | Both plausible; never compared head to head | same budget, two intermediates |
| Q4 | Is a JEPA-style auxiliary head inside an LM load-bearing? | Never tried inside a token model — only in an external sentence space | ablate the head at equal params |
| Q5 | Does an **untrained** inserted block learn something the same block cannot learn without the JEPA objective? | The user's live idea; needs a trained/un-trained × with/without-objective 2×2 | small-scale, fixed token budget |

**Assignment in the 2026-11 programme.** Q1 is tested at **G3** (E4-S1); Q4 and Q5
at **G2** (E1 cells A/B for Q5, A − D for Q4). Q2 and Q3 are out of scope: the
renderer upgrade is not funded, and paragraph centroids vs text plans is a
study with no clean independent endpoint. See [PROTOCOL.md](PROTOCOL.md).

## Artifacts whose headline verdicts are misleading

Do not cite these without reading the corresponding ledger row.

- `artifacts/metrics.json` — `reconstruction_cosine: 1.0` is a retrieval
  artifact (F1); `next_vector_cosine: 0.384` is from a different run than the
  0.148 in the diagnostics and is not comparable (F3).
- `artifacts/sentence_pred/RESULTS.md` — the `FAIL` verdict is honest; the
  **`PROMOTE_MDN`** decision above it rests on the strawman in F14.
- `artifacts/jepa_1m_diffusion/` — reported judge scores near zero; any
  "diffusion improves it" reading comes from the heuristic judge on the Austen
  variant, which is unanchored.
- `artifacts/ablate_{none,vicreg}.log` — filenames are ambiguous about which arm
  removes VICReg (F10). Resolve before citing.
- `artifacts/pivot_stageA/gate_g1_epoch01.json` — `fluency_guard.pass: true`
  with infinite NLL (F27).

## Corrections already applied

- The `adj − within` gate is mis-specified, not a property of bad encoders; the
  self-pair-corrected value is +0.016, not −0.017 (F6).
- The "+54 nats conditional gain" is 50.4 capacity + 0.4 conditioning (F13/F14).
- "Novelty rate 1.0" was measured on steer vectors; the text was never novel
  (F25).
- Coherence failed because of the renderer, not the leash (F18/F22/F23).
- The E1 sample-size constant is **10.5**, not 11.6, and the 0.015 planning row is
  **n = 32**, not 40. The code was right; the protocol table and the test
  expectation were corrected together (PROTOCOL §5.4).
- Holm on `[0.01, 0.02, 0.03]` rejects all three (thresholds 0.0167 / 0.025 /
  0.05). `[True, False, False]` is Bonferroni, not Holm. A diagnostic-only bug in
  `holm` that overwrote the thresholds of non-rejected tests was fixed in
  `experiments/protocol/stats.py`.
