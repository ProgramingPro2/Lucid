# Walls

Measured constraints, not opinions. Each one cost a real experiment to find, and
each one killed at least one architecture. **Rule: before proposing a run, name
the wall it survives.** If it survives none, it is re-running a test that has
already failed.

Status marks: `MEASURED` (numbers in hand), `MEASURED-ONCE` (one run, not
replicated), `INFERRED` (follows from geometry/arithmetic, no direct run).

---

## Terms used here

- **latent / vector** — a fixed-length list of numbers standing in for meaning.
  "Concept space" is the space those lists live in.
- **cosine** — alignment of two vectors, 1 = same direction, 0 = unrelated.
  Symmetric: `cos(a,b) = cos(b,a)`.
- **EMA target** — a slowly-updated copy of a network used as a teacher, so the
  student cannot cheat by moving the target.
- **ablation** — delete a component and see whether quality drops. The only way
  to show something is load-bearing.
- **control arm** — the do-nothing version: identity (copy input), zero (feed
  zeros), or shuffle. A steered result must beat *this*.
- **2-AFC** — two-alternative forced choice: a judge picks which of two outputs
  matches the target. The one metric here that has never been gamed.
- **nats** — information units, ≈1.44 bits. Always ask "+N nats *compared to
  what*."

---

## W1 — The sentence-embedding space has no order to predict

**Status: MEASURED (three independent ways).**

Narrative order is not weakly present in this space; it is absent to within
measurement error, for two structural reasons.

Evidence:

| Measurement | Value | Meaning |
|---|---|---|
| Consecutive-delta autocorrelation | **−0.4865** (n=19,982) | Exchangeability bound is −0.5. The sequence is 97.3% of the way to "unordered bag around a topic centroid." Implied AR coefficient ρ ≈ 0.04. |
| `adj − within` cosine | **−0.017** official, **+0.016** after removing self-pairs from the shuffle | Either way ≈ 0. A healthy space also fails this gate on this data, so the gate itself was mis-specified. |
| adjacent cos vs within-doc cos | 0.361 vs 0.378 | Adjacent sentences are *not more* similar than same-document non-adjacent ones. |
| cos(sentence, own-doc centroid) | 0.629 | Document identity dominates everything else. |
| next-sentence prediction | cosine 0.148; top-1 retrieval **0.05** over 512 candidates; centered cos(μ, x_next) 0.040 vs identity 0.102 | The predictor loses to "copy the current vector." |

Two mechanisms, both fatal at once:

1. **Objective family.** VICReg / uniformity / InfoNCE select features that are
   *stable* across the positive set. Discourse state — the thing that differs
   between sentence *i* and *i+1* while topic is constant — is exactly what
   these losses treat as noise and remove. This was tested directly: the repair
   run paid for adjacency with an InfoNCE align term at weight 0.25, and the
   adjacency margin never moved over 15 epochs (`val_adj_cos − val_shuf_cos`
   stayed in +0.005..+0.009). The term's optimum was a *doc-identity* code,
   because its negatives were only cross-window.
2. **Geometry.** Cosine is symmetric. "A precedes B" cannot be a proximity
   relation. At best a space can host an ordered *curve*; the delta measurement
   shows not even the curve formed.

**Forbids:** next-sentence vector prediction as a target, in any head (point,
delta, diffusion, MDN-as-transition); judging coherence by closeness to a
predicted latent; spending GPU on encoder surgery in this objective family.

**Permits:** the same encoder as a *topic* codec at paragraph scale (see W3);
as a critic/reranker; as a destination that is set once per paragraph.

**Overturns if:** a space shows ρ ≥ 0.3 on structure-intact data (W4), or an
asymmetric conditional model beats an *unconditional* mixture by ≥ 5 nats.

Evidence: `.hermes/plans/2026-08-08_100000-fable5-reassessment.md` §1b, §1d,
Part 2; `artifacts/sentence_pred/RESULTS.md`; `artifacts/jepa_repair_1m/train.db`.

---

## W2 — The LM floor: context alone already scores what steering claims

**Status: MEASURED (two independent stages).**

A pretrained LM conditioned on the previous text already produces output that
re-encodes near the target. So "our steering scored 0.66" is not evidence.

| Measurement | Steered | Unsteered control |
|---|---|---|
| `fed_vs_concept` (Phase-A gate) | 0.663 | **0.667** (Stage-B baseline) |
| MiniLM fidelity | 0.454 | 0.445 |
| 2-AFC identification | 0.755 (gate 0.85) | **0.813** |

And in the from-scratch writer, the do-nothing arms *won*:

| Experiment | Steered | Identity | Zero |
|---|---|---|---|
| retain@k8 (forward two-channel) | 0.118 | **0.171** | 0.113 |
| retain@k8 (coarse steer) | 0.105 | 0.112 | **0.124** |

The concept channel was never load-bearing. "Outputs differ, therefore steering
works" is false: divergence measures *influence*, not *control*.

**Forbids:** gates expressed as "steered ≥ unsteered baseline + ε"; claims from
cross-arm divergence; any result without a matched ablation.

**Requires:** a matched-ablation control (identity/zero/shuffle at the same
tokens and seeds), a discriminative test (2-AFC or blind pairwise), and a
**necessity** score — how much quality drops when the channel is ablated.

**Overturns if:** steering beats identity on 2-AFC with CI lower bound above
chance, or necessity > 0 at matched compute.

Evidence: `artifacts/pivot_stageA/gate_g1_epoch01.json`;
`artifacts/pivot_stageB/gate_b0.json`; `artifacts/sentence_pred/forward_gen/RESULTS_forward.md`;
`artifacts/sentence_pred/coarse_steer/RESULTS_coarse.md`.

---

## W3 — The render wall: pooled topic → text is the bottleneck

**Status: MEASURED.**

The concept machinery *works in concept space*. The renderer destroys it. This
is the single most important localization in the repo: the signal dies in the
last step, not the first.

| Measurement | Value |
|---|---|
| Real mid-sentence vs its own paragraph context (MiniLM, G=5) — **human ceiling** | **0.650** (chance 0.053) |
| Oracle: writer given the *true* paragraph centroid | **0.185** |
| Phase-3 fine-tuned writer, pooled topics | **0.221** (gate 0.25) |
| Anchored leash in JEPA space | 0.355 vs human **0.336** |
| Human retain@k8 − shuffled (text) | **+0.251** (bootstrap p5–p95 +0.180..+0.319) |

Causes, all measured:

- **FiLM is a position-uniform tint.** One soft token plus per-block γ/β applied
  identically at every position can bias register; it cannot *place* content
  ("this paragraph is about Port Vale").
- **The token-prefix channel measured −0.041** (never helped) yet is chained at
  inference, feeding the writer's own drift back in.
- **Training starvation.** Phase 3 saw 80k examples ≈ 35% of one epoch; val loss
  was still falling at the last epoch (3.659 → 3.449).
- **Backbone prior.** A 161.7M scratch writer trained on Gutenberg-heavy text
  emits 19th-century characters regardless of seed.

**Forbids:** concluding "latent steering doesn't work" from a failed renderer;
FiLM as the sole content channel; CE-only fine-tunes of the scratch writer;
fewer than 2 full epochs.

**Permits:** pretrained 1–4B base + LoRA; K ≈ 16–32 soft prefix tokens or
per-layer memory-KV; ≥ 2 epochs; a re-encode contrastive auxiliary.

**Overturns if:** a renderer reaches ≥ 0.40 MiniLM fidelity (≈ 0.62 × human) from
a held-out pooled centroid.

Evidence: reassessment §1c, §1f, Part 2; `artifacts/writer_p3/train3.log`;
`artifacts/sentence_pred/{coarse_steer,forward_gen}/`.

---

## W4 — The corpus never contained the phenomenon

**Status: MEASURED.**

| Measurement | Value |
|---|---|
| Corpus shape | 113,216 single-paragraph blocks; article/chapter structure destroyed before any model saw it |
| Shuffle check | adjacent same-register rate 0.3714 vs 0.3715 expected under independence |
| Register skew | 81% Wikipedia by sentence; the `min_sents ≥ 4` filter keeps 93.8% of wiki but **40.9% of fiction** |
| Window starvation | with ctx=32, **1,274 of 31,000 docs (4.1%)** yield any training window → **1,677 windows**, 14.7% of sentences; 95.4% of eligible docs are Wikipedia |

Every architecture was asked to learn long-range coherence from data in which
no document exceeds one shuffled paragraph, using a predictor trained on 1,677
windows of an unrepresentative 4% subset.

**Forbids:** any sequence-learning claim from the ctx-32 runs; any
cross-register generalization claim.

**Permits:** reuse of the segmenter, doc-disjoint split machinery, `data/raw/`
originals, and the paragraph-centroid evidence.

**Overturns if:** a structure-intact, single-register corpus (corpus v2) shows
measurable adjacency signal.

Evidence: reassessment §1a; `experiments/sentence_pred/data.py:199,266`;
`configs/sentence_pred.json:8`.

---

## W5 — Metrics that cannot fail

**Status: MEASURED.** This is the wall that produced the other four: bad metrics
kept bad architectures alive for months.

| Instance | What happened |
|---|---|
| MDN "+54 nats" | **50.4 nats were mixture capacity, 0.4 nats were conditioning.** The gate compared a K=8 mixture to a K=1 vMF — a strawman. |
| Fluency guard | Passed with `mean_nll = Infinity`, `ppl_ratio = NaN`, win 0.469 ≥ 0.45. |
| `self_bleu4` "must not increase" | Baseline 5e-5; trivially passable. |
| `reconstruction_cosine = 1.000` | Retrieval decoder returns a bank sentence, so its own embedding is the target. True by construction, not evidence. |
| `val_pred_cos` | 0.80 (anisotropy 0.61) in one regularizer arm vs 0.10 (anisotropy 0.003) in the other. **Collapse inflates it 8×.** ⚠ the two logs are ambiguously named — confirm arm mapping before citing. |
| `novelty_rate = 1.0` | Measured on off-manifold *steer vectors*. Re-encoded text novelty is **0.750 vs human median 0.745** — i.e. no novelty at all. |
| 8-seed sweep | Best of 49 configs at SE ≈ 0.07; the 20-seed confirm then killed it. Winner's curse by construction. |
| `fed_vs_concept ≥ 0.55` | Passes from context continuity alone (W2). |
| Single-seed demo | "Moon is closer than the Sun" (n=1) is not evidence. |

**Forbids:** a gate without a named control; NaN/Inf treated as pass; thresholds
without CIs; any metric that a degenerate model scores *better* on.

**Requires:** gate spec as data with its control arm declared; NaN/Inf → FAIL,
run void; decisions on the CI lower bound; per-corpus human reference bands;
novelty measured on re-encoded text, two-sided.

Evidence: reassessment §1d, §1f; `gate_g1_epoch01.json`;
`artifacts/ablate_{none,vicreg}.log`.

---

## Which wall killed which architecture

| Architecture | Died to |
|---|---|
| Point-regression next-vector (×3) | W1 (+ W5: 0.38 cosine looked like progress) |
| Delta predictor "+0.39" | W1/W5: the answer is −0.5·Δt by arithmetic |
| MDN forward generation | W1 + W5: 0.4 nats conditioning, graded against K=1 vMF |
| Diffusion head | W1: Lemonade judge 0.0, max cos to bank 0.34, consecutive cos 0.02 |
| Two-channel writer | W3 (+ W2: identity arm beat it) |
| Coarse-anchor sampler | W3: leash correct in concept space (0.355 vs human 0.336), text Δ +0.013 |
| Scratch writer (161.7M) | W3 |
| Phase-A soft-prefix adapter | W2: 0.663 vs 0.667 unsteered; 2-AFC 0.755 vs 0.85 |
| Stage-B RAFT / GRPO | W2: 2-AFC 0.805/0.797 vs 0.88; heldout judge +0.006/+0.008 |
| Entire ctx-32 family | W4: 4.1% of docs, 1,677 windows |
