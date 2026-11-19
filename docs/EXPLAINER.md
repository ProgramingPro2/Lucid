# Explainer — the vocabulary, with pictures

Companion to [WALLS.md](WALLS.md), [FINDINGS.md](FINDINGS.md) and
[PROTOCOL.md](PROTOCOL.md). It exists because the programme is written in a
vocabulary that mixes machine learning and clinical-trial statistics, and a reader
who knows one of those should not have to guess the other.

Nothing here is a new claim. Every number is a pointer to a row in
[FINDINGS.md](FINDINGS.md) or a plan in [PROTOCOL.md](PROTOCOL.md).

---

# Layer 1 — The model

## Picture 1 — what an LLM is, and where a block gets inserted

```
 text ──► tokens ──► [block 1] ──► [block 2] ──► ... ──► [block N] ──► P(next token)
                        │             │                      │
                        └─────────────┴──────────────────────┘
                          hidden state at each position
                          ("what the model represents here")
```

```
 ... ──► [block k] ──► ┌─────────────────┐ ──► [block k+1] ──► ...
                       │ INSERTED BLOCK  │
                       └────────┬────────┘
                                │ output added to the residual stream
                                │
                                └── JEPA head predicts the model's own
                                    state Δ tokens ahead
                                          ▲
                                     EMA copy (slow teacher,
                                     gradients cut)
```

| Term | Plain meaning | Why it appears here |
|---|---|---|
| **token** | A word-piece; the model's unit of text (~¾ of a word). | "Per-token" = per output step. |
| **base model** | Pretrained, not instruction-tuned. | Less RLHF mode collapse; better for research. |
| **LoRA** | Small trainable matrices bolted onto frozen weights. | Lets a 3090 adapt a 1.7B model without full fine-tuning. |
| **QLoRA** | LoRA on top of a 4-bit quantised model. | Fits larger models in 24 GB of VRAM. |
| **hidden state** | The vector at one position in one layer. | The prediction target for the JEPA head. |
| **residual stream** | A running sum that every block reads from and writes to. | "Inserting" a block means adding to this stream. |
| **EMA target** | A slowly-updated copy of the network used as a teacher, with gradients cut. | Stops the student cheating by moving the target. |
| **JEPA** | Predict in *representation* space instead of reconstructing the input. | The core idea under test. |
| **collapse** | All vectors become identical, so prediction is trivially easy. | Inflates prediction scores 8× (F10). |
| **anisotropy** | Vectors crowd into a narrow cone, so distances mean less. | Measured 0.003 (healthy) vs 0.61 (collapsed). |
| **effective rank** | How many independent directions actually carry variance. | 110/512 is low; 456/512 means off-manifold. |
| **VICReg** | A loss (variance + covariance terms) that prevents collapse. | Required, or the metric becomes meaningless. |

## Picture 2 — latents, and the two ways to inject one

```
   paragraph ──► encoder ──► z (512 numbers) ──► "the gist, as coordinates"

   Injection A — soft prefix:              Injection B — FiLM:
   z ──► projector ──► 16 fake tokens      z ──► scale + shift applied
        [t1][t2]...[t16][real text...]          identically at EVERY position
        (can carry content)                     (a global tint; cannot place content)

                                  ▲
                      this is why FiLM failed (0.185 vs 0.650 human, F18)
```

| Term | Plain meaning | Why it matters |
|---|---|---|
| **latent / vector / embedding** | A fixed-length list of numbers standing in for meaning. | "Concept space" = the space these live in. |
| **cosine similarity** | Alignment of two vectors: 1 = same direction, 0 = unrelated. **Symmetric.** | Cannot express "before/after" — W1. |
| **soft prefix** | K fake token-embeddings prepended to the real input, carrying a latent. | The only injection channel that can carry *content*. |
| **adapter / projector** | A small network mapping latent → prefix tokens. | It must be **fitted**; an untrained one emits noise (norm 0.064 vs 0.9). That is why E4-S1 fits three of them. |
| **FiLM** | Per-channel scale and shift applied at every position. | A tint, not a message. |
| **teacher forcing** | Feeding true previous tokens during training. | Standard; also why held-out evaluation must be separate. |
| **greedy vs sampling** | Always take the top token, vs draw from the distribution. | Greedy is deterministic — which is why an "identical pair" null has zero variance. |

---

# Layer 2 — The measurement

## Picture 3 — how a metric becomes circular

```
   BROKEN
   ┌─────────────────────────────────────────────────────┐
   │                                                     │
   target z ──► generator ──► text ──► SAME encoder ─────┘──► cos = "score"
                                       (the one that
                                        defined z)
   The scorer shares the mechanism with the treatment.
   It rewards self-consistency, not content.

   FIXED
   target z ──► generator ──► text ──► INDEPENDENT measure ──► score
                                       (entity recall, a different
                                        encoder, or a calibrated human)
```

This is the defect in the repo's existing 2-AFC: the same codec supplies the target
and judges the text. It is why [PROTOCOL.md](PROTOCOL.md) §1 lists an independent
endpoint as non-negotiable, and why E4's primary endpoint is entity recall rather
than cosine alone.

## Picture 4 — the central quantity: necessity

```
   necessity = Q(channel present) − Q(channel ablated)

   Q(true draft)  −  Q(shuffled draft)

        0 ────────────────► decorative. Remove it; nothing happens.
       >0 ────────────────► load-bearing. This is the only thing worth building on.
```

| Term | Plain meaning | Why it appears here |
|---|---|---|
| **NLL** | Negative log-likelihood: average surprise per token. Lower is better. | The standard language-modelling endpoint. |
| **bits / nats** | Units of information; 1 nat ≈ 1.44 bits. | "+54 nats" turned out to be 50.4 capacity + 0.4 conditioning (F13/F14). |
| **perplexity** | `e^NLL`; "how many options the model is torn between." | Same information, friendlier scale. |
| **context gain** | NLL with shuffled context − NLL with true context. | Isolates *use of context* from raw knowledge. The primary endpoint for E1. |
| **necessity** | Quality drop when the channel is ablated. | The metric the repo never computed (Q1). |
| **ablation** | Delete, zero, or shuffle a component and re-measure. | The only way to prove something is load-bearing. |
| **retention@k8** | Cosine between paragraph *k* and the seed paragraph. | Coherence proxy; human band is +0.251. |
| **2-AFC** | Two-alternative forced choice: pick which of two outputs matches the target. Chance = 0.5. | Cleanest discriminative test — *if* the scorer is independent. |
| **z-identification** | Pick the true target from N distractors. Chance = 1/N. | What the repo's "2-AFC" actually computed. |
| **human band** | The distribution of a metric on real human text. | The natural reference; 0.650 for sentence ↔ paragraph. |
| **contamination / leakage** | Test data was in the model's training set. | WikiText-103 is almost certainly inside Qwen3. |
| **sealed split** | A held-out set examined exactly once. | Prevents the 8-seed → 20-seed mirage (+0.132 → +0.013). |

---

# Layer 3 — The comparison

## Picture 5 — what each arm in E1 is for

```
                        JEPA loss applied        JEPA loss OFF
   block trained         A  TREATMENT             B  isotype control
                         (the claim)              (same params, same init,
                                                   loss switched off)

   block frozen random   C  ─── loss never changes anything ─── C′
                            (the two cells are the SAME system → decorative)

   no block              —                        D  VEHICLE
```

What each contrast identifies:

```
   A − B    = effect of the objective        (parameters held constant)
   A − D    = effect of block + objective
   A − E    = effect of using real correspondences   (E = shuffled targets)
   A − A′   = architecture vs objective      (A′ = objective on the trunk, no block)
```

## Picture 6 — what *n* counts: the nesting problem

```
   DOCUMENT  (n = 271)              ← experimental unit  ("animal")
     ├── paragraph 1                ← technical replicate ("section")
     │      ├── token …
     │      └── token …
     ├── paragraph 2
     └── paragraph 3

   SEED  (n = 12)                   ← experimental unit for training ("batch")
     └── one trained model
           └── evaluated on the same fixed test set
```

**Pseudoreplication** is counting 2,000 paragraphs as n = 2,000 when they come from
271 documents. Paragraphs within a document correlate ~0.79 (F8), so this shrinks
every confidence interval by roughly √(paragraphs per document) — fake precision.
It is also why E4-S1 takes **one item per document**: it makes the clustering
factor ≈ 1 for free.

## Picture 7 — blocking and pairing

```
            seed 1          seed 2          seed 3
          ┌────────┐      ┌────────┐      ┌────────┐
  arm A   │  run   │      │  run   │      │  run   │
  arm B   │  run   │      │  run   │      │  run   │
  arm D   │  run   │      │  run   │      │  run   │
          └────────┘      └────────┘      └────────┘
        same data order within a column
        arm order randomised within each column
```

Analyse the **per-seed difference** A − B, then average across seeds. Between-seed
variance cancels. This is the same logic as a paired clinical design, and it is why
the shared-trunk variant in [PROTOCOL.md](PROTOCOL.md) §5.2 can cut *n* by 4–9×.

## Picture 8 — the E4 crossover: necessity as an interaction

```
                      target = i          target = j
     draft = z_i      MATCH   (a)         MISMATCH (b)
     draft = z_j      MISMATCH (c)        MATCH    (d)

     necessity = (a + d) − (b + c)      ← interaction, not a main effect
```

This separates "the draft carries **content**" from "the draft sets the
**register/topic**." A cross-document shuffle cannot do that, because 79 % of the
variance is document identity.

| Term | Plain meaning | Why it matters |
|---|---|---|
| **replicate** | An independent repeat of the whole unit. | Seeds for training; documents for evaluation. |
| **experimental unit** | The thing *n* counts. | Using the wrong unit is the most common statistical error in ML. |
| **pseudoreplication** | Nested units counted as independent. | Inflates significance by √(nesting). |
| **blocking** | Group similar units so nuisance variance cancels. | Here: seed. |
| **pairing** | Same block, different arms. | Turns a between-group comparison into a within-group one. |
| **randomisation** | Random arm order / assignment. | Stops GPU thermal drift being mistaken for an effect. |

---

# Layer 4 — The decision

## Picture 9 — three verdicts carrying one rule

```
  FAIL                    INCONCLUSIVE                   PASS
  |<---- CI ---->|        |<------- CI ------->|      |<---- CI ---->|
  ───────────────●──────────────────●─────────────────────●──────────► effect
                 0                 MPID                  2×MPID
   CI entirely below MPID   CI straddles MPID      CI lower bound > MPID
```

The rule is **not** "CI excludes zero." It is "CI lower bound beats a
pre-registered **MPID**."

## Picture 10 — VOID is not a soft FAIL

```
   FAIL : the question was asked correctly, and the answer is "no".
   VOID : the instrument was broken, so no answer exists.
          Fix the instrument and re-ask. Never record it as a result.
```

## Picture 11 — why E3 was cut: the horn

```
                    ┌── condition on previous TEXT ──► W2 floor:
   E3 policy ───────┤     every arm lands near 0.445 ──► all arms tie
                    │
                    └── condition on LATENT only ────► W3 ceiling:
                          every arm lands near 0.19 ──► all arms tie

   No third path. The null is guaranteed before the first GPU-hour.
```

| Term | Biology analogue | Meaning |
|---|---|---|
| **endpoint** | Primary outcome | The number the decision uses. |
| **primary / secondary** | Primary vs exploratory outcome | Exactly one decision metric; the rest are descriptive. |
| **effect size** | Treatment effect | Magnitude of the difference. |
| **SD / SE** | Spread / precision | SE = SD/√n. |
| **CI** | Confidence interval | Range of effects compatible with the data. |
| **bootstrap / cluster bootstrap** | Resampling | Resample **documents**, not paragraphs, to respect nesting. |
| **power** | Study power | Probability of detecting a real effect of a given size. |
| **MPID** | **Minimal clinically important difference (MCID)** | The smallest effect worth acting on — fixed *before* the data. |
| **alpha / FDR / Holm** | Multiple-comparison correction | Controls false positives across arms and endpoints. |
| **pre-registration** | ClinicalTrials.gov | Hypothesis, arms, endpoint and decision rule fixed in advance. |
| **fixed sequence** | Pre-specified test order | Test 1 at α; test 2 at α only if test 1 passes. Controls FWER with no penalty. |
| **decision rule** | Stopping/analysis plan | PASS / FAIL / INCONCLUSIVE / VOID thresholds. |
| **stopping rule** | Interim-analysis plan | Prevents optional stopping (peeking until it looks significant). |
| **blinding** | Blinded assessment | Arm labels hidden from the judge. |
| **ICC** | Inter-rater reliability | Agreement between raters/judges; ≥ 0.70 human–human, ≥ 0.60 judge–human. |
| **LODO** | Leave-one-out sensitivity | Does the decision reverse if one document is removed? |

---

# The numbers that keep recurring

| Number | What it is | Read it as |
|---|---|---|
| **0.650** | Human ceiling, sentence ↔ own paragraph (MiniLM) | The bar for "renders content" |
| **0.445** | Unsteered null, paragraph-level (MiniLM) | The floor. Any gate below this cannot fail — hence N1 |
| **0.185 / 0.221** | Oracle render / phase-3 writer | The render wall — 30 % of the ceiling |
| **0.663 vs 0.667** | Steered vs unsteered `fed_vs_concept` | Steering added nothing (F26) |
| **0.755 / 0.813** | 2-AFC, steered / unsteered | The control was *better* (F26) |
| **−0.4865** | Sentence delta autocorrelation (bound −0.5) | ~4 % order signal (W1) |
| **50.4 vs 0.4** | Nats of capacity vs conditioning | "+54 nats" was mostly capacity (F13) |
| **1,677 / 4.1 %** | Training windows / documents that produced any | The corpus starved the model (F32) |
| **0.0112 nats** | E1's MPID (0.5 % relative NLL) | The smallest effect worth reporting |
| **32 / 24–64** | E1 seeds per cell | Set by a 10-seed pilot, not by preference |
| **~1,500 h** | Programme GPU time | ~9–11 weeks serial on one 3090 |

---

# How to read a result in this repo

1. Find the **row** in [FINDINGS.md](FINDINGS.md). If the mechanism is the same as
   a FAIL, it does not matter how large *n* was.
2. Find the **wall** in [WALLS.md](WALLS.md). If the proposal does not name a wall
   it survives, it is a re-run.
3. Ask the **four questions**: independent endpoint? null that varies?
   pre-registered MPID? sealed split?
4. Only then look at the number.

A result that fails any of the four is not a weak result. It is not a result.
