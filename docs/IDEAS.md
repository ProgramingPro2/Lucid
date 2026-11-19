# Ideas — live bets

These are research bets, not commitments. Each entry states the idea, the walls
it must survive, the control that makes it falsifiable, the cheapest first
experiment, and the kill condition. Nothing here is funded until it passes
[PLAN.md](PLAN.md)'s decision rules.

Read [WALLS.md](WALLS.md) first. The one-line summary of all four bets below:

> Every one of them lives or dies on **necessity** — whether the latent path
> changes the answer when it is ablated. No steered arm in this repo has ever
> beaten its own identity/zero control (W2). That is the fact to design against.

## Status after the 2026-11 review

| # | Idea | Disposition | Where |
|---|---|---|---|
| 1 | JEPA nodes in an LLM | **In programme** — confirmatory claim C1 | [PROTOCOL.md](PROTOCOL.md) §5, gate G2 |
| 2 | Optional per-token thinking vector | **In programme** — confirmatory claim C3, three task families | [PROTOCOL.md](PROTOCOL.md) §6, gate G2b |
| 3 | Reconstruction-reward distillation | **Cut permanently** — not deferred | §3 below |
| 4 | JEPA drafts, LLM refines | **In programme** — confirmatory claim C2, behind an interface stage | [PROTOCOL.md](PROTOCOL.md) §7, gates G3–G4 |

The programme is pre-registered. "In programme" means the design, *n*, MPID, gate
and kill condition are fixed; it does not mean funded beyond the gate.

---

## 1. JEPA nodes in the middle of an LLM

**Statement.** Insert a small untrained network (or an auxiliary predictive head)
between blocks of a pretrained LLM, give it a JEPA-style objective, and watch
whether training it produces any improvement over the same model without it.

**Survives W1 — and this is the important part.** The old project predicted in
an *external sentence-embedding space*, which has no order (W1). Here the
prediction domain is the LM's **own hidden state at token resolution**, where
next-token prediction is well-posed by construction. This is the one place in
the repo's history where order is natively present.

**Design constraints (each is a wall in disguise):**

- The EMA target must be the LM's own representation — not MiniLM, not SONAR.
  Using an external sentence encoder re-imports W1.
- Normalize/whiten before the loss. LLM hidden states are anisotropic with a few
  high-norm dimensions ("rogue dims"); raw MSE/cosine is dominated by them.
- Insert *between* blocks, not as a parallel branch, so the downstream trunk
  must actually consume its output.
- Add a VICReg/SIGReg term. The regularizer ablation (F10) shows prediction
  cosine can be inflated 8× by collapse — the metric must not reward it.

**Control (this is where the idea usually dies).** A 2×2, at matched parameter
count and matched FLOPs:

|  | JEPA objective | no JEPA objective |
|---|---|---|
| **trained** block | treatment | parameter-count control |
| **untrained** block | random-feature control | null |

The "no JEPA objective" arms are essential. Inserting *any* block adds
parameters and noise; without them a small loss change will be misread as a
result. This is exactly the W5 failure mode.

**Metric.** Held-out NLL on unseen text at matched compute, and one
discriminative downstream task (2-AFC). **Never** cosine-to-a-latent.

**Cheapest first experiment.** Qwen3-1.7B-Base on one 3090: freeze the trunk,
LoRA, insert one block at ~1/3 depth, train only the block + LoRA for a fixed
token budget. Four arms above, 3 seeds. Days, not a week.

**Kill condition.** CI lower bound of the NLL improvement ≤ 0 at matched
compute, or the gain vanishes in the "no JEPA objective" arm (i.e. it was just
extra parameters).

**Why it is first.** It is the cheapest idea that escapes W1, it reuses the
existing Qwen3 setup, and its null result is still informative.

---

## 2. An optional per-token "thinking vector"

**Statement.** The model may emit a latent thought at a position instead of a
token, when it chooses. Variable computation, interleaved with output.

**Survives W1** for the same reason as idea 1: the state is the LM's own.

**The hard condition.** This only works with a **budget**. The measured evidence
is unambiguous: an auxiliary channel with free token emission gets ignored or
cancelled — identity and zero arms matched or beat the steered arms in every
table (F23/F24). If emitting a token costs the same as emitting a thought, the
model will always emit the token, because per-token supervision gives it exact
credit at every position while the latent path gets only diffuse credit.

Fixes, in order of strength:

1. **Charge for tokens** (bits, FLOPs, or a halting policy with a token budget
   in the objective). Without this, "prefer tokens" is optimal and the latent
   path is decorative.
2. **Discrete gate** — straight-through or top-1 with a capacity cap
   (Mixture-of-Depths style). Continuous interpolation produces states that serve
   neither path.
3. **Stage it.** Train the latent path first with the token path frozen; unfreeze
   only after the latent path beats a text baseline.

**Control.** Same FLOPs/token budget with *no* latent path (spend the budget on
more tokens instead). Anything else measures "more compute helps."

**Metric.** Quality at **fixed compute** — a Pareto curve of quality vs tokens
emitted, not raw quality. That curve is also the honest way to present the
result.

**Kill condition.** The latent path is used below a pre-registered fraction, or
quality at fixed FLOPs is not better than the no-latent baseline.

---

## 3. Distill an LLM into JEPA by reconstruction reward

**Statement.** The LLM writes a sentence; the JEPA model tries to reconstruct
it; reward is proportional to closeness.

**Verdict: CUT permanently.** Not parked, not deferred. It re-enters three walls
at once:

- **W1** — it is again a prediction problem in a sentence-embedding space, the
  space measured to have ~4% order signal (F5).
- **W5** — "closeness" is the pooled-cosine family that produced five false
  positives in this repo (F1, F17, F25). It cannot distinguish "understood the
  idea" from "same topic."
- **W3** — the optimum is a vector the *decoder* can render, not a state the
  *predictor* should hold. You end up training a lossy autoencoder and calling
  it a predictor.

There is a defensible residue: use the LLM to **generate targets** (a corpus of
context → next-state pairs) and train on the LLM's own internal states. That is
just idea 1 with better data, and it costs no new objective.

**Why unlimited compute makes this worse, not better.** The reward is a cosine in
the JEPA space; the endpoint would be a 2-AFC in the same or a neighbouring space.
A stronger optimiser finds the codec's blind spot faster. F29 is the local proof
that this repo's reward loops already do exactly that — train judge 0.235 → 0.333
while held-out stayed flat. The "oracle" arm is a **ceiling**, not a control, and
an identity arm is a metric that cannot fail. There is no reward-sham available.

**If it is ever reopened** — it should not be — the only valid version: reward
measured on *re-encoded text* against a per-corpus human band, two-sided, with a
shuffled-target control, and gated on the CI lower bound. Even then, W3 must be
cleared first, so the GPU is better spent on the renderer.

---

## 4. JEPA drafts, LLM refines — one model, mixed emission

**Statement.** JEPA produces a draft (a plan / gist); the LLM fills in exact
tokens where precision is needed. One model, not a router between two models.

**Survives W1.** The draft is paragraph-scale gist, and paragraph centroids are
the one thing measured to persist (Δ +0.131–0.138, F8).

**Hits W3 head-on.** This is the "can a pooled latent drive text" question, and
the repo has a measurement: yes in concept space (0.355 vs human 0.336), no in
text (0.185–0.221 vs human 0.650). So the idea is not "invent a hybrid"; it is
"fix the renderer, then put the draft in front of it."

**The "balance" worry is already empirically answered.** The concern was that
the model would prefer token precision because it is easier. The measured
mechanism is different and worse: the latent branch gets diffuse credit, so it
becomes a residual the token path cancels. That is exactly why identity/zero
arms beat steered arms (F24). The fix is not a λ — it is:

1. **Freeze the draft** and make the refiner condition *only* on it (no free
   context to route around it). This is the staged approach that already worked
   once in this repo (frozen codec, trained predictor).
2. **Measure necessity** — the quality drop when the draft is ablated or
   shuffled. That number is the whole idea.
3. Only then unfreeze and allow a learned gate, with a token cost.

**Design note.** Reframe "balance" as **capacity**, not difficulty: a sentence
latent carries gist (~tens of bits of propositional content), not surface form.
No amount of training makes one vector carry exact tokens. Split by *information
type* — gist in latents, surface in tokens — not by which is easier.

**Cheapest first experiment.** Two stages, **no joint training**: freeze a draft,
use an off-the-shelf small LM as the refiner, and measure (a) render fidelity
against the 0.650 human band and (b) necessity by ablation. This measures the
*interface* before any architecture is built.

**Kill condition.** Necessity CI includes zero, or render fidelity < 0.40
MiniLM (≈ 0.62 × human) from a held-out centroid.

---

## Ranking now

| Rank | Idea | Why | Gate |
|---|---|---|---|
| 1 | **JEPA nodes in an LLM** (idea 1) | Cheapest escape from W1; reuses Qwen3; null is informative | G2 |
| 2 | **Draft/refine** (idea 4) | Its interface stage costs 16 h and answers Q1 before anything expensive runs | G3 → G4 |
| 3 | **Optional thinking vector** (idea 2) | The only fully independent endpoint, but 290 h and effectively a second paper | G2b |
| 4 | **Reconstruction-reward distillation** (idea 3) | Re-enters three walls; **cut permanently** | — |

Ranking is by *order of execution*, not by promise. Idea 4 moves up because its
cheapest stage is the repo's largest open question (Q1), not because the idea got
better.

## What all four share

None of them needs the old per-sentence predictor. The old stack tried to make
the *concept space* carry order; all four bets put order where it is natively
available (the token stream) and use the concept space only for gist, prior, and
critique. If an idea needs "predict the next sentence vector," it is already
dead (W1) — no matter how it is dressed up.
