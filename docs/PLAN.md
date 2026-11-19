# Plan and Key Decisions

**Status: reset, 2026-11-17.** The original 2026-06 plan is archived at
[history/PLAN-2026-06.md](history/PLAN-2026-06.md). It described a framing that
has since been measured and falsified. Read [WALLS.md](WALLS.md) before
proposing work, and [FINDINGS.md](FINDINGS.md) before assuming something is
untested.

## What was the project

Predict the next *sentence-concept vector*, decode it back to text, and improve
the predictor with an LLM-judge + human-rating reward loop. The concept-level
framing was chosen because sequences are short (cheap on one GPU) and a frozen
codec gives a decoder for free.

## What is now established

Three premises were false, each measured:

1. **There was no order to predict.** The sentence-embedding space is a bag of
   sentences: delta autocorrelation −0.4865 against an exchangeability bound of
   −0.5, i.e. ≈ 4% sequential variance. The objective family removes discourse
   state by construction, and cosine cannot express precedence. (W1)
2. **The corpus never contained the phenomenon.** 113k shuffled single
   paragraphs; 4.1% of documents yielded any ctx-32 training window, 1,677 in
   total, 81% Wikipedia by sentence. (W4)
3. **The gates could not fail.** The "+54 nats conditional gain" was 50.4 nats
   mixture capacity and 0.4 nats conditioning; the fluency gate passed with
   infinite NLL; the unsteered baseline scored exactly what the steered run
   scored. (W5, W2)

The render step is where the real signal died: concept-space coherence is at
human level (0.355 vs 0.336) while text rendering reaches 0.185–0.221 against a
human ceiling of 0.65. (W3)

## What survived

- The frozen JEPA encoder as a **paragraph topic codec** (Δ +0.131–0.138
  persistence).
- `anchored_sample` / slerp as a **paragraph-level leash** (0.876 cos@k8).
- The vMF/MDN machinery as an **explicit prior** over paragraph centroids — not
  as a transition model.
- The **anti-mirage eval protocol**: doc-disjoint splits, shuffled controls,
  centering/anisotropy floors, distinct-candidate counts, 20-seed confirms,
  blind sheets.
- The **negative measurements** themselves, which are the strongest asset in the
  repo and the basis of the analysis write-up.

## Direction

Use each representation only at the timescale where it measurably carries
signal:

| Representation | Timescale | Role |
|---|---|---|
| Token autoregression | every token | order, syntax, entity carry-over |
| Pooled concept vector | once per paragraph | topic identity and destination |
| Concept density | once per paragraph | prior to sample from, critic to rerank with |

The generator seat belongs to a pretrained 1–4B base model with LoRA, because a
161.7M scratch writer cannot pass the render wall (W3). The concept machinery is
demoted to prior + arc controller + critic. This is the architecture the
2026-08 reassessment recommended; it has not yet been tested at the gate.

## Next step

The programme is specified, pre-registered and not yet run. It lives in
[PROTOCOL.md](PROTOCOL.md); the run manifests are in
[EXPERIMENTS.md](EXPERIMENTS.md); the data stage is in [CORPUS.md](CORPUS.md).

In one line: **G-1 corpus → G0 assay → G3 E4-S1 (is a latent necessary at all?) →
G1/G2 E1 (is a JEPA block load-bearing?) → G4 E4-S2, G2b E2, G5 OLMo.**

Three claims are confirmatory (C1 E1, C2 E4-S2, C3 E2); E0 and E4-S1 are assay
stages; E3 is cut permanently. The four things that decide whether the programme
can return a real answer — independent endpoint, a null that varies,
a pre-registered MPID, a sealed confirmatory split — are fixed before run 1 and
are not negotiable at any compute budget.

## Decision rules (pre-registered, non-negotiable)

- Every experiment names the wall it survives and the row in `FINDINGS.md` it
  would repeat.
- Every gate names its control arm. Allowed types: absolute floor,
  control-differential, discriminative (2-AFC / blind pairwise), cost.
- **NaN or Inf in any gate metric = FAIL, run void.** No fallback path may pass.
- Gate on the **CI lower bound against a pre-registered MPID**, not on the median
  and not merely on "CI excludes zero". Seed count is set by the pilot power
  calculation in [PROTOCOL.md](PROTOCOL.md) §5.4, not by a fixed threshold.
- Two FAILs on the same wall ⇒ the wall is real. Write it in `WALLS.md`; the
  third attempt needs a *new mechanism*, not a new hyperparameter.
- Verdicts are **PASS / FAIL / INCONCLUSIVE / VOID**. `FAIL_BUT_NOT_KILL` is not a
  verdict; pair each result with `next_action: stop | fix_metric | one_more_attempt`.
- Budget stops are absolute and set before the run.
- An amendment after data has been seen on the affected contrast is a protocol
  deviation and is reported as one.

## Doc map

| File | Purpose |
|---|---|
| [WALLS.md](WALLS.md) | The five measured constraints. Read first. |
| [FINDINGS.md](FINDINGS.md) | Every claim + number + verdict. Append, don't rewrite. |
| [IDEAS.md](IDEAS.md) | Live research bets, each mapped to the walls it survives. |
| [PROTOCOL.md](PROTOCOL.md) | The confirmatory programme: arms, *n*, MPIDs, gates, decision rules. |
| [CORPUS.md](CORPUS.md) | G-1 data stage: sources, licences, splits, contamination screen. |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Run manifests, spawn keys, run order, pipeline qualification. |
| [EXPLAINER.md](EXPLAINER.md) | The vocabulary and diagrams, for readers new to the terms. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the surviving code fits together (updated 2026-11). |
| [LIMITATIONS.md](LIMITATIONS.md) | Honest account of scope, now with measured numbers. |
| [history/PLAN-2026-06.md](history/PLAN-2026-06.md) | The superseded plan, kept for context. |
