# COARSE STEER + FINE SAMPLE — Design Spec

Direction: composed forward-generation stack (Fable review §3.4) for `/home/jpearce/Documents/Lucid`.
Deliverable: brainstorm + ranked mechanisms + buildable spec + implementation map + gates + probes.
Status: plan only. No source edited.

---

## 0. TL;DR (the opinionated verdict)

The topic chaos is not a prediction failure — it is a **sampling-policy and closed-loop-control failure**. The
sentence MDN is a well-calibrated multimodal density over "where sentences in this topic region live"
(NLL cond−uncond = +54.3 nats), but each rollout step (a) samples i.i.d. from that cloud, including from
modes that are *adjacent topics*, and (b) feeds the re-encoded result back, so the implied topic centroid
random-walks. Genre-jump within 4 steps is exactly what compounding centroid drift looks like.

The fix that follows from the measurements: **maintain an explicit slow anchor (pooled topic centroid — the
object we proved is stable at G=5–10), and constrain the fine MDN sample to live inside it** via
(1) exponential tilting of mixture weights toward the anchor, (2) best-of-n anchored rejection, and
(3) a bounded slerp of the emitted concept toward the anchor. **v1 requires zero training** — it reuses
`mdn.pt`, `writer_two_channel.pt`, and the repaired codec unchanged. A trained coarse-MDN
(paragraph-level anchor *predictor*) is a contingent v2, gated hard against the identity-paragraph baseline
so we do not repeat the mean-seeking trap.

Expected outcome, honestly: v1 has a real shot at retain@k8−shuf ≥ +0.10 because the identity arm already
achieves +0.087 with zero novelty machinery; the design question is whether we can hold that retention while
keeping novelty_rate ≥ 0.5. That is a tunable frontier (η, β), and the spec includes the sweep that finds
the operating point or proves none exists.

---

## 1. Measured reality → mechanistic model (why coarse works, why fine is chaos)

Everything below is from `artifacts/sentence_pred/RESULTS.md`, `artifacts/sentence_pred/forward_gen/RESULTS_forward.md`,
`artifacts/paragraph_reasoner.metrics.json`, `experiments/test_delta_verify.py`, and the corpus itself.

### 1.1 The encoder carries "topic + noise", not "sequence"

| Measurement | Value | Implication |
|---|---|---|
| adjacent cos | 0.361 | |
| within-doc **shuffled** cos | 0.378 | sentence i and random same-doc j are as close as i and i+1 |
| cross-doc cos | 0.269 | strong shared register (WikiText + novels) |
| anisotropy floor | 0.044 | |
| M3-identity in-doc top1 | 0.107 | identity ≈ ceiling of in-doc retrieval |
| MDN centered cos(mu, x) | 0.040 | conditional mean knows ~nothing about the *exact* next sentence |
| MDN NLL cond−uncond | +54.3 nats | context is worth a lot **as a density** (topic region), not as a point |

Working model: `z_t = c(t) + e_t`, where `c(t)` is a **slowly drifting doc/topic centroid** and `e_t` is a
per-sentence residual that is, to this encoder, **noise** (unpredictable in direction; only its magnitude
statistics matter). Evidence that `e_t` is mean-reverting noise: the delta-predictor history — consecutive
within-doc deltas are anti-correlated, and the celebrated ridge delta→delta test cos ≈ 0.39 is what you get
when `next_delta ≈ −prev_delta + drift`, i.e. the ridge was mostly learning the *reversal artifact* of
i.i.d. residuals around a centroid (`delta_t = e_{t+1} − e_t` ⇒ corr(delta_t, delta_{t+1}) ≈ −0.5 for
i.i.d. e). That "first real forward signal" was real, but it was `−e_t`, not a direction of travel.

### 1.2 Why pooling makes prediction reappear

Mean-pooling G sentences shrinks `e` by ~1/√G while `c` survives ⇒ paragraph vectors ≈ centroids.
Measured: paragraph-scale M3 retrieval 36–47% (reasoner at G=5 hit 0.36 top1 / 0.37 MRR, 2.6M params,
395s of training) vs sentence-scale ~5–11%. The predictable object in this latent space **is the centroid
and its slow drift**, full stop.

Caveat the plan must handle: the 36–47% was measured on the **old codec** (`jepa_v3_128t`; see
`artifacts/paragraph_concepts_cache/paragraphs_G5_*.npz` and `experiments/paragraph_concepts/data.py`
defaults). The repaired-codec sentence vectors already exist
(`artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1/sent_vecs.npy`, 377,477 × 512, 31k docs,
median 9 sents/doc), so re-verifying coarse stability on the repaired encoder is a **numpy-only, minutes,
zero-GPU probe** (P0 below). This is the single cheapest way to de-risk the whole direction.

### 1.3 Why free-running generation genre-jumps

Two compounding mechanisms, both visible in the numbers:

1. **Per-step mode roulette.** 100% of contexts are multimodal, mean weight entropy 1.80 ≈ 6 live
   components. If even some components encode *neighboring topic regions* (cross-doc cos 0.269 says
   regions overlap heavily), then flat sampling flips genre with some per-step probability, and 8 steps
   nearly guarantees a flip. Seed 1 in RESULTS.md is the smoking gun: baseball → hockey → "worst sports"
   → Simpsons episode — each hop is a plausible *mode* of "wiki-article about competitive/media topic".
2. **Closed-loop centroid drift.** `fed = encode(rendered_text)` re-enters the context. Writer fidelity is
   finite (oracle MiniLM ≈ 0.35–0.40), so `fed` lands off the intended concept; the context window (8)
   accumulates the offset; the MDN's implied topic region moves; the next sample is drawn from the *moved*
   cloud. Errors multiply. Note consecutive-cos is sane (mean 0.255, p90 0.456) — steps are locally smooth;
   it's the *integrated* drift that kills retention (sample retain@k8 0.118 vs shuf 0.084).

3. **The frontier we must beat, not just one endpoint.** identity steer: retain 0.171 (Δ+0.087) but
   novelty 0.344. flat sample: novelty 1.0 but Δ+0.034. Success = both at once (Δ ≥ +0.10 AND
   novelty ≥ 0.5). Any design that only slides along the existing interpolation line has to prove it can
   reach that corner — hence the sweep + frontier report is part of the build, not an afterthought.

---

## 2. Brainstorm: mechanisms considered, with verdicts

### Ranked proposals

**#1 — Two-timescale anchored sampler (Option C as chassis, Option E as the sampling rule, Option F
additions: exponential tilt + best-of-n rejection). PICK. Training-free.**
Maintain an explicit slow anchor `a` (unit 512-d): initialized from the seed concept, updated as a slow
EMA / block-pool of the rollout's own trajectory — i.e. exactly the G-pooling operation that we *proved*
produces a stable, identifiable object. Per step, sample the MDN **through the anchor**: tilt the mixture
weights toward components aligned with `a` (this is the principled version of Option E — we don't need to
track mode identity across steps, the anchor selects the consistent mode family each step), draw n
candidates, keep the best-aligned, then slerp the emitted concept a bounded distance from `a` toward the
sample. Novelty lives in (i) which component, (ii) the vMF shell around the component mean (kappa), and
(iii) the slerp budget η — all *inside* the coarse arc.
Why it's most likely to work: it attacks both measured failure mechanisms directly (mode roulette →
tilt+rejection; loop drift → the emitted concept is leashed to a slowly-moving pooled anchor, so
`fed ≈ near a`, so the loop self-stabilizes); it needs no training, so the calibrated MDN and the live
writer are untouched; and its degenerate corners reproduce the existing arms (β=0, η=1 ⇒ flat sample;
η=0 ⇒ ~identity-of-anchor), so the A/B is airtight.

**#2 — Trained coarse anchor predictor (Option A), as contingent v2a.**
A second MDN with the *same* `SentenceDensityPredictor` class, trained on **pooled paragraph sequences**
(self-supervised, pooling is the only label; built from existing `sent_vecs.npy`, no re-encoding). Used
only to *update the anchor* once per G steps — "where is this going" at the scale where that question has
an answer (36–47%). Not used per-sentence, and never as retrieval. Gated before integration against the
identity-paragraph baseline (the boss that killed the sentence predictor). Reason it is #2 and not #1:
median doc = 9 sentences, so an 8-step rollout spans ~2 paragraphs — a *quasi-static anchor already
captures most of the value*; the trained predictor only matters for longer rollouts (16–24 steps) or if
the EMA anchor proves too inert. Build it only if v1 passes coherence but the arc is boring/static.

**#3 — Concept-AR / anchored-to-previous (Option D), demoted to an ablation arm.**
`concept = slerp(z_t, sample, η)` is v1 with anchor window = 1 and λ = 1. It is strictly dominated by
pooling on the project's own evidence: `z_t` contains the full residual `e_t` (the noise), while the EMA
anchor averages it out — the same reason paragraph retrieval works and sentence retrieval doesn't. Keep it
as one row of the sweep (window=1) so the report *shows* the pooling win rather than asserting it.

### Declared dead (and why)

- **Option B (slow direction-of-travel / delta predictor).** Dead as a primary mechanism. The measured
  delta signal was the reversal artifact (§1.1); genuine centroid drift over 8 sentences is tiny relative
  to `e_t`, unestimable within a rollout, and a *wrong* drift direction actively steers off-topic — worse
  than a static anchor. Directional motion belongs at paragraph scale (v2a samples the next-paragraph
  distribution), not as a per-sentence velocity term. Do not resurrect the trend head.
- **Option E standalone (mode persistence without an anchor).** The component identities are not stable
  objects across steps — the head is re-evaluated on a *shifting context*, and "component 3" today is not
  "component 3" tomorrow. Persisting an index is meaningless; persisting a *direction* (the anchor) and
  re-selecting the nearest mode each step is the coherent version — which is #1.
- **Exact-next retrieval anything.** Proven dead (G3/G6). Nothing here retrieves.
- **Retraining the sentence MDN with coherence auxiliaries.** The density is the one component that
  passed its gates (+45.3 nats vs vMF, calibrated, multimodal). The failure is downstream of it. Leave it frozen.

### Honest ceiling statement

Under this encoder, "coherent multi-sentence novel documents" means: **smooth topic arc + novel in-arc
sentences**. Plot-level structure (setups, callbacks, entity persistence beyond what the token prefix
carries) is not in these vectors — adjacent ≈ within-doc-shuffled guarantees that. The design aims for
"no genre-jump, on-topic novelty", and the gates are written for exactly that, not for more.

---

## 3. Architecture spec (v1: Anchored Two-Timescale Sampler)

Forward-generation stack. No retraining. All modules frozen: `mdn.pt`, `writer_two_channel.pt`,
repaired codec `jepa_repair_1m/jepa/joint.pt`.

### 3.1 State

| Object | Shape / type | Init | Update |
|---|---|---|---|
| `a` (coarse anchor) | (1, 512) unit fp32 | `encode(seed_text)` | EMA or block-pool (below) |
| `history_vecs` | list of (512,) unit | `[z_seed]` | append `fed` per step (unchanged from baseline) |
| `history_texts` | list of str | `[seed_text]` | append rendered sentence (unchanged) |

Anchor update, two variants (flag `anchor_mode`):
- `ema` (default): every step, `a ← normalize((1−λ)·a + λ·u)` with λ = 0.10; `u` is the update source.
- `paragraph`: every `anchor_every` = 4 steps, `a ← normalize(mean(last 4 u's))`; frozen between.

Update source `u` (flag `anchor_source`): `fed` = re-encoded rendered sentence (default — anchors to what
was actually written, honest and self-correcting) or `concept` = emitted steer (anchors to intent, immune
to re-encode noise). Ablate both; if `fed` drifts while `concept` holds, the leak is renderability (see risks).

### 3.2 Per-step sampling (the core)

Inputs per step: context `ctx` = last 8 `history_vecs` stacked → (1, T≤8, 512), `pad` (1, T) all-False,
anchor `a` (1, 512). Uses only existing public modules of `SentenceDensityPredictor`
(`predictor.trunk`, `predictor.head`, `ContextTransformer.last_real_token`, `heads.sample_vmf`).

```python
# (1) mixture params at last position  — one new 12-line method on SentenceDensityPredictor
h      = predictor.trunk(ctx, pad)                                   # (1, T, 512)
h_last = ContextTransformer.last_real_token(h, pad).unsqueeze(1)     # (1, 1, 512)
pi_logits, mu, kappa = predictor.head(h_last)                        # (1,1,K=8), (1,1,8,512), (1,1,8)
pi_logits, mu, kappa = pi_logits.squeeze(1), mu.squeeze(1), kappa.squeeze(1) / steer_temp

# (2) exponential tilt of mixture weights toward the anchor (Option E, done right)
align  = (mu @ a.unsqueeze(-1)).squeeze(-1)                          # (1, 8) = cos(mu_k, a)
tilted = torch.log_softmax(pi_logits, dim=-1) + beta * align         # beta default 8.0; beta=0 → flat MDN
pi_t   = torch.softmax(tilted, dim=-1)

# (3) best-of-n anchored rejection (novelty preserved: candidates are raw MDN samples)
comps  = torch.multinomial(pi_t, n_cand, replacement=True)           # (1, n_cand), n_cand default 8
s      = sample_vmf(mu[0, comps[0]], kappa[0, comps[0]])             # (n_cand, 512) unit — reuse heads.sample_vmf
score  = s @ a[0]                                                    # (n_cand,)
# optional novelty guard: drop candidates with n(s_i) > ref_median (needs bank; harness supplies it)
s_star = s[score.argmax()][None, :]                                  # (1, 512)

# (4) bounded slerp toward the sample (η = novelty budget along the arc)
concept = slerp(a, s_star, eta)                                      # eta default 0.35; unit (1, 512)
```

`slerp(a, b, t)`: `Ω = arccos(clamp(a·b, −1+1e−7, 1−1e−7))`; if `sin(Ω) < 1e−4` return
`normalize((1−t)a + t·b)`; else `(sin((1−t)Ω)·a + sin(tΩ)·b) / sin(Ω)`. Both inputs unit; output unit.

Then exactly the existing pipeline (unchanged): `text = _render_sentence(writer, tokenizer, concept,
history_texts[-1], temperature=0.8, top_k=50, max_tokens=96)`; `fed = _encode_one(codec, text)`;
append; update anchor. The writer consumes the composed concept through the **existing all-block FiLM
channel** — no second channel, no writer change in v1. Rationale: FiLM was trained on unit sentence
vectors with 10% zero-drop; a slerp of two unit in-distribution-ish vectors stays on the sphere inside the
convex cone of real concepts. Whether a *pooled* anchor is FiLM-legible is probe P1's job, not an assumption.

### 3.3 Config dataclass (defaults frozen after sweep)

```python
@dataclass
class CoarseSteerConfig:
    anchor_mode: Literal["ema", "paragraph"] = "ema"
    anchor_lambda: float = 0.10       # EMA rate
    anchor_every: int = 4             # paragraph mode block size
    anchor_source: Literal["fed", "concept"] = "fed"
    beta: float = 8.0                 # mixture tilt strength (0 = flat MDN)
    n_cand: int = 8                   # best-of-n rejection
    eta: float = 0.35                 # slerp budget toward sample (1.0 = raw sample)
    use_novelty_guard: bool = False   # drop candidates with n(x) > ref median
    steer_temp: float = 1.0
    context_len: int = 8
```

Sweep grid (run on 8 *separate* seeds, `--seed 777`, before the frozen run — anti-mirage discipline):
`eta ∈ {0.25, 0.35, 0.5, 1.0} × beta ∈ {0, 4, 8, 16} × anchor_lambda ∈ {0.05, 0.10, 0.25}`, plus the
Option-D row (window=1 ⇒ `anchor_lambda=1.0`). Pick by: maximize `retain@k8 − shuf` subject to
`novelty_rate ≥ 0.5`; write every row to `frontier.jsonl`-style output. `eta=1.0, beta>0` is the
"pure rejection, untouched samples" corner — if it alone clears both gates, prefer it (cleanest novelty story).

### 3.4 Novelty accounting (be honest about what is measured)

`n_x_steer` is computed on the **emitted composed concept** (the thing that steers FiLM), same semantics
as the baseline arms (which logged the steer vector). Log the raw sample's `n(x)` separately
(`n_x_raw_sample`). Expected: identity arm measured mean n(x)=0.771 (novelty 0.344), flat sample 0.383
(novelty 1.0); slerp at η=0.35 should land between — the sweep exists because the ≥0.5 gate is genuinely
tight. Reference distribution unchanged: held-vs-train bank, median 0.7451 / p99 0.9673, same code path
(`evalsuite.reference_distribution`, bank cap 50k, held cap 5k, rng 1234).

### 3.5 Instrumentation added per step (extends the existing row dict)

`anchor_vs_seed_cos` = cos(a_k, a_0); `concept_vs_anchor_cos`; `sample_vs_anchor_cos` (pre-slerp);
`fed_vs_concept_cos` (renderability leak detector); `chosen_component`; `tilt_moved_argmax` (bool);
`n_x_raw_sample`. Keep all existing fields (`consecutive_cos`, `minilm_retention`, `minilm_consecutive`,
`n_x_steer`, `n_x_fed`, `steer_vs_zt_cos`).

---

## 4. v2 extensions (contingent, each behind a gate)

### 4.2a Trained coarse anchor predictor (Option A materialized)

Build **only if** v1 passes G-CS1/2 but the anchor is too static (qualitative: arcs never develop;
instrumented: `anchor_vs_seed_cos` ≥ 0.95 at k=16 on long rollouts) — or if 16–24-step rollouts are the next goal.

- **Data (self-supervised, no labels, no encoding):** pool existing repaired-codec `sent_vecs` into
  non-overlapping G=4 blocks per doc (mean → L2-normalize; recipe = `_group_doc_sentences` in
  `experiments/paragraph_concepts/data.py`, re-implemented over `SentenceCorpus` arrays). G=4 because
  median doc = 9 sentences; docs ≥ 16 sentences (needed for context 3 + target) number 7,835 —
  tens of thousands of windows. Package as a `SentenceCorpus` instance (`sent_vecs`=paragraph vecs,
  `doc_offsets`=paragraph offsets, `sentences`=joined block text, same train/held doc split) ⇒
  **`train_head.train_density_head` runs verbatim** with `context_len=3`.
- **Model/loss:** `SentenceDensityPredictor(head="mdn", K=4)`, same trunk config, pure mixture NLL —
  the measured antidote to mean-seeking (+45.3 nats says the mixture escapes the conditional-mean trap);
  pooling has already removed ~1/√G of the residual noise that made the sentence conditional flat.
- **Gate before any integration** (run on pooled held set, identity-paragraph is the boss baseline):
  centered cos(mu, x) ≥ 0.15; M3-indoc top1 ≥ identity-paragraph top1 + 5pts; NLL cond−uncond ≥ 5 nats.
  Fail ⇒ keep the EMA anchor; v1 remains valid and shipped.
- **Integration:** every G steps, `ctx_pooled` = G-block means of rollout `history_vecs`;
  `a ← slerp(a, coarse_sample, η_c=0.3)` with one anchored sample from the coarse MDN (same tilt machinery,
  tilted toward current `a`). This is "predict the next paragraph centroid, move the anchor partway there" —
  directed arc, still leashed.

### 4.2b Dual-channel FiLM writer (only if P1 kills or renderability is the proven leak)

If pooled/slerped concepts are FiLM-illegible (P1) or `fed_vs_concept_cos` stays low while
`concept_vs_anchor_cos` is high (the writer can't execute the steer): finetune the writer with
`film_dim = 1024` = concat(coarse anchor, fine concept), zero-init FiLM (identity start), phase-2.5 recipe
(`train_phase2.py`: freeze all but FiLM + top-3 blocks + concept_proj), targets unchanged (same-sentence
CE), coarse input = pooled centroid of the G-block containing the target sentence, drop arms extended:
10% drop-coarse, 10% drop-fine, 10% drop-prefix, never all. This is the only training in the entire v1+v2b
path and reuses an existing, proven trainer. Do not build it preemptively.

---

## 5. Implementation map (this codebase, exact)

### Reused, unchanged
- `artifacts/sentence_pred/mdn.pt` via `forward_gen.load_mdn_predictor`
- `artifacts/writer_p25/writer_two_channel.pt` via `conceptpoc.persistence.load_writer`
- `artifacts/jepa_repair_1m/jepa/joint.pt` via `load_joint` + `JepaCodec` (max_tokens=128)
- corpus `artifacts/sentence_pred/corpus_repair/corpus_754d65cd97f3bfb1` via `load_sentence_corpus`
- `experiments/sentence_pred/forward_gen.py` module-level helpers imported as-is: `_render_sentence`,
  `_encode_one`, `_pick_seeds`, `_load_minilm`, `_agg_retention_curve`, `_shuffled_retention_curve`,
  `_novelty_summary`, `_pairwise_text_divergence`, `ForwardGenResult`, and `generate_document_twochannel`
  (for the baseline arms)
- `experiments/sentence_pred/evalsuite.py`: `centering`, `novelty_scores`, `reference_distribution`,
  `novelty_rate`, `copy_rate`, `load_split`
- `artifacts/sentence_pred/forward_gen/seeds.json` — **load these exact 20 seeds** for the head-to-head run

### Modified (one file, one method)
- `experiments/sentence_pred/heads.py`: add to `SentenceDensityPredictor`:

```python
@torch.no_grad()
def mixture_params_last(self, context, pad_mask=None):
    """MDN params at last real position: (pi_logits (B,K), mu (B,K,D), kappa (B,K))."""
    assert self.head_type == "mdn"
    h = self.trunk(context, pad_mask)
    h_last = ContextTransformer.last_real_token(h, pad_mask).unsqueeze(1)
    pi_logits, mu, kappa = self.head(h_last)
    return pi_logits.squeeze(1), mu.squeeze(1), kappa.squeeze(1)
```

  No parameters added ⇒ `mdn.pt` loads unchanged (`strict=True` still passes).

### Created
1. `experiments/sentence_pred/coarse_probe.py` — probes P0–P3 (§7), one argparse CLI, writes
   `artifacts/sentence_pred/coarse_steer/PROBES.md` + `probes.json` with explicit PASS/KILL lines.
2. `experiments/sentence_pred/coarse_steer.py` — the experiment:
   - `CoarseSteerConfig` (§3.3), `slerp`, `anchored_sample(predictor, ctx, pad, anchor, cfg, bank=None)`
   - `generate_document_coarse(seed_text, predictor_mdn, writer, tokenizer, codec, steps=8, cfg=..., train_bank=None, ref_dist=None, minilm=None, writer_temperature=0.8, writer_top_k=50, max_tokens=96) -> ForwardGenResult`
     (mirrors `generate_document_twochannel`'s signature and return type exactly)
   - `run_experiment(args)`: arms `{coarse, sample, identity, zero}` — baseline arms produced by calling
     the existing `generate_document_twochannel` so the comparison shares every code path; same MiniLM,
     same shuffled-seed control (`rng_seed = seed+7`), same report structure
   - `--sweep` mode: grid of §3.3 on 8 seeds (`--seed 777`), rows appended to
     `artifacts/sentence_pred/coarse_steer/frontier_coarse.jsonl`; `--run` mode: frozen config, the 20
     `seeds.json` seeds, plus a `--steps 16` secondary table
   - writes `artifacts/sentence_pred/coarse_steer/RESULTS_coarse.md` with the gates table (§6) quoting the
     baseline numbers inline, `rollouts.json`, and the blind-read sheet (40 shuffled rollouts, arm labels
     held out to `blind_key.json`)
3. (v2a only) `experiments/sentence_pred/coarse_data.py` — `build_pooled_corpus(corpus, G=4) -> SentenceCorpus`
   + cache under `artifacts/sentence_pred/coarse_steer/pooled_G4/`; training via existing
   `train_head.train_density_head(pooled, head="mdn", K=4, context_len=3, out=".../mdn_coarse_G4.pt")`.

### Build order (each step gated by the previous)
1. P0 probe (CPU, minutes) → KILL/GO for the whole direction on the repaired codec.
2. P3 + P2 probes (GPU, ~5 min) → fixes β default; validates composition math in concept space.
3. P1 probe (GPU, ~5 min) → pooled/slerped concepts are FiLM-legible, or route to v2b later.
4. `heads.py` method + `coarse_steer.py` core + `--sweep` (≈1 GPU-hr).
5. Frozen `--run` head-to-head (≈1 GPU-hr, 20 seeds × 8 steps × 4 arms) → `RESULTS_coarse.md`.
6. Blind qualitative read (human, ~20 min).
7. Contingent: v2a, then re-run step 5 with `anchor_update=coarse_mdn` as a fifth arm.

Total new code ≈ 400–500 lines across two experiment files; zero retraining on the critical path.

---

## 6. Success gates (frozen before the run; same metrics as baseline)

Baselines being beaten (from `RESULTS_forward.md`): sample Δretain=+0.034 / novelty 1.00;
identity Δ=+0.087 / novelty 0.344; zero Δ=+0.029 / novelty 1.00; mean consec 0.255, p90 0.456;
divergences sample↔identity 0.173, sample↔zero 0.145.

| Gate | Metric (coarse arm, 20 seeds × 8 steps) | Threshold | Baseline |
|---|---|---|---|
| G-CS1 coherence | retain@k8 − shuffled-seed control | ≥ +0.10 | +0.034 (sample) |
| G-CS2 novelty | novelty_rate of emitted steer; copy_rate | ≥ 0.50; ≤ 0.02 | 1.00 / 0.00 (sample), 0.34 (identity) |
| G-CS3 sane band | mean consecutive MiniLM; p90; jump_rate = frac(consec < 0.05); distinct-2 | ∈ [0.15, 0.60]; ≤ 0.90; ≤ 0.15; ≥ 0.30 | 0.255 / 0.456 / n.a. / 0.904 |
| G-CS4 steer alive | mean MiniLM cos coarse↔zero AND coarse↔identity | both ≤ 0.85 | 0.145 / 0.173 (sample arm) |
| G-CS5 genre-hold | blind read: seeds with ≥1 hard domain flip; anchor stability | ≤ 5/20 AND < ½ of sample arm's count; median cos(fed_8, a_0) ≥ sample arm's + 0.15 (computed in-run for both arms) | sample arm ≈ most seeds flip |

**PASS = G-CS1 ∧ G-CS2 ∧ G-CS3 ∧ G-CS4**, with G-CS5 as the qualitative confirmation. Report the full
(novelty_rate, Δretain) frontier from the sweep regardless of pass/fail — if no (η, β, λ) point satisfies
G-CS1 ∧ G-CS2 simultaneously, that is the finding: coarse anchoring on this encoder trades novelty for
coherence 1:1 and the composed-stack hypothesis fails. Say it plainly in RESULTS_coarse.md.

---

## 7. Risks, fail-modes, and the cheap probes that catch them first

Hard-won lesson honored: every mechanism gets a minutes-scale probe with a numeric KILL before any build.

**P0 — Coarse stability replicates on the repaired codec (CPU, numpy only, ~2 min).**
From `sent_vecs.npy` + `doc_offsets.npy` (held docs): (a) consecutive-paragraph cos (G=5 non-overlapping
pools) minus cross-doc paragraph cos; (b) split-half doc-centroid cos minus cross-doc centroid cos;
(c) consecutive sentence-delta cos (expect < −0.2, confirming mean reversion).
PASS: (a) ≥ +0.10 and (b) ≥ +0.15. KILL: (a) < +0.05 and (b) < +0.10 ⇒ the pooled centroid is not stable
on this codec; the direction dies here for the cost of a numpy script. (Context: the sentence-level analog
of (a) measured **−0.017**; pooling must flip its sign decisively or there is no coarse signal to anchor to.)
Risk covered: 36–47% was old-codec evidence; repaired codec failed cross_doc ≤ 0.10 (0.269), so this is genuinely open.

**P3 — Mode geometry (GPU, ~2 min).** For 500 held contexts (context_len 8): spread of cos(mu_k, doc
centroid) across the 8 components — report max−min gap and share of contexts where the anchor-tilted
argmax ≠ flat argmax. If gap < 0.05 the tilt is inert ⇒ set β=0 and rely on rejection+slerp (informs
config; not a kill). If gap is large, the mode-roulette mechanism of §1.3 is confirmed and β matters.

**P2 — Concept-space anchored rollout, no writer, no re-encode (GPU, ~3 min).**
Closed loop with `fed := concept` (skip render+encode): 20 seeds × 8 steps, flat vs anchored sampler;
compare median cos(concept_k, a_0) at k=8. PASS: anchored ≥ flat + 0.20 and ≥ 0.50 absolute.
KILL: anchored ≤ flat + 0.05 ⇒ the composition math doesn't hold even without the writer in the loop.
Isolates sampler correctness from renderability before any expensive text generation.

**P1 — FiLM legibility of pooled/slerped concepts (GPU, ~5 min).**
200 held paragraph items; render with concept ∈ {same-sentence z_i (reference), pooled G=5 centroid,
slerp(centroid, MDN sample, 0.35)}; prefix = previous sentence; measure MiniLM cos(render, source
paragraph text). PASS: pooled ≥ 0.25 (reference: concept-only renderability measured 0.395).
KILL: pooled ≤ 0.10 ⇒ FiLM ignores pooled vectors ⇒ v1 can still run with `anchor_source=concept` and
η ≥ 0.5 (emitted concepts stay near real samples), and v2b (dual-FiLM finetune) becomes the mainline fix.

**Fail-modes at run time and their handles:**
- *Anchor freezes the topic* (retain high, distinct-2 < 0.3 or consec > 0.60, repetitive text): λ too
  small / β too large / η too small — the sweep grid brackets all three; G-CS3 catches it.
- *Novelty collapses below 0.5*: raise η toward the `eta=1.0` pure-rejection corner (samples untouched,
  novelty ≈ 1.0 by construction; coherence then rests on tilt+rejection alone). If even that corner fails
  G-CS1, report the frontier and stop.
- *Renderability leak* (concept_vs_anchor high, fed_vs_concept low, retention still poor): the writer
  can't execute the steer — route to v2b; do not tune the sampler harder against a writer bottleneck.
- *Seed-anchor is noisy* (single-sentence init: a_0 = c + e_seed): expected and self-healing (EMA pools it
  out within ~1/λ steps); if early-step flips dominate, warm-start λ=0.3 for k ≤ 2 (one-line change, note in code).
- *v2a mean-seeks at paragraph level*: caught by its own gate (centered cos ≥ 0.15, M3 ≥ identity-paragraph
  + 5pts) before integration; the fallback (EMA anchor) is already shipped by then.

---

## 8. What I'd bet

- P0 passes: pooling is arithmetic, and the doc-topic component (within-shuf 0.378 vs cross 0.269) is
  exactly what survives averaging. ~85%.
- P2 passes: the sampler math is a leash by construction. ~90%.
- The real fight is G-CS1 ∧ G-CS2 jointly. The identity arm's +0.087 at novelty 0.34 says the retention
  budget exists; tilt+rejection add retention *without* touching sample novelty, which is headroom the
  old interpolation line didn't have. I'd put ~55–65% on some sweep point clearing both gates at 8 steps,
  and lower (~40%) at 16 steps without v2a. If it fails, the frontier plot plus P-probe results pin
  *which* link broke (sampler / writer / codec), which is precisely the diagnostic the project has been
  paying for with every prior negative result.
