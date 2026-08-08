# Paragraph-Scale Generation for Lucid: Architecture + Implementation Spec

Repo: `/home/jpearce/Documents/Lucid`. Design-only deliverable; no source was modified. Three new read-only measurements were run against cached artifacts during this design pass (marked NEW below); they materially change the framing and are reported honestly.

---

## 0. Executive decision (TL;DR)

Build a **paragraph-commitment generator**: a paragraph-scale MDN reasoner over G=5 pooled sentence centroids proposes ONE concept per paragraph through the already-proven `anchored_sample` leash; a **phase-3 fine-tuned writer** (the crux, and the only genuinely new training) renders G sentences under that single frozen concept with per-sentence prefix chaining; re-encoding and re-planning happen **only at paragraph boundaries**. This is Option B's decoding loop with Option A's planning object and Option C's decomposition — and it is gated by two cheap kill-probes before the full build, because one of the load-bearing numbers in the prompt (M3 36-47%) turns out to be an artifact.

The single most important fact discovered in this audit: **the current writer cannot read pooled paragraph centroids (P1 = 0.175, gate was 0.25), and no one has ever trained it to.** Every composed-stack failure to date is consistent with that one deficiency. Fixing a train/inference contract mismatch is the highest-yield, cheapest class of fix available. That, not a fancier planner, is the bet.

Honest probability this produces gate-passing coherent novel multi-paragraph documents: **~25%** (see §7). It is still the right last bet, with a hard stop-condition, ~3-4 days of work, and two kill-points inside the first day and a half.

---

## 1. Evidence audit — what is actually true (with corrections)

### 1.1 The five sentence-scale negatives (confirmed, trusted)

All numbers from `artifacts/sentence_pred/RESULTS.md`, `RESULTS_coarse.md`, `probes_P0.json`, `probes_P12.json`, `writer_check.json` on the repaired codec (`artifacts/jepa_repair_1m/jepa/joint.pt`) and corpus `corpus_754d65cd97f3bfb1`:

- **Exact next-sentence prediction**: M3-global top1 0.2% vs identity 0.6% (predictor loses to copying); centered cos(mu, x) 0.040 vs identity 0.102. The killer diagnostic: adjacent cos 0.361 but *within-doc-shuffled* cos 0.378 — `adj − within = −0.017`. **There is no sentence-order signal in this embedding space at all**; only doc-identity signal.
- **Delta predictor**: consecutive sentence-delta cos = **−0.483** (re-confirmed in P0(c)); the "+0.39 delta signal" was learning `−e_t` (mean reversion).
- **MDN density head** (`mdn.pt`): val NLL −1003.5 vs vMF −958.2 (Δ45.3 nats), weight entropy 1.80 with 100% of contexts multimodal, novelty 1.0 / copy 0.0 — a perfectly calibrated model of a distribution whose predictable component is "somewhere in this document's topic blob" (NLL beats unconditional by 54.3 nats but per-position kappa is uninformative, Spearman 0.0004).
- **Two-channel forward gen**: FiLM steer is live (sample/identity/zero text divergence 0.10-0.17 MiniLM cos), but retain@k8−shuf = +0.034 (sample), +0.087 (identity). Genre-jumps.
- **Coarse-anchor sampler**: concept-space P2 **PASS** (anchored median cos@k8 0.876 vs flat 0.246, Δ+0.63); P0 **GO** (below); frozen 20-seed head-to-head **FAIL** G-CS1 (Δretain +0.013 vs need +0.10). The 8-seed sweep showed Δ+0.127 which collapsed to +0.013 at 20 seeds — the anti-mirage protocol worked. fed8-vs-a0: coarse 0.355 vs sample 0.235 (Δ+0.121, just under the +0.15 genre-hold gate).

### 1.2 The paragraph-scale positives — one is real, one is a mirage

**Real (P0, cross-doc controlled, repaired codec, `probes_P0.json`):**

- G=5: consecutive-paragraph cos 0.7054 vs cross-doc-paragraph 0.5680 → **Δ +0.137**
- G=10: 0.8192 vs 0.6842 → **Δ +0.135**
- Doc-half centroids: 0.7932 vs 0.6554 → **Δ +0.138**

**Mirage (M3 "36-47%") — the repo already flags it and this audit quantifies it.** `docs/architecture.ascii` line 136: `[~] Paragraph ranker (retrieval-only) "36%" — a MIRAGE, superseded`. NEW measurement against `artifacts/paragraph_concepts_cache/paragraphs_G5_c9918d626200019d.npz` (the exact corpus the 36% was trained/evaled on): docs average **3.57 paragraphs**; the "100-candidate" in-doc eval draws distractors *with replacement* from a mean pool of ~1.7 distinct future paragraphs, giving **mean 2.67 distinct candidates**. On that pool the trivial identity baseline scores **64.4% top-1** and uniform-among-distinct chance is ~60%. The trained ranker's 36% is *below chance on distinct candidates*. The 36-47% number carries no evidence that the next paragraph is predictable. Do not design around it.

**The honest replacement number (NEW, measured this session on the repaired codec, held docs, 1000 cross-doc candidates, identity query = current paragraph centroid):**

- G=5: **top1 3.3%, top10 13.0%, MRR 0.071** (2,000 pairs; chance 0.1%)
- G=10: **top1 4.4%, top10 18.9%, MRR 0.092** (772 pairs)
- Sentence-scale equivalent (from RESULTS.md): identity top1 0.6%, top10 4.1%

So: paragraph objects are ~5x more forward-predictable-by-copy than sentence objects and 33-44x above chance — real, but the mechanism demonstrated is **topic persistence**, not directional order information. P0 and this retrieval measure the same thing: "the next paragraph is near the current one." No experiment to date shows that paragraph *order* (p+1 vs p−1) is decodable. The design below therefore does NOT bet on predicting a direction; it bets on **holding a topic while sampling novel content inside it** — which is exactly what `anchored_sample` already does correctly in concept space (P2).

### 1.3 The renderability deficit — the actual root cause of failure #5

`probes_P12.json` P1: rendering one sentence from a **pooled G=5 centroid** through the current two-channel writer scores MiniLM cos to the source paragraph of **0.175** (pass gate 0.25, kill 0.10). Critically, even the *same-sentence* concept scores only 0.186 against the paragraph, while the same-sentence concept against its *own* sentence scores ~0.395 (`writer_check.json` concept_only). The writer was trained (phase-2.5) exclusively on `(prev-sentence tokens, same-sentence concept → same-sentence tokens)`. A pooled centroid is out-of-distribution for its FiLM channel. **The leash worked in concept space and failed in text space because the writer has never been taught what a pooled topic vector means.** This is trainable, self-supervised, and cheap — and it is the crux probe (P3-B, §6).

### 1.4 Corpus structure facts that constrain the design (NEW, measured)

`corpus_754d65cd97f3bfb1`: 31,000 docs / 377,477 sentences / 512-dim fp16, split 27,900/3,100 document-disjoint. Doc length: median 9, mean 12.2, p90 24 sentences. Docs are blank-line blocks of `data/corpus_1m.txt` (WikiText articles + Gutenberg chunks). Two consequences:

- **Option D is half-right and half-dead.** The blocks ARE paragraph-sized (median 9), but adjacent doc_ids carry **zero** signal: cos(last G-block of doc i, first G-block of doc i+1) = 0.5485 vs random-pair 0.5481 (**Δ +0.0004**, n=22.5k; within-doc Δ was +0.137). The corpus was shuffled at build time. There is no chaining across doc boundaries; the unit must be within-doc G-blocks.
- **Data volume at paragraph granularity** (NEW, exact counts): at G=5, train has 13,470 docs with ≥2 blocks → 32,833 next-block transitions (phase-0), **134,036 with G-phase augmentation** (start-offset s∈0..4; blocks within one phase are disjoint, so no leakage); held: 3,535 / 14,471. At G=8: 12,475 / 71,215 train. Context depth is shallow: p90 = 6 prior blocks at G=5. This sizes the reasoner (small) and fixes context_len (4) and G (=5 primary; G=8 ablation only).

---

## 2. Design decision (Section 1 of the brief)

### 2.1 Option analysis

- **Option A** (paragraph reasoner → sample paragraph concept → writer renders whole paragraph as one 150-token unit): Right *planning* object; wrong *decoding* mechanics. One-shot 150+-token generation abandons per-sentence prefix chaining — the only channel ever measured to carry local coherence (G5: prefix present in the loop is what makes sentences connect; the 152M writer's long-range token quality is also unproven past ~96 tokens). Keep A's planner; reject A's monolithic render as primary (kept as an ablation arm).
- **Option B** (sentence-token generation, paragraph-committed concept, coarse arc locked): Right *decoding* mechanics; but as stated it is training-free, and P1 (0.175) says the frozen writer cannot read the committed pooled concept. B without new writer training is predicted to fail exactly like failure #5 — it *is* failure #5 with `anchor_every`-style commitment. Keep B's loop; add the missing writer training. B-without-training is retained as the free ablation arm ARM-PF that isolates the phase-3 writer's value.
- **Option C** (hierarchical: paragraph centroid predictor + trained within-paragraph residual sampler): The decomposition z = c_para + e is correct and is the physics of the whole project (pooling shrinks e by ~1/sqrt(G)). But a *trained* residual sampler is dead weight: e_t is measured to be mean-reverting noise (delta cos −0.48); its useful entropy is exactly what the writer's own token sampling already provides. C collapses into "commit c_para, let the writer's token distribution be the residual sampler." Adopt C's decomposition, delete C's second trained module.
- **Option D** (unit = the actual corpus block): Adopted in spirit — G-blocks within docs are the unit, and G=5 is chosen to match the data (median doc = 1.8 blocks). Rejected in letter: variable whole-block units add complexity for zero measured gain, and the cross-boundary chaining that would have made D powerful is measured dead (+0.0004).

### 2.2 The chosen design

**Paragraph-commitment generation.** One sampled concept per paragraph; the concept is a *topic*, not a content spec; the writer is fine-tuned so that a pooled topic is in-distribution for its FiLM channel; text-level novelty comes from token sampling under the topic, concept-level novelty from the MDN sample tilt; coherence comes from (i) the slow anchor leash at paragraph granularity (proven, P2), (ii) prefix chaining across every sentence including paragraph boundaries (proven, G5/writer_check), (iii) *not re-planning inside a paragraph* — which removes the per-step leakage of sentence-level chaos into the planning loop (the mechanism common to failures #1-#5).

Why this specific combination is the most likely to produce coherent multi-sentence novel documents:

1. Every measured failure is explained by exactly two defects: per-sentence re-planning in a space with no sentence-order signal (adj−within = −0.017), and a writer illiterate in pooled topics (P1 0.175). This design removes both and changes nothing else.
2. Every measured success is reused verbatim: the MDN family's calibration (Δ54 nats), `anchored_sample`'s concept-space leash (0.876 @ k8), the two-channel writer's prefix coherence and live FiLM (concept-shuffle 0.162), and paragraph-centroid stability (+0.137).
3. The one new trained component (phase-3 writer) is trained on a self-supervised target that exactly matches its inference contract — the same fix (same-sentence pairing, "fixes the Phase-2 D1 off-by-one") that took the writer from oracle −0.004 to +0.354 in phase-2.5. There is precedent in this repo that fixing the writer's contract produces step-change gains.

```mermaid
flowchart TB
    subgraph planning [Paragraph planning - once per paragraph]
        PH[para history feds] --> MDN[para MDN reasoner]
        MDN -->|"pi, mu, kappa (K=4)"| AS[anchored_sample: tilt + best-of-8 + slerp]
        ANC[slow anchor EMA] --> AS
        AS -->|"c_k (1,512) frozen for G sentences"| WR
    end
    subgraph rendering [Paragraph rendering - G sentences]
        WR[phase-3 writer FiLM c_k] -->|sentence j| TX[text_j]
        TX -->|prefix chain| WR
    end
    TX --> ENC[repaired codec encode]
    ENC -->|"pool G, L2"| FED[fed_k paragraph centroid]
    FED --> PH
    FED -->|"EMA lambda 0.25"| ANC
