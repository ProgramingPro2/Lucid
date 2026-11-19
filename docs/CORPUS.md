# Corpus — G-1 register

**Status: live harvest running; NLL table pending.** The pipeline, sources,
licence rules, JATS full-text parser, resumable JSONL harvest, document-level
split, token-stream hash and corpus card are implemented in `experiments/corpus/`
and exercised end to end by `--source fake --dry-run` (35 synthetic documents
retained from 60, 64-hex reproducible stream) and by a live bioRxiv window.

**Measured live yield (2026-11-18).** One week of postings: 40 metadata rows →
19 licence-allowed → **16 full-text documents** (84 % fetch success; 2 withdrawn,
3 blocked by the host). Median **86 k characters ≈ 21 k text tokens** per
document. Full text is the JATS XML linked per paper; abstracts alone are ~1.6 k
characters and are not enough. Extrapolating, the 500 M-token target needs
≈ 23 k full-text documents, which the 2025-06→2026-11 posting window can supply
(~100 k papers, ~47 % CC-BY). The binding constraint is licence-filtered posting
volume, not download speed. Harvest is append-only and resumable (`--resume`).

Two live-only bugs were found and fixed before the main run: the API returns 30
rows/page (the cursor had advanced by 100, skipping 70 %), and licences are
machine tokens (`cc_by_nc_nd`) that the prose normaliser did not classify
([FINDINGS.md](FINDINGS.md) F46).

**NLL and a pilot build.** `experiments/corpus/nll.py` scores four evenly spaced
2,048-token windows per document with the base model and writes a resumable table
(measured at ~460 documents/min on this card). A 40-document live pilot then went
through every gate: **13 documents retained** (~167 k tokens across the five
strata). The surviving drops were 26 pre-cutoff dates, 24 abstract-only or
withdrawn documents, and 1 low-NLL document; the counts overlap because
`not_fulltext` contains the withdrawn ones. This proves the whole chain
(harvest → NLL → gates → split → hash → card) on real data. What is *not* done is
the full 500 M-token harvest; it is running and resumable. This is the data stage
of the programme in [PROTOCOL.md](PROTOCOL.md); it has no GPU cost and must finish
before G0 because E0's items are drawn from it.

---

## 1. Why the corpus changes

The current corpus cannot carry a confirmatory claim, for two independent reasons.

| Reason | Measurement | Consequence |
|---|---|---|
| **Register skew** | 81 % Wikipedia by sentence; the filter keeps 93.8 % of wiki text and only 40.9 % of fiction | Every result is a Wikipedia result. Cross-register generalisation is untestable. (W4, F33) |
| **Contamination** | WikiText-103 is 100 % Wikipedia and is almost certainly inside Qwen3's pretraining | A held-out NLL on it may measure recall, not generalisation. Qwen's training data are opaque, so the confound cannot be measured — only avoided. |

A third reason is structural: the old corpus was **113,216 shuffled single
paragraphs**, of which 4.1 % yielded any ctx-32 window (1,677 windows in total).
Documents were not intact, so document-level splitting and document identity were
both impossible (W4, F31/F32).

The replacement fixes all three: **dated, post-cutoff, document-intact, licence-checked.**

---

## 2. Sources

| Role | Source | Licence | Size | Why |
|---|---|---|---|---|
| **Primary** | bioRxiv / medRxiv, postings dated **> model cutoff + 30 days** | CC-BY 4.0 for most; **read the per-item `license` field from the API**, exclude NC/ND if redistributing | ~120M tokens/month; 3–4 months ≈ 400M+ | Research-manuscript prose with natural sub-registers; short and datable exposure window — the best contamination profile available |
| **Second register (held out)** | FDA prescribing information, DailyMed / Drugs@FDA, 2026 label versions keyed on `effective_time` | **US Government work — public domain** | ~1,500 labels × 10–30k ≈ 20–45M | A genuinely different register, which is what makes the cross-register endpoint a real generalisation test rather than a re-split |
| **Replication only** | WikiText-103 test | as distributed | — | Kept solely as the comparability anchor to every prior number in the repo. **Never primary.** Its contamination is quantified (§6), not hidden. |

**Licence rule: per item, machine-checked.** A repository's default licence does not
apply to an individual item. The build script records the licence string for every
document and refuses to include anything it cannot classify.

---

## 3. Size requirements

- **≥ 500M tokens** in the primary pool.
- **≥ 3,000 documents**, with a median ≥ 3,000 tokens per document.
- **≥ 8× the largest planned run.** Largest run is 30M tokens; the pre-registered
  extension rule allows 60M. 8 × 60M = 480M, hence the 500M target.
- Enough documents that **no seed repeats a token stream**, and the extension rule
  never has to redraw.

---

## 4. Construction pipeline

Each step is a gate. A document that fails any step is dropped and the drop is
counted; the counts go into the corpus card.

| # | Step | Rule |
|---|---|---|
| 1 | **Cutoff** | Read the base model's training-cutoff date from its model card and record it verbatim in the corpus card. Every downstream date test refers to this value, not to "recent". |
| 2 | **Harvest** | Query by posting date. Keep only items with `date > cutoff + 30 days`. |
| 3 | **Licence** | Read the per-item licence field. Keep CC-BY / CC0. Drop NC / ND unless the redistribution question is settled in writing. |
| 4 | **Extract body text** | Strip tables, figure captions, reference lists, supplementary material, author lists, funding statements. Keep running prose only. |
| 5 | **Segment into strata** | Retain the section structure: `abstract`, `introduction`, `methods`, `results`, `discussion`. Strata are used for the stratified primary endpoint and for the register-balanced sensitivity analysis. |
| 6 | **Deduplicate** | Near-duplicate detection at the document level (preprint → journal version pairs are common). Keep the earliest. |
| 7 | **Memorisation screen** | Compute base-model NLL per document. Drop the bottom decile. A document the model finds implausibly easy is either leaked or boilerplate; both corrupt a held-out estimate. Record the dropped count and the NLL distribution. |
| 8 | **Split by document** | Sealed test / dev / train, **document-level, never paragraph-level**. Stratify by stratum and by NLL decile so the split is not confounded with difficulty. |
| 9 | **Freeze** | Write a manifest with a SHA-256 per document and one SHA-256 over the token stream. This hash is asserted identical across every training arm ([PROTOCOL.md](PROTOCOL.md) §10). |
| 10 | **Seal** | Encrypt or move the test split out of the working tree; the key is recorded in the amendment log. One look. |

Recommended split: 80 % train / 10 % dev / 10 % sealed test, by document.

---

## 5. Gates

| Gate | Criterion | If it fails |
|---|---|---|
| **G-1a size** | ≥ 500M tokens, ≥ 3,000 documents | harvest another month; do not shrink the run budget |
| **G-1b licence** | 100 % of retained documents carry a machine-read licence string | drop the unclassifiable; if that exceeds 20 %, change source |
| **G-1c integrity** | Every document has ≥ 1 complete section and no shuffled paragraphs | re-extract |
| **G-1d split** | Zero documents shared across train / dev / test | re-split |
| **G-1e hash** | Token-stream SHA-256 reproducible from a clean rebuild | fix the build; a non-reproducible stream voids every downstream arm |
| **G-1f register** | Sealed second register (FDA) present and untouched | required for the cross-register endpoint only |

---

## 6. Contamination probe

This is a **reported measurement**, not a gate, because there is no threshold that
would make a leaked corpus acceptable.

1. Compute base-model NLL on the new post-cutoff corpus and on WikiText-103 test,
   **at matched register** (Wikipedia-style prose is not comparable to methods
   sections; compare like with like, or report both and say so).
2. The gap is the contamination estimate. Report it with a CI.
3. Run the same probe on OLMo-2-1B, whose training data are open. If Qwen's gap on
   WikiText is substantially larger than OLMo's, that is direct evidence of
   contamination and belongs in the paper.

The figure to report: **nats/token on post-cutoff prose minus nats/token on
WikiText-103 test**. A large positive value means the model has seen the benchmark.

---

## 7. Register handling

- **Primary endpoint** is computed on the primary register only, stratified by
  section.
- **Cross-register endpoint** (secondary) is computed on the sealed FDA labels. It
  is a generalisation test and is reported whether or not the primary passes.
- **Register-balanced sensitivity analysis**: reweight the primary test split so
  each section contributes equally. If the primary contrast changes sign under
  reweighting, that is a finding about the corpus, not about the mechanism, and it
  blocks promotion of the claim.

---

## 8. What is forbidden

- Paragraph-level splits. (W4; this is the error that produced 1,677 windows.)
- WikiText-103 or any Wikipedia-derived text as the **primary** corpus.
- Shuffled or block-shuffled documents.
- A licence inferred from the repository rather than the item.
- Any test-set inspection before the analysis script is frozen.
- Re-drawing the test split after a result.

---

## 9. Corpus card (to be filled at build time)

| Field | Value |
|---|---|
| Base model and recorded cutoff | _pending_ |
| Sources and per-source licence counts | _pending_ |
| Documents harvested / dropped, by reason | _pending_ |
| Tokens per stratum | _pending_ |
| NLL decile distribution | _pending_ |
| Contamination gap (primary vs WikiText) | _pending_ |
| Token-stream SHA-256 | _pending_ |
| Split counts and sealed-test key holder | _pending_ |

The card ships with the paper. A corpus without a card is not citable.
