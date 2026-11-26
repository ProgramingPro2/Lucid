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

## Harvest operational window

Wifi on this machine is off from **23:30 to 06:00 America/New_York** every night. It is a scheduled local outage (the radio is powered off) and it recurs every night. A network-bound harvest still running at 23:30 dies mid-request; the 2026-11-25 run aborted at 23:32 with `Errno 101 Network is unreachable`. Network-bound harvests have to run inside the allowed window, **06:00–23:30** (17.5 hours). The morning timer starts the supervisor at 06:05, five minutes after the radio is back.

`scripts/harvest_supervisor.py` enforces the window. Launched inside the quiet hours, it sleeps until 06:00 and starts nothing. Inside the window it runs `experiments.corpus.harvest` with `--resume`, and if that process exits (crash, abort, or a clean finish) it starts it again after a bounded backoff. Resume is per page and per document id: the cursor sidecar is rewritten through a temp file and `rename`, and document ids already in the JSONL are skipped, so a restart does not lose or duplicate a completed document. At 23:30 the supervisor sends SIGTERM, waits for the child to exit, and does not send SIGKILL.

`harvest.py` installs no SIGTERM handler. Each finished JSONL line is flushed, and the cursor file is not truncated by a kill during its temp write. A signal between the 8 KiB chunks of one large flush can still leave a truncated final line with no newline, and the next append can glue onto it. `load_documents` skips a malformed final line and still raises `JSONDecodeError` on a malformed line earlier in the file. `_load_done` skips any line that is not valid JSON, so resume continues.

Any long run whose wall clock could cross 23:30 has to be started early enough to finish, or it dies mid-run. At the measured rate (13.7 docs/min, 3.5 full-text docs/min, about 17.7k tokens per full-text document) one window yields about **65M tokens**. The 3,000-document gate fits in a single window. The **500M-token gate is the binding one**, at about eight days of windowed harvesting. One 30M-token E1 arm is about 2.7 GPU-hours and fits in an evening (latest start 20:48). A six-arm batch is about 16.1 hours and has to start by **07:22** to finish before 23:30.

The user systemd timer `lucid-harvest.timer` starts the supervisor at 06:05. The process exits at 23:30, so it is not left running through the outage. Lingering is already on for this user. The unit files in `scripts/systemd/` are symlinked from `~/.config/systemd/user/`.

Verify:

    systemctl --user list-timers lucid-harvest.timer
    systemctl --user is-active lucid-harvest.service

Remove:

    systemctl --user disable --now lucid-harvest.timer
    rm ~/.config/systemd/user/lucid-harvest.service ~/.config/systemd/user/lucid-harvest.timer
    systemctl --user daemon-reload

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
| **Primary** | bioRxiv / medRxiv, postings dated **after 2025-05-29** (step 1) | CC-BY 4.0 for most; **read the per-item `license` field from the API**, exclude NC/ND if redistributing | ~120M tokens/month; 3–4 months ≈ 400M+ | Research-manuscript prose with natural sub-registers; short and datable exposure window — the best contamination profile available |
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
| 1 | **Cutoff** | Record the base model's training cutoff in the corpus card and use that date, not "recent". For `Qwen/Qwen3-1.7B-Base` the card discloses none. The release-date proxy is **2025-04-29** (`release_date_proxy`). The in-progress harvest's operative cutoff is **2025-06-01** (`operative_harvest_cutoff`), which is later and therefore strictly more conservative. Keep postings strictly after the date the build was handed, plus 30 days. Revise the proxy only upward. The rule, the evidence, and the cost of the in-progress file are below. |
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

### Step 1 rule — undisclosed cutoff

`Qwen/Qwen3-1.7B-Base` does not state a training cutoff. The release-date proxy is therefore not a date copied out of the card. It is a conservative proxy, and the card has to say so next to the operative harvest cutoff.

Evidence, checked 2026-11-26:

- The local weight snapshot `~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (the same revision under `HF_HOME=/mnt/keepstar-extra/hf-cache`) has `config.json`, `generation_config.json`, the tokenizer files and `model.safetensors`. It has no `README.md`. `config.json` has no cutoff field.
- The Hub card `https://huggingface.co/Qwen/Qwen3-1.7B-Base/raw/main/README.md` says "Qwen3 is pre-trained on 36 trillion tokens across 119 languages" and names coding, STEM, reasoning, books, multilingual and synthetic data. It states no knowledge-cutoff date. `cardData` is only `license` and `library_name`. The Hub `createdAt` is `2025-04-28T05:05:55.000Z` and the revision is `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- The release announcement `https://qwenlm.github.io/blog/qwen3/` is dated April 29, 2025. The Qwen3 GitHub news line is "2025.04.29: We released the Qwen3 series." The pre-training section describes about 36 trillion tokens and three stages and states no cutoff date.
- The technical report, arXiv:2505.09388 §3.1, says "All Qwen3 models are trained on a large and diverse dataset consisting of 119 languages and dialects, with a total of 36 trillion tokens." The report text does not contain the word "cutoff".

Third-party pages that name a March 2025 or June 2024 cutoff were not used. They are not the model card, and both dates are earlier than the release. The proxy may not move earlier on the strength of them.

Rule. When the model card discloses no cutoff, the programme cutoff is the model **release date**, 2025-04-29. That date is after the weights were already on the Hub (2025-04-28T05:05:55Z), so it is after any data that could have been in the training run. The existing 30-day grace still applies: the build keeps a posting only when its date is strictly after 2025-05-29, so the first retained posting is **2025-05-30**. This discards papers from the gap between the unknown true cutoff and the release, instead of risking contamination. The proxy can be revised only **upward** (a later date, which drops still more text). It is never revised downward.

`2025-06-01` is not the proxy. It is the operative harvest cutoff: the date the running supervisor and `experiments.corpus.harvest` were opened with (`DEFAULT_CUTOFF` in `scripts/harvest_supervisor.py`, and the `--cutoff` on the live command). It also appears as the month in the volume phrase "2025-06→2026-11". It is not on the model card. The card records it as `operative_harvest_cutoff`. That name is the harvest cutoff, not the model-card cutoff (`model_card_cutoff` stays null) and not the release-date proxy (`release_date_proxy` stays 2025-04-29). `recorded_cutoff` is only the date the build's date gate was handed. A build handed 2025-06-01 records that date as an upward revision and says it is not a model-card cutoff. A build handed the proxy for the in-progress file `artifacts/corpus/biorxiv_2026.jsonl` still writes `operative_harvest_cutoff` as 2025-06-01, so the harvest date cannot disappear behind the proxy. A build handed `_pending_`, an empty cutoff, or no cutoff at all refuses to run.

The operative harvest cutoff 2025-06-01 is later than the release-date proxy 2025-04-29, so it is strictly more conservative, and that is what makes the rows in the current file safe despite the two dates.

The in-progress file was opened with `--cutoff 2025-06-01`. Harvest windows start at that date plus 30 days, so the file begins on 2025-07-01. Do not retarget the running harvest and do not change its `--cutoff`. Filling the 2025-05-30..2025-07-01 gap needs a separate corpus generation, not an append onto this file. Do not backfill. The build's date gate, handed that same 2025-06-01, keeps a posting only when its date is strictly after 2025-07-01, so full-text rows posted on the window-start day are dropped as `date`. That drop is the grace boundary, not a second cutoff. Two cutoffs in one JSONL cannot be separated later. Relative to the proxy, the audit measured roughly **1,700 full-text documents and ~31M tokens** posted from 2025-05-30 up to this file's start on 2025-07-01 that are absent. Rows already in the file are dated on or after 2025-07-01, which is after 2025-05-29, so they satisfy the proxy date gate. The missing slice is a coverage gap, not contamination of the rows that are present. The generated card says this in `coverage_note`, which names the gap and the separate generation, when the earliest row is after 2025-05-30.

---

## 5. Gates

| Gate | Criterion | If it fails |
|---|---|---|
| **G-1a size** | ≥ 500M tokens, ≥ 3,000 documents | harvest another month; do not shrink the run budget |
| **G-1b licence** | 100 % of retained documents carry a machine-read licence string | drop the unclassifiable; if that exceeds 20 %, change source |
| **G-1c integrity** | Every retained document has ≥ 1 complete section and no shuffled paragraphs. On a harvested bioRxiv/medRxiv row the fulltext status is exactly `ok` | re-extract. A failed fetch (`error:*`) or an `abstract_only` row is not a document |
| **G-1d split** | Zero documents shared across train / dev / test | re-split |
| **G-1e hash** | Token-stream SHA-256 reproducible from a clean rebuild | fix the build; a non-reproducible stream voids every downstream arm |
| **G-1f register** | Sealed second register (FDA) present and untouched | required for the cross-register endpoint only |

The fulltext-status check is its own gate. It runs on every harvested row before the licence gate, and the card records both `drops_by_reason.not_fulltext` and `fulltext_status_counts` (the per-status census of the whole harvest). A cc-by abstract is dropped because its status is `abstract_only`. The same row is dropped when the licence would have allowed it and when the licence would have rejected it, and the census counts it either way. Raising the length floor does not close this: failed fetches in the live file include rows over 2,000 characters.

`--allow-degraded` is the only way to keep a non-`ok` status. It prints a warning on stderr and sets `g1c_compliant` to false with a verdict that says **NOT G-1c compliant**. A degraded card must not be cited as a gate pass. The default build does not take that flag.

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
| Base model | `Qwen/Qwen3-1.7B-Base`. The model card discloses no cutoff (`model_card_cutoff` is null). |
| Release-date proxy | `release_date_proxy`: **2025-04-29**. This is not a date copied out of the model card. Weights uploaded 2025-04-28T05:05:55Z, snapshot `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, grace 30 days, first retained posting under the proxy 2025-05-30, revision policy upward-only. Evidence: the local snapshot has no README and `config.json` has no cutoff field; the Hub README (36 trillion tokens, 119 languages) states no cutoff; the 2025-04-29 announcement and arXiv:2505.09388 §3.1 state none either. Justification: no model card, config, or Qwen3 paper states a cutoff, so the proxy is the 2025-04-29 release date; with the 30-day grace this keeps only postings after 2025-05-29, and the date may move later but never earlier. |
| Operative harvest cutoff | `operative_harvest_cutoff`: **2025-06-01** for `artifacts/corpus/biorxiv_2026.jsonl`. This is the cutoff the running harvest was opened with. It is not a model-card value. It is later than 2025-04-29, so it is strictly more conservative, and that is what makes the rows in the current file safe despite the two dates. `recorded_cutoff` is the date the build's date gate was handed; for this file that is also 2025-06-01, recorded as an upward revision (`kind: upward-revision`). |
| Coverage | `coverage_note` names the absent slice 2025-05-30..2025-07-01. Filling that gap needs a separate corpus generation, not an append. This file is not backfilled, and the running harvest's `--cutoff` is not changed. |
| G-1c verdict | Written by the build as `g1c_compliant` and `g1c_verdict`. The default drops non-`ok` fulltext. `--allow-degraded` is NOT G-1c compliant. |
| Sources and per-source licence counts | _pending_ |
| Documents harvested / dropped, by reason | Written by the build into `card.json` as `drops_by_reason` and `fulltext_status_counts`. Totals move until the harvest is finished. |
| Tokens per stratum | _pending_ |
| NLL decile distribution | _pending_ |
| Contamination gap (primary vs WikiText) | _pending_ |
| Token-stream SHA-256 | _pending_ |
| Split counts and sealed-test key holder | _pending_ |

The card ships with the paper. A corpus without a card is not citable.
