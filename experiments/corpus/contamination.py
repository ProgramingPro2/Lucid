"""Contamination probe for the post-cutoff corpus (docs/CORPUS.md §6).

Section 6 asks for three things. The figure is base-model NLL, in nats/token,
on the post-cutoff corpus minus the same NLL on WikiText-103. A large positive
value means WikiText is easier, which is what memorisation of the benchmark
looks like. The gap is reported with a bootstrap CI. The same gap on
OLMo-2-1B is the open-data comparison: if Qwen's gap is substantially larger
than OLMo's, that is evidence about Qwen, and the protocol does not define
"substantially", so this module reports the difference and does not turn it
into a pass.

The probe also puts each document against a reference distribution, because an
absolute floor by itself is an arbitrary number. The reference used for that
fence is a set of fluent passages written for this probe, with nonce names, so
they are not in any training set frozen at the Qwen3 release. The fence is
median - 3 * 1.4826 * MAD of those passages. It is never estimated from the
corpus under test.

What this cannot do, and must not be cited as doing:

- An NLL above the floor, or above the fence, does not prove the document was
  absent from training. Low NLL flags suspicious text. It does not certify
  the rest.
- Qwen3-1.7B-Base does not publish its training mix. A famous public passage
  is not a confirmed training member. The positive control can show that the
  screen is sensitive to *something*. It cannot validate a contamination test.
  The probe is a screening heuristic.
- WikiText is encyclopedia prose and the corpus is manuscript prose. Section 6
  says those registers are not comparable. The gap mixes register with
  contamination. The OLMo difference is the part that can cancel a register
  shift both models share. It was not run unless the weights were actually
  scored.
- The §4 bottom decile drops about a tenth of documents whatever their NLL.
  That trim is applied by the build screen. It is not this probe's verdict.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.corpus.contamination \\
        --docs-jsonl artifacts/corpus/biorxiv_2026.jsonl \\
        --wikitext /tmp/contamination-probe/wikitext-103-raw-v1-test.parquet \\
        --out /tmp/contamination-probe/report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from .nll import (
    CRITERION_SOURCE,
    MEMORISATION_NLL_FLOOR,
    N_WINDOWS,
    REFERENCE_Z,
    WINDOW_TOKENS,
    InvalidNLLScreen,
    apply_memorisation_screen,
    coerce_nll,
    nll_distribution,
    sampling_rule,
)

PROBE_SAMPLE_N = 50
WIKITEXT_SAMPLE_N = 40
WIKITEXT_MIN_CHARS = 1500
PROBE_SEED = 20261126
N_BOOT = 2000
OLMO_MODEL_ID = "allenai/OLMo-2-0425-1B"
QWEN_MODEL_ID = "Qwen/Qwen3-1.7B-Base"

_ARTICLE_HEADING = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")
_ANY_HEADING = re.compile(r"^\s*=+\s+.+=+\s*$")


def detok_wikitext(text: str) -> str:
    """Undo WikiText raw spacing. Same substitutions as ``corpus_v2.detok_wikitext``."""
    text = re.sub(r"\s*@-@\s*", "-", text)
    text = re.sub(r"\s*@,@\s*", ",", text)
    text = re.sub(r"\s*@\.@\s*", ".", text)
    for punct in ",.!?;:":
        text = re.sub(rf"\s+{re.escape(punct)}", punct, text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"(?=\W|$)', '"', text)
    return text.rstrip()


def wikitext_articles(lines: list[str]) -> list[tuple[str, str]]:
    """Split WikiText-raw lines into ``(title, body)`` articles.

    A level-1 heading (``= Title =``) starts an article. Section headings and
    blank lines are not body text. Body lines are detokenized and joined.
    """
    articles: list[tuple[str, str]] = []
    title: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal title, parts
        body = " ".join(parts).strip()
        if title and body:
            articles.append((title, body))
        title = None
        parts = []

    for line in lines:
        raw = line.strip()
        if raw and _ARTICLE_HEADING.match(raw) and not raw.lstrip().startswith("=="):
            flush()
            title = detok_wikitext(raw.strip().strip("=").strip())
            parts = []
            continue
        if title is None or not raw or _ANY_HEADING.match(raw):
            continue
        parts.append(detok_wikitext(raw))
    flush()
    return articles


def read_wikitext_lines(path: str | Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path), columns=["text"])
    column = table.column("text")
    return [
        "" if column[i].as_py() is None else str(column[i].as_py())
        for i in range(table.num_rows)
    ]


def select_wikitext_articles(
    articles: list[tuple[str, str]],
    n: int = WIKITEXT_SAMPLE_N,
    min_chars: int = WIKITEXT_MIN_CHARS,
) -> list[tuple[str, str]]:
    """First ``n`` articles at or above ``min_chars``, in file order."""
    chosen = [(title, body) for title, body in articles if len(body) >= min_chars]
    return chosen[:n]


def select_fulltext(docs: list, n: int = PROBE_SAMPLE_N, seed: int = PROBE_SEED) -> list:
    """A deterministic sample of harvested rows whose fulltext status is ``ok``."""
    ok = [
        doc
        for doc in docs
        if (doc.meta or {}).get("fulltext") == "ok" and (doc.text or "").strip()
    ]
    ok.sort(key=lambda doc: doc.doc_id)
    rng = random.Random(seed)
    rng.shuffle(ok)
    chosen = ok[:n]
    chosen.sort(key=lambda doc: doc.doc_id)
    return chosen


def shuffle_words(text: str, seed: str) -> str:
    """Word-shuffle ``text``. Same words, destroyed order, fixed seed."""
    words = text.split()
    rng = random.Random(f"contamination-shuffle:{seed}")
    rng.shuffle(words)
    return " ".join(words)


def robust_lower_fence(values: list[float], z: float = REFERENCE_Z) -> float:
    """Lower fence of a reference sample: median - z * 1.4826 * MAD.

    A zero MAD (every reference value equal) puts the fence on the reference
    itself, so only a strictly lower score is flagged. The width is not invented.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise InvalidNLLScreen(
            "INVALID: the reference distribution is empty or non-finite. "
            "The probe is rejected. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad == 0.0:
        return median
    return median - z * 1.4826 * mad


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        raise InvalidNLLScreen(
            "INVALID: NLL weights sum to nothing. The probe is rejected. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    return float(np.sum(values * weights) / total)


def gap_nats(
    corpus_nll: list[float],
    corpus_weights: list[float],
    reference_nll: list[float],
    reference_weights: list[float],
    *,
    n_boot: int = N_BOOT,
    seed: int = PROBE_SEED,
) -> dict:
    """Post-cutoff mean minus reference mean, with a percentile bootstrap CI.

    The unit is the document (or article). Weights are tokens scored, so the
    point estimate is nats/token rather than an unweighted mean of documents.
    Resampling draws documents, not tokens.
    """
    corpus = np.asarray(corpus_nll, dtype=float)
    reference = np.asarray(reference_nll, dtype=float)
    corpus_w = np.asarray(corpus_weights, dtype=float)
    reference_w = np.asarray(reference_weights, dtype=float)
    if (
        corpus.size == 0
        or reference.size == 0
        or not np.all(np.isfinite(corpus))
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(corpus_w))
        or not np.all(np.isfinite(reference_w))
    ):
        raise InvalidNLLScreen(
            "INVALID: a gap input is empty or non-finite. The probe is rejected "
            "and is not reported as a pass. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    point_corpus = _weighted_mean(corpus, corpus_w)
    point_reference = _weighted_mean(reference, reference_w)
    point = point_corpus - point_reference
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ic = rng.integers(0, corpus.size, corpus.size)
        ir = rng.integers(0, reference.size, reference.size)
        diffs[i] = _weighted_mean(corpus[ic], corpus_w[ic]) - _weighted_mean(
            reference[ir], reference_w[ir]
        )
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "gap_nats": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_corpus": int(corpus.size),
        "n_reference": int(reference.size),
        "corpus_mean_nats": float(point_corpus),
        "reference_mean_nats": float(point_reference),
        "weighting": "tokens_scored",
        "n_boot": n_boot,
        "seed": seed,
        "definition": (
            "nats/token on post-cutoff prose minus nats/token on the reference "
            "(docs/CORPUS.md §6). A large positive value means the reference is "
            "easier, which is the signature that the model has seen the reference. "
            "It is not a signature that the post-cutoff corpus is clean."
        ),
    }


def is_suspicious(nll: float, *, floor: float, fence: float | None) -> dict:
    """Flag a score. Does not prove the complementary score is clean."""
    number = coerce_nll(nll, "probe")
    below_floor = number < floor
    below_fence = fence is not None and number < fence
    return {
        "nll": number,
        "below_floor": below_floor,
        "below_reference_fence": below_fence,
        "flagged": below_floor or below_fence,
    }


def interpret_controls(
    positive: list[float],
    negative: list[float],
    *,
    floor: float,
    fence: float | None,
) -> dict:
    """Say what the controls show, including when they show nothing.

    ``confirmed_training_member`` is always false. Qwen3-1.7B-Base's training
    data are not public, so a famous passage is not a known member, and this
    function will not claim otherwise because the scores separated.
    """
    for value in list(positive) + list(negative):
        coerce_nll(value, "control")
    pos_flags = [is_suspicious(v, floor=floor, fence=fence)["flagged"] for v in positive]
    neg_flags = [is_suspicious(v, floor=floor, fence=fence)["flagged"] for v in negative]
    scores_separate = bool(positive) and bool(negative) and max(positive) < min(negative)
    sensitivity = bool(pos_flags) and all(pos_flags) and not any(neg_flags)
    return {
        "confirmed_training_member": False,
        "probe_kind": "screening_heuristic",
        "positive_all_flagged": bool(pos_flags) and all(pos_flags),
        "negative_none_flagged": not any(neg_flags),
        "scores_separate": bool(scores_separate),
        "sensitivity_check_passed": bool(sensitivity),
        "reason": (
            "Qwen3-1.7B-Base does not publish its training mix, so none of the "
            "positive passages is a confirmed training member. A gap between "
            "those passages and the nonce controls shows the screen is sensitive "
            "to whatever made the famous text easier. It does not validate a "
            "contamination test. If the scores do not separate, the screen is "
            "not even a reliable detector of these famous passages."
        ),
    }


def g1_contamination_verdict(*, n_corpus: int, n_below_floor: int) -> dict:
    """Section 6 is a measurement. There is no pass to award."""
    return {
        "passes": None,
        "n_corpus": n_corpus,
        "n_below_floor": n_below_floor,
        "criterion_source": CRITERION_SOURCE,
        "reason": (
            "docs/CORPUS.md §5 gates G-1a through G-1f do not contain a "
            "contamination pass/fail. §6 says the probe is reported because "
            "there is no threshold that would make a leaked corpus acceptable. "
            "n_below_floor is how many scored documents the absolute floor "
            "flagged. Documents the floor did not flag are not certified clean, "
            "and a sample of a few dozen documents does not make the criterion "
            "a test of the retained corpus."
        ),
    }


def compare_model_gaps(qwen_gap: dict | None, olmo_gap: dict | None) -> dict:
    """§6 step 3. The difference is reported. It is not thresholded."""
    if not qwen_gap or not olmo_gap:
        return {
            "status": "not_run",
            "note": (
                "OLMo-2-1B was not scored, so the open-data comparison in "
                "docs/CORPUS.md §6 step 3 is not available."
            ),
        }
    qwen = float(qwen_gap["gap_nats"])
    olmo = float(olmo_gap["gap_nats"])
    return {
        "status": "scored",
        "qwen_gap_nats": qwen,
        "olmo_gap_nats": olmo,
        "qwen_minus_olmo_nats": qwen - olmo,
        "interpretation": (
            "docs/CORPUS.md §6 step 3: a Qwen gap on WikiText that is "
            "substantially larger than OLMo's is evidence Qwen has seen the "
            "benchmark more than the open model. The protocol does not define "
            "substantially. A register shift that both models share cancels "
            "in the difference only to the extent the two models experience "
            "that shift equally. This number is not a pass/fail."
        ),
    }


# --------------------------------------------------------------------------
# Controls. The positive passages are public domain. None is a confirmed
# member of Qwen3's training mix. The negative passages were written for
# this probe and use nonce names so they are not a published paragraph.
# --------------------------------------------------------------------------

POSITIVE_PASSAGES: tuple[tuple[str, str], ...] = (
    (
        "dickens_two_cities",
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        "was the epoch of incredulity, it was the season of Light, it was the "
        "season of Darkness, it was the spring of hope, it was the winter of "
        "despair, we had everything before us, we had nothing before us, we were "
        "all going direct to Heaven, we were all going direct the other way — in "
        "short, the period was so far like the present period, that some of its "
        "noisiest authorities insisted on its being received, for good or for "
        "evil, in the superlative degree of comparison only.",
    ),
    (
        "gettysburg",
        "Four score and seven years ago our fathers brought forth on this "
        "continent, a new nation, conceived in Liberty, and dedicated to the "
        "proposition that all men are created equal. Now we are engaged in a "
        "great civil war, testing whether that nation, or any nation so "
        "conceived and so dedicated, can long endure.",
    ),
    (
        "hamlet_to_be",
        "To be, or not to be, that is the question: Whether 'tis nobler in the "
        "mind to suffer the slings and arrows of outrageous fortune, or to take "
        "arms against a sea of troubles and by opposing end them.",
    ),
)

NEGATIVE_PASSAGES: tuple[tuple[str, str], ...] = (
    (
        "neg_fibroblast",
        "We cultured primary fibroblasts from the invented donor line "
        "Qellibrax-17 and exposed them to 40 nanomolar flornexin for six hours. "
        "Viability was read with a resazurin reduction that we prepared in "
        "house. The vehicle was 0.02 percent dimethyl sulfoxide. Each of the "
        "eight nonce wells, batch lucid-probe-neg-1, was background-subtracted "
        "before the mean was taken. No public dataset was joined to this plate.",
    ),
    (
        "neg_canopy",
        "The field crew at station Brumble-12 recorded canopy temperature every "
        "nine minutes during the 2026-11-26 overpass. A handheld probe, serial "
        "korvax-77, was calibrated against melting ice the same morning. We "
        "excluded three gusts above 14 metres per second. The remaining series "
        "was smoothed with a five-point median and stored as lucid-probe-neg-2. "
        "Nothing in the series was copied from a weather archive.",
    ),
    (
        "neg_grating",
        "Participants in cohort Zintar-9 completed a two-alternative forced "
        "choice on grating contrast. The grating was generated in session "
        "software written for this probe and was never posted. Contrast ranged "
        "from 1.2 to 9.6 percent in seven steps. Response keys were "
        "counterbalanced. We analysed only the nonce session tagged "
        "lucid-probe-neg-3, and we did not pool it with any published "
        "psychometric function.",
    ),
    (
        "neg_polymer",
        "Sectioning of the synthetic polymer plimwort-4 was done at 8 "
        "micrometres on a cryostat set to minus 22 celsius. Slides were stained "
        "with a dye mixture we labelled umbral-blue, which is not a catalogue "
        "product. Three fields per section were photographed at 20 times. "
        "Counts of voids were entered twice and reconciled. The sheet "
        "lucid-probe-neg-4 is the only record of those counts.",
    ),
    (
        "neg_reactor",
        "A closed-loop controller held the invented reactor Vellimor-3 at 41.5 "
        "celsius while a peristaltic pump delivered quench buffer at 0.8 "
        "millilitres per minute. Pressure was sensed by a gauge whose offset we "
        "measured that hour. The run lasted 54 minutes and ended when the nonce "
        "marker lucid-probe-neg-5 was written to the log. No chromatogram from "
        "this run has been deposited.",
    ),
    (
        "neg_interview",
        "We transcribed interviews with twelve hypothetical respondents about a "
        "device called the quorflan kettle. The interview guide was drafted for "
        "this measurement and uses no items from a published scale. Two coders "
        "marked mentions of heat, cost, and noise. Disagreements were left as "
        "disagreements. The codebook lucid-probe-neg-6 was not derived from any "
        "atlas or manual.",
    ),
)

PARAPHRASE_GETTYSBURG = (
    "Eighty-seven years before the speech, the speaker's predecessors had "
    "founded a country on this continent, committed to freedom and to the claim "
    "that people are equal. A civil war was underway, and it asked whether a "
    "country founded on that claim could last."
)

REPETITION_DIAGNOSTIC = " ".join(["the"] * 80)


def control_items() -> list[dict]:
    """Every string the probe scores that is not the corpus or WikiText."""
    items: list[dict] = []
    for name, text in POSITIVE_PASSAGES:
        items.append({"name": name, "kind": "positive", "text": text})
        items.append(
            {
                "name": f"shuffled_{name}",
                "kind": "shuffled_positive",
                "text": shuffle_words(text, name),
            }
        )
    for name, text in NEGATIVE_PASSAGES:
        items.append({"name": name, "kind": "negative", "text": text})
    items.append(
        {
            "name": "paraphrase_gettysburg",
            "kind": "paraphrase",
            "text": PARAPHRASE_GETTYSBURG,
        }
    )
    items.append(
        {
            "name": "repetition_the",
            "kind": "diagnostic_repetition",
            "text": REPETITION_DIAGNOSTIC,
        }
    )
    return items


def _index_rows(rows: list[dict]) -> dict[str, dict]:
    return {row["doc_id"]: row for row in rows}


def _kind_nlls(rows_by_id: dict[str, dict], items: list[dict], kind: str) -> list[float]:
    return [float(rows_by_id[item["name"]]["nll"]) for item in items if item["kind"] == kind]


def assemble_report(
    *,
    control_rows: list[dict],
    corpus_rows: list[dict],
    wikitext_rows: list[dict],
    olmo_control_rows: list[dict] | None = None,
    olmo_corpus_rows: list[dict] | None = None,
    olmo_wikitext_rows: list[dict] | None = None,
    timing: dict | None = None,
) -> dict:
    """Turn scored rows into the probe record. No model, no I/O."""
    items = control_items()
    by_id = _index_rows(control_rows)
    positive = _kind_nlls(by_id, items, "positive")
    negative = _kind_nlls(by_id, items, "negative")
    fence = robust_lower_fence(negative)
    fence_source = (
        "median - REFERENCE_Z * 1.4826 * MAD of the fluent nonce negative "
        "controls in experiments.corpus.contamination.NEGATIVE_PASSAGES. "
        "Not estimated from the corpus or from WikiText."
    )
    controls = []
    for item in items:
        row = by_id[item["name"]]
        flag = is_suspicious(row["nll"], floor=MEMORISATION_NLL_FLOOR, fence=fence)
        controls.append(
            {
                "name": item["name"],
                "kind": item["kind"],
                "nll": flag["nll"],
                "tokens_scored": row["tokens_scored"],
                "flagged": flag["flagged"],
                "below_floor": flag["below_floor"],
                "below_reference_fence": flag["below_reference_fence"],
            }
        )
    assessment = interpret_controls(positive, negative, floor=MEMORISATION_NLL_FLOOR, fence=fence)
    shuffled = {
        item["name"].removeprefix("shuffled_"): by_id[item["name"]]["nll"]
        for item in items
        if item["kind"] == "shuffled_positive"
    }
    order_gaps = []
    for item in items:
        if item["kind"] != "positive":
            continue
        order_gaps.append(
            {
                "name": item["name"],
                "verbatim_nll": by_id[item["name"]]["nll"],
                "shuffled_nll": shuffled[item["name"]],
                "shuffled_minus_verbatim": shuffled[item["name"]] - by_id[item["name"]]["nll"],
            }
        )
    paraphrase_nll = by_id["paraphrase_gettysburg"]["nll"]
    gettysburg_nll = by_id["gettysburg"]["nll"]

    corpus_nll = [float(row["nll"]) for row in corpus_rows]
    corpus_w = [float(row["tokens_scored"]) for row in corpus_rows]
    wiki_nll = [float(row["nll"]) for row in wikitext_rows]
    wiki_w = [float(row["tokens_scored"]) for row in wikitext_rows]
    gap = gap_nats(corpus_nll, corpus_w, wiki_nll, wiki_w)
    n_below_floor = sum(1 for value in corpus_nll if value < MEMORISATION_NLL_FLOOR)
    n_below_fence = sum(1 for value in corpus_nll if value < fence)

    olmo_gap = None
    if olmo_corpus_rows and olmo_wikitext_rows:
        olmo_gap = gap_nats(
            [float(row["nll"]) for row in olmo_corpus_rows],
            [float(row["tokens_scored"]) for row in olmo_corpus_rows],
            [float(row["nll"]) for row in olmo_wikitext_rows],
            [float(row["tokens_scored"]) for row in olmo_wikitext_rows],
        )

    screen_payload = {
        "model": QWEN_MODEL_ID,
        "max_tokens": WINDOW_TOKENS,
        "n_windows": N_WINDOWS,
        "nll": {row["doc_id"]: row["nll"] for row in corpus_rows},
        "tokens_scored": {row["doc_id"]: row["tokens_scored"] for row in corpus_rows},
        "reference_fence_nats": fence,
        "reference_fence_source": fence_source,
        "contamination_gap_nats": gap,
    }
    # The screen needs objects with doc_id. The rows themselves qualify.
    class _Id:
        def __init__(self, doc_id: str) -> None:
            self.doc_id = doc_id

    screened = apply_memorisation_screen(
        [_Id(row["doc_id"]) for row in corpus_rows], screen_payload
    )

    return {
        "criterion_source": CRITERION_SOURCE,
        "sampling": sampling_rule(),
        "threshold_nats": MEMORISATION_NLL_FLOOR,
        "reference_fence_nats": fence,
        "reference_fence_source": fence_source,
        "reference_z": REFERENCE_Z,
        "cannot_prove_absence": True,
        "probe_kind": "screening_heuristic",
        "controls": controls,
        "control_assessment": assessment,
        "order_gaps": order_gaps,
        "paraphrase_minus_gettysburg": paraphrase_nll - gettysburg_nll,
        "corpus_distribution": nll_distribution(corpus_nll),
        "wikitext_distribution": nll_distribution(wiki_nll),
        "negative_distribution": nll_distribution(negative),
        "positive_distribution": nll_distribution(positive),
        "corpus_n_below_floor": n_below_floor,
        "corpus_n_below_fence": n_below_fence,
        "gap": gap,
        "olmo_gap": olmo_gap,
        "model_comparison": compare_model_gaps(gap, olmo_gap),
        "screen": screened.record,
        "g1": g1_contamination_verdict(
            n_corpus=len(corpus_rows), n_below_floor=n_below_floor
        ),
        "timing": timing or {},
        "registers": (
            "WikiText-103 test is encyclopedia prose. The corpus slice is "
            "bioRxiv full text. docs/CORPUS.md §6 says those registers are not "
            "comparable; both are reported and the gap is not a pure "
            "contamination estimate."
        ),
        "olmo_controls": olmo_control_rows,
    }


def _score_model(pairs: list[tuple[str, str]], model_id: str) -> tuple[list[dict], dict]:
    from .nll import _free_cuda, load_lm, score_labeled_texts

    started = time.time()
    lm, device, free = load_lm(model_id)
    peak = None
    try:
        rows, used_length = score_labeled_texts(
            pairs, lm, device=device, length=WINDOW_TOKENS, n_windows=N_WINDOWS
        )
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    finally:
        lm = None
        _free_cuda()
    seconds = time.time() - started
    tokens = sum(row["tokens_scored"] for row in rows)
    timing = {
        "model": model_id,
        "device": device,
        "free_mib_before_load": free,
        "peak_allocated_mib": peak,
        "seconds": round(seconds, 3),
        "documents": len(rows),
        "tokens_scored": tokens,
        "docs_per_min": round(len(rows) / max(seconds, 1e-9) * 60, 2),
        "window_tokens_used": used_length,
        "n_windows": N_WINDOWS,
    }
    return rows, timing


def _olmo_ready() -> bool:
    """True when the OLMo snapshot is on disk. Does not start a download."""
    import os

    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    name = "models--allenai--OLMo-2-0425-1B"
    for root in roots:
        snaps = root / name / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in snaps.iterdir():
            if any(snap.glob("*.safetensors")) or (snap / "model.safetensors.index.json").exists():
                return True
    return False


def run_probe(
    docs_jsonl: str | Path,
    wikitext_path: str | Path,
    *,
    n_corpus: int = PROBE_SAMPLE_N,
    n_wiki: int = WIKITEXT_SAMPLE_N,
    score_olmo: bool = True,
) -> dict:
    """Score a bounded slice and assemble the probe. Read-only on the JSONL."""
    from .harvest import load_documents

    docs = select_fulltext(load_documents(docs_jsonl), n=n_corpus)
    if len(docs) < 40:
        raise RuntimeError(
            f"only {len(docs)} full-text documents were available; the probe "
            "asked for a slice of 40 to 60"
        )
    articles = select_wikitext_articles(
        wikitext_articles(read_wikitext_lines(wikitext_path)), n=n_wiki
    )
    if len(articles) < 10:
        raise RuntimeError(f"only {len(articles)} WikiText articles cleared the length floor")

    controls = control_items()
    pairs: list[tuple[str, str]] = [(item["name"], item["text"]) for item in controls]
    pairs.extend((doc.doc_id, doc.text) for doc in docs)
    pairs.extend((f"wikitext:{title}", body) for title, body in articles)

    qwen_rows, qwen_timing = _score_model(pairs, QWEN_MODEL_ID)
    by_id = _index_rows(qwen_rows)
    control_rows = [by_id[item["name"]] for item in controls]
    corpus_rows = [by_id[doc.doc_id] for doc in docs]
    wikitext_rows = [by_id[f"wikitext:{title}"] for title, _ in articles]

    olmo_controls = olmo_corpus = olmo_wiki = None
    olmo_timing = None
    olmo_note = None
    if score_olmo and _olmo_ready():
        try:
            olmo_rows, olmo_timing = _score_model(pairs, OLMO_MODEL_ID)
        except Exception as exc:  # a failed second model must not erase the Qwen numbers
            import traceback

            traceback.print_exc()
            olmo_note = f"{type(exc).__name__}: {exc}"
        else:
            olmo_by = _index_rows(olmo_rows)
            olmo_controls = [olmo_by[item["name"]] for item in controls]
            olmo_corpus = [olmo_by[doc.doc_id] for doc in docs]
            olmo_wiki = [olmo_by[f"wikitext:{title}"] for title, _ in articles]
    elif score_olmo:
        olmo_note = (
            f"{OLMO_MODEL_ID} weights were not on disk, so §6 step 3 was not run. "
            "The probe does not download a second model in the scoring path."
        )

    report = assemble_report(
        control_rows=control_rows,
        corpus_rows=corpus_rows,
        wikitext_rows=wikitext_rows,
        olmo_control_rows=olmo_controls,
        olmo_corpus_rows=olmo_corpus,
        olmo_wikitext_rows=olmo_wiki,
        timing={"qwen": qwen_timing, "olmo": olmo_timing, "olmo_note": olmo_note},
    )
    report["sample"] = {
        "corpus_jsonl": str(docs_jsonl),
        "n_corpus_requested": n_corpus,
        "n_corpus_scored": len(docs),
        "corpus_doc_ids": [doc.doc_id for doc in docs],
        "corpus_selection": (
            f"fulltext status ok, seed {PROBE_SEED}, first {n_corpus} after a "
            "shuffle of doc_id order, then sorted by doc_id"
        ),
        "wikitext_path": str(wikitext_path),
        "wikitext_split": "wikitext-103-raw-v1 test",
        "n_wikitext": len(articles),
        "wikitext_titles": [title for title, _ in articles],
        "scored_text": (
            "Harvested document.text, before build's clean_text. The same "
            "string the standalone NLL scorer uses. WikiText bodies are "
            "detokenized articles from the test split."
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-jsonl", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-corpus", type=int, default=PROBE_SAMPLE_N)
    parser.add_argument("--n-wiki", type=int, default=WIKITEXT_SAMPLE_N)
    parser.add_argument("--skip-olmo", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_probe(
            args.docs_jsonl,
            args.wikitext,
            n_corpus=args.n_corpus,
            n_wiki=args.n_wiki,
            score_olmo=not args.skip_olmo,
        )
    except InvalidNLLScreen as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[probe] wrote {out}", flush=True)
    gap = report["gap"]
    print(
        f"[probe] gap {gap['gap_nats']:.3f} nats "
        f"CI [{gap['ci95_low']:.3f}, {gap['ci95_high']:.3f}] "
        f"floor flags {report['corpus_n_below_floor']}/{gap['n_corpus']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
