"""E4-S1 endpoints and the G3 decision.

Computed, and nothing else is allowed to flip PASS:

- Primary. Paired necessity on target-only keyword recall. MPID +0.05.
- Primary. Paired necessity on retrieval wins (closer to the item's own
  leave-one-out document centroid than to the next document's). The protocol
  names no third MPID; the proportion MPID +0.05 is used, and that choice is
  recorded on the result rather than left implicit.
- Supporting. MiniLM cosine to the leave-one-out document centroid, as a
  paired difference only, MPID +0.03. An absolute 0.40 comparison is refused.
- Fidelity. Mean cosine of the true-draft arm ≥ 0.40, a precondition that the
  output contains content. It is not evidence the draft caused it. If the
  no-draft arm already sits above 0.40, ``absolute_gate_is_valid`` is false
  and the floor cannot be cited as a result (FINDINGS F38).
- Format control. The no-draft arm "does not match" when the paired
  (true − none) recall interval has lower bound > 0. A tie, or a no-draft arm
  that wins, is not a PASS.

PASS iff both primary lower bounds exceed their MPID and the no-draft arm
does not match, and only when the positive control and the null pair behave
and the run is the pre-registered 192 × 3 design. Otherwise the headline is
``HARNESS_BROKEN`` or ``NOT_MEASURED``. A smoke run never calls G3.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from experiments.e4.metrics import entity_recall
from experiments.e4.necessity import (
    absolute_gate_is_valid,
    combine_endpoints,
    evaluate_necessity,
    fidelity_gate,
)
from experiments.protocol.stats import cluster_bootstrap_ci

MPID_RECALL = 0.05
MPID_RETRIEVAL = 0.05
MPID_COSINE = 0.03
FIDELITY_THRESHOLD = 0.40

# Where this file is deliberately narrower or more explicit than §7.3.
PROTOCOL_DIFFERENCES = (
    "Retrieval win-rate has no numeric MPID in PROTOCOL §7.3. The proportion "
    "MPID +0.05 (the one named for absolute recall) is applied to both primary "
    "proportion endpoints. Cosine keeps +0.03.",
    "Cosine necessity is reported and is not in the PASS conjunction. §7.3 "
    "calls it supporting and forbids the absolute 0.40 gate. G3's 'both "
    "endpoints' is read as the two N1 primary endpoints (recall and retrieval).",
    "Fidelity ≥ 0.40 voids the call when it fails (no content, so no question). "
    "It never produces PASS. If the no-draft mean is already ≥ 0.40 the floor "
    "is marked non-discriminative (F38) and is not cited as evidence.",
    "'The no-draft arm does not match' means the paired (true − none) recall "
    "interval has lower bound > 0. A no-draft arm that scores higher is a "
    "match-failure in the wrong direction and is not a PASS.",
    "The text-prefix positive control is judged on paired MiniLM cosine, the "
    "quantity F42 measured. Target-only recall is the wrong readout for it: "
    "the gold keywords are absent from the prefix by construction.",
    "The null pair is a paired or cluster difference of two fresh sampled "
    "decodes on disjoint items. Its cosine interval must contain zero and "
    "must not be a point (N2). A point, or an interval that excludes zero, "
    "means the harness is broken and no scientific verdict is reported.",
)


class AbsoluteCosineGateRefused(RuntimeError):
    """Raised when a caller asks cosine to act as an absolute gate."""


def absolute_cosine_gate(*_args, **_kwargs):
    """The deleted gate. Calling it is a bug, not a result.

    FINDINGS F38: the unsteered null was 0.445, above the 0.40 threshold, so
    the gate could not fail. MiniLM cosine is a paired difference only.
    """
    raise AbsoluteCosineGateRefused(
        "absolute MiniLM cosine gate is refused (FINDINGS F38). "
        "Cosine enters E4-S1 only as a paired difference. "
        "The 0.40 fidelity floor is a content precondition, not this gate."
    )


def _as_float(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-d score vector, got {a.shape}")
    return a


def _finite(name: str, x: np.ndarray) -> None:
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite value in {name} (VOID)")


def retrieval_wins(
    gen_emb: np.ndarray, own_centroid: np.ndarray, other_centroid: np.ndarray
) -> np.ndarray:
    """1 when the generation is closer to its own document than to the next.

    Ties lose. The comparison is strict so a constant embedding does not win
    by equality, and the rule does not draw a random foil (N2).
    """
    g = _unit_rows(gen_emb)
    own = _unit_rows(own_centroid)
    other = _unit_rows(other_centroid)
    cos_own = np.sum(g * own, axis=1)
    cos_other = np.sum(g * other, axis=1)
    return (cos_own > cos_other).astype(np.float64)


def cosine_to_centroid(gen_emb: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    g = _unit_rows(gen_emb)
    c = _unit_rows(centroid)
    return np.sum(g * c, axis=1).astype(np.float64)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-norm embedding")
    return a / n


def keyword_recalls(texts: Sequence[str], gold: Sequence[Sequence[str]]) -> np.ndarray:
    if len(texts) != len(gold):
        raise ValueError("texts and gold lists must align")
    return np.asarray(
        [entity_recall(t, list(g)) for t, g in zip(texts, gold)], dtype=np.float64
    )


def _necessity(name, true, shuf, clusters, mpid, n_boot, seed):
    return evaluate_necessity(
        name, true, shuf, clusters, mpid, n_boot=n_boot, seed=seed
    )


def _paired_null(a, b, clusters, n_boot, seed) -> dict:
    """CI of the paired difference. A point (zero width) is a broken null."""
    diff = _as_float(a) - _as_float(b)
    _finite("null difference", diff)
    summary = cluster_bootstrap_ci(diff, clusters, n_boot=n_boot, seed=seed)
    width = float(summary.ci_high - summary.ci_low)
    contains = bool(summary.ci_low <= 0.0 <= summary.ci_high)
    is_point = bool(width <= 1e-12)
    return {
        "difference": float(summary.mean),
        "ci_low": float(summary.ci_low),
        "ci_high": float(summary.ci_high),
        "width": width,
        "contains_zero": contains,
        "is_point": is_point,
        "n": int(diff.size),
        "n_clusters": int(np.unique(clusters).size),
        "behaves": bool(contains and not is_point),
    }


def _unpaired_null(a, ca, b, cb, n_boot, seed) -> dict:
    """Difference of means when the two arms do not share documents."""
    a = _as_float(a)
    b = _as_float(b)
    _finite("null_a", a)
    _finite("null_b", b)
    rng = np.random.default_rng(seed)
    ca = np.asarray(ca)
    cb = np.asarray(cb)
    groups_a = _groups(a, ca)
    groups_b = _groups(b, cb)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ma = _resample_mean(groups_a, rng)
        mb = _resample_mean(groups_b, rng)
        means[i] = ma - mb
    lo, hi = np.quantile(means, [0.025, 0.975])
    diff = float(a.mean() - b.mean())
    width = float(hi - lo)
    contains = bool(lo <= 0.0 <= hi)
    is_point = bool(width <= 1e-12)
    return {
        "difference": diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "width": width,
        "contains_zero": contains,
        "is_point": is_point,
        "n": int(a.size + b.size),
        "n_clusters": int(np.unique(ca).size + np.unique(cb).size),
        "behaves": bool(contains and not is_point),
        "paired": False,
    }


def _groups(values: np.ndarray, clusters: np.ndarray) -> list[np.ndarray]:
    uniq, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(uniq.size + 1))
    return [values[order[bounds[i] : bounds[i + 1]]] for i in range(uniq.size)]


def _resample_mean(groups: list[np.ndarray], rng: np.random.Generator) -> float:
    pick = rng.integers(0, len(groups), size=len(groups))
    return float(np.concatenate([groups[i] for i in pick]).mean())


def assess_positive(true_cos, rand_cos, clusters, *, n_boot: int, seed: int) -> dict:
    """The readout control. A real difference has lower bound above zero."""
    res = _necessity(
        "text_prefix_cosine", true_cos, rand_cos, clusters, mpid=0.0, n_boot=n_boot, seed=seed
    )
    width = float(res.ci_high - res.ci_low)
    behaves = bool(res.ci_low > 0.0 and res.difference > 0.0 and width > 1e-12)
    return {
        "endpoint": "minilm_cosine",
        "difference": res.difference,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "width": width,
        "true_mean": res.true_mean,
        "random_mean": res.shuffled_mean,
        "n": res.n,
        "reference_f42": {"difference": 0.186, "ci_low": 0.109, "ci_high": 0.263},
        "behaves": behaves,
    }


def assess_null(a, b, clusters_a, clusters_b=None, *, n_boot: int, seed: int) -> dict:
    """Noise floor. Must cover zero and must have width."""
    if clusters_b is None or (
        len(a) == len(b) and np.array_equal(np.asarray(clusters_a), np.asarray(clusters_b))
    ):
        out = _paired_null(a, b, clusters_a, n_boot, seed)
        out["paired"] = True
        return out
    return _unpaired_null(a, clusters_a, b, clusters_b, n_boot, seed)


def decide_g3(
    *,
    recall_true,
    recall_shuf,
    recall_none,
    win_true,
    win_shuf,
    cos_true,
    cos_shuf,
    cos_none,
    clusters,
    pos_true,
    pos_rand,
    pos_clusters,
    null_a,
    null_b,
    null_clusters,
    null_clusters_b=None,
    n_boot: int = 2_000,
    seed: int = 0,
    confirmatory: bool = False,
) -> dict:
    """Stand-behind verdict plus every number that produced it.

    ``confirmatory`` is true only for the pre-registered 192 × 3 run. A smoke
    records its arithmetic and reports ``NOT_MEASURED``.
    """
    recall_true = _as_float(recall_true)
    recall_shuf = _as_float(recall_shuf)
    recall_none = _as_float(recall_none)
    win_true = _as_float(win_true)
    win_shuf = _as_float(win_shuf)
    cos_true = _as_float(cos_true)
    cos_shuf = _as_float(cos_shuf)
    cos_none = _as_float(cos_none)
    pos_true = _as_float(pos_true)
    pos_rand = _as_float(pos_rand)
    null_a = _as_float(null_a)
    null_b = _as_float(null_b)

    positive = assess_positive(
        pos_true, pos_rand, pos_clusters, n_boot=n_boot, seed=seed
    )
    null = assess_null(
        null_a, null_b, null_clusters, null_clusters_b, n_boot=n_boot, seed=seed
    )
    controls_ok = bool(positive["behaves"] and null["behaves"])

    base = {
        "experiment": "E4-S1",
        "gate": "G3",
        "question": "Q1",
        "mpid": {
            "entity_recall": MPID_RECALL,
            "retrieval_win": MPID_RETRIEVAL,
            "minilm_cosine": MPID_COSINE,
        },
        "protocol_differences": list(PROTOCOL_DIFFERENCES),
        "cosine_absolute_gate": "refused",
        "controls": {"positive": positive, "null": null, "ok": controls_ok},
        "confirmatory": bool(confirmatory),
    }

    if not controls_ok:
        failed = []
        if not positive["behaves"]:
            failed.append("positive control (text-prefix true vs random) showed no real difference")
        if not null["behaves"]:
            failed.append("null pair CI does not contain zero or is a point")
        base.update(
            verdict="HARNESS_BROKEN",
            trustworthy=False,
            reason="; ".join(failed),
            arithmetic_verdict=None,
        )
        return base

    try:
        for name, arr in (
            ("recall_true", recall_true),
            ("recall_shuf", recall_shuf),
            ("recall_none", recall_none),
            ("win_true", win_true),
            ("win_shuf", win_shuf),
            ("cos_true", cos_true),
            ("cos_shuf", cos_shuf),
            ("cos_none", cos_none),
        ):
            _finite(name, arr)
    except ValueError as exc:
        base.update(
            verdict="VOID",
            trustworthy=False,
            reason=str(exc),
            arithmetic_verdict="VOID",
        )
        return base

    recall = _necessity(
        "entity_recall", recall_true, recall_shuf, clusters, MPID_RECALL, n_boot, seed
    )
    retrieval = _necessity(
        "retrieval_win", win_true, win_shuf, clusters, MPID_RETRIEVAL, n_boot, seed
    )
    cosine = _necessity(
        "minilm_cosine", cos_true, cos_shuf, clusters, MPID_COSINE, n_boot, seed
    )
    no_draft = _necessity(
        "true_minus_none_recall", recall_true, recall_none, clusters, 0.0, n_boot, seed
    )
    # Lower bound > 0 means the true draft beats the empty prefix. Anything
    # else — a tie, or the empty prefix winning — is "the no-draft arm matches"
    # for the purpose of refusing PASS.
    no_draft_separated = bool(no_draft.ci_low > 0.0)

    fid_mean = float(np.mean(cos_true))
    none_mean = float(np.mean(cos_none))
    fid_pass = bool(fidelity_gate(fid_mean, FIDELITY_THRESHOLD))
    gate_valid = bool(absolute_gate_is_valid(FIDELITY_THRESHOLD, none_mean))
    fidelity = {
        "threshold": FIDELITY_THRESHOLD,
        "true_mean_cosine": fid_mean,
        "none_mean_cosine": none_mean,
        "passed": fid_pass,
        "absolute_gate_valid": gate_valid,
        "used_as_pass_criterion": False,
        "note": (
            "Content floor only. PASS is never awarded because this mean "
            "cleared 0.40. absolute_gate_valid is false when the no-draft arm "
            "already exceeds the floor (F38)."
        ),
    }

    primary = combine_endpoints([recall, retrieval])
    if not fid_pass:
        arithmetic = "VOID"
        reason = (
            f"fidelity {fid_mean:.3f} < {FIDELITY_THRESHOLD:.2f}; the output "
            "does not contain enough content to ask whether the draft caused it"
        )
    elif primary == "PASS" and no_draft_separated:
        arithmetic = "PASS"
        reason = (
            "recall and retrieval necessity lower bounds exceed MPID, and the "
            "no-draft arm does not match"
        )
    elif primary == "PASS" and not no_draft_separated:
        arithmetic = "FAIL"
        reason = (
            "necessity cleared MPID but the no-draft arm matches the true draft, "
            "so the channel is not load-bearing"
        )
    elif primary == "FAIL":
        arithmetic = "FAIL"
        reason = "necessity interval lies entirely below the MPID on both primary endpoints"
    elif primary == "VOID":
        arithmetic = "VOID"
        reason = "a primary endpoint interval was non-finite"
    else:
        arithmetic = "INCONCLUSIVE"
        reason = (
            f"primary combination is {primary}; a CI straddles its MPID, "
            "or the two endpoints disagree"
        )

    if not confirmatory:
        verdict = "NOT_MEASURED"
        reason = (
            "scale is not the pre-registered 192 items × 3 adapter fits; "
            f"arithmetic verdict is {arithmetic} and is not a G3 call. " + reason
        )
        trustworthy = False
    else:
        verdict = arithmetic
        trustworthy = arithmetic in {"PASS", "FAIL", "INCONCLUSIVE"}

    base.update(
        verdict=verdict,
        arithmetic_verdict=arithmetic,
        trustworthy=trustworthy,
        reason=reason,
        no_draft_matches=not no_draft_separated,
        no_draft=no_draft.as_dict(),
        fidelity=fidelity,
        endpoints={
            "entity_recall": recall.as_dict(),
            "retrieval_win": retrieval.as_dict(),
            "minilm_cosine": cosine.as_dict(),
        },
    )
    return base


def embed_documents(texts: Sequence[str], embed_fn) -> np.ndarray:
    """Mean of the two halves, so a long context is not only its first 256 tokens."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        raw = (text or " ").strip() or " "
        if len(raw) < 600:
            parts = [raw]
        else:
            mid = len(raw) // 2
            parts = [raw[:mid], raw[mid:]]
        spans.append((len(pieces), len(pieces) + len(parts)))
        pieces.extend(parts)
    emb = np.asarray(embed_fn(pieces), dtype=np.float64)
    if emb.shape[0] != len(pieces):
        raise ValueError("embedder returned the wrong number of rows")
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float64)
    for i, (a, b) in enumerate(spans):
        m = emb[a:b].mean(axis=0)
        n = float(np.linalg.norm(m))
        out[i] = m / max(n, 1e-12)
    return out


def _rows(records: Sequence[dict], arm: str) -> list[dict]:
    rows = [r for r in records if r["arm"] == arm]
    rows.sort(key=lambda r: (int(r["item_index"]), int(r["fit"])))
    return rows


def _align_key(rows: list[dict]) -> list[tuple[int, int]]:
    return [(int(r["item_index"]), int(r["fit"])) for r in rows]


def score_and_decide(
    records: Sequence[dict],
    embed_fn,
    *,
    confirmatory: bool = False,
    n_boot: int = 2_000,
    seed: int = 0,
) -> dict:
    """Score generations and apply :func:`decide_g3`.

    Each record needs ``arm``, ``fit``, ``item_index``, ``doc_id``, ``text``,
    ``gold`` and ``context``. Latent arms (true, shuf, none) must align on
    ``(item_index, fit)``. Cosine and retrieval use the leave-one-out context
    the record carries (the document with the target span removed). The foil
    for retrieval is the next document in sorted ``doc_id`` order among the
    true-draft rows.
    """
    true = _rows(records, "true")
    shuf = _rows(records, "shuf")
    none = _rows(records, "none")
    if not true:
        raise ValueError("no true-draft generations")
    key = _align_key(true)
    if _align_key(shuf) != key or _align_key(none) != key:
        raise ValueError("true, shuffled and no-draft rows do not align")

    contexts = [r["context"] for r in true]
    centroids = embed_documents(contexts, embed_fn)
    doc_ids = [str(r["doc_id"]) for r in true]
    # One centroid per (item, fit). Documents repeat across fits; the context
    # is the same text, so the centroid matches. Foils are by document.
    uniq = sorted(set(doc_ids))
    if len(uniq) < 2:
        raise ValueError("need at least 2 documents")
    doc_centroid: dict[str, np.ndarray] = {}
    for doc, cen in zip(doc_ids, centroids):
        doc_centroid.setdefault(doc, cen)
    foil = {doc: uniq[(i + 1) % len(uniq)] for i, doc in enumerate(uniq)}

    def pack(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        gold = [tuple(r["gold"]) for r in rows]
        recall = keyword_recalls([r["text"] for r in rows], gold)
        gen = np.asarray(embed_fn([r["text"] or " " for r in rows]), dtype=np.float64)
        own = np.stack([doc_centroid[str(r["doc_id"])] for r in rows])
        other = np.stack([doc_centroid[foil[str(r["doc_id"])]] for r in rows])
        cos = cosine_to_centroid(gen, own)
        wins = retrieval_wins(gen, own, other)
        clusters = [str(r["doc_id"]) for r in rows]
        return recall, wins, cos, clusters

    recall_t, win_t, cos_t, clusters = pack(true)
    recall_s, win_s, cos_s, _ = pack(shuf)
    recall_n, _win_n, cos_n, _ = pack(none)

    pos_t = _rows(records, "textpc_true")
    pos_r = _rows(records, "textpc_rand")
    if _align_key(pos_t) != _align_key(pos_r):
        raise ValueError("text-prefix arms do not align")
    pos_gen_t = np.asarray(embed_fn([r["text"] or " " for r in pos_t]), dtype=np.float64)
    pos_gen_r = np.asarray(embed_fn([r["text"] or " " for r in pos_r]), dtype=np.float64)
    pos_own = np.stack([doc_centroid[str(r["doc_id"])] for r in pos_t])
    pos_true = cosine_to_centroid(pos_gen_t, pos_own)
    pos_rand = cosine_to_centroid(pos_gen_r, pos_own)
    pos_clusters = [str(r["doc_id"]) for r in pos_t]

    null_a_rows = _rows(records, "null_a")
    null_b_rows = _rows(records, "null_b")
    null_emb_a = embed_documents([r["context"] for r in null_a_rows], embed_fn)
    null_emb_b = embed_documents([r["context"] for r in null_b_rows], embed_fn)
    gen_a = np.asarray(embed_fn([r["text"] or " " for r in null_a_rows]), dtype=np.float64)
    gen_b = np.asarray(embed_fn([r["text"] or " " for r in null_b_rows]), dtype=np.float64)
    null_a = cosine_to_centroid(gen_a, null_emb_a)
    null_b = cosine_to_centroid(gen_b, null_emb_b)
    null_ca = [str(r["doc_id"]) for r in null_a_rows]
    null_cb = [str(r["doc_id"]) for r in null_b_rows]

    result = decide_g3(
        recall_true=recall_t,
        recall_shuf=recall_s,
        recall_none=recall_n,
        win_true=win_t,
        win_shuf=win_s,
        cos_true=cos_t,
        cos_shuf=cos_s,
        cos_none=cos_n,
        clusters=clusters,
        pos_true=pos_true,
        pos_rand=pos_rand,
        pos_clusters=pos_clusters,
        null_a=null_a,
        null_b=null_b,
        null_clusters=null_ca,
        null_clusters_b=null_cb,
        n_boot=n_boot,
        seed=seed,
        confirmatory=confirmatory,
    )
    result["n_records"] = len(records)
    return result
