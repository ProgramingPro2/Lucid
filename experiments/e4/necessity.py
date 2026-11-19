"""Necessity: the only quantity that decides whether a latent is load-bearing.

necessary = score(channel present) - score(channel ablated)

Docs/PROTOCOL.md §7.3 fixes the endpoints and the MPID; this module implements
the arithmetic, the clustered interval, and the two guards that stop the repo's
two historical mistakes from recurring.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from experiments.protocol.stats import Summary, cluster_bootstrap_ci, decide


@dataclass(frozen=True)
class NecessityResult:
    endpoint: str
    true_mean: float
    shuffled_mean: float
    difference: float
    ci_low: float
    ci_high: float
    mpid: float
    verdict: str
    n: int
    n_clusters: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_necessity(
    endpoint: str,
    true_scores,
    shuffled_scores,
    clusters,
    mpid: float,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> NecessityResult:
    """Paired necessity with a cluster bootstrap and the pre-registered rule."""
    t = np.asarray(true_scores, dtype=float)
    s = np.asarray(shuffled_scores, dtype=float)
    if t.shape != s.shape:
        raise ValueError("true and shuffled scores must align")
    diffs = t - s
    ci: Summary = cluster_bootstrap_ci(diffs, clusters, n_boot=n_boot, seed=seed)
    verdict = decide(ci, mpid)["verdict"]
    return NecessityResult(
        endpoint=endpoint,
        true_mean=float(t.mean()),
        shuffled_mean=float(s.mean()),
        difference=float(diffs.mean()),
        ci_low=float(ci.ci_low),
        ci_high=float(ci.ci_high),
        mpid=float(mpid),
        verdict=verdict,
        n=int(t.size),
        n_clusters=int(np.unique(clusters).size),
    )


def fidelity_gate(mean_cosine: float, threshold: float = 0.40) -> bool:
    """Does the output contain content at all, before asking about necessity?

    This is a floor on *fidelity*, not evidence of conditioning. A model that
    reproduces the draft verbatim is a copier; a model that ignores it and writes
    plausible prose fails here. Neither statement is about causality.
    """
    return bool(np.isfinite(mean_cosine) and mean_cosine >= threshold)


def absolute_gate_is_valid(threshold: float, null_mean: float) -> bool:
    """Guard against F38: a threshold below the null cannot fail.

    Returns False when the gate would be passed by doing nothing.
    """
    return bool(threshold > null_mean)


def combine_endpoints(results: list[NecessityResult], rule: str = "all") -> str:
    """Aggregate per-endpoint verdicts. PASS only if every endpoint passes."""
    if not results:
        raise ValueError("no endpoints")
    verdicts = [r.verdict for r in results]
    if rule != "all":
        raise ValueError("only rule='all' is pre-registered")
    if any(v == "VOID" for v in verdicts):
        return "VOID"
    if all(v == "PASS" for v in verdicts):
        return "PASS"
    if all(v == "FAIL" for v in verdicts):
        return "FAIL"
    return "INCONCLUSIVE"
