"""Z-bank drift audit for E4-S1.

``artifacts/corpus_v2/zbank_meta.json`` records ``drift_mean`` 0.7725 against
the legacy band [0.655, 0.755] and ``drift_in_band: false``. That bank is the
frozen JEPA encoder's paragraph centroids on WikiText corpus-v2. E4-S1's drafts
are encodings of harvested bioRxiv documents, not rows looked up from this
bank. The audit still has to be run before those encodings are used, because
the encoder is the same one and the flag says its calibration is out of band.

What the numbers mean
---------------------
``drift_mean`` is the mean cosine of *consecutive* paragraphs. The legacy band
is the mixed-corpus value this encoder was originally checked against. A mean
above the band on Wikipedia is topical persistence (FINDINGS F8), not by itself
an arm-specific bias.

Paired contrasts
----------------
E4-S1 necessity is ``score(true draft) - score(shuffled draft)`` on entity
recall, retrieval wins, and MiniLM cosine. None of those scores read the JEPA
cosine. Every latent arm is produced by one frozen encoder, so a global
anisotropy is common mode: it cannot make the true arm look better than the
shuffled arm unless true and shuffled vectors actually differ. The failure
mode of a too-narrow cone is the opposite one — necessity driven toward zero.
The audit reports the consecutive-minus-cross-document gap so that claim is a
measurement, not an assumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# The band written into zbank_meta.json. It is the legacy mixed-corpus check,
# not a property of the harvested bioRxiv documents.
LEGACY_BAND = (0.655, 0.755)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = _ROOT / "artifacts" / "corpus_v2" / "zbank.npz"
DEFAULT_META = _ROOT / "artifacts" / "corpus_v2" / "zbank_meta.json"


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def audit_zbank(
    bank_path: str | Path = DEFAULT_BANK,
    meta_path: str | Path = DEFAULT_META,
    *,
    n_cross: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Recompute consecutive and cross-document cosine from the stored bank.

    This does not re-encode. It checks that the recorded drift is the drift of
    the vectors on disk, and it measures how far a random other document sits
    from a paragraph. That gap is what a shuffled draft can still see.
    """
    bank_path = Path(bank_path)
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bank = np.load(bank_path)
    z = bank["paragraph_z"]
    doc = bank["doc_id"]
    para = bank["para_idx"]

    same = doc[:-1] == doc[1:]
    nxt = para[:-1] + 1 == para[1:]
    mask = same & nxt
    a = _unit_rows(z[:-1][mask].astype(np.float32))
    b = _unit_rows(z[1:][mask].astype(np.float32))
    consec = np.sum(a * b, axis=1).astype(np.float64)

    stored = None
    if "consec_pair_cos" in bank.files:
        stored = np.asarray(bank["consec_pair_cos"], dtype=np.float64)

    rng = np.random.default_rng(seed)
    i = rng.integers(0, z.shape[0], size=n_cross * 2)
    j = rng.integers(0, z.shape[0], size=n_cross * 2)
    ok = doc[i] != doc[j]
    i, j = i[ok][:n_cross], j[ok][:n_cross]
    cross = np.sum(
        _unit_rows(z[i].astype(np.float32)) * _unit_rows(z[j].astype(np.float32)),
        axis=1,
    ).astype(np.float64)

    drift_mean = float(consec.mean())
    drift_std = float(consec.std())
    cross_mean = float(cross.mean())
    lo, hi = LEGACY_BAND
    gap = drift_mean - cross_mean
    # A collapsed encoder has consecutive ≈ cross. The shuffled arm is then
    # the same vector as the true arm and necessity is zero by construction.
    separable = bool(gap > 0.02)

    report: dict[str, Any] = {
        "bank": str(bank_path),
        "n_paragraphs": int(z.shape[0]),
        "n_docs": int(np.unique(doc).size),
        "dim": int(z.shape[1]),
        "n_consecutive_pairs": int(consec.size),
        "drift_mean": drift_mean,
        "drift_std": drift_std,
        "drift_p05": float(np.quantile(consec, 0.05)),
        "drift_p50": float(np.quantile(consec, 0.50)),
        "drift_p95": float(np.quantile(consec, 0.95)),
        "meta_drift_mean": float(meta.get("drift_mean")),
        "meta_matches_recompute": bool(
            abs(drift_mean - float(meta.get("drift_mean", -1))) < 1e-4
        ),
        "stored_consec_matches": (
            None
            if stored is None
            else bool(
                stored.shape == consec.shape
                and float(np.max(np.abs(stored - consec))) < 1e-3
            )
        ),
        "legacy_band": [lo, hi],
        "drift_in_band": bool(lo <= drift_mean <= hi),
        "drift_above_band": float(drift_mean - hi),
        "cross_doc_mean": cross_mean,
        "cross_doc_std": float(cross.std()),
        "n_cross_pairs": int(cross.size),
        "consecutive_minus_cross": gap,
        "drafts_separable": separable,
        "used_as_e4_draft_source": False,
        "affects_paired_contrasts": False,
        "reason": _reason(drift_mean, cross_mean, gap, separable),
    }
    return report


def _reason(drift_mean: float, cross_mean: float, gap: float, separable: bool) -> str:
    lo, hi = LEGACY_BAND
    return (
        f"Recomputed consecutive-paragraph cosine is {drift_mean:.4f}, "
        f"{drift_mean - hi:.4f} above the legacy band [{lo:.3f}, {hi:.3f}]. "
        f"Cross-document pairs sit at {cross_mean:.4f} "
        f"(gap {gap:+.4f}). "
        + (
            "The gap is large enough that a shuffled draft is not the same "
            "vector as the true draft, so the out-of-band flag is topical "
            "persistence of the wiki bank, not encoder collapse. "
            if separable
            else "The gap is small enough that true and shuffled drafts may "
            "be interchangeable. That would push necessity toward zero; it "
            "cannot manufacture a positive necessity. "
        )
        + "E4-S1 endpoints do not read this cosine: entity recall is lexical "
        "and the continuous endpoint is MiniLM. Every latent arm of one item "
        "is encoded by the same frozen encoder, so a global anisotropy is "
        "common mode in the paired difference. The wiki bank is not the draft "
        "source for harvested documents; those drafts are encoded fresh. "
        "The out-of-band drift therefore does not enter the paired contrast "
        "as an arm-specific bias, and it is not a reason to call PASS."
    )


def write_audit(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def encode_sample_drift(
    documents: list[dict[str, Any]],
    *,
    jepa_path: str | Path | None = None,
    n_docs: int = 8,
    paragraphs_per_doc: int = 4,
    device: str = "cpu",
) -> dict[str, Any]:
    """Encode a few harvested documents with the current JEPA weights.

    This is the "from the current encoder" check. It is deliberately small:
    the full bank already records what this checkpoint did to wiki text, and
    re-encoding 262k paragraphs is a GPU job this audit does not need.
    """
    import sys

    import torch

    root = _ROOT
    for p in (str(root / "src"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from conceptpoc.codec.jepa_codec import JepaCodec
    from conceptpoc.persistence import load_joint

    from experiments.e4.data import chunk_document, pool_texts

    jepa = Path(jepa_path) if jepa_path else root / "artifacts" / "jepa_repair_1m" / "jepa" / "joint.pt"
    model, tok = load_joint(str(jepa), device=device)
    codec = JepaCodec(model, tok, max_tokens=128, device=device)

    vectors: list[np.ndarray] = []
    doc_ids: list[str] = []
    used = 0
    for doc in documents:
        chunks = [text for _, text in chunk_document(doc)]
        if len(chunks) < paragraphs_per_doc:
            continue
        for text in chunks[:paragraphs_per_doc]:
            vectors.append(pool_texts(codec.encode, text))
            doc_ids.append(str(doc["doc_id"]))
        used += 1
        if used >= n_docs:
            break

    del codec, model, tok
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if used < 2:
        raise RuntimeError(f"only {used} documents had enough paragraphs to encode")

    z = _unit_rows(np.stack(vectors))
    ids = np.asarray(doc_ids)
    consec: list[float] = []
    for i in range(len(z) - 1):
        if ids[i] == ids[i + 1]:
            consec.append(float(z[i] @ z[i + 1]))
    # One cross pair per consecutive pair, from a different document.
    rng = np.random.default_rng(0)
    cross: list[float] = []
    for i in range(len(z)):
        choices = np.flatnonzero(ids != ids[i])
        if choices.size == 0:
            continue
        j = int(choices[int(rng.integers(0, choices.size))])
        cross.append(float(z[i] @ z[j]))

    c = np.asarray(consec, dtype=np.float64)
    x = np.asarray(cross, dtype=np.float64)
    lo, hi = LEGACY_BAND
    mean = float(c.mean())
    return {
        "source": "current JEPA encoder on harvested documents",
        "jepa": str(jepa),
        "device": device,
        "n_docs": used,
        "n_paragraphs": int(z.shape[0]),
        "n_consecutive_pairs": int(c.size),
        "drift_mean": mean,
        "drift_std": float(c.std()) if c.size else float("nan"),
        "cross_doc_mean": float(x.mean()) if x.size else float("nan"),
        "consecutive_minus_cross": float(mean - x.mean()) if x.size else float("nan"),
        "legacy_band": [lo, hi],
        "drift_in_band": bool(lo <= mean <= hi),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Audit zbank drift before E4-S1")
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--out", type=Path, default=_ROOT / "artifacts" / "e4s1" / "zbank_drift_audit.json")
    p.add_argument("--corpus", type=Path, default=_ROOT / "artifacts" / "corpus" / "biorxiv_2026.jsonl")
    p.add_argument("--encode-sample", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    report = audit_zbank(args.bank, args.meta)
    if args.encode_sample:
        from experiments.e4.data import load_documents

        docs = load_documents(args.corpus, min_chars=4000)
        report["harvest_sample"] = encode_sample_drift(docs, device=args.device)
    write_audit(args.out, report)
    print(json.dumps({k: report[k] for k in ("drift_mean", "drift_in_band", "cross_doc_mean", "consecutive_minus_cross", "affects_paired_contrasts")}, indent=2))


if __name__ == "__main__":
    main()
