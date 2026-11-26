"""Score harvested documents with the base model's own NLL, and screen them.

The memorisation screen (docs/CORPUS.md §4 step 7) drops documents the base
model finds implausibly easy. The contamination probe (docs/CORPUS.md §6) is a
separate measurement: the gap between post-cutoff prose and a reference, with
a CI. Section 6 is explicit that no threshold would make a leaked corpus
acceptable. An NLL above the floor does not prove the document was absent
from training.

Sampling, the absolute floor, and the bottom-decile quantile are defined once,
here. ``build`` and ``contamination`` import them. They are not restated at
the call site.

A full bioRxiv paper in the live file is on the order of 17k tokens. Scoring
every token of every document is a long GPU pass and is not what a mean-NLL
screen needs. Each document is reduced to at most :data:`N_WINDOWS` evenly
spaced windows of :data:`WINDOW_TOKENS` (2048 tokens, about an eighth of a
median paper), placed from the start of the document to the end so the
abstract is not the whole score. Short documents are one window. There is no
RNG, so the same text always yields the same windows.

The decision is per document against :data:`MEMORISATION_NLL_FLOOR`, then a
per-corpus bottom decile on whatever survived the floor (and, when the table
supplies one, a reference fence). The decile always discards about a tenth of
a clean corpus. It cannot reject a corpus that is contaminated in the bulk.
The floor can: a document at 0.01 nats is rejected even when it is the only
low score, and a run of non-finite scores is INVALID rather than a pass.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.corpus.nll \
        --docs-jsonl artifacts/corpus/biorxiv_2026.jsonl \
        --out artifacts/corpus/nll_qwen3_1.7b.json \
        --limit 50
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Policy. One place.
# --------------------------------------------------------------------------

# 512-token windows keep the activation footprint small on a shared 24 GB
# card (batch size 1, no training). Four of them cover the document instead
# of scoring one span at the front.
WINDOW_TOKENS = 512
N_WINDOWS = 4

# Natural-log cross-entropy. 1.0 nat is a geometric-mean next-token
# probability of exp(-1) ≈ 0.37, sustained across every scored window.
# The 2026-11 pilot on this corpus family (artifacts/corpus/nll_pilot.json,
# 40 documents, 1024×4 windows) ranges from 1.43 to 3.07 nats/token. The
# floor sits below that whole band, so it does not trim ordinary manuscripts.
# It rejects the near-deterministic regime. It is not fitted to the corpus
# under test, and text above it is not certified clean.
MEMORISATION_NLL_FLOOR = 1.0

# docs/CORPUS.md §4 step 7: "Drop the bottom decile."
BOTTOM_DECILE = 0.10

# Robust z for a reference fence supplied by the contamination probe.
# The screen never estimates this fence from the documents it is judging.
REFERENCE_Z = 3.0
MAD_TO_SIGMA = 1.4826

# Below this much free GPU memory the 1.7B bf16 weights do not fit beside
# the other tenants. 5120 MiB is 5 GiB. The scorer then uses CPU.
MIN_FREE_MIB = 5120

CRITERION_SOURCE = (
    "docs/CORPUS.md §4 step 7 (per-document base-model NLL; drop the bottom "
    "decile; record the distribution) and §6 (the contamination figure is a "
    "gap against a reference, not a threshold that can certify a corpus)"
)

_DECILE_QS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class InvalidNLLScreen(RuntimeError):
    """A non-finite NLL makes the memorisation screen INVALID.

    The run is rejected. NaN and Infinity must not be dropped quietly and
    must not be treated as a pass (docs/FINDINGS.md F27).
    """


def sampling_rule() -> dict:
    """The subsample the scorer and the card both cite."""
    return {
        "window_tokens": WINDOW_TOKENS,
        "n_windows": N_WINDOWS,
        "max_tokens_scored_per_document": WINDOW_TOKENS * N_WINDOWS,
        "placement": (
            "up to n_windows evenly spaced windows covering the token span; "
            "a document shorter than one window is scored whole; deterministic, "
            "no RNG"
        ),
        "unit": "per-document token-weighted mean NLL, nats/token",
        "decision": (
            "per document, against MEMORISATION_NLL_FLOOR"
            + (
                " and against a reference fence when the NLL table supplies "
                "one that was computed on held-out text"
            )
            + "; then the bottom decile of the survivors, which is a "
            "per-corpus rank trim and is not itself a contamination test"
        ),
    }


def decide_device(free_mib: int | None, requested: str = "auto") -> str:
    """Pick cuda only when the card has at least :data:`MIN_FREE_MIB` free.

    ``requested`` other than ``"auto"`` is still subject to the free-memory
    check. Loading into a nearly full card is how a second scorer evicts
    someone else's process.
    """
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device request {requested!r}")
    if requested == "cpu":
        return "cpu"
    if free_mib is None or free_mib < MIN_FREE_MIB:
        return "cpu"
    return "cuda"


def gpu_free_mib() -> int | None:
    """Free GPU memory in MiB, from ``nvidia-smi``, or None if it cannot be read.

    This is the figure that includes other processes. ``torch``'s own cache
    does not.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = out.strip().splitlines()
    if not line:
        return None
    token = line[0].strip().split()[0]
    try:
        return int(token)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def token_windows(ids: list[int], length: int, n_windows: int = N_WINDOWS) -> list[list[int]]:
    """Up to ``n_windows`` windows of ``length``.

    Short documents are one window. Long documents are sampled evenly from start
    to end, so methods and discussion are both represented. Deterministic: no
    RNG, so the table can be regenerated exactly.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if len(ids) <= length:
        return [ids]
    # Do not ask for more windows than there are distinct start positions; a
    # short document would otherwise score the same tokens several times.
    n = max(1, min(n_windows, len(ids) - length + 1))
    if n == 1:
        return [ids[:length]]
    starts = np.linspace(0, len(ids) - length, num=n, dtype=int)
    return [ids[int(s) : int(s) + length] for s in starts]


def _usable_windows(
    ids: list[int], length: int, n_windows: int
) -> list[list[int]]:
    return [w for w in token_windows(ids, length, n_windows) if len(w) > 1]


def scored_token_count(
    ids: list[int], length: int = WINDOW_TOKENS, n_windows: int = N_WINDOWS
) -> int:
    """Predicted tokens (the length-1 targets), not the raw window width."""
    return sum(len(w) - 1 for w in _usable_windows(ids, length, n_windows))


def score_document(
    ids: list[int], loss_fn, *, length: int = WINDOW_TOKENS, n_windows: int = N_WINDOWS
) -> float:
    """Token-weighted mean NLL over the selected windows.

    ``loss_fn`` maps a non-empty id list to a mean NLL. Injecting it keeps the
    windowing testable without a GPU. A non-finite ``loss_fn`` result propagates
    as a non-finite mean; the screen then rejects the run.
    """
    windows = _usable_windows(ids, length, n_windows)
    total_tokens = sum(len(w) - 1 for w in windows)
    if total_tokens <= 0:
        return float("nan")
    return float(sum(loss_fn(w) * (len(w) - 1) for w in windows) / total_tokens)


def nll_distribution(values: list[float]) -> dict:
    """Median and deciles of a finite NLL sample. Empty input is explicit."""
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None, "deciles": {}}
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise InvalidNLLScreen(
            "INVALID: NLL distribution contains a non-finite value. "
            "The run is rejected and no document from it can enter the corpus. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    deciles = {
        f"{q:.1f}": float(np.quantile(arr, q)) for q in _DECILE_QS
    }
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.quantile(arr, 0.5)),
        "max": float(arr.max()),
        "deciles": deciles,
    }


def coerce_nll(value, doc_id: str) -> float:
    """Return a finite NLL or raise :class:`InvalidNLLScreen`."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidNLLScreen(
            f"INVALID: NLL for {doc_id} is not a float ({value!r}). "
            "The run is rejected and the document cannot enter the corpus. "
            f"Criterion: {CRITERION_SOURCE}."
        ) from exc
    if not math.isfinite(number):
        raise InvalidNLLScreen(
            f"INVALID: NLL for {doc_id} is {number!r}. A non-finite NLL "
            "rejects the run; the document cannot enter the corpus and the "
            "screen does not pass. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    return number


def _table_from_payload(payload) -> dict:
    if not isinstance(payload, dict):
        raise InvalidNLLScreen(
            "INVALID: NLL payload is not an object. The run is rejected. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    table = payload.get("nll", payload) if "nll" in payload else payload
    if not isinstance(table, dict):
        raise InvalidNLLScreen(
            "INVALID: NLL table is not an object. The run is rejected. "
            f"Criterion: {CRITERION_SOURCE}."
        )
    return table


def _optional_fence(payload: dict) -> float | None:
    """A fence is applied only when the table carries one.

    It must have been computed on text that is not the corpus under screen.
    This function does not estimate a fence from ``payload["nll"]``.
    """
    if "nll" not in payload or "reference_fence_nats" not in payload:
        return None
    raw = payload.get("reference_fence_nats")
    if raw is None:
        return None
    fence = coerce_nll(raw, "reference_fence")
    return fence


@dataclass
class ScreenResult:
    kept: list
    rejected: list
    n_rejected: int
    record: dict
    contamination_gap: object = "pending"


def not_scored_record() -> dict:
    """What the card says when no base-model table was supplied.

    Dry runs use this. A hash of the document id is not an NLL, and it must
    not produce a ``low_nll`` drop that looks like a screen.
    """
    return {
        "status": "not_scored",
        "documents_scored": 0,
        "tokens_scored": 0,
        "threshold_nats": MEMORISATION_NLL_FLOOR,
        "reference_fence_nats": None,
        "bottom_decile_quantile": BOTTOM_DECILE,
        "sampling": sampling_rule(),
        "distribution": None,
        "rejected_below_threshold": 0,
        "rejected_below_reference": 0,
        "rejected_bottom_decile": 0,
        "rejected": 0,
        "criterion_source": CRITERION_SOURCE,
        "cannot_prove_absence": True,
        "note": (
            "No base-model NLL table was supplied. Nothing was dropped for "
            "memorisation. A hash stub is not a screen."
        ),
    }


def apply_memorisation_screen(docs: list, payload: dict) -> ScreenResult:
    """Apply the real screen to documents that already have base-model NLLs.

    Missing ids raise ``KeyError`` (the build names the first one). Non-finite
    values raise :class:`InvalidNLLScreen` and the caller must not write a
    corpus. A finite score below the floor is rejected. A finite score below
    a supplied reference fence is rejected. The bottom decile of what remains
    is then dropped, per §4 step 7.
    """
    from .pipeline import drop_low_nll

    table = _table_from_payload(payload)
    missing = [d.doc_id for d in docs if d.doc_id not in table]
    if missing:
        raise KeyError(
            f"{len(missing)} harvested documents have no NLL score, e.g. "
            f"{missing[0]}; rerun experiments.corpus.nll over this JSONL"
        )

    scored: list[tuple[float, object]] = []
    for doc in docs:
        scored.append((coerce_nll(table[doc.doc_id], doc.doc_id), doc))

    fence = _optional_fence(payload) if isinstance(payload, dict) else None
    tokens_table = {}
    if isinstance(payload, dict) and isinstance(payload.get("tokens_scored"), dict):
        tokens_table = payload["tokens_scored"]

    values = [nll for nll, _ in scored]
    token_values = []
    tokens_complete = True
    for _, doc in scored:
        if doc.doc_id not in tokens_table:
            tokens_complete = False
            break
        token_values.append(coerce_nll(tokens_table[doc.doc_id], f"tokens:{doc.doc_id}"))
    tokens_scored = int(sum(token_values)) if tokens_complete else None

    below_floor = [(nll, doc) for nll, doc in scored if nll < MEMORISATION_NLL_FLOOR]
    floor_ids = {doc.doc_id for _, doc in below_floor}
    below_fence = [
        (nll, doc)
        for nll, doc in scored
        if fence is not None and nll < fence and doc.doc_id not in floor_ids
    ]
    rejected_ids = floor_ids | {doc.doc_id for _, doc in below_fence}
    survivors = [doc for _, doc in scored if doc.doc_id not in rejected_ids]

    def _known(doc) -> float:
        return coerce_nll(table[doc.doc_id], doc.doc_id)

    kept, decile_dropped = drop_low_nll(survivors, _known, quantile=BOTTOM_DECILE)
    rejected = [doc for _, doc in below_floor]
    rejected.extend(doc for _, doc in below_fence)
    rejected.extend(decile_dropped)
    n_rejected = len(rejected)
    if n_rejected != len(docs) - len(kept):
        raise AssertionError("memorisation screen lost or double-counted a document")

    gap = "pending"
    if isinstance(payload, dict) and "nll" in payload and "contamination_gap_nats" in payload:
        gap = payload["contamination_gap_nats"]
        if isinstance(gap, (int, float)) and not isinstance(gap, bool):
            gap = coerce_nll(gap, "contamination_gap")

    record = {
        "status": "scored",
        "documents_scored": len(docs),
        "tokens_scored": tokens_scored,
        "threshold_nats": MEMORISATION_NLL_FLOOR,
        "reference_fence_nats": fence,
        "reference_z": REFERENCE_Z if fence is not None else None,
        "bottom_decile_quantile": BOTTOM_DECILE,
        "sampling": sampling_rule(),
        "scored_with": {
            "max_tokens": payload.get("max_tokens") if isinstance(payload, dict) else None,
            "n_windows": payload.get("n_windows") if isinstance(payload, dict) else None,
            "model": payload.get("model") if isinstance(payload, dict) else None,
        },
        "distribution": nll_distribution(values),
        "rejected_below_threshold": len(below_floor),
        "rejected_below_reference": len(below_fence),
        "rejected_bottom_decile": len(decile_dropped),
        "rejected": n_rejected,
        "rejected_ids": [d.doc_id for d in rejected],
        "criterion_source": CRITERION_SOURCE,
        "cannot_prove_absence": True,
        "note": (
            "A score at or above the floor is not evidence the document was "
            "absent from training. The floor only flags near-deterministic "
            "text. The bottom decile drops about a tenth of the survivors "
            "whether or not any of them were memorised."
        ),
    }
    return ScreenResult(
        kept=kept,
        rejected=rejected,
        n_rejected=n_rejected,
        record=record,
        contamination_gap=gap,
    )


# --------------------------------------------------------------------------
# Model scoring
# --------------------------------------------------------------------------


class _GpuOOM(RuntimeError):
    pass


def _model_loss_fn(lm, device: str):
    import torch

    def loss_fn(ids: list[int]) -> float:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                out = lm.model(input_ids=x, labels=x)
            return float(out.loss.detach().float().cpu().item())
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise _GpuOOM(str(exc)) from exc
            raise
        finally:
            del x

    return loss_fn


def load_lm(model_id: str, *, requested_device: str = "auto"):
    """Load the base model. Check free GPU memory at the moment of the load.

    One process, bf16 on CUDA, fp32 on CPU. No second attempt on CUDA after
    an OOM: the caller falls back to CPU.
    """
    import torch

    from experiments.runner.model import LM

    free = gpu_free_mib()
    device = decide_device(free, requested_device)
    if device == "cpu" and requested_device != "cpu":
        print(
            f"[nll] GPU free memory is {free} MiB "
            f"(need {MIN_FREE_MIB}); scoring on CPU",
            flush=True,
        )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[nll] loading {model_id} on {device} ({dtype})", flush=True)
    try:
        lm = LM(model_id, device=device, dtype=dtype)
    except RuntimeError as exc:
        if device == "cuda" and "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[nll] OOM during CUDA load; retrying on CPU", flush=True)
            lm = LM(model_id, device="cpu", dtype=torch.float32)
            device = "cpu"
        else:
            raise
    return lm, device, free


def _free_cuda() -> None:
    """Return cached CUDA blocks after the caller has dropped the model."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_labeled_texts(
    pairs: list[tuple[str, str]],
    lm,
    *,
    device: str,
    length: int = WINDOW_TOKENS,
    n_windows: int = N_WINDOWS,
    allow_shrink: bool = True,
) -> tuple[list[dict], int]:
    """Score ``(doc_id, text)`` pairs. Returns rows and the window length used.

    On CUDA OOM, shrink the window once and retry the unscored suffix. A
    second OOM is the caller's signal to reload on CPU.
    """
    loss_fn = _model_loss_fn(lm, device)
    rows: list[dict] = []
    started = time.time()
    index = 0
    while index < len(pairs):
        doc_id, text = pairs[index]
        ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"]
        try:
            value = score_document(ids, loss_fn, length=length, n_windows=n_windows)
        except _GpuOOM:
            if allow_shrink and length > 128 and device == "cuda":
                _free_cuda()
                shrunk = max(128, length // 2)
                print(
                    f"[nll] CUDA OOM at window {length}; retrying once at {shrunk}",
                    flush=True,
                )
                length = shrunk
                allow_shrink = False
                continue
            raise
        tokens = scored_token_count(ids, length, n_windows)
        # A non-finite loss is not "unscored". Recording it as absent would
        # let the build treat the hole as a missing-file error, or worse,
        # skip the document. Reject here.
        coerce_nll(value, doc_id)
        rows.append(
            {"doc_id": doc_id, "nll": value, "tokens_scored": tokens}
        )
        index += 1
        done = len(rows)
        if done % 10 == 0:
            rate = done / max(1e-9, time.time() - started)
            print(f"[nll] {done}/{len(pairs)}  {rate * 60:.1f} docs/min", flush=True)
    return rows, length


def _envelope(
    rows: list[dict],
    *,
    model_id: str,
    length: int,
    n_windows: int,
    device: str,
    seconds: float,
    free_mib: int | None,
) -> dict:
    import torch

    peak = None
    if device == "cuda" and torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    known = [r["tokens_scored"] for r in rows if "tokens_scored" in r]
    return {
        "model": model_id,
        "max_tokens": length,
        "n_windows": n_windows,
        "device": device,
        "free_mib_before_load": free_mib,
        "peak_allocated_mib": peak,
        "seconds": round(seconds, 3),
        "sampling": sampling_rule(),
        "threshold_nats": MEMORISATION_NLL_FLOOR,
        "criterion_source": CRITERION_SOURCE,
        "nll": {r["doc_id"]: r["nll"] for r in rows},
        "tokens_scored": {r["doc_id"]: r["tokens_scored"] for r in rows if "tokens_scored" in r},
        "documents_scored": len(rows),
        "tokens_scored_total": int(sum(known)) if len(known) == len(rows) else None,
    }


def score_jsonl(
    docs_path: str | Path,
    out_path: str | Path,
    *,
    model_id: str = "Qwen/Qwen3-1.7B-Base",
    max_tokens: int = WINDOW_TOKENS,
    n_windows: int = N_WINDOWS,
    limit: int | None = None,
    requested_device: str = "auto",
) -> dict:
    from .harvest import load_documents

    docs = load_documents(docs_path)
    if limit is not None:
        docs = docs[:limit]
    out_path = Path(out_path)
    table: dict[str, float] = {}
    tokens: dict[str, int] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        table = dict(existing.get("nll", {}))
        tokens = dict(existing.get("tokens_scored", {}))
        for doc_id, value in table.items():
            coerce_nll(value, doc_id)

    todo = [d for d in docs if d.doc_id not in table]
    print(f"[nll] {len(docs)} documents, {len(todo)} to score", flush=True)
    if not todo:
        return table

    started = time.time()
    lm = None
    try:
        lm, device, free = load_lm(model_id, requested_device=requested_device)
        try:
            rows, used_length = score_labeled_texts(
                [(d.doc_id, d.text) for d in todo],
                lm,
                device=device,
                length=max_tokens,
                n_windows=n_windows,
                allow_shrink=True,
            )
        except _GpuOOM:
            if device == "cpu":
                raise
            print("[nll] OOM after one shrink; reloading on CPU", flush=True)
            # Drop the CUDA weights before the CPU load. Assigning None is
            # what drops the reference; deleting a callee's parameter does not.
            lm = None
            _free_cuda()
            lm, device, free = load_lm(model_id, requested_device="cpu")
            rows, used_length = score_labeled_texts(
                [(d.doc_id, d.text) for d in todo],
                lm,
                device=device,
                length=max_tokens,
                n_windows=n_windows,
                allow_shrink=False,
            )
        for row in rows:
            table[row["doc_id"]] = row["nll"]
            tokens[row["doc_id"]] = row["tokens_scored"]
        all_rows = []
        for doc_id, nll in table.items():
            row = {"doc_id": doc_id, "nll": nll}
            if doc_id in tokens:
                row["tokens_scored"] = tokens[doc_id]
            all_rows.append(row)
        payload = _envelope(
            all_rows,
            model_id=model_id,
            length=used_length,
            n_windows=n_windows,
            device=device,
            seconds=time.time() - started,
            free_mib=free,
        )
    finally:
        lm = None
        _free_cuda()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[nll] wrote {len(table)} scores to {out_path}", flush=True)
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--docs-jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--max-tokens", type=int, default=WINDOW_TOKENS)
    p.add_argument("--n-windows", type=int, default=N_WINDOWS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args(argv)
    try:
        score_jsonl(
            args.docs_jsonl,
            args.out,
            model_id=args.model,
            max_tokens=args.max_tokens,
            n_windows=args.n_windows,
            limit=args.limit,
            requested_device=args.device,
        )
    except InvalidNLLScreen as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
