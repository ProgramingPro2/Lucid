#!/usr/bin/env python3
"""Bounded nightly audit of the G-1 corpus harvest.

One run, CPU only. It does not write the harvest JSONL, it does not signal
the harvester, and it does not load the base model.

Steps, in order:

1. Read the harvest file. The default does not wait for the harvest to
   finish (it is running at 06:35). ``--wait-quiescent SECONDS`` waits up to
   that long for the size to sit still, then reads anyway. A truncated final
   line is skipped. A malformed earlier line is counted and left out of the
   snapshot. The source file is opened read-only.
2. Census every well-formed row (documents, fulltext-ok, whitespace tokens,
   dates). This is a stream, not the gated build.
3. Write a snapshot of at most ``--max-docs`` rows and run
   ``experiments.corpus.build`` on that snapshot. ``--window tail`` (the
   default) takes the last rows, which is where an append-only harvest puts
   new text. ``--window prefix`` takes the first rows. The snapshot of the
   in-progress file keeps the name ``biorxiv_2026.jsonl`` so the build still
   records the operative harvest cutoff.
4. Run the build a second time and compare token-stream hashes (G-1e),
   unless ``--no-rebuild``.
5. Judge G-1a..G-1f, the NLL screen, and the contamination screen. A check
   that is missing, stubbed, or empty is NOT_MEASURED. That is unknown, not
   a pass. One check raising does not abort the others.
6. Write ``<out-root>/<YYYY-MM-DD>/card.json`` and ``summary.txt``, and
   print the summary. The date is America/New_York.
7. Compare gate status with the latest earlier card. pass→fail is a
   REGRESSION. The first night says "no previous run" and invents nothing.

What the cap does not prove is in :data:`FIDELITY` and in the run note.
This card is not a FINDINGS row. See :data:`CITATION`.

Exit status: 0 card written, 2 card written and a gate regressed, 1 the
card could not be written. A gate that is FAIL because the corpus is still
small (G-1a today) is not by itself a non-zero exit. The card carries it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ZONE = ZoneInfo("America/New_York")
DEFAULT_JSONL = REPO / "artifacts" / "corpus" / "biorxiv_2026.jsonl"
DEFAULT_OUT_ROOT = REPO / "artifacts" / "corpus" / "nightly"
DEFAULT_NLL_NAME = "nll_qwen3_1.7b.json"
DEFAULT_CUTOFF = "2025-06-01"
DEFAULT_MAX_DOCS = 40
G1A_MIN_DOCUMENTS = 3000
G1A_MIN_TOKENS = 500_000_000

# Contamination is a reported measurement with no accept/reject threshold
# (docs/CORPUS.md §6). REPORTED is deliberately not PASS.
STATUSES = ("PASS", "FAIL", "NOT_MEASURED", "REPORTED")
CHECK_ORDER = (
    "G-1a",
    "G-1b",
    "G-1c",
    "G-1d",
    "G-1e",
    "G-1f",
    "nll_screen",
    "contamination",
)

CITATION = (
    "This card measures the corpus on a bounded snapshot. It is not an "
    "experimental result. No gate line here may be cited as a FINDINGS row. "
    "The analysis/verdict tool is the only authorised path to a FINDINGS row."
)

FIDELITY = (
    "The build is capped at --max-docs documents. --window tail (the default) "
    "takes the last rows in the file, which is where an append-only harvest "
    "adds text, and it does not re-check the head. --window prefix takes the "
    "first rows; because the file is append-only that prefix is stable and "
    "does not see documents written later. Neither window is the corpus. "
    f"G-1a requires at least {G1A_MIN_DOCUMENTS} documents and "
    f"{G1A_MIN_TOKENS} tokens. A cap below that cannot pass it. The streaming "
    "census can fail G-1a when the ungated fulltext-ok upper bound is already "
    "short, because later gates only drop text. The census cannot pass G-1a. "
    "A PASS on G-1b, G-1c, or G-1e covers only the capped window, and only "
    "with the evidence stored on that line. G-1e is two builds of that window "
    "with the build's whitespace tokeniser. While NLL is the stub, that hash "
    "is not the base-model token stream and it is not a protocol freeze. "
    "A stub NLL, a missing contamination module, a missing split module, and "
    "a missing or not-yet-sealed FDA register are NOT_MEASURED. A missing "
    "check is not a pass. Whitespace tokens are the build's own count, not "
    "the model tokeniser."
)

FIDELITY_SUMMARY = (
    "Cap: this build is not the corpus. It cannot pass G-1a, and a PASS on "
    "G-1b, G-1c, or G-1e covers only the capped window. A stub NLL, a missing "
    "contamination or split module, and an unsealed FDA register stay "
    "NOT_MEASURED (unknown, not a pass)."
)

_API: dict | None = None


def corpus_api() -> dict:
    """Import the corpus types the audit reads. Build is imported later."""
    global _API
    if _API is None:
        from experiments.corpus.pipeline import fulltext_status_reason
        from experiments.corpus.sources import (
            IN_PROGRESS_HARVEST_NAME,
            LICENSE_ALLOWLIST,
            OPERATIVE_HARVEST_CUTOFF,
            Document,
        )

        _API = {
            "Document": Document,
            "fulltext_status_reason": fulltext_status_reason,
            "LICENSE_ALLOWLIST": LICENSE_ALLOWLIST,
            "IN_PROGRESS_HARVEST_NAME": IN_PROGRESS_HARVEST_NAME,
            "OPERATIVE_HARVEST_CUTOFF": OPERATIVE_HARVEST_CUTOFF,
        }
    return _API


def make_gate(status: str, evidence: str, **extra) -> dict:
    """A gate line. PASS with empty evidence is refused.

    F27 passed with ``mean_nll = Infinity`` and F38's cosine gate sat below
    its own null. A status with nothing behind it is not a pass.
    """
    evidence = "" if evidence is None else str(evidence).strip()
    if status not in STATUSES:
        evidence = (
            f"unrecognised status {status!r} kept as NOT_MEASURED. {evidence}"
        ).strip()
        status = "NOT_MEASURED"
    if status == "PASS" and not evidence:
        status = "NOT_MEASURED"
        evidence = (
            "refusing PASS with empty evidence; a missing measurement is not a pass"
        )
    out = {"status": status, "evidence": evidence}
    out.update(extra)
    return out


def canonical_status(status: object) -> str:
    """Map a stored status onto pass / fail / unknown.

    Anything that is not exactly a pass or a fail is unknown. NOT_MEASURED,
    a missing key, and "not available" are unknown. They are never pass.
    """
    if status is None:
        return "unknown"
    text = str(status).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text == "pass":
        return "pass"
    if text == "fail":
        return "fail"
    return "unknown"


def card_path_for(out_root: Path, day: date) -> Path:
    return Path(out_root) / day.isoformat() / "card.json"


def _empty_read_report() -> dict:
    return {
        "missing": False,
        "documents": 0,
        "truncated_final_line": False,
        "malformed_lines": 0,
        "blank_lines": 0,
        "grew_during_read": False,
        "size_bytes": 0,
    }


def _document_from_obj(obj: object):
    if not isinstance(obj, dict):
        return None
    try:
        doc = corpus_api()["Document"](**obj)
    except TypeError:
        return None
    if not isinstance(doc.doc_id, str) or not doc.doc_id:
        return None
    if not isinstance(doc.text, str):
        return None
    if not isinstance(doc.sections, dict):
        return None
    if not isinstance(doc.meta, dict):
        return None
    return doc


def _classify_line(raw: bytes, *, final: bool) -> tuple[str, object]:
    if not raw.strip():
        return "blank", None
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ("truncated", None) if final else ("malformed", None)
    doc = _document_from_obj(obj)
    if doc is None:
        return "malformed", None
    return "doc", doc


def _iter_records(path: Path):
    """Yield ``(kind, payload)`` for one read-only pass.

    ``kind`` is ``doc``, ``malformed``, ``blank``, or ``truncated``. The
    harvest appends concurrently. A final partial line has no newline; that
    remainder is ``truncated`` and is not parsed into a document. The file
    is never opened for write.
    """
    pending = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            pending += chunk
            while True:
                nl = pending.find(b"\n")
                if nl < 0:
                    break
                raw = pending[:nl]
                pending = pending[nl + 1 :]
                yield _classify_line(raw, final=False)
    if pending.strip():
        # A torn tail is not a document. Do not try to repair it in place.
        yield "truncated", None
    elif pending:
        yield "blank", None


def read_jsonl_tolerant(path: Path) -> tuple[list[dict], dict]:
    """Every well-formed document, plus a report. Does not write ``path``."""
    path = Path(path)
    report = _empty_read_report()
    if not path.is_file():
        report["missing"] = True
        return [], report
    report["size_bytes"] = path.stat().st_size
    docs: list[dict] = []
    for kind, payload in _iter_records(path):
        if kind == "doc":
            docs.append(asdict(payload))
            report["documents"] += 1
        elif kind == "truncated":
            report["truncated_final_line"] = True
        elif kind == "malformed":
            report["malformed_lines"] += 1
        elif kind == "blank":
            report["blank_lines"] += 1
    return docs, report


def _words(doc) -> int:
    if isinstance(doc.text, str) and doc.text.strip():
        return len(doc.text.split())
    total = 0
    for value in doc.sections.values():
        if isinstance(value, str):
            total += len(value.split())
    return total


def _scan_once(path: Path, *, max_docs: int, window: str) -> tuple[list, dict]:
    report = _empty_read_report()
    report["size_bytes"] = path.stat().st_size
    reason_for = corpus_api()["fulltext_status_reason"]
    selected: list = []
    tail: deque | None = deque(maxlen=max_docs) if window == "tail" else None
    status_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    whitespace_tokens = 0
    eligible_documents = 0
    eligible_tokens = 0
    fulltext_ok = 0

    for kind, payload in _iter_records(path):
        if kind == "truncated":
            report["truncated_final_line"] = True
            continue
        if kind == "malformed":
            report["malformed_lines"] += 1
            continue
        if kind == "blank":
            report["blank_lines"] += 1
            continue
        doc = payload
        try:
            reason = reason_for(doc)
            words = _words(doc)
        except Exception:
            report["malformed_lines"] += 1
            continue
        report["documents"] += 1
        whitespace_tokens += words
        stamped = str(doc.date or "")[:10]
        if stamped:
            date_counts[stamped] += 1
        status = doc.meta.get("fulltext")
        if status == "ok":
            fulltext_ok += 1
        if reason is None:
            eligible_documents += 1
            eligible_tokens += words
        else:
            status_counts[str(reason)] += 1
        if tail is not None:
            tail.append(doc)
        elif len(selected) < max_docs:
            selected.append(doc)

    if tail is not None:
        selected = list(tail)
    report["whitespace_tokens"] = whitespace_tokens
    report["eligible_documents"] = eligible_documents
    report["eligible_whitespace_tokens"] = eligible_tokens
    report["fulltext_ok"] = fulltext_ok
    report["fulltext_status_counts"] = dict(sorted(status_counts.items()))
    report["date_counts"] = dict(sorted(date_counts.items()))
    report["date_min"] = min(date_counts) if date_counts else None
    report["date_max"] = max(date_counts) if date_counts else None
    report["distinct_dates"] = len(date_counts)
    return selected, report


def scan_jsonl(
    path: Path, *, max_docs: int, window: str, attempts: int = 3
) -> tuple[list, dict]:
    """Census the file and keep at most ``max_docs`` documents.

    If the harvest appends during the read, the pass is repeated, up to
    ``attempts``. The last pass is kept either way. This does not write
    ``path`` and it does not wait.
    """
    path = Path(path)
    if window not in {"prefix", "tail"}:
        raise ValueError(f"window must be prefix or tail, got {window!r}")
    if not path.is_file():
        report = _empty_read_report()
        report["missing"] = True
        report.update(
            {
                "whitespace_tokens": 0,
                "eligible_documents": 0,
                "eligible_whitespace_tokens": 0,
                "fulltext_ok": 0,
                "fulltext_status_counts": {},
                "date_counts": {},
                "date_min": None,
                "date_max": None,
                "distinct_dates": 0,
            }
        )
        return [], report
    last: tuple[list, dict] | None = None
    tries = max(1, attempts)
    for _ in range(tries):
        size_before = path.stat().st_size
        selected, report = _scan_once(path, max_docs=max_docs, window=window)
        try:
            size_after = path.stat().st_size
        except OSError:
            size_after = size_before
        report["grew_during_read"] = size_after != size_before
        report["size_bytes"] = size_after
        last = (selected, report)
        if size_after == size_before:
            break
    assert last is not None
    return last


def harvest_writer_pid(out: Path) -> int | None:
    """Pid of a harvest whose command line names ``out``, or None.

    Read-only. This never signals that process. Matching is on the absolute
    path and, when the file lives in the repo, its repo-relative path. The
    bare filename is not a needle: a short name would match unrelated
    command lines.
    """
    out = Path(out)
    needles = {str(out)}
    try:
        resolved = out.resolve()
    except OSError:
        resolved = out
    needles.add(str(resolved))
    try:
        needles.add(str(resolved.relative_to(REPO)))
    except ValueError:
        pass
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    me = os.getpid()
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        text = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        if "experiments.corpus.harvest" not in text and "corpus/harvest.py" not in text:
            continue
        if any(needle and needle in text for needle in needles):
            return pid
    return None


def wait_for_quiescent(path: Path, seconds: float) -> bool:
    """True if ``path`` stayed the same size for a short stretch.

    Returns False when ``seconds`` is 0, the file is missing, or the size
    was still changing when the budget ran out. Never writes and never
    signals the writer.
    """
    if seconds <= 0 or not path.is_file():
        return False
    deadline = time.monotonic() + seconds
    last = path.stat().st_size
    stable_for = 0.0
    last_check = time.monotonic()
    need = min(2.0, seconds)
    while True:
        now = time.monotonic()
        if now >= deadline:
            return False
        time.sleep(min(0.5, deadline - now))
        try:
            size = path.stat().st_size
        except OSError:
            return False
        tick = time.monotonic()
        delta = tick - last_check
        last_check = tick
        if size == last:
            stable_for += delta
            if stable_for >= need:
                return True
        else:
            last = size
            stable_for = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_fda(repo: Path = REPO) -> Path | None:
    folder = repo / "artifacts" / "corpus"
    if not folder.is_dir():
        return None
    hits = sorted(p for p in folder.glob("fda*.jsonl") if p.is_file())
    return hits[0] if hits else None


def resolve_nll_argument(spec: str | None, repo: Path = REPO) -> Path | None:
    """``auto`` uses the canonical table if it is on disk, else nothing.

    A missing table is not an error and not a pass. ``none`` skips it.
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if text.lower() in {"", "none", "off"}:
        return None
    if text.lower() == "auto":
        candidate = repo / "artifacts" / "corpus" / DEFAULT_NLL_NAME
        return candidate if candidate.is_file() else None
    return Path(text)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_snapshot(docs: list, dest: Path, source: Path) -> None:
    """Write the capped window to ``dest``. Refuses to overwrite ``source``."""
    if dest.resolve() == source.resolve():
        raise RuntimeError(
            f"refusing to write the snapshot onto the harvest file {source}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    if tmp.resolve() == source.resolve():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"refusing to replace the harvest file {source} with the snapshot"
        )
    tmp.replace(dest)


def screen_module(
    module_name: str,
    *,
    kind: str,
    card: dict | None = None,
    snapshot: Path | None = None,
) -> dict:
    """Call a CPU-side audit hook if that module has one.

    A missing module is NOT_MEASURED and the evidence says it is not
    available yet. Importing is allowed to fail. A hook whose parameters
    ask for a model is not called: this process stays off the GPU.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        return make_gate(
            "NOT_MEASURED",
            f"not available yet: {module_name} is not importable ({exc})",
        )
    except Exception as exc:
        return make_gate(
            "NOT_MEASURED",
            "not available yet: importing "
            f"{module_name} failed ({type(exc).__name__}: {exc})",
        )
    for name in ("audit", "check", "measure", "screen", "run"):
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        blob = " ".join(sig.parameters)
        if any(word in blob for word in ("model", "device", "cuda", "lm")):
            return make_gate(
                "NOT_MEASURED",
                f"not available yet: {module_name}.{name} asks for a model "
                f"({blob}). This audit is CPU-only and will not load one.",
            )
        kwargs: dict = {}
        if "card" in sig.parameters and card is not None:
            kwargs["card"] = card
        for key in ("snapshot", "docs_jsonl", "path", "jsonl"):
            if key in sig.parameters and snapshot is not None:
                kwargs[key] = str(snapshot)
                break
        missing = [
            p.name
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.name not in kwargs
        ]
        if missing:
            continue
        try:
            value = fn(**kwargs)
        except (Exception, SystemExit) as exc:
            return make_gate(
                "NOT_MEASURED",
                f"{kind} check {module_name}.{name} failed "
                f"({type(exc).__name__}: {exc}). The failure is this check's "
                "status, not a pass.",
            )
        return _interpret_screen(value, kind)
    return make_gate(
        "NOT_MEASURED",
        f"not available yet: {module_name} imported but exposes no CPU-side "
        "audit/check/measure/screen/run this audit can call without a model",
    )


def _normalise_reported_status(value: str) -> str:
    text = value.strip().upper().replace("-", "_").replace(" ", "_")
    if text in STATUSES:
        return text
    if text in {"NOT_AVAILABLE", "UNAVAILABLE", "UNKNOWN", "SKIPPED", "PENDING"}:
        return "NOT_MEASURED"
    return "NOT_MEASURED"


def _interpret_screen(value: object, kind: str) -> dict:
    if isinstance(value, dict):
        if value.get("status"):
            evidence = (
                value.get("evidence")
                or value.get("reason")
                or value.get("detail")
                or ""
            )
            status = _normalise_reported_status(str(value["status"]))
            if status == "NOT_MEASURED" and "not available" not in str(evidence).lower():
                evidence = f"not available yet: {evidence}".strip()
            return make_gate(status, str(evidence) if evidence else "")
        if "gap" in value or "contamination_gap_nats" in value:
            gap = value.get("gap", value.get("contamination_gap_nats"))
            return _gap_gate(kind, gap)
    if isinstance(value, bool):
        # A bare boolean has no evidence. Do not promote True to PASS.
        return make_gate(
            "NOT_MEASURED",
            f"not available yet: {kind} returned {value!r} with no evidence. "
            "A bare boolean is not a pass.",
        )
    if isinstance(value, (int, float)):
        return _gap_gate(kind, value)
    return make_gate(
        "NOT_MEASURED",
        f"not available yet: {kind} returned {type(value).__name__}, "
        "which is not a measurement this audit will score as a pass",
    )


def _gap_gate(kind: str, gap: object) -> dict:
    try:
        number = float(gap)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return make_gate(
            "NOT_MEASURED",
            f"not available yet: {kind} gap {gap!r} is not a number",
        )
    if not math.isfinite(number):
        return make_gate(
            "FAIL",
            f"{kind} returned a non-finite value {number!r}. "
            "Infinity or NaN is not a measurement and is not a pass.",
        )
    return make_gate(
        "REPORTED",
        f"{kind} reported {number}. docs/CORPUS.md §6 defines no threshold, "
        "so this is a measurement, not a pass or a fail.",
    )


def inspect_nll_table(path: Path | None, needed_ids: list[str]) -> tuple[dict, bool]:
    """Decide whether a base-model NLL table may drive the build.

    The second value is True only when every id in the window has a finite
    score. The stub in ``build._stub_nll`` is not that table. A non-finite
    score is a FAIL of the screen (F27 passed on Infinity).
    """
    if path is None:
        return (
            make_gate(
                "NOT_MEASURED",
                "not available yet: no base-model NLL table. "
                "The build stub hashes the document id; it is not base-model "
                "NLL and it cannot pass the memorisation screen. "
                "This audit does not load the model.",
            ),
            False,
        )
    path = Path(path)
    if not path.is_file():
        return (
            make_gate(
                "NOT_MEASURED",
                f"not available yet: NLL table {path} does not exist. "
                "A missing table is not a pass.",
            ),
            False,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            make_gate(
                "NOT_MEASURED",
                f"NLL table {path} could not be read ({type(exc).__name__}: {exc}). "
                "The screen was not applied.",
            ),
            False,
        )
    table = payload.get("nll", payload) if isinstance(payload, dict) else None
    if not isinstance(table, dict):
        return (
            make_gate(
                "NOT_MEASURED",
                f"NLL table {path} is not an object of doc_id to score. "
                "The screen was not applied.",
            ),
            False,
        )
    bad: list[str] = []
    finite: dict[str, float] = {}
    for key, value in table.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            bad.append(str(key))
            continue
        if not math.isfinite(number):
            bad.append(str(key))
            continue
        finite[str(key)] = number
    if bad:
        return (
            make_gate(
                "FAIL",
                f"{len(bad)} NLL scores in {path} are not finite "
                f"(example {bad[0]}). Infinity or NaN is not a memorisation "
                "measurement and is not a pass. The table was not applied.",
            ),
            False,
        )
    missing = [doc_id for doc_id in needed_ids if doc_id not in finite]
    if missing:
        return (
            make_gate(
                "NOT_MEASURED",
                f"NLL table {path} covers {len(needed_ids) - len(missing)} of "
                f"{len(needed_ids)} documents in this window "
                f"(missing example {missing[0]}). The screen was not applied. "
                "Partial coverage is not a pass.",
            ),
            False,
        )
    if not needed_ids:
        return (
            make_gate(
                "NOT_MEASURED",
                f"NLL table {path} was not applied because the window has no "
                "documents. An unused table is not a pass.",
            ),
            False,
        )
    return (
        make_gate(
            "PASS",
            f"finite scores for all {len(needed_ids)} documents in this window "
            f"from {path}. The build drops the lowest decile. This is the "
            "memorisation screen on the window, not a contamination verdict "
            "and not a result outside the cap.",
        ),
        True,
    )


def judge_g1a(
    *,
    eligible_documents: int,
    eligible_tokens: int,
    retained_documents: int | None,
    retained_tokens: int | None,
    saw_every_document: bool,
    file_documents: int,
    file_tokens: int,
    grew_during_read: bool,
) -> dict:
    """G-1a is ≥ 3,000 documents and ≥ 500M tokens in the retained pool.

    The census is an upper bound (fulltext-ok, before licence, dedup, and
    NLL). An upper bound under the bar is a FAIL. An upper bound over the
    bar is not a PASS. PASS requires an uncapped build whose retained pool
    meets both numbers.
    """
    grew = " The file grew during the read, so these counts can be short by the rows appended in that pass." if grew_during_read else ""
    if file_documents <= 0:
        return make_gate(
            "FAIL",
            "0 documents. G-1a requires at least "
            f"{G1A_MIN_DOCUMENTS} documents and {G1A_MIN_TOKENS} tokens. "
            "An empty harvest is a failure, not a vacuous pass.",
        )
    if eligible_documents < G1A_MIN_DOCUMENTS or eligible_tokens < G1A_MIN_TOKENS:
        return make_gate(
            "FAIL",
            "eligible upper bound is "
            f"{eligible_documents} fulltext-ok documents and "
            f"{eligible_tokens} whitespace tokens, before licence, dedup, and "
            f"NLL. G-1a requires at least {G1A_MIN_DOCUMENTS} documents and "
            f"{G1A_MIN_TOKENS} tokens. Later gates only drop text, so the "
            "retained pool cannot clear a bar the ungated eligible rows miss. "
            f"Whole file: {file_documents} rows, {file_tokens} whitespace "
            f"tokens.{grew}",
        )
    if not saw_every_document or retained_documents is None or retained_tokens is None:
        return make_gate(
            "NOT_MEASURED",
            "eligible upper bound meets the bar "
            f"({eligible_documents} documents, {eligible_tokens} whitespace "
            "tokens) but this run did not build every document, so the "
            "retained pool was not measured. A cap cannot pass G-1a. "
            f"Bounded retained: {retained_documents} documents, "
            f"{retained_tokens} whitespace tokens.{grew}",
        )
    if (
        retained_documents >= G1A_MIN_DOCUMENTS
        and retained_tokens >= G1A_MIN_TOKENS
    ):
        return make_gate(
            "PASS",
            "uncapped build retained "
            f"{retained_documents} documents and {retained_tokens} whitespace "
            f"tokens, meeting {G1A_MIN_DOCUMENTS} and {G1A_MIN_TOKENS}. "
            "Counts are the build's whitespace tokens, not the model tokeniser.",
        )
    return make_gate(
        "FAIL",
        "uncapped build retained "
        f"{retained_documents} documents and {retained_tokens} whitespace "
        f"tokens, below {G1A_MIN_DOCUMENTS} documents and {G1A_MIN_TOKENS} "
        f"tokens. Eligible upper bound was {eligible_documents} documents "
        f"and {eligible_tokens} tokens.{grew}",
    )


def _scope(window: str, n_window: int, max_docs: int, saw_every: bool) -> str:
    return (
        f" Scope: {window} window of {n_window} documents, max_docs={max_docs}, "
        f"saw_every_document={saw_every}. This does not cover rows outside the cap."
    )


def judge_g1b(
    build: dict | None,
    *,
    window: str,
    n_window: int,
    max_docs: int,
    saw_every: bool,
) -> dict:
    """100% of retained documents carry a machine-read allowlisted licence.

    Zero retained documents is not 100% of an empty set. Unclassifiable
    drops above 20% of the licence stage fail the gate: the criterion says
    to change source.
    """
    if not build or not build.get("ran"):
        return make_gate(
            "NOT_MEASURED",
            "not available yet: the bounded build did not produce a card, "
            "so the licence gate was not measured.",
        )
    retained = int(build.get("documents_retained") or 0)
    if retained <= 0:
        return make_gate(
            "NOT_MEASURED",
            "no retained documents in this window. 100% of an empty retained "
            "set is not a licence pass, and it says nothing about rows the "
            "cap did not build.",
        )
    sources = build.get("sources") or {}
    if not isinstance(sources, dict) or not sources:
        return make_gate(
            "NOT_MEASURED",
            "the build retained documents but recorded no per-licence source "
            "counts. The licence gate is not established.",
        )
    listed = sum(int(v) for v in sources.values())
    if listed != retained:
        return make_gate(
            "NOT_MEASURED",
            f"source counts sum to {listed} and documents_retained is "
            f"{retained}. The licence gate is not established.",
        )
    allow = corpus_api()["LICENSE_ALLOWLIST"]
    bad: list[str] = []
    for key in sources:
        _source, licence = str(key).rsplit(":", 1) if ":" in str(key) else (str(key), "unknown")
        if licence not in allow:
            bad.append(f"{key} ({licence})")
    if bad:
        return make_gate(
            "FAIL",
            "retained documents include a licence that is missing or not "
            f"allowlisted: {bad[:5]}. G-1b needs a machine-read allowlisted "
            "licence on every retained document.",
        )
    drops = build.get("drops_by_reason") or {}
    unclass = int(drops.get("licence:unclassifiable", 0))
    licence_drops = sum(
        int(v) for k, v in drops.items() if str(k).startswith("licence:")
    )
    considered = retained + licence_drops
    fraction = (unclass / considered) if considered else 0.0
    if considered and fraction > 0.20:
        return make_gate(
            "FAIL",
            f"unclassifiable {unclass}/{considered} "
            f"({fraction:.1%}) of the licence stage exceeds 20%. "
            "The retained rows are classified, and the criterion still says "
            "to change source. "
            f"Sources: {sources}.",
        )
    return make_gate(
        "PASS",
        "every retained document in this window carries an allowlisted "
        f"machine-read licence. Sources: {sources}. "
        f"Unclassifiable {unclass}/{considered} of the licence stage "
        f"(the 20% source-change line is not crossed)."
        + _scope(window, n_window, max_docs, saw_every),
    )


def judge_g1c(
    build: dict | None,
    *,
    window: str,
    n_window: int,
    max_docs: int,
    saw_every: bool,
) -> dict:
    """G-1c from the build's flag and verdict.

    ``build._g1c_verdict`` returns a pass when nothing was retained ("no
    retained document violates the gate"). That sentence cannot fail, so
    this audit does not treat it as a pass.
    """
    if not build or not build.get("ran"):
        return make_gate(
            "NOT_MEASURED",
            "not available yet: the bounded build did not produce a card, "
            "so integrity was not measured.",
        )
    retained = int(build.get("documents_retained") or 0)
    compliant = bool(build.get("g1c_compliant"))
    verdict = str(build.get("g1c_verdict") or "").strip()
    if not compliant:
        return make_gate(
            "FAIL",
            verdict or "g1c_compliant is false and the build gave no verdict.",
        )
    if retained <= 0:
        return make_gate(
            "NOT_MEASURED",
            "no retained documents in this window. The build's own verdict "
            f"was: {verdict or '(empty)'}. A vacuous G-1c pass cannot fail "
            "and is not a pass. This does not by itself fail rows outside "
            "the cap.",
        )
    if verdict.startswith("G-1c pass") and "NOT G-1c" not in verdict:
        return make_gate(
            "PASS",
            verdict + _scope(window, n_window, max_docs, saw_every),
        )
    return make_gate(
        "NOT_MEASURED",
        "g1c_compliant is true but the verdict is not a pass this audit "
        f"will cite: {verdict or '(empty)'}",
    )


def judge_g1d(build: dict | None, split_gate: dict) -> dict:
    """G-1d is zero documents shared across train / dev / test.

    ``pipeline.split_documents`` asserts that inside the build. That
    assertion is not ``experiments.corpus.split``. Until the split module
    is importable and returns its own check, this gate stays NOT_MEASURED.
    The build's split counts are quoted so the gap is visible.
    """
    counts = ""
    if build and build.get("ran"):
        counts = (
            f" The build recorded split_counts={build.get('split_counts')} "
            f"for {build.get('documents_retained')} retained documents. "
            "Those counts are not a disjointness measurement."
        )
    status = split_gate.get("status")
    if status in {"PASS", "FAIL", "REPORTED"}:
        evidence = str(split_gate.get("evidence") or "").strip()
        if counts and counts.strip() not in evidence:
            evidence = (evidence + counts).strip()
        return make_gate(str(status), evidence)
    evidence = str(split_gate.get("evidence") or "not available yet").strip()
    if "not available" not in evidence.lower():
        evidence = f"not available yet: {evidence}"
    return make_gate("NOT_MEASURED", evidence + counts)


def judge_g1e(
    *,
    hashes: list[str],
    errors: list[str],
    n_window: int,
    window: str,
    max_docs: int,
    saw_every: bool,
    nll_applied: bool,
    rebuild: bool,
) -> dict:
    """Two clean builds of the same snapshot must hash the same stream."""
    if n_window <= 0:
        return make_gate(
            "NOT_MEASURED",
            "no documents in the window. An empty token stream is not a "
            "reproducibility pass.",
        )
    if not rebuild:
        return make_gate(
            "NOT_MEASURED",
            "second build was not run (--no-rebuild). One hash is not a "
            "reproducibility measurement and is not a pass.",
        )
    good = [h for h in hashes if _is_sha256(h)]
    if errors and len(good) < 2:
        return make_gate(
            "NOT_MEASURED",
            "the rebuild did not finish, so G-1e was not measured "
            f"({'; '.join(errors[:2])}). A crashed rebuild is not a pass.",
        )
    if len(good) < 2:
        return make_gate(
            "NOT_MEASURED",
            f"expected two SHA-256 digests, got {hashes!r}. "
            "A missing hash is not a pass.",
        )
    if good[0] != good[1]:
        return make_gate(
            "FAIL",
            "two builds of the same snapshot disagreed: "
            f"{good[0]} vs {good[1]}. A non-reproducible stream fails G-1e.",
        )
    return make_gate(
        "PASS",
        f"two builds of this {window} window ({n_window} documents) produced "
        f"token-stream SHA-256 {good[0]}. The tokeniser is the build's "
        "whitespace tokeniser, not the model tokeniser. "
        f"Base-model NLL applied: {nll_applied}. "
        "This is reproducibility of the capped window, not a freeze of the "
        "corpus."
        + _scope(window, n_window, max_docs, saw_every),
    )


def judge_g1f(
    path: Path | None,
    digest: str | None,
    previous_digest: str | None,
) -> dict:
    """Sealed FDA register present and untouched.

    Present without a prior digest is not "untouched": the first night
    records the digest and stays NOT_MEASURED. Absence is not a pass.
    """
    if path is None or not Path(path).is_file():
        return make_gate(
            "NOT_MEASURED",
            "FDA second register is not present "
            f"({path or 'no artifacts/corpus/fda*.jsonl'}). "
            "Absence is not a pass. G-1f is required for the cross-register "
            "endpoint only.",
        )
    if not _is_sha256(digest):
        return make_gate(
            "NOT_MEASURED",
            f"FDA file {path} is present but it was not hashed. Untouched "
            "is not established.",
        )
    if not _is_sha256(previous_digest):
        return make_gate(
            "NOT_MEASURED",
            f"FDA file {path} is present, sha256 {digest}. No previous "
            "night's digest is on record, so untouched is not established. "
            "The digest is stored for the next night. This is not a pass.",
        )
    if digest == previous_digest:
        return make_gate(
            "PASS",
            f"FDA file {path} sha256 {digest} is unchanged since the previous "
            "night's card.",
        )
    return make_gate(
        "FAIL",
        f"FDA register changed. Previous sha256 {previous_digest}, now {digest} "
        f"({path}).",
    )


def resolve_g1f(
    *,
    fda_jsonl: Path | None,
    discover: bool,
    previous_digest: object,
) -> tuple[dict, str | None]:
    """Locate the FDA file, hash it, and judge G-1f. Returns ``(gate, digest)``."""
    path = fda_jsonl
    if path is None and discover:
        path = discover_fda(REPO)
    if path is None or not Path(path).is_file():
        return judge_g1f(None, None, None), None
    try:
        digest = sha256_file(Path(path))
    except OSError as exc:
        return (
            make_gate(
                "NOT_MEASURED",
                f"FDA file {path} could not be hashed ({exc}). "
                "Untouched is not established.",
            ),
            None,
        )
    prior = previous_digest if _is_sha256(previous_digest) else None
    return judge_g1f(Path(path), digest, prior), digest


def _run_build(
    snapshot: Path,
    out_dir: Path,
    *,
    cutoff: str,
    seed: int,
    nll_json: Path | None,
) -> dict:
    from experiments.corpus.build import build

    return build(
        source="biorxiv",
        cutoff=cutoff,
        out=out_dir,
        nll_json=str(nll_json) if nll_json else None,
        dry_run=False,
        max_docs=max(1, 10**9),
        seed=seed,
        docs_jsonl=snapshot,
        require_fulltext=False,
        allow_degraded=False,
    )


def _summarise_build(result: dict, *, nll_applied: bool) -> dict:
    card = result.get("card") or {}
    tokens = card.get("tokens_per_stratum") or {}
    retained_tokens = sum(int(v) for v in tokens.values()) if isinstance(tokens, dict) else 0
    return {
        "ran": True,
        "error": None,
        "nll_applied": nll_applied,
        "documents_harvested": card.get("documents_harvested"),
        "documents_retained": card.get("documents_retained"),
        "retained_whitespace_tokens": retained_tokens,
        "tokens_per_stratum": tokens,
        "drops_by_reason": card.get("drops_by_reason"),
        "sources": card.get("sources"),
        "fulltext_status_counts": card.get("fulltext_status_counts"),
        "split_counts": card.get("split_counts"),
        "token_stream_sha256": card.get("token_stream_sha256"),
        "g1c_compliant": card.get("g1c_compliant"),
        "g1c_verdict": card.get("g1c_verdict"),
        "nll_decile": card.get("nll_decile"),
        "recorded_cutoff": card.get("recorded_cutoff"),
        "release_date_proxy": card.get("release_date_proxy"),
        "operative_harvest_cutoff": card.get("operative_harvest_cutoff"),
        "cutoff_compliant": card.get("cutoff_compliant"),
        "coverage_note": card.get("coverage_note"),
    }


def run_builds(
    snapshot: Path,
    out_dir: Path,
    *,
    cutoff: str,
    seed: int,
    nll_json: Path | None,
    nll_applied: bool,
    rebuild: bool,
) -> tuple[dict | None, list[str], list[str]]:
    """One or two builds. Failures are returned, not raised."""
    hashes: list[str] = []
    errors: list[str] = []
    summary: dict | None = None
    passes = 2 if rebuild else 1
    for index in range(passes):
        dest = out_dir / f"build-{index + 1}"
        try:
            result = _run_build(
                snapshot,
                dest,
                cutoff=cutoff,
                seed=seed,
                nll_json=nll_json if nll_applied else None,
            )
        except Exception as exc:
            errors.append(f"build-{index + 1}: {type(exc).__name__}: {exc}")
            continue
        current = _summarise_build(result, nll_applied=nll_applied)
        digest = current.get("token_stream_sha256")
        if isinstance(digest, str):
            hashes.append(digest)
        if summary is None:
            summary = current
    if summary is not None:
        summary["rebuild_hashes"] = list(hashes)
        summary["rebuild_errors"] = list(errors)
    return summary, hashes, errors


def compare_gates(
    previous: dict[str, str] | None,
    current: dict[str, str],
    *,
    previous_date: str | None = None,
    previous_note: str | None = None,
    counts_previous: dict | None = None,
    counts_current: dict | None = None,
) -> dict:
    """Day-over-day gate delta.

    pass→fail is a regression. fail→pass is newly passed. A missing status
    is unknown, never pass. The first night (``previous is None`` and no
    unreadability note) does not call today's failures "newly failed".
    """
    names = list(CHECK_ORDER)
    for key in list(previous or {}) + list(current or {}):
        if key not in names:
            names.append(key)
    no_previous = previous is None and not previous_note
    rows: dict[str, dict] = {}
    regressions: list[str] = []
    newly_passed: list[str] = []
    newly_failed: list[str] = []
    lost: list[str] = []
    for name in names:
        prev_raw = None if previous is None else previous.get(name)
        cur_raw = None if current is None else current.get(name)
        prev = canonical_status(prev_raw)
        cur = canonical_status(cur_raw)
        rows[name] = {
            "previous": prev,
            "current": cur,
            "previous_raw": prev_raw,
            "current_raw": cur_raw,
        }
        if no_previous or previous is None:
            continue
        if prev == "pass" and cur == "fail":
            regressions.append(name)
        if prev == "fail" and cur == "pass":
            newly_passed.append(name)
        if cur == "fail" and prev != "fail":
            newly_failed.append(name)
        if prev == "pass" and cur == "unknown":
            lost.append(name)

    def _count(key: str) -> dict:
        prev_v = None if not counts_previous else counts_previous.get(key)
        cur_v = None if not counts_current else counts_current.get(key)
        delta = None
        if isinstance(prev_v, int) and isinstance(cur_v, int):
            delta = cur_v - prev_v
        return {"previous": prev_v, "current": cur_v, "delta": delta}

    if previous_note:
        label = previous_note
    elif no_previous:
        label = "no previous run"
    else:
        label = previous_date or "previous run"
    return {
        "previous": label,
        "previous_date": previous_date,
        "documents": _count("documents"),
        "fulltext_ok": _count("fulltext_ok"),
        "whitespace_tokens": _count("whitespace_tokens"),
        "eligible_documents": _count("eligible_documents"),
        "eligible_whitespace_tokens": _count("eligible_whitespace_tokens"),
        "newly_passed": newly_passed,
        "newly_failed": newly_failed,
        "regressions": regressions,
        "lost_measurement": lost,
        "regression": bool(regressions),
        "gates": rows,
    }


def _statuses_from_card(card: dict | None) -> dict[str, str]:
    if not isinstance(card, dict):
        return {}
    gates = card.get("gates") or {}
    out: dict[str, str] = {}
    if not isinstance(gates, dict):
        return out
    for name, payload in gates.items():
        if isinstance(payload, dict):
            out[str(name)] = str(payload.get("status"))
        else:
            out[str(name)] = str(payload)
    return out


def _tracked_from_card(card: dict | None) -> dict:
    if not isinstance(card, dict):
        return {}
    tracked = card.get("tracked")
    return dict(tracked) if isinstance(tracked, dict) else {}


def find_previous_card(out_root: Path, day: date) -> tuple[Path | None, dict | None, str | None]:
    """Latest card strictly before ``day``. Unreadable JSON is reported."""
    root = Path(out_root)
    if not root.is_dir():
        return None, None, None
    best: tuple[date, Path] | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            stamped = date.fromisoformat(child.name)
        except ValueError:
            continue
        if stamped >= day:
            continue
        card = child / "card.json"
        if card.is_file() and (best is None or stamped > best[0]):
            best = (stamped, card)
    if best is None:
        return None, None, None
    path = best[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return path, None, f"previous card unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return path, None, "previous card unreadable: JSON value is not an object"
    return path, payload, None


def _fmt_int(value: object) -> str:
    if not isinstance(value, int):
        return "n/a"
    return f"{value:,}"


def _fmt_change(block: dict | None) -> str:
    if not isinstance(block, dict):
        return "n/a"
    prev, cur, delta = block.get("previous"), block.get("current"), block.get("delta")
    if not isinstance(cur, int):
        return "n/a"
    if not isinstance(prev, int) or not isinstance(delta, int):
        return f"{_fmt_int(cur)} (no previous run)"
    return f"{_fmt_int(prev)} -> {_fmt_int(cur)} ({delta:+,})"


def _one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def regression_banner(delta: dict) -> str:
    names = delta.get("regressions") or []
    if not names:
        return ""
    listed = ", ".join(str(name) for name in names)
    return (
        f"REGRESSION {listed} changed from pass to fail. "
        "A gate that had passed no longer does. The build or the corpus "
        "broke under a check that used to hold."
    )


def format_summary(card: dict) -> str:
    """One screen. Full evidence stays in the card."""
    delta = card.get("delta") or {}
    lines: list[str] = []
    banner = regression_banner(delta)
    if banner:
        lines.append(banner)
        lines.append("")
    lines.append(
        f"NIGHTLY CORPUS AUDIT  {card.get('audit_date')}  America/New_York"
    )
    lines.append(CITATION)
    lines.append(FIDELITY_SUMMARY)
    overall = card.get("overall")
    failure = card.get("explicit_failure")
    if failure:
        lines.append(f"overall {overall}  EXPLICIT FAILURE: {failure}")
    else:
        lines.append(f"overall {overall}")
    harvest = card.get("harvest") or {}
    lines.append(
        f"harvest  {harvest.get('path')}  "
        f"writer_pid={harvest.get('writer_pid')}  "
        f"read_only=yes  "
        f"truncated_final_line={harvest.get('truncated_final_line')}  "
        f"malformed_lines={harvest.get('malformed_lines')}  "
        f"grew_during_read={harvest.get('grew_during_read')}"
    )
    census = card.get("census") or {}
    cap = card.get("cap") or {}
    lines.append(
        f"census   rows={_fmt_int(census.get('documents'))}  "
        f"fulltext_ok={_fmt_int(census.get('fulltext_ok'))}  "
        f"eligible={_fmt_int(census.get('eligible_documents'))}  "
        f"eligible_tokens={_fmt_int(census.get('eligible_whitespace_tokens'))}  "
        f"all_tokens={_fmt_int(census.get('whitespace_tokens'))}"
    )
    lines.append(
        f"dates    n={census.get('distinct_dates')}  "
        f"min={census.get('date_min')}  max={census.get('date_max')}  "
        f"{census.get('date_counts')}"
    )
    lines.append(
        f"cap      max_docs={cap.get('max_docs')}  window={cap.get('window')}  "
        f"in_window={cap.get('documents_in_window')}  "
        f"saw_every_document={cap.get('saw_every_document')}  "
        f"cutoff={cap.get('cutoff')}"
    )
    build = card.get("build") or {}
    if build.get("ran"):
        lines.append(
            f"build    harvested={build.get('documents_harvested')}  "
            f"retained={build.get('documents_retained')}  "
            f"retained_tokens={build.get('retained_whitespace_tokens')}  "
            f"nll_applied={build.get('nll_applied')}  "
            f"nll_decile={build.get('nll_decile')}  "
            f"operative_cutoff={build.get('operative_harvest_cutoff')}"
        )
    elif build.get("error"):
        lines.append(f"build    NOT RUN  {_one_line(str(build.get('error')), 180)}")
    else:
        lines.append("build    not run")
    lines.append("")
    lines.append(f"{'CHECK':<16} {'STATUS':<14} EVIDENCE")
    gates = card.get("gates") or {}
    for name in list(CHECK_ORDER) + [k for k in gates if k not in CHECK_ORDER]:
        payload = gates.get(name) or {}
        status = payload.get("status", "NOT_MEASURED")
        evidence = _one_line(payload.get("evidence") or "")
        lines.append(f"{name:<16} {status:<14} {evidence}")
    lines.append("")
    lines.append(f"DELTA  {delta.get('previous')}")
    lines.append(f"  documents     {_fmt_change(delta.get('documents'))}")
    lines.append(f"  fulltext ok   {_fmt_change(delta.get('fulltext_ok'))}")
    lines.append(f"  tokens        {_fmt_change(delta.get('whitespace_tokens'))}  whitespace, whole file")
    lines.append(
        "  newly passed  "
        + (", ".join(delta.get("newly_passed") or []) or "none")
    )
    lines.append(
        "  newly failed  "
        + (", ".join(delta.get("newly_failed") or []) or "none")
    )
    lines.append(
        "  regressions   "
        + (", ".join(delta.get("regressions") or []) or "none")
    )
    lost = delta.get("lost_measurement") or []
    if lost:
        lines.append("  lost measure  " + ", ".join(lost))
    lines.append(f"card  {((card.get('paths') or {}).get('card'))}")
    text = "\n".join(lines).rstrip() + "\n"
    return text


def _overall(gates: dict, explicit_failure: str | None) -> str:
    if explicit_failure:
        return "FAIL"
    statuses = [g.get("status") for g in gates.values()]
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS"
    return "INCOMPLETE"


def _not_measured_gates(reason: str) -> dict:
    return {name: make_gate("NOT_MEASURED", reason) for name in CHECK_ORDER}


def run_audit(
    *,
    jsonl: Path,
    out_root: Path,
    max_docs: int = DEFAULT_MAX_DOCS,
    cutoff: str = DEFAULT_CUTOFF,
    seed: int = 0,
    nll_json: Path | None = None,
    when: date | None = None,
    window: str = "tail",
    wait_quiescent: float = 0.0,
    fda_jsonl: Path | None = None,
    discover_fda_register: bool = False,
    contamination_module: str = "experiments.corpus.contamination",
    split_module: str = "experiments.corpus.split",
    rebuild: bool = True,
    echo: bool = True,
    scan_writer: bool = True,
) -> dict:
    """Run one audit and write the dated card. Does not write ``jsonl``."""
    started = time.perf_counter()
    jsonl = Path(jsonl)
    out_root = Path(out_root)
    if max_docs < 1:
        raise ValueError("max_docs must be >= 1; this audit has no uncapped mode")
    if window not in {"prefix", "tail"}:
        raise ValueError("window must be 'prefix' or 'tail'")
    day = when or datetime.now(ZONE).date()
    dated = out_root / day.isoformat()
    card_path = dated / "card.json"
    summary_path = dated / "summary.txt"

    quiescent = False
    if wait_quiescent > 0:
        quiescent = wait_for_quiescent(jsonl, wait_quiescent)

    explicit: str | None = None
    selected: list = []
    census: dict
    import_error: str | None = None
    try:
        corpus_api()
        selected, census = scan_jsonl(jsonl, max_docs=max_docs, window=window)
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
        census = _empty_read_report()
        census.update(
            {
                "whitespace_tokens": 0,
                "eligible_documents": 0,
                "eligible_whitespace_tokens": 0,
                "fulltext_ok": 0,
                "fulltext_status_counts": {},
                "date_counts": {},
                "date_min": None,
                "date_max": None,
                "distinct_dates": 0,
            }
        )
        if not jsonl.is_file():
            census["missing"] = True

    if census.get("missing") or int(census.get("documents") or 0) <= 0:
        explicit = "zero-document corpus"

    saw_every = (
        explicit is None
        and int(census.get("documents") or 0) <= max_docs
        and not census.get("grew_during_read")
        and int(census.get("malformed_lines") or 0) == 0
        and import_error is None
    )
    writer = harvest_writer_pid(jsonl) if scan_writer and jsonl.exists() else None

    snapshot_path: Path | None = None
    build_summary: dict | None = None
    hashes: list[str] = []
    build_errors: list[str] = []
    nll_gate, nll_applied = inspect_nll_table(
        nll_json, [doc.doc_id for doc in selected]
    )
    if import_error:
        nll_gate = make_gate(
            "NOT_MEASURED",
            "not available yet: the corpus package could not be imported "
            f"({import_error}). The NLL screen was not run.",
        )
        nll_applied = False

    if selected and import_error is None:
        try:
            api = corpus_api()
            snap_name = (
                jsonl.name
                if jsonl.name == api["IN_PROGRESS_HARVEST_NAME"]
                else "snapshot.jsonl"
            )
            snapshot_path = dated / "snapshot" / snap_name
            write_snapshot(selected, snapshot_path, jsonl)
            build_summary, hashes, build_errors = run_builds(
                snapshot_path,
                dated,
                cutoff=cutoff,
                seed=seed,
                nll_json=nll_json,
                nll_applied=nll_applied,
                rebuild=rebuild,
            )
        except Exception as exc:
            build_errors.append(f"{type(exc).__name__}: {exc}")
            build_summary = None

    if build_summary is None:
        build_view: dict = {
            "ran": False,
            "error": "; ".join(build_errors) if build_errors else import_error,
            "nll_applied": False,
            "documents_retained": 0 if explicit else None,
            "retained_whitespace_tokens": None,
        }
    else:
        build_view = build_summary
        if build_errors:
            build_view["error"] = "; ".join(build_errors)

    window_n = len(selected)
    if explicit:
        gates = _not_measured_gates(
            "no documents; a missing corpus is not a pass"
        )
        gates["G-1a"] = judge_g1a(
            eligible_documents=0,
            eligible_tokens=0,
            retained_documents=0,
            retained_tokens=0,
            saw_every_document=False,
            file_documents=0,
            file_tokens=0,
            grew_during_read=bool(census.get("grew_during_read")),
        )
        if census.get("missing"):
            gates["G-1a"] = make_gate(
                "FAIL",
                f"harvest file {jsonl} is not present, so there are 0 documents. "
                f"G-1a requires at least {G1A_MIN_DOCUMENTS} documents and "
                f"{G1A_MIN_TOKENS} tokens. A missing file is a failure, not a pass.",
            )
        elif census.get("truncated_final_line"):
            gates["G-1a"]["evidence"] += (
                " The file had a truncated final line and no complete document."
            )
    else:
        gates = {
            "G-1a": judge_g1a(
                eligible_documents=int(census.get("eligible_documents") or 0),
                eligible_tokens=int(census.get("eligible_whitespace_tokens") or 0),
                retained_documents=(
                    None
                    if not build_view.get("ran")
                    else int(build_view.get("documents_retained") or 0)
                ),
                retained_tokens=(
                    None
                    if not build_view.get("ran")
                    else int(build_view.get("retained_whitespace_tokens") or 0)
                ),
                saw_every_document=saw_every and bool(build_view.get("ran")),
                file_documents=int(census.get("documents") or 0),
                file_tokens=int(census.get("whitespace_tokens") or 0),
                grew_during_read=bool(census.get("grew_during_read")),
            ),
            "G-1b": judge_g1b(
                build_view if build_view.get("ran") else None,
                window=window,
                n_window=window_n,
                max_docs=max_docs,
                saw_every=saw_every,
            ),
            "G-1c": judge_g1c(
                build_view if build_view.get("ran") else None,
                window=window,
                n_window=window_n,
                max_docs=max_docs,
                saw_every=saw_every,
            ),
        }
        try:
            split_gate = screen_module(
                split_module,
                kind="split",
                card=build_view if build_view.get("ran") else None,
                snapshot=snapshot_path,
            )
        except Exception as exc:
            split_gate = make_gate(
                "NOT_MEASURED",
                f"not available yet: split probe crashed ({type(exc).__name__}: {exc})",
            )
        gates["G-1d"] = judge_g1d(
            build_view if build_view.get("ran") else None, split_gate
        )
        gates["G-1e"] = judge_g1e(
            hashes=hashes,
            errors=build_errors,
            n_window=window_n,
            window=window,
            max_docs=max_docs,
            saw_every=saw_every,
            nll_applied=nll_applied and bool(build_view.get("ran")),
            rebuild=rebuild,
        )
        gates["nll_screen"] = nll_gate
        try:
            gates["contamination"] = screen_module(
                contamination_module,
                kind="contamination",
                card=build_view if build_view.get("ran") else None,
                snapshot=snapshot_path,
            )
        except Exception as exc:
            gates["contamination"] = make_gate(
                "NOT_MEASURED",
                "not available yet: contamination probe crashed "
                f"({type(exc).__name__}: {exc})",
            )

    prev_path, prev_card, prev_note = find_previous_card(out_root, day)
    prev_tracked = _tracked_from_card(prev_card)
    fda_digest: str | None = None
    if explicit is None:
        try:
            gates["G-1f"], fda_digest = resolve_g1f(
                fda_jsonl=Path(fda_jsonl) if fda_jsonl else None,
                discover=discover_fda_register,
                previous_digest=prev_tracked.get("fda_sha256"),
            )
        except Exception as exc:
            gates["G-1f"] = make_gate(
                "NOT_MEASURED",
                "not available yet: FDA check failed "
                f"({type(exc).__name__}: {exc}). Absence of a result is not a pass.",
            )
            fda_digest = None

    tracked = {
        "documents": int(census.get("documents") or 0),
        "fulltext_ok": int(census.get("fulltext_ok") or 0),
        "whitespace_tokens": int(census.get("whitespace_tokens") or 0),
        "eligible_documents": int(census.get("eligible_documents") or 0),
        "eligible_whitespace_tokens": int(census.get("eligible_whitespace_tokens") or 0),
        "malformed_lines": int(census.get("malformed_lines") or 0),
        "fda_sha256": fda_digest,
    }

    delta = compare_gates(
        None if prev_card is None else _statuses_from_card(prev_card),
        {name: gates[name]["status"] for name in gates},
        previous_date=prev_card.get("audit_date") if isinstance(prev_card, dict) else (
            prev_path.parent.name if prev_path is not None else None
        ),
        previous_note=prev_note,
        counts_previous=prev_tracked if prev_card is not None else None,
        counts_current=tracked,
    )
    if prev_path is not None:
        delta["previous_card"] = str(prev_path)

    overall = _overall(gates, explicit)
    # A green card is every check PASS. Refuse that if any evidence is empty
    # (make_gate already did) or if the explicit failure was set.
    if overall == "PASS" and any(
        g.get("status") != "PASS" or not str(g.get("evidence") or "").strip()
        for g in gates.values()
    ):
        overall = "INCOMPLETE"

    card = {
        "kind": "nightly-corpus-audit",
        "audit_date": day.isoformat(),
        "timezone": "America/New_York",
        "citation": CITATION,
        "fidelity": FIDELITY,
        "overall": overall,
        "explicit_failure": explicit,
        "regression": bool(delta.get("regression")),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "harvest": {
            "path": str(jsonl),
            "writer_pid": writer,
            "read_only": True,
            "missing": bool(census.get("missing")),
            "truncated_final_line": bool(census.get("truncated_final_line")),
            "malformed_lines": int(census.get("malformed_lines") or 0),
            "blank_lines": int(census.get("blank_lines") or 0),
            "grew_during_read": bool(census.get("grew_during_read")),
            "size_bytes": int(census.get("size_bytes") or 0),
            "wait_quiescent_seconds": wait_quiescent,
            "quiescent": quiescent,
            "import_error": import_error,
        },
        "cap": {
            "max_docs": max_docs,
            "window": window,
            "documents_in_window": window_n,
            "saw_every_document": saw_every,
            "cutoff": cutoff,
            "seed": seed,
            "rebuild": rebuild,
        },
        "census": {
            "documents": int(census.get("documents") or 0),
            "fulltext_ok": int(census.get("fulltext_ok") or 0),
            "whitespace_tokens": int(census.get("whitespace_tokens") or 0),
            "eligible_documents": int(census.get("eligible_documents") or 0),
            "eligible_whitespace_tokens": int(
                census.get("eligible_whitespace_tokens") or 0
            ),
            "fulltext_status_counts": census.get("fulltext_status_counts") or {},
            "date_counts": census.get("date_counts") or {},
            "date_min": census.get("date_min"),
            "date_max": census.get("date_max"),
            "distinct_dates": int(census.get("distinct_dates") or 0),
        },
        "tracked": tracked,
        "gates": {name: gates[name] for name in CHECK_ORDER if name in gates},
        "delta": delta,
        "build": build_view,
        "paths": {
            "card": str(card_path),
            "summary": str(summary_path),
            "snapshot": str(snapshot_path) if snapshot_path else None,
        },
    }
    # Any gate that wasn't in CHECK_ORDER (should not happen) still ships.
    for name, payload in gates.items():
        card["gates"].setdefault(name, payload)

    summary = format_summary(card)
    _write_json(card_path, card)
    _write_text(summary_path, summary)
    if echo:
        sys.stdout.write(summary)
        sys.stdout.flush()
        if delta.get("regression"):
            sys.stderr.write(regression_banner(delta) + "\n")
            sys.stderr.flush()
    return card


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nightly bounded G-1 audit. Reads the harvest JSONL, never writes "
            "it, and writes a dated card. CPU only. Not a FINDINGS row."
        )
    )
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help=(
            f"documents in the gated build (default {DEFAULT_MAX_DOCS}). "
            "The census still covers the whole file. There is no uncapped mode."
        ),
    )
    parser.add_argument(
        "--window",
        choices=("tail", "prefix"),
        default="tail",
        help="tail: last --max-docs rows (default). prefix: first rows",
    )
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nll-json",
        default="auto",
        help=(
            "base-model NLL table, 'auto' "
            f"(artifacts/corpus/{DEFAULT_NLL_NAME} if present), or 'none'"
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD in America/New_York; default is today there",
    )
    parser.add_argument(
        "--wait-quiescent",
        type=float,
        default=0.0,
        help=(
            "seconds to wait for the harvest file size to stop changing. "
            "0 reads immediately and tolerates a truncated final line"
        ),
    )
    parser.add_argument(
        "--fda-jsonl",
        default=None,
        help="sealed FDA register; default searches artifacts/corpus/fda*.jsonl",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="build once; G-1e stays NOT_MEASURED",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # The 3090 is not ours. Set this before any optional module import.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = build_parser().parse_args(argv)
    if args.max_docs < 1:
        print(
            "refusing: --max-docs must be >= 1. This audit has no uncapped build.",
            file=sys.stderr,
        )
        return 2
    if args.date:
        try:
            when = date.fromisoformat(args.date)
        except ValueError as exc:
            print(f"refusing: --date {exc}", file=sys.stderr)
            return 2
    else:
        when = None
    nll = resolve_nll_argument(args.nll_json)
    fda = Path(args.fda_jsonl) if args.fda_jsonl else None
    try:
        card = run_audit(
            jsonl=Path(args.jsonl),
            out_root=Path(args.out_root),
            max_docs=args.max_docs,
            cutoff=args.cutoff,
            seed=args.seed,
            nll_json=nll,
            when=when,
            window=args.window,
            wait_quiescent=args.wait_quiescent,
            fda_jsonl=fda,
            discover_fda_register=fda is None,
            rebuild=not args.no_rebuild,
            echo=True,
        )
    except Exception as exc:
        print(
            f"audit failed before writing a card: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    written = (card.get("paths") or {}).get("card")
    if not written or not Path(written).is_file():
        print("audit failed: card was not written", file=sys.stderr)
        return 1
    return 2 if card.get("regression") else 0


if __name__ == "__main__":
    sys.exit(main())
