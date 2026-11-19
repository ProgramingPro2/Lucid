"""Harvest a resumable JSONL register from bioRxiv/medRxiv.

Harvesting is separated from ``build`` because the gates need a per-document NLL
table, and re-downloading a multi-hour corpus to re-run a gate is wasteful. This
CLI only fetches and writes; ``build.py --docs-jsonl`` applies every gate.

Full text is fetched only for licences in ``LICENSE_ALLOWLIST``. A failed paper
is written with ``meta.fulltext`` set to the reason, so the failure is counted
rather than hidden. The file is append-only and flushed per document; rerunning
with ``--resume`` skips doc ids already present.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.corpus.harvest \
        --server biorxiv --cutoff 2025-06-01 \
        --out artifacts/corpus/biorxiv.jsonl --max-docs 200 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .sources import (
    LICENSE_ALLOWLIST,
    BioRxivClient,
    Document,
    cutoff_plus_grace,
    load_cutoff,
)


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (KeyError, json.JSONDecodeError):
                continue
    return done


def _windows(start: date, end: date, days: int):
    lo = start
    while lo <= end:
        hi = min(lo + timedelta(days=days - 1), end)
        yield lo.isoformat(), hi.isoformat()
        lo = hi + timedelta(days=1)


def harvest(
    *,
    server: str,
    since: str,
    until: str,
    out: Path,
    max_docs: int,
    window_days: int = 7,
    delay: float = 1.0,
    fulltext: bool = True,
    resume: bool = True,
) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out) if resume else set()
    client = BioRxivClient(server=server, delay=delay, fulltext=fulltext)

    counts = {
        "metadata": 0,
        "skipped_existing": 0,
        "licence_allowed": 0,
        "fulltext_ok": 0,
        "abstract_only": 0,
        "fulltext_error": 0,
        "written": 0,
    }
    started = time.time()
    with out.open("a") as fh:
        for lo, hi in _windows(
            cutoff_plus_grace(since), date.fromisoformat(until), window_days
        ):
            remaining = max_docs - counts["written"]
            if remaining <= 0:
                break
            print(f"[harvest] {server} {lo}..{hi}", flush=True)
            try:
                for doc in client.iter_metadata(lo, hi):
                    if counts["written"] >= max_docs:
                        print("[harvest] reached --max-docs", flush=True)
                        return counts
                    counts["metadata"] += 1
                    if doc.doc_id in done:
                        counts["skipped_existing"] += 1
                        continue
                    allowed = doc.license in LICENSE_ALLOWLIST
                    counts["licence_allowed"] += int(allowed)
                    if fulltext and allowed and doc.meta.get("jatsxml"):
                        doc = client.fetch_fulltext(doc)
                    status = doc.meta.get("fulltext", "abstract_only")
                    if status == "ok":
                        counts["fulltext_ok"] += 1
                    elif status == "abstract_only":
                        counts["abstract_only"] += 1
                    else:
                        counts["fulltext_error"] += 1
                    fh.write(json.dumps(asdict(doc)) + "\n")
                    fh.flush()
                    done.add(doc.doc_id)
                    counts["written"] += 1
                    if counts["written"] % 25 == 0:
                        rate = counts["written"] / max(1e-9, time.time() - started)
                        print(
                            f"[harvest] written={counts['written']} "
                            f"fulltext_ok={counts['fulltext_ok']} "
                            f"{rate * 60:.1f} docs/min",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001 - keep earlier documents
                print(
                    f"[harvest] window stopped: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
    return counts


def load_documents(path: str | Path) -> list[Document]:
    """Read a harvest JSONL back into ``Document`` objects."""
    docs: list[Document] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(Document(**json.loads(line)))
    return docs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", choices=["biorxiv", "medrxiv"], default="biorxiv")
    p.add_argument("--cutoff", required=True, help="ISO date or JSON with 'cutoff'")
    p.add_argument("--until", default=date.today().isoformat())
    p.add_argument("--out", required=True, help="JSONL path")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--abstract-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    counts = harvest(
        server=args.server,
        since=load_cutoff(args.cutoff),
        until=args.until,
        out=Path(args.out),
        max_docs=args.max_docs,
        window_days=args.window_days,
        delay=args.delay,
        fulltext=not args.abstract_only,
        resume=not args.no_resume,
    )
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
