"""Harvest a resumable JSONL register from bioRxiv/medRxiv.

Harvesting is separated from ``build`` because the gates need a per-document NLL
table, and re-downloading a multi-hour corpus to re-run a gate is wasteful. This
CLI only fetches and writes; ``build.py --docs-jsonl`` applies every gate.

Full text is fetched only for licences in ``LICENSE_ALLOWLIST``. A failed paper
is written with ``meta.fulltext`` set to the reason, so the failure is counted
rather than hidden and is never relabelled ``abstract_only``. The file is
append-only and flushed per document; rerunning with ``--resume`` skips doc ids
already present. The last completed page cursor for each window is stored beside
the JSONL, so a retry continues at the next page instead of cursor 0.

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
    HostRefused,
    backoff_seconds,
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


def cursor_path_for(out: Path) -> Path:
    """Sidecar mapping ``server|lo|hi`` to the next page cursor to request."""
    return out.with_name(out.name + ".cursors.json")


def _load_cursors(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _flush_cursors(path: Path, cursors: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cursors))
    tmp.replace(path)


def _format_exc(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"cause={type(cause).__name__}: {cause}")
    return " | ".join(parts)


def harvest(
    *,
    server: str,
    since: str,
    until: str,
    out: Path,
    max_docs: int,
    window_days: int = 7,
    delay: float = 3.0,
    fulltext: bool = True,
    resume: bool = True,
    window_retries: int = 3,
    cooldown: float = 120.0,
) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out) if resume else set()
    client = BioRxivClient(server=server, delay=delay, fulltext=fulltext)
    cursors_file = cursor_path_for(out)
    cursors = _load_cursors(cursors_file) if resume else {}

    counts: dict = {
        "metadata": 0,
        "skipped_existing": 0,
        "licence_allowed": 0,
        "licence_excluded": 0,
        "no_jats_link": 0,
        "fulltext_ok": 0,
        "fulltext_empty": 0,
        "fulltext_error": 0,
        "fulltext_error_reasons": {},
        "abstract_only": 0,
        "written": 0,
    }
    print(
        f"[harvest] delay={delay}s retries={window_retries} cooldown_cap={cooldown}s "
        f"ua={BioRxivClient.UA}",
        flush=True,
    )
    started = time.time()

    def _tally(doc: Document) -> None:
        status = doc.meta.get("fulltext", "")
        if status == "ok":
            counts["fulltext_ok"] += 1
        elif status == "empty":
            counts["fulltext_empty"] += 1
        elif status == "no_jats_link":
            counts["no_jats_link"] += 1
        elif status == "abstract_only":
            counts["abstract_only"] += 1
        elif status == "licence_excluded":
            counts["licence_excluded"] += 1
        else:
            counts["fulltext_error"] += 1
            reasons = counts["fulltext_error_reasons"]
            reasons[status] = reasons.get(status, 0) + 1

    with out.open("a") as fh:
        for lo, hi in _windows(
            cutoff_plus_grace(since), date.fromisoformat(until), window_days
        ):
            if counts["written"] >= max_docs:
                break
            key = f"{server}|{lo}|{hi}"
            written_at_window = counts["written"]
            for attempt in range(1, window_retries + 1):
                client.start_cursor = int(cursors.get(key, 0) or 0)

                def _on_page(cursor: int, key: str = key) -> None:
                    cursors[key] = int(cursor)
                    if resume:
                        _flush_cursors(cursors_file, cursors)

                client.on_page_complete = _on_page
                where = (
                    f" at cursor {client.start_cursor}"
                    if client.start_cursor
                    else ""
                )
                print(
                    f"[harvest] {server} {lo}..{hi}{where} "
                    f"(attempt {attempt}/{window_retries})",
                    flush=True,
                )
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
                        if not allowed:
                            doc.meta["fulltext"] = "licence_excluded"
                        elif not fulltext:
                            doc.meta["fulltext"] = "abstract_only"
                        elif not doc.meta.get("jatsxml"):
                            doc.meta["fulltext"] = "no_jats_link"
                        else:
                            doc = client.fetch_fulltext(doc)
                        _tally(doc)
                        fh.write(json.dumps(asdict(doc)) + "\n")
                        fh.flush()
                        done.add(doc.doc_id)
                        counts["written"] += 1
                        if counts["written"] % 25 == 0:
                            elapsed = max(1e-9, time.time() - started)
                            rate = counts["written"] / elapsed
                            ft_rate = counts["fulltext_ok"] / elapsed
                            print(
                                f"[harvest] written={counts['written']} "
                                f"fulltext_ok={counts['fulltext_ok']} "
                                f"{rate * 60:.1f} docs/min "
                                f"fulltext {ft_rate * 60:.1f}/min",
                                flush=True,
                            )
                    break
                except HostRefused as exc:
                    print(
                        f"[harvest] host refused; stopping rather than retrying: "
                        f"{_format_exc(exc)}",
                        flush=True,
                    )
                    counts["aborted"] = "host_refused"
                    return counts
                except Exception as exc:  # noqa: BLE001 - keep earlier documents
                    print(
                        f"[harvest] window stopped: {_format_exc(exc)}",
                        flush=True,
                    )
                    made_progress = counts["written"] > written_at_window
                    if attempt < window_retries:
                        wait = backoff_seconds(
                            attempt - 1, base=5.0, cap=cooldown
                        )
                        print(
                            f"[harvest] retrying window in {wait:.0f}s from cursor "
                            f"{cursors.get(key, 0)} "
                            "(documents already written are kept)",
                            flush=True,
                        )
                        time.sleep(wait)
                        continue
                    if not made_progress:
                        print(
                            "[harvest] no documents written for this window after "
                            "retries; stopping so later windows are not hammered",
                            flush=True,
                        )
                        counts["aborted"] = "no_progress"
                        return counts
                    print("[harvest] giving up on this window", flush=True)
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
    p.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="seconds between requests (polite default 3)",
    )
    p.add_argument("--abstract-only", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="skip doc ids already in the JSONL (this is the default)",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--window-retries",
        type=int,
        default=3,
        help="retry a rate-limited window before giving up on it",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=120.0,
        help="cap in seconds on the exponential backoff between window retries",
    )
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
        resume=args.resume and not args.no_resume,
        window_retries=args.window_retries,
        cooldown=args.cooldown,
    )
    print(json.dumps(counts, indent=2))
    if counts.get("aborted"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
