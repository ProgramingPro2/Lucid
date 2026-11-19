"""Build the G-1 corpus register (docs/CORPUS.md).

The CLI only sequences ``pipeline`` functions and writes the corpus card. It is
deliberately possible to run the whole thing with ``--source fake`` and no
network, which is what makes the machinery testable before a real harvest.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.corpus.build \
        --source fake --cutoff 2025-01-01 --dry-run --out artifacts/corpus_dryrun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from .pipeline import (
    build_card,
    clean_text,
    deduplicate,
    drop_low_nll,
    filter_by_date,
    filter_by_license,
    split_documents,
    token_stream_sha256,
)
from .sources import (
    LICENSE_ALLOWLIST,
    BioRxivClient,
    Document,
    FakeClient,
    FdaLabelClient,
    load_cutoff,
)

_STRATA = ("abstract", "introduction", "methods", "results", "discussion")

_WORDS = (
    "cohort assay receptor pathway protein expression inhibition sample "
    "measurement analysis significant baseline treatment response variant "
    "sequence transcription cellular membrane signal kinetics dose endpoint "
    "randomised placebo longitudinal correlation regression tissue culture"
).split()


def _synthetic_documents(cutoff: str, n: int = 60) -> list[Document]:
    """Deterministic stand-in documents for ``--dry-run``.

    They have real section structure, a real date spread, and a mix of licences
    including ones that must be dropped. Nothing about their *content* is meant
    to be scientific; they exist to exercise every gate.
    """
    rng = random.Random(f"corpus-dryrun:{cutoff}:{n}")
    base = date.fromisoformat(cutoff)
    licenses = ["CC BY 4.0"] * 46 + ["CC BY-NC 4.0"] * 7 + ["unknown"] * 4 + ["CC0"] * 3

    docs: list[Document] = []
    for i in range(n):
        # Day offsets straddle the grace boundary so the date gate fires.
        day_offset = rng.randint(-5, 160)
        d = base + timedelta(days=day_offset)

        sections: dict[str, str] = {}
        for stratum in _STRATA:
            body = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(60, 140)))
            sections[stratum] = f"{stratum.capitalize()}. {body}."

        text = "\n\n".join(sections.values())
        # A reference list and a caption that clean_text must remove.
        text += "\n\nReferences\n" + " ".join(f"Ref{i + 1}." for i in range(12))
        text += "\nFigure 1. A synthetic caption that must not survive."

        docs.append(
            Document(
                doc_id=f"fake:{i:04d}",
                source="fake",
                title=f"Synthetic study {i}",
                text=text,
                sections=sections,
                license=licenses[i % len(licenses)],
                date=d.isoformat(),
                meta={"stratum": _STRATA[i % len(_STRATA)]},
            )
        )
    return docs


def _stub_nll(doc: Document) -> float:
    """Deterministic pseudo-NLL in [0, 1) keyed on the document id."""
    h = hashlib.sha256(doc.doc_id.encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _word_tokeniser(text: str) -> list[int]:
    """Stable integer tokenisation for hashing, without a model tokenizer."""
    out: list[int] = []
    for word in text.split():
        digest = hashlib.sha1(word.encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:4], "big"))
    return out


def _acquire(source: str, cutoff: str, max_docs: int, fulltext: bool = False):
    if source == "fake":
        return FakeClient(_synthetic_documents(cutoff, max_docs)), "fake"
    if source == "biorxiv":
        since = cutoff
        until = date.today().isoformat()
        return BioRxivClient(fulltext=fulltext), (since, until)
    if source == "fda":
        return FdaLabelClient(), None
    raise ValueError(source)


def build(
    *,
    source: str = "fake",
    cutoff: str,
    out: str | Path = "artifacts/corpus",
    nll_json: str | Path | None = None,
    dry_run: bool = False,
    max_docs: int = 60,
    seed: int = 0,
    docs_jsonl: str | Path | None = None,
    require_fulltext: bool = False,
    min_doc_chars: int = 2000,
) -> dict:
    """Run every gate and write ``card.json`` + ``manifest.json`` under ``out``."""
    out_dir = Path(out)
    cutoff = load_cutoff(cutoff)

    drops: Counter[str] = Counter()
    if docs_jsonl:
        from .harvest import load_documents

        harvested = load_documents(docs_jsonl)
        source = f"{source}:jsonl"
    else:
        client, spec = _acquire(source, cutoff, max_docs, fulltext=require_fulltext)
        if source == "fake":
            harvested = client.fetch()
        elif source == "biorxiv":
            harvested = client.fetch(spec[0], spec[1], max_docs=max_docs)
        else:
            harvested = client.fetch(max_docs=max_docs)

    n_harvested = len(harvested)
    if require_fulltext:
        full = [d for d in harvested if d.meta.get("fulltext") == "ok"]
        drops["not_fulltext"] += n_harvested - len(full)
        withdrawn = [
            d
            for d in full
            if d.title.upper().startswith("WITHDRAWN")
            or "withdrawal statement" in d.text[:500].lower()
        ]
        drops["withdrawn"] += len(withdrawn)
        withdrawn_ids = {d.doc_id for d in withdrawn}
        full = [d for d in full if d.doc_id not in withdrawn_ids]
        short = [d for d in full if len(d.text) < min_doc_chars]
        drops["too_short"] += len(short)
        harvested = [d for d in full if len(d.text) >= min_doc_chars]

    dated = filter_by_date(harvested, cutoff, grace_days=30)
    drops["date"] += n_harvested - len(dated)

    licensed, licence_drops = filter_by_license(dated, LICENSE_ALLOWLIST)
    for _, reason in licence_drops:
        drops[reason] += 1

    cleaned = [clean_text(d) for d in licensed]
    dense = [d for d in cleaned if d.text.strip()]
    drops["empty_after_clean"] += len(cleaned) - len(dense)

    deduped = deduplicate(dense, threshold=0.95)
    drops["near_duplicate"] += len(dense) - len(deduped)

    if nll_json:
        table = json.loads(Path(nll_json).read_text())
        nll_fn = lambda d: float(table[d.doc_id])  # noqa: E731
    else:
        nll_fn = _stub_nll
    kept, _ = drop_low_nll(deduped, nll_fn, quantile=0.10)
    drops["low_nll"] += len(deduped) - len(kept)

    splits = split_documents(kept, seed=seed)
    stream_hash = token_stream_sha256(kept, _word_tokeniser)

    tokens_per_stratum: Counter[str] = Counter()
    for doc in kept:
        for stratum, text in doc.sections.items():
            tokens_per_stratum[stratum] += len(_word_tokeniser(text))

    card = build_card(
        kept,
        splits,
        dict(drops),
        cutoff,
        extra={
            "base_model": "pending (operator must record the model card)",
            "documents_harvested": n_harvested,
            "token_stream_sha256": stream_hash,
            "nll_decile": "stub" if not nll_json else nll_json,
            "sealed_test_key_holder": "pending",
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    card_path.write_text(json.dumps(card, indent=2))
    manifest = {
        "source": source,
        "cutoff": cutoff,
        "dry_run": dry_run,
        "seed": seed,
        "documents_harvested": n_harvested,
        "documents_retained": len(kept),
        "drops_by_reason": dict(sorted(drops.items())),
        "split_counts": card["split_counts"],
        "tokens_per_stratum": dict(sorted(tokens_per_stratum.items())),
        "token_stream_sha256": stream_hash,
        "card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "card": card,
        "manifest": manifest,
        "drops": dict(sorted(drops.items())),
        "tokens_per_stratum": dict(sorted(tokens_per_stratum.items())),
    }


def _print_report(result: dict) -> None:
    card = result["card"]
    print("=" * 68)
    print("CORPUS BUILD REPORT" + (" (DRY RUN — SYNTHETIC INPUT)" if card else ""))
    print("=" * 68)
    print(f"documents harvested : {card['documents_harvested']}")
    print("drops by reason:")
    for reason, count in result["drops"].items():
        print(f"  {reason:24s} {count}")
    print(f"documents retained  : {card['documents_retained']}")
    print("tokens per stratum:")
    for stratum, n in result["tokens_per_stratum"].items():
        print(f"  {stratum:16s} {n}")
    print(f"split counts        : {card['split_counts']}")
    print(f"token-stream sha256 : {card['token_stream_sha256']}")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["biorxiv", "fda", "fake"], default="fake")
    p.add_argument("--cutoff", required=True, help="ISO date or JSON with 'cutoff'")
    p.add_argument("--out", default="artifacts/corpus")
    p.add_argument("--nll-json", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-docs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--docs-jsonl",
        default=None,
        help="build from a harvest JSONL instead of the network",
    )
    p.add_argument(
        "--require-fulltext",
        action="store_true",
        help="drop abstract-only documents (live harvest with JATS enabled)",
    )
    p.add_argument(
        "--min-doc-chars",
        type=int,
        default=2000,
        help="with --require-fulltext, drop documents shorter than this",
    )
    args = p.parse_args(argv)

    if not args.dry_run and args.source == "fake" and not args.docs_jsonl:
        print("refusing: --source fake requires --dry-run", file=sys.stderr)
        return 2
    if not args.dry_run and not args.nll_json:
        print(
            "refusing a live build without --nll-json: the memorisation screen "
            "requires base-model NLL per document (docs/CORPUS.md §4 step 7)",
            file=sys.stderr,
        )
        return 2

    result = build(
        source=args.source,
        cutoff=args.cutoff,
        out=args.out,
        nll_json=args.nll_json,
        dry_run=args.dry_run,
        max_docs=args.max_docs,
        seed=args.seed,
        docs_jsonl=args.docs_jsonl,
        require_fulltext=args.require_fulltext,
        min_doc_chars=args.min_doc_chars,
    )
    _print_report(result)
    print(f"wrote {Path(args.out) / 'card.json'} and manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
