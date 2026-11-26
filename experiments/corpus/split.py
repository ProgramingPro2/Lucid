"""Document-disjoint split and the reproducible token-stream hash.

G-1d and G-1e (docs/CORPUS.md §5). The split is a hash bucket, not a shuffle:
the assignment of a document is a function of the seed, that document's
stratum, and its id. Adding a document later cannot move a document that was
already assigned, which is the property a growing harvest needs. A random
permutation of the current id list does not have it.

The token-stream digest is taken over documents sorted by ``doc_id``. Harvest
arrival order is not an input. Token ids are written as unsigned 64-bit
integers, so the digest does not depend on ``str`` of a float, dict order, or
the wall clock.

A human can re-check the rebuild path on a cheap prefix of a JSONL::

    python -m experiments.corpus.split verify-rebuild \\
        --jsonl artifacts/corpus/biorxiv_2026.jsonl \\
        --cutoff 2025-06-01 \\
        --limit 200 \\
        --out-a /tmp/g1e-a \\
        --out-b /tmp/g1e-b

``--limit`` freezes the first N complete lines and builds that prefix twice.
Matching hashes prove this code path on that prefix. They do not prove the
hash of the full retained corpus, they do not rederive an older card, and
they do not stay equal once the prefix itself changes. ``--full`` drops the
cap and is no longer a cheap check; it still only covers the file bytes read
on that invocation, and this command applies the stub NLL screen unless
``--nll-json`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .sources import Document

# One document, one bucket. 1e7 keeps the default 80/10/10 cuts on integers
# (8_000_000 and 9_000_000) so the threshold is not a float formatting accident.
SPLIT_MODULUS = 10_000_000
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "test")

SPLIT_ASSIGNMENT_RULE = (
    f"hash-bucket modulus {SPLIT_MODULUS}; "
    "payload is the length-prefixed utf-8 triple (decimal seed, stratum, doc_id); "
    "bucket = sha256(payload)[:8] as a big-endian integer modulo the modulus; "
    "train if bucket < round(train_fraction*modulus), "
    "dev if bucket < that cut plus round(dev_fraction*modulus), else test; "
    "default fractions 0.8/0.1/0.1; "
    "stratum = document_stratum (meta override, else the alphabetically first "
    "section name, else 'unknown'); "
    "the assignment of a document depends only on seed, stratum and doc_id, "
    "so appending a document cannot reshuffle documents already assigned; "
    "ids within each split are sorted by doc_id; "
    "a repeated doc_id raises"
)

TOKEN_STREAM_HASH_RULE = (
    "tokeniser: one id per whitespace-separated word, "
    "sha1(utf-8 word) first 4 bytes as a big-endian int; "
    "documents sorted by doc_id; "
    "each document framed as u32be(len(utf-8 doc_id)) || doc_id || "
    "u64be(token count) || u64be(token id) repeated; "
    "sha256 of that byte string; "
    "token ids must be non-negative ints (bool rejected); "
    "the hashed body is doc.text; "
    "harvest arrival order, section-dict order, float formatting and "
    "wall-clock are not inputs; a repeated doc_id is refused"
)

# What a bounded rebuild actually establishes. Kept here so the command and
# the tests report the same limit, rather than a softer claim in one of them.
BOUNDED_REBUILD_PROVES = (
    "two independent builds of this frozen JSONL prefix produce the same "
    "token_stream_sha256 and the same split counts"
)
BOUNDED_REBUILD_DOES_NOT_PROVE = (
    "the hash of the full retained corpus, reproducibility of a later longer "
    "file, the historical card digests, or a build that uses a real NLL table"
)


def corpus_word_tokeniser(text: str) -> list[int]:
    """Stable integer tokenisation for hashing, without a model tokenizer.

    One id per whitespace-separated word. The id is the first four bytes of
    SHA-1, so the mapping does not depend on interpreter dict order or on
    how a float would render.
    """
    out: list[int] = []
    for word in text.split():
        digest = hashlib.sha1(word.encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:4], "big"))
    return out


def _length_prefixed(*parts: bytes) -> bytes:
    blob = bytearray()
    for part in parts:
        blob += len(part).to_bytes(4, "big")
        blob += part
    return bytes(blob)


def _assignment_bucket(seed: int, stratum: str, doc_id: str) -> int:
    payload = _length_prefixed(
        str(int(seed)).encode("utf-8"),
        stratum.encode("utf-8"),
        doc_id.encode("utf-8"),
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % SPLIT_MODULUS


def _fraction_cuts(fractions: tuple[float, float, float]) -> tuple[int, int]:
    if len(fractions) != 3:
        raise ValueError("fractions must be train, dev, test")
    if any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be non-negative and sum to 1")
    train_cut = int(round(fractions[0] * SPLIT_MODULUS))
    dev_width = int(round(fractions[1] * SPLIT_MODULUS))
    if train_cut + dev_width > SPLIT_MODULUS:
        raise ValueError("rounded fraction cuts exceed the modulus")
    return train_cut, train_cut + dev_width


def verify_document_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    """Raise if any document id is shared across splits or repeated in one.

    Identity is exact string equality. Ids that only look alike (``doc``,
    ``doc:1``, ``doc:10``, a DOI that is a prefix of another DOI) are different
    documents. This function does not compare prefixes, and it does not drop
    duplicates: a repeated id is an error.
    """
    missing = [name for name in SPLIT_NAMES if name not in splits]
    if missing:
        raise ValueError(f"splits missing {missing}")
    extra = [name for name in splits if name not in SPLIT_NAMES]
    if extra:
        raise ValueError(f"unknown splits {extra}")
    owner: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for doc_id in splits[name]:
            if not isinstance(doc_id, str):
                raise TypeError(
                    f"document id in {name} must be a str, got {type(doc_id).__name__}"
                )
            previous = owner.get(doc_id)
            if previous is None:
                owner[doc_id] = name
                continue
            if previous == name:
                raise ValueError(f"document {doc_id!r} appears twice in {name}")
            raise ValueError(
                f"document {doc_id!r} is in both {previous} and {name}"
            )


def split_documents(
    docs: list[Document],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    strata_key: str = "stratum",
) -> dict[str, list[str]]:
    """Stratified document split that does not reshuffle when the corpus grows.

    Each document is bucketed by ``sha256(seed, stratum, doc_id)`` into train,
    dev, or test at the requested fractions (default 80/10/10). The stratum is
    :func:`experiments.corpus.pipeline.document_stratum`. Nothing about the
    other documents — not their number, not harvest order, not a permutation
    of the id list — enters the bucket. A document added later gets its own
    bucket and leaves every existing assignment where it was.

    Balance across strata is the expected value of that per-document draw, not
    a quota. A small stratum can miss dev or test; the card records the counts
    that actually occurred. Quotas would require looking at the rest of the
    list, and that is the reshuffle this function exists to avoid.

    Ids are sorted within each split. A document id that appears twice raises
    via :func:`verify_document_disjoint` rather than being silently dropped.
    """
    # Imported lazily: pipeline imports this module.
    from .pipeline import document_stratum

    train_cut, dev_cut = _fraction_cuts(fractions)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for doc in docs:
        stratum = document_stratum(doc, strata_key)
        bucket = _assignment_bucket(seed, stratum, doc.doc_id)
        if bucket < train_cut:
            splits["train"].append(doc.doc_id)
        elif bucket < dev_cut:
            splits["dev"].append(doc.doc_id)
        else:
            splits["test"].append(doc.doc_id)
    for name in SPLIT_NAMES:
        splits[name].sort()
    verify_document_disjoint(splits)
    assigned = [doc_id for name in SPLIT_NAMES for doc_id in splits[name]]
    if sorted(assigned) != sorted(doc.doc_id for doc in docs):
        raise AssertionError("split did not assign every document exactly once")
    return splits


def counts_by_split_and_stratum(
    docs: list[Document],
    splits: Mapping[str, Sequence[str]],
    strata_key: str = "stratum",
) -> dict[str, dict[str, int]]:
    """Counts per split per stratum, with an explicit zero for a missing cell.

    The stratum set is the strata of the assigned documents. A stratum that
    landed entirely in train still appears under dev and test with count 0,
    so a starved cell is visible on the card.
    """
    from .pipeline import document_stratum

    verify_document_disjoint(splits)
    by_id: dict[str, str] = {}
    for doc in docs:
        if doc.doc_id in by_id:
            raise ValueError(f"document {doc.doc_id!r} appears twice")
        by_id[doc.doc_id] = document_stratum(doc, strata_key)
    assigned = [doc_id for name in SPLIT_NAMES for doc_id in splits[name]]
    missing = [doc_id for doc_id in assigned if doc_id not in by_id]
    if missing:
        raise ValueError(f"split contains unknown document {missing[0]!r}")
    strata = sorted({by_id[doc_id] for doc_id in assigned})
    out: dict[str, dict[str, int]] = {}
    for name in SPLIT_NAMES:
        counts: Counter[str] = Counter(by_id[doc_id] for doc_id in splits[name])
        out[name] = {stratum: int(counts.get(stratum, 0)) for stratum in strata}
    return out


def _encode_token(token: object) -> bytes:
    if isinstance(token, bool) or not isinstance(token, int):
        raise TypeError(
            f"token id must be a non-bool int, got {type(token).__name__}"
        )
    if token < 0 or token > 2**64 - 1:
        raise ValueError(f"token id {token} does not fit in uint64")
    return token.to_bytes(8, "big")


def token_stream_sha256(docs: list[Document], tokeniser) -> str:
    """SHA-256 of the token ids and the document boundaries, in doc_id order.

    ``tokeniser`` maps ``doc.text`` to a sequence of token ids. Documents are
    sorted by ``doc_id`` before any byte is hashed, so two rebuilds of the
    same documents agree even when the harvest appended them in a different
    order or a directory listing changed. The framing is length-prefixed and
    the ids are uint64, so the digest covers the ids in order and the boundary
    between documents, and it does not depend on float formatting.

    A repeated ``doc_id`` is refused. Sort is stable for ties, and hashing
    that tie would make the digest depend on input order.
    """
    ordered = sorted(docs, key=lambda d: d.doc_id)
    ids = [doc.doc_id for doc in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "token stream has a repeated doc_id; refusing an order-dependent hash"
        )
    hasher = hashlib.sha256()
    for doc in ordered:
        doc_id = doc.doc_id.encode("utf-8")
        hasher.update(len(doc_id).to_bytes(4, "big"))
        hasher.update(doc_id)
        token_ids = [_encode_token(token) for token in tokeniser(doc.text)]
        hasher.update(len(token_ids).to_bytes(8, "big"))
        for encoded in token_ids:
            hasher.update(encoded)
    return hasher.hexdigest()


def snapshot_jsonl_prefix(path: str | Path, limit: int | None) -> bytes:
    """Raw bytes of the first ``limit`` complete JSONL records.

    A line counts only when it ends in a newline, which drops a torn tail the
    live harvest may still be writing. Blank lines are kept in the byte string
    (the loader skips them) and do not count toward ``limit``. ``limit is
    None`` reads every newline-terminated line and still drops a torn tail.
    This function does not write, and it does not parse.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1, or None for every complete line")
    chunks: list[bytes] = []
    kept = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            chunks.append(line)
            if line.strip():
                kept += 1
                if limit is not None and kept >= limit:
                    break
    return b"".join(chunks)


def verify_clean_rebuild(
    jsonl: str | Path,
    out_a: str | Path,
    out_b: str | Path,
    *,
    cutoff: str,
    limit: int | None = 200,
    seed: int = 0,
    nll_json: str | Path | None = None,
) -> dict:
    """Build one JSONL prefix twice, into two directories, and compare hashes.

    The prefix is read twice. If those reads differ, the source moved and the
    function does not build; a mismatch of that kind proves nothing about the
    hash. When they agree, each copy is built with :func:`experiments.corpus.build.build`.
    Without ``nll_json`` that build applies the stub NLL screen.

    See :data:`BOUNDED_REBUILD_PROVES` and :data:`BOUNDED_REBUILD_DOES_NOT_PROVE`.
    """
    # Local import: build imports the pipeline, which imports this module.
    from .build import build

    source = Path(jsonl)
    first = snapshot_jsonl_prefix(source, limit)
    second = snapshot_jsonl_prefix(source, limit)
    report = {
        "jsonl": str(source),
        "limit": limit,
        "cutoff": cutoff,
        "seed": seed,
        "nll": str(nll_json) if nll_json else "stub quantile 0.10",
        "prefix_bytes": len(first),
        "proves": BOUNDED_REBUILD_PROVES,
        "does_not_prove": BOUNDED_REBUILD_DOES_NOT_PROVE,
    }
    if first != second:
        report["equal"] = False
        report["reason"] = "source changed between the two reads"
        return report

    dir_a = Path(out_a)
    dir_b = Path(out_b)
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    slice_a = dir_a / "slice.jsonl"
    slice_b = dir_b / "slice.jsonl"
    slice_a.write_bytes(first)
    slice_b.write_bytes(second)
    built_a = build(
        source="biorxiv",
        cutoff=cutoff,
        out=dir_a,
        nll_json=nll_json,
        seed=seed,
        docs_jsonl=slice_a,
    )
    built_b = build(
        source="biorxiv",
        cutoff=cutoff,
        out=dir_b,
        nll_json=nll_json,
        seed=seed,
        docs_jsonl=slice_b,
    )
    hash_a = built_a["card"]["token_stream_sha256"]
    hash_b = built_b["card"]["token_stream_sha256"]
    splits_a = built_a["card"]["split_counts"]
    splits_b = built_b["card"]["split_counts"]
    report.update(
        {
            "equal": hash_a == hash_b and splits_a == splits_b,
            "hash_a": hash_a,
            "hash_b": hash_b,
            "split_counts_a": splits_a,
            "split_counts_b": splits_b,
            "documents_harvested_a": built_a["card"]["documents_harvested"],
            "documents_retained_a": built_a["card"]["documents_retained"],
            "documents_retained_b": built_b["card"]["documents_retained"],
            "reason": "ok" if hash_a == hash_b else "token_stream_sha256 differs",
        }
    )
    return report


def _print_rebuild_report(report: dict) -> None:
    print("clean rebuild")
    print(f"  jsonl            : {report['jsonl']}")
    print(f"  limit            : {report['limit']}")
    print(f"  nll              : {report['nll']}")
    print(f"  proves           : {report['proves']}")
    print(f"  does not prove   : {report['does_not_prove']}")
    if "hash_a" in report:
        print(f"  hash A           : {report['hash_a']}")
        print(f"  hash B           : {report['hash_b']}")
        print(f"  retained         : {report['documents_retained_a']}")
        print(f"  split counts     : {report['split_counts_a']}")
    print(f"  equal            : {report['equal']}")
    if report.get("reason") and report["reason"] != "ok":
        print(f"  reason           : {report['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a JSONL prefix twice and compare token_stream_sha256. "
            "A bounded run proves the code path on that prefix only."
        )
    )
    parser.add_argument("command", choices=["verify-rebuild"])
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-a", required=True)
    parser.add_argument("--out-b", required=True)
    parser.add_argument("--cutoff", required=True, help="ISO date passed to the build")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="complete JSONL lines to rebuild (default 200). Ignored with --full",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "rebuild every newline-terminated line. Not a cheap check, and "
            "still not a proof about a file that keeps growing"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nll-json", default=None)
    args = parser.parse_args(argv)
    limit = None if args.full else args.limit
    try:
        report = verify_clean_rebuild(
            args.jsonl,
            args.out_a,
            args.out_b,
            cutoff=args.cutoff,
            limit=limit,
            seed=args.seed,
            nll_json=args.nll_json,
        )
    except (OSError, ValueError) as exc:
        print(f"verify-rebuild failed: {exc}", file=sys.stderr)
        return 2
    _print_rebuild_report(report)
    if report.get("reason") == "source changed between the two reads":
        return 2
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
