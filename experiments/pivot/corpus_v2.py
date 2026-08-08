"""Build CORPUS V2 with real document / paragraph boundaries.

Documents are separated by a ``===DOC===`` delimiter line; paragraphs within a
document are separated by blank lines. Sentence order is preserved. Sentences
are produced by the repo's ``segment_sentences`` so downstream tooling stays
consistent with the existing JEPA / sentence_pred pipelines.

Sources (priority):
  1. ``wikitext-103`` — WikiText-103-raw-v1 train (real Wikipedia articles)
  2. ``chain`` — synthetic-chain corpus v2b: greedy MiniLM nearest-neighbor
     ordering of existing ``corpus_1m`` blocks (topic persistence, not narrative)

Cache downloads under ``data/corpus_v2/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.data.segment import segment_sentences  # noqa: E402

DOC_DELIM = "===DOC==="
DEFAULT_OUT = _ROOT / "data" / "corpus_v2.txt"
DEFAULT_CACHE = _ROOT / "data" / "corpus_v2"
DEFAULT_CHAIN_SRC = _ROOT / "data" / "corpus_1m.txt"
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_FILES = (
    "wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "wikitext-103-raw-v1/train-00001-of-00002.parquet",
)

# Level-1 WikiText heading: " = Title = " (not "== Section ==")
_ARTICLE_HEADING = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")
_ANY_HEADING = re.compile(r"^\s*=+\s+.+=+\s*$")


def detok_wikitext(s: str) -> str:
    """WikiText-style detokenization (matches sentence_pred.data.detok_text)."""
    s = re.sub(r"\s*@-@\s*", "-", s)
    s = re.sub(r"\s*@,@\s*", ",", s)
    s = re.sub(r"\s*@\.@\s*", ".", s)
    for punct in ",.!?;:":
        s = re.sub(rf"\s+{re.escape(punct)}", punct, s)
    s = re.sub(r'"\s+', '"', s)
    s = re.sub(r'\s+"(?=\W|$)', '"', s)
    return s.rstrip()


@dataclass
class CorpusReport:
    source: str
    out: str
    n_docs: int
    n_paragraphs: int
    n_sentences: int
    median_paragraphs_per_doc: float
    median_sentences_per_doc: float
    label: str
    elapsed_s: float


def _is_article_heading(line: str) -> bool:
    s = line.strip()
    if not s or not _ARTICLE_HEADING.match(s):
        return False
    # Reject multi-equals section headings that somehow matched
    return not s.lstrip().startswith("==")


def _is_any_heading(line: str) -> bool:
    return bool(_ANY_HEADING.match(line.strip()))


def _paragraph_from_lines(lines: Sequence[str]) -> list[str]:
    """Join non-heading content lines into one paragraph; return segment_sentences."""
    parts: list[str] = []
    for ln in lines:
        t = ln.strip()
        if not t or _is_any_heading(t):
            continue
        parts.append(detok_wikitext(t))
    if not parts:
        return []
    text = " ".join(parts)
    return segment_sentences(text)


def iter_wikitext_raw_lines(paths: Sequence[Path]) -> Iterator[str]:
    """Yield text lines from WikiText parquet shards in order."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pyarrow required to read WikiText parquet") from e

    for path in paths:
        table = pq.read_table(path, columns=["text"])
        col = table.column("text")
        for i in range(table.num_rows):
            val = col[i].as_py()
            yield "" if val is None else str(val)


def parse_wikitext_articles(
    lines: Iterable[str],
    min_paras: int = 4,
    max_docs: int = 200_000,
) -> list[list[list[str]]]:
    """Parse WikiText-raw into documents of paragraphs of sentences.

    Article boundary = level-1 heading ``= Title =``. Blank lines flush the
    current paragraph. Section headings are skipped (not paragraphs).
    Returns list[doc] where doc = list[paragraph] and paragraph = list[sentence].
    """
    docs: list[list[list[str]]] = []
    cur_paras: list[list[str]] = []
    buf: list[str] = []
    in_article = False

    def flush_para() -> None:
        nonlocal buf
        sents = _paragraph_from_lines(buf)
        buf = []
        if sents:
            cur_paras.append(sents)

    def flush_doc() -> None:
        nonlocal cur_paras, in_article
        flush_para()
        if len(cur_paras) >= min_paras:
            docs.append(cur_paras)
        cur_paras = []
        in_article = False

    for raw in lines:
        # Parquet cells may contain embedded newlines (title line ends with \n)
        for line in str(raw).splitlines() or [""]:
            if _is_article_heading(line):
                if in_article or cur_paras or buf:
                    flush_doc()
                in_article = True
                buf = []
                continue
            if not in_article:
                continue
            if not line.strip():
                flush_para()
                continue
            if _is_any_heading(line):
                flush_para()
                continue
            buf.append(line)
        if len(docs) >= max_docs:
            break

    if in_article or cur_paras or buf:
        flush_doc()

    return docs[:max_docs]


def download_wikitext_raw(
    cache_dir: Path,
    timeout_s: float = 600.0,
) -> list[Path]:
    """Download WikiText-103-raw train shards into cache_dir. Raises on timeout."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = [cache_dir / f for f in WIKITEXT_FILES]
    if all(p.is_file() and p.stat().st_size > 0 for p in existing):
        return existing

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    paths: list[Path] = []
    for rel in WIKITEXT_FILES:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"WikiText download exceeded {timeout_s}s; use source=chain"
            )
        local = cache_dir / rel
        if local.is_file() and local.stat().st_size > 0:
            paths.append(local)
            continue
        p = hf_hub_download(
            WIKITEXT_REPO,
            rel,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        paths.append(Path(p))
        # Size guard: refuse >2GB total
        total = sum(x.stat().st_size for x in paths if x.is_file())
        if total > 2 * 1024**3:
            raise RuntimeError(
                f"WikiText download exceeds 2GB ({total} bytes); use source=chain"
            )
    return paths


def load_wikitext_from_raw_path(path: Path) -> Iterator[str]:
    """Load a plain-text wiki.train.raw (or similar) line file."""
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n")


def write_corpus(
    docs: Sequence[Sequence[Sequence[str]]],
    out: Path,
) -> CorpusReport:
    """Write docs as ===DOC=== + blank-line paragraphs; return counts."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_paras = 0
    n_sents = 0
    paras_per_doc: list[int] = []
    sents_per_doc: list[int] = []

    with out.open("w", encoding="utf-8") as f:
        for di, doc in enumerate(docs):
            if di:
                f.write(f"\n{DOC_DELIM}\n\n")
            else:
                f.write(f"{DOC_DELIM}\n\n")
            doc_sents = 0
            for pi, para in enumerate(doc):
                if pi:
                    f.write("\n")
                # One paragraph = sentences joined by space on a single block;
                # blank line between paragraphs. Keep each sentence on its own
                # line inside the block for easier inspection / downstream split.
                f.write(" ".join(para).strip() + "\n")
                n_paras += 1
                n_sents += len(para)
                doc_sents += len(para)
            paras_per_doc.append(len(doc))
            sents_per_doc.append(doc_sents)

    return CorpusReport(
        source="",
        out=str(out),
        n_docs=len(docs),
        n_paragraphs=n_paras,
        n_sentences=n_sents,
        median_paragraphs_per_doc=float(median(paras_per_doc)) if paras_per_doc else 0.0,
        median_sentences_per_doc=float(median(sents_per_doc)) if sents_per_doc else 0.0,
        label="",
        elapsed_s=0.0,
    )


def load_corpus_v2(path: Path | str) -> list[list[list[str]]]:
    """Load corpus_v2.txt back into docs of paragraphs of sentences."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    raw_docs = text.split(DOC_DELIM)
    docs: list[list[list[str]]] = []
    for raw in raw_docs:
        raw = raw.strip()
        if not raw:
            continue
        paras: list[list[str]] = []
        for block in re.split(r"\n\s*\n", raw):
            block = block.strip()
            if not block:
                continue
            sents = segment_sentences(block)
            if sents:
                paras.append(sents)
        if paras:
            docs.append(paras)
    return docs


def _load_minilm():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def _encode_texts(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 2000,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_chain_corpus(
    blocks_path: Path,
    min_paras: int = 4,
    max_docs: int = 200_000,
    seed: int = 42,
    max_blocks: int | None = 80_000,
    chain_len: int = 8,
) -> list[list[list[str]]]:
    """Synthetic-chain v2b: greedy MiniLM NN ordering of existing corpus blocks.

    Each corpus_1m blank-line block becomes one paragraph. Blocks are chained
    by cosine nearest-neighbor so consecutive paragraphs share topic signal.
    """
    blocks_path = Path(blocks_path)
    raw = blocks_path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    rng = np.random.RandomState(seed)
    if max_blocks is not None and len(blocks) > max_blocks:
        idx = rng.choice(len(blocks), size=max_blocks, replace=False)
        blocks = [blocks[i] for i in sorted(idx.tolist())]

    paras: list[list[str]] = []
    texts: list[str] = []
    for b in blocks:
        sents = segment_sentences(detok_wikitext(b))
        if len(sents) < 1:
            continue
        paras.append(sents)
        texts.append(" ".join(sents))

    if len(paras) < min_paras:
        raise ValueError(f"Not enough usable blocks in {blocks_path}")

    print(f"[chain] encoding {len(texts)} blocks with MiniLM...", flush=True)
    model = _load_minilm()
    emb = _encode_texts(model, texts)
    n = emb.shape[0]
    used = np.zeros(n, dtype=bool)
    docs: list[list[list[str]]] = []

    # Greedy: pick unused seed, then repeatedly append nearest unused neighbor
    order = rng.permutation(n)
    for seed_i in order:
        if used[seed_i]:
            continue
        chain = [int(seed_i)]
        used[seed_i] = True
        cur = int(seed_i)
        target_len = int(chain_len)
        while len(chain) < target_len:
            sims = emb @ emb[cur]
            sims[used] = -np.inf
            nxt = int(np.argmax(sims))
            if not np.isfinite(sims[nxt]):
                break
            used[nxt] = True
            chain.append(nxt)
            cur = nxt
        if len(chain) >= min_paras:
            docs.append([paras[i] for i in chain])
        if len(docs) >= max_docs:
            break
        if used.all():
            break

    return docs


def build_wikitext_corpus(
    cache_dir: Path,
    wikitext_raw: Path | None,
    min_paras: int,
    max_docs: int,
    download_timeout_s: float = 600.0,
) -> list[list[list[str]]]:
    if wikitext_raw is not None:
        path = Path(wikitext_raw)
        if not path.exists():
            raise FileNotFoundError(f"--wikitext-raw not found: {path}")
        print(f"[wikitext] reading local raw: {path}", flush=True)
        lines = load_wikitext_from_raw_path(path)
        return parse_wikitext_articles(lines, min_paras=min_paras, max_docs=max_docs)

    print(f"[wikitext] ensuring shards under {cache_dir}", flush=True)
    paths = download_wikitext_raw(cache_dir, timeout_s=download_timeout_s)
    # Small frac verify then full: parse first shard first
    print(f"[wikitext] verifying parse on first shard ({paths[0].name})...", flush=True)
    sample_docs = parse_wikitext_articles(
        iter_wikitext_raw_lines(paths[:1]),
        min_paras=min_paras,
        max_docs=min(50, max_docs),
    )
    if not sample_docs:
        raise RuntimeError("WikiText sample parse produced 0 docs; aborting")
    print(
        f"[wikitext] sample ok: {len(sample_docs)} docs, "
        f"median paras={median(len(d) for d in sample_docs):.0f}",
        flush=True,
    )
    print("[wikitext] parsing full train shards...", flush=True)
    return parse_wikitext_articles(
        iter_wikitext_raw_lines(paths),
        min_paras=min_paras,
        max_docs=max_docs,
    )


def build_corpus_v2(
    source: str = "wikitext-103",
    out: str | Path = DEFAULT_OUT,
    min_paras: int = 4,
    max_docs: int = 200_000,
    seed: int = 42,
    wikitext_raw: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE,
    chain_blocks: str | Path = DEFAULT_CHAIN_SRC,
    download_timeout_s: float = 600.0,
    allow_fallback: bool = True,
) -> CorpusReport:
    """Build corpus v2 and write ``out``. Returns a ``CorpusReport``.

    If ``source='wikitext-103'`` fails (timeout / size / parse), and
    ``allow_fallback`` is True, falls back to synthetic-chain v2b.
    """
    t0 = time.time()
    source = source.strip().lower()
    used = source
    label = ""
    docs: list[list[list[str]]]

    if source in ("wikitext-103", "wikitext", "wiki"):
        try:
            docs = build_wikitext_corpus(
                cache_dir=Path(cache_dir),
                wikitext_raw=Path(wikitext_raw) if wikitext_raw else None,
                min_paras=min_paras,
                max_docs=max_docs,
                download_timeout_s=download_timeout_s,
            )
            used = "wikitext-103"
            label = "WikiText-103-raw-v1 (real articles)"
        except Exception as e:
            if not allow_fallback:
                raise
            print(
                f"[wikitext] FAILED ({type(e).__name__}: {e}); "
                f"falling back to synthetic-chain v2b",
                flush=True,
            )
            docs = build_chain_corpus(
                Path(chain_blocks),
                min_paras=min_paras,
                max_docs=max_docs,
                seed=seed,
            )
            used = "chain"
            label = "synthetic-chain corpus v2b (MiniLM NN over corpus_1m)"
    elif source in ("chain", "v2b", "synthetic-chain"):
        docs = build_chain_corpus(
            Path(chain_blocks),
            min_paras=min_paras,
            max_docs=max_docs,
            seed=seed,
        )
        used = "chain"
        label = "synthetic-chain corpus v2b (MiniLM NN over corpus_1m)"
    else:
        raise ValueError(f"Unknown source={source!r}; use wikitext-103|chain")

    if not docs:
        raise RuntimeError(f"No documents produced for source={used}")

    report = write_corpus(docs, Path(out))
    report.source = used
    report.label = label
    report.elapsed_s = round(time.time() - t0, 2)

    meta_path = Path(out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Lucid CORPUS V2")
    p.add_argument(
        "--source",
        default="wikitext-103",
        choices=["wikitext-103", "chain"],
        help="Corpus source (default: wikitext-103)",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--min-paras", type=int, default=4)
    p.add_argument("--max-docs", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--wikitext-raw",
        default=None,
        help="Path to local wiki.train.raw (skip HF download)",
    )
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--chain-blocks", default=str(DEFAULT_CHAIN_SRC))
    p.add_argument("--download-timeout", type=float, default=600.0)
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fall back to chain if wikitext fails",
    )
    args = p.parse_args(argv)

    build_corpus_v2(
        source=args.source,
        out=args.out,
        min_paras=args.min_paras,
        max_docs=args.max_docs,
        seed=args.seed,
        wikitext_raw=args.wikitext_raw,
        cache_dir=args.cache_dir,
        chain_blocks=args.chain_blocks,
        download_timeout_s=args.download_timeout,
        allow_fallback=not args.no_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
