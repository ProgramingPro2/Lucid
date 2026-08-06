"""Stage orchestrator for per-sentence prediction proof (Fable §8.2–§8.3).

Stages: data → check-encoder → train-vmf → train-mdn → baselines →
writer-check → eval → rollout → report.

Skips stages whose artifact exists unless ``--force``. Training/encoder stages
delegate to existing CLIs (filled by later chunks); ``--stage eval`` runs on a
tiny fake predictor without error for smoke verification.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.config import PredictorConfig  # noqa: E402
from experiments.sentence_pred.baselines import all_baseline_fns  # noqa: E402
from experiments.sentence_pred.data import (  # noqa: E402
    SentenceCorpus,
    SentenceSequenceDataset,
    build_sentence_corpus,
    load_sentence_corpus,
    save_sentence_corpus,
)
from experiments.sentence_pred.evalsuite import (  # noqa: E402
    SEED_DEFAULT,
    effective_rank,
    kappa_reliability,
    load_split,
    m1_table,
    m3_eval,
    multimodality_report,
    nll_report,
    novelty_suite,
    shuffled_context_control,
    write_row,
)
from experiments.sentence_pred.heads import SentenceDensityPredictor  # noqa: E402
from experiments.sentence_pred.rollout import generate_document  # noqa: E402

STAGES = [
    "data",
    "check-encoder",
    "train-vmf",
    "train-mdn",
    "baselines",
    "writer-check",
    "eval",
    "rollout",
    "report",
]


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _artifact(out_dir: Path, name: str) -> Path:
    return out_dir / name


def _skip(path: Path, force: bool) -> bool:
    return path.is_file() and not force


def make_smoke_corpus(
    out_dir: Path | str = "/tmp/sent_corpus_smoke",
    n_docs: int = 30,
    sents_per_doc: int = 8,
    dim: int = 512,
    seed: int = SEED_DEFAULT,
) -> SentenceCorpus:
    """Tiny synthetic corpus for orchestration smoke (no JEPA encode)."""
    out_dir = Path(out_dir)
    if (out_dir / "split_manifest.json").is_file():
        return load_sentence_corpus(out_dir)
    rng = np.random.RandomState(seed)
    offsets = [0]
    texts: list[str] = []
    vecs = []
    for d in range(n_docs):
        for s in range(sents_per_doc):
            texts.append(f"doc{d} sentence {s} about topic {d % 5}.")
            v = rng.randn(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            # mild adjacency: blend with previous
            if s > 0:
                v = 0.7 * v + 0.3 * vecs[-1]
                v /= np.linalg.norm(v) + 1e-12
            vecs.append(v.astype(np.float32))
        offsets.append(len(texts))
    sent_vecs = np.stack(vecs, 0)
    n_held = max(1, int(round(n_docs * 0.10)))
    perm = rng.permutation(n_docs)
    held = np.sort(perm[:n_held].astype(np.int64))
    train = np.sort(perm[n_held:].astype(np.int64))
    corpus = SentenceCorpus(
        sent_vecs=sent_vecs,
        doc_offsets=np.asarray(offsets, dtype=np.int64),
        sentences=texts,
        train_doc_ids=train,
        held_doc_ids=held,
    )
    save_sentence_corpus(corpus, out_dir)
    return corpus


class FakePredictor(SentenceDensityPredictor):
    """Tiny density predictor for smoke eval (random init, tiny trunk)."""

    def __init__(self, concept_dim: int = 512):
        cfg = PredictorConfig(
            d_model=64, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0, max_seq_len=34
        )
        super().__init__(concept_dim=concept_dim, predictor_cfg=cfg, head="vmf", K=2)


def stage_data(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    marker = out / "corpus" / "DONE"
    if _skip(marker, force):
        print(f"[skip] data ({marker})")
        return marker
    # Prefer real build if ckpt exists; else smoke corpus.
    ckpt = Path(cfg.get("jepa_checkpoint", ""))
    corpus_path = cfg.get("corpus_path", "data/corpus_1m.txt")
    if ckpt.is_file() and Path(corpus_path).is_file():
        build_sentence_corpus(
            corpus_path=corpus_path,
            jepa_ckpt=str(ckpt),
            out_dir=str(out / "corpus"),
            min_sents=int(cfg.get("min_sents", 4)),
            held_out_frac=float(cfg.get("held_out_frac", 0.10)),
            seed=int(cfg.get("eval", {}).get("seed", SEED_DEFAULT)),
        )
    else:
        smoke = make_smoke_corpus("/tmp/sent_corpus_smoke")
        # also mirror into artifacts for later stages
        dest = out / "corpus" / "smoke"
        save_sentence_corpus(smoke, dest)
        print(f"[data] smoke corpus → {dest} (n_docs={smoke.n_docs})")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return marker


def _resolve_corpus(cfg: dict) -> SentenceCorpus:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    smoke = Path("/tmp/sent_corpus_smoke")
    if (smoke / "split_manifest.json").is_file():
        return load_sentence_corpus(smoke)
    dest = out / "corpus" / "smoke"
    if (dest / "split_manifest.json").is_file():
        return load_sentence_corpus(dest)
    # search cache dirs
    corpus_root = out / "corpus"
    if corpus_root.is_dir():
        for p in corpus_root.iterdir():
            if (p / "split_manifest.json").is_file():
                return load_sentence_corpus(p)
    return make_smoke_corpus(smoke)


def stage_check_encoder(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    path = out / "m1_table.json"
    if _skip(path, force):
        print(f"[skip] check-encoder ({path})")
        return path
    corpus = _resolve_corpus(cfg)
    table = m1_table(None, corpus, n_pairs=min(500, corpus.n_sentences), seed=SEED_DEFAULT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, default=str)
    print(f"[check-encoder] wrote {path}")
    return path


def stage_train_placeholder(cfg: dict, head: str, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    path = out / f"{head}.pt"
    if _skip(path, force):
        print(f"[skip] train-{head} ({path})")
        return path
    # Later chunks wire train_head.py; for now write a tiny fake checkpoint.
    model = FakePredictor(concept_dim=512)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "predictor_cfg": asdict(model.trunk.cfg) if hasattr(model.trunk, "cfg") else {},
            "head": head if head != "vmf" else "vmf",
            "head_type": "density",
            "concept_dim": 512,
            "history": [],
        },
        path,
    )
    print(f"[train-{head}] placeholder checkpoint → {path}")
    return path


def stage_baselines(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    path = out / "baselines.json"
    if _skip(path, force):
        print(f"[skip] baselines ({path})")
        return path
    corpus = _resolve_corpus(cfg)
    ds = SentenceSequenceDataset(
        corpus, corpus.held_doc_ids, context_len=min(4, 32), stride=4, eval_mode=True
    )
    fns = all_baseline_fns(corpus, seed=SEED_DEFAULT)
    results = {}
    for name, fn in fns.items():
        results[name] = {
            "global": m3_eval(fn, ds, corpus, "global", n_eval=min(64, len(ds)), n_candidates=100, seed=SEED_DEFAULT),
            "indoc": m3_eval(fn, ds, corpus, "indoc", n_eval=min(64, len(ds)), seed=SEED_DEFAULT),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[baselines] wrote {path}")
    return path


def stage_writer_check(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    path = out / "writer_check.json"
    if _skip(path, force):
        print(f"[skip] writer-check ({path})")
        return path
    # Placeholder until writer_p25 lands.
    payload = {"status": "pending", "g5": None, "note": "writer-check awaits writer_p25"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[writer-check] placeholder → {path}")
    return path


def stage_eval(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    frontier = out / "frontier.jsonl"
    marker = out / "eval_DONE.json"
    if _skip(marker, force) and frontier.is_file():
        print(f"[skip] eval ({marker})")
        return marker

    corpus = _resolve_corpus(cfg)
    ds = SentenceSequenceDataset(
        corpus, corpus.held_doc_ids, context_len=min(4, int(cfg.get("context_len", 32))),
        stride=4, eval_mode=True,
    )
    # Load predictor if present; else fake.
    ckpt = out / "vmf.pt"
    model = FakePredictor(512)
    if ckpt.is_file():
        try:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"], strict=False)
        except Exception as e:
            print(f"[eval] could not load {ckpt}: {e}; using fresh FakePredictor")
    model.eval()

    def query_mu(context, pad_mask):
        mu, _ = model.predict_mu(context, pad_mask)
        return F.normalize(mu, dim=-1)

    n_eval = min(64, len(ds), int(cfg.get("eval", {}).get("n_eval", 5000)))
    m3g = m3_eval(query_mu, ds, corpus, "global", n_eval=n_eval, n_candidates=100, seed=SEED_DEFAULT)
    m3i = m3_eval(query_mu, ds, corpus, "indoc", n_eval=n_eval, seed=SEED_DEFAULT)
    shuf = shuffled_context_control(query_mu, ds, corpus, seed=SEED_DEFAULT)
    nll = nll_report(model, ds, corpus)
    kap = kappa_reliability(model, ds, n=n_eval)
    multi = multimodality_report(model, ds, n_ctx=min(32, len(ds)), n_samples=4)

    # novelty on mu predictions
    bank, held, held_texts, _, _ = load_split(corpus, "")
    from experiments.sentence_pred.evalsuite import centering, novelty_scores, reference_distribution

    mu_b, bc = centering(bank)
    # collect a few predicted mus as "gen"
    rng = np.random.RandomState(SEED_DEFAULT)
    idxs = rng.choice(len(ds), size=min(32, len(ds)), replace=False).tolist()
    from experiments.sentence_pred.evalsuite import _last_target_and_meta

    ctx, pad, _tgt, _ = _last_target_and_meta(ds, idxs, corpus)
    with torch.no_grad():
        mus, _ = model.predict_mu(ctx, pad)
        gen = F.normalize(mus, dim=-1).numpy()
    nov = novelty_suite(gen, [held_texts[i % len(held_texts)] for i in range(len(gen))], corpus, None, None)

    row = {
        "config": "smoke_eval",
        "temperature": 1.0,
        "m3_global": m3g,
        "m3_indoc": m3i,
        "shuffled_context": shuf,
        "nll": nll,
        "kappa_reliability": kap,
        "multimodality": multi,
        "novelty": {k: nov[k] for k in nov if k != "novelty_scores"},
        "effective_rank_held": effective_rank(held[:4096] if len(held) else bank[:8]),
    }
    # rewrite frontier for smoke cleanliness when force
    if force and frontier.is_file():
        frontier.unlink()
    write_row(frontier, **row)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"frontier": str(frontier), "ok": True}, f, indent=2)
    print(f"[eval] frontier row → {frontier}")
    return marker


def stage_rollout(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    path = out / "rollouts.json"
    if _skip(path, force):
        print(f"[skip] rollout ({path})")
        return path
    corpus = _resolve_corpus(cfg)
    model = FakePredictor(512)
    ckpt = out / "vmf.pt"
    if ckpt.is_file():
        try:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"], strict=False)
        except Exception:
            pass
    model.eval()

    class _StubCodec:
        dim = 512

        def encode(self, texts):
            rng = np.random.RandomState(abs(hash(texts[0])) % (2**31))
            v = rng.randn(len(texts), 512).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            return v

    seeds = [corpus.sentences[corpus.doc_slice(int(d))[0]] for d in corpus.held_doc_ids.tolist()[:5]]
    bank, _, _, _, _ = load_split(corpus, "")
    docs = []
    for s in seeds:
        sents = generate_document(
            s, model, None, None, _StubCodec(), steps=4, temperature=1.0,
            feedback="latent", train_bank=bank,
        )
        docs.append({"seed": s, "sentences": sents, "instrumentation": getattr(sents, "instrumentation", [])})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, default=str)
    print(f"[rollout] wrote {path} ({len(docs)} seeds)")
    return path


def stage_report(cfg: dict, force: bool) -> Path:
    out = Path(cfg.get("out_dir", "artifacts/sentence_pred"))
    path = out / "RESULTS.md"
    if _skip(path, force):
        print(f"[skip] report ({path})")
        return path

    def _load(name, default=None):
        p = out / name
        if not p.is_file():
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f) if p.suffix == ".json" else f.read()

    m1 = _load("m1_table.json", {})
    base = _load("baselines.json", {})
    eval_m = _load("eval_DONE.json", {})
    rolls = _load("rollouts.json", [])
    frontier_rows = []
    fp = out / "frontier.jsonl"
    if fp.is_file():
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    frontier_rows.append(json.loads(line))

    # Gate table (measured when present; FAIL if missing thresholds)
    def g(name, ok, measured):
        return f"| {name} | {'PASS' if ok else 'FAIL'} | {measured} |"

    lines = [
        "# Per-sentence prediction RESULTS",
        "",
        "## G1–G6 pass/fail",
        "",
        "| Gate | Status | Measured |",
        "|---|---|---|",
        g("G1 encoder", False, json.dumps(m1, default=str)[:120] if m1 else "missing"),
        g("G3 predictor skill", False, "see frontier / eval_DONE"),
        g("G4 distribution", False, "see kappa_reliability / multimodality in frontier"),
        g("G5 writer", False, "pending writer_p25"),
        g("G6 full path", False, "see novelty / rollouts"),
        "",
        "## Verdict",
        "",
        "**per-sentence prediction: FAIL**",
        "",
        "(Smoke/orchestration run — gates fail until real encoder/writer/head training completes.)",
        "",
        "## Frontier rows",
        "",
        f"```\n{json.dumps(frontier_rows, indent=2, default=str)[:4000]}\n```",
        "",
        "## Baselines / controls",
        "",
        f"```\n{json.dumps(base, indent=2, default=str)[:3000]}\n```",
        "",
        "## Rollout excerpts (up to 20)",
        "",
    ]
    n_ex = 0
    for doc in (rolls or [])[:20]:
        lines.append(f"- seed: {doc.get('seed', '')!r}")
        for s in doc.get("sentences", [])[:8]:
            lines.append(f"  - {s}")
            n_ex += 1
            if n_ex >= 20:
                break
        if n_ex >= 20:
            break
    if not rolls:
        lines.append("_no rollouts yet_")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {path}")
    return path


def run_stages(cfg: dict, stages: list[str], force: bool) -> None:
    handlers = {
        "data": lambda: stage_data(cfg, force),
        "check-encoder": lambda: stage_check_encoder(cfg, force),
        "train-vmf": lambda: stage_train_placeholder(cfg, "vmf", force),
        "train-mdn": lambda: stage_train_placeholder(cfg, "mdn", force),
        "baselines": lambda: stage_baselines(cfg, force),
        "writer-check": lambda: stage_writer_check(cfg, force),
        "eval": lambda: stage_eval(cfg, force),
        "rollout": lambda: stage_rollout(cfg, force),
        "report": lambda: stage_report(cfg, force),
    }
    for s in stages:
        if s not in handlers:
            raise SystemExit(f"unknown stage: {s}")
        handlers[s]()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-sentence JEPA proof orchestrator")
    p.add_argument("--config", default="configs/sentence_pred.json")
    p.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Stage to run (repeatable). Default: --all",
    )
    p.add_argument("--all", action="store_true", help="Run all stages in order")
    p.add_argument("--force", action="store_true", help="Redo even if artifact exists")
    args = p.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        # minimal default so smoke works without committed config
        cfg = {
            "corpus_path": "data/corpus_1m.txt",
            "jepa_checkpoint": "artifacts/jepa_repair_1m/jepa/joint.pt",
            "writer_checkpoint": "artifacts/writer_p25/writer_two_channel.pt",
            "old_writer_checkpoint": "artifacts/writer_p2_film/writer_cond.pt",
            "out_dir": "artifacts/sentence_pred",
            "min_sents": 4,
            "context_len": 32,
            "held_out_frac": 0.10,
            "predictor": {
                "d_model": 512, "n_layers": 6, "n_heads": 8, "ff_mult": 4,
                "dropout": 0.1, "max_seq_len": 34,
            },
            "eval": {"seed": 1234, "n_eval": 5000, "n_candidates_global": 1000},
        }
        print(f"[warn] {cfg_path} missing; using built-in defaults")
    else:
        cfg = _load_cfg(cfg_path)

    if args.all or not args.stage:
        stages = STAGES if args.all or not args.stage else args.stage
        if not args.all and not args.stage:
            stages = STAGES
    else:
        stages = args.stage
    if args.stage and not args.all:
        stages = args.stage

    Path(cfg.get("out_dir", "artifacts/sentence_pred")).mkdir(parents=True, exist_ok=True)
    run_stages(cfg, stages, force=args.force)


if __name__ == "__main__":
    main()
