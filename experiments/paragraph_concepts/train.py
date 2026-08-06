"""Train the paragraph-scale ranking reasoner.

CLI:
  python -m experiments.paragraph_concepts \\
    --group-size 5 --window 8 --epochs 20 --batch-size 256 --lr 3e-4 \\
    --dev auto --out artifacts/paragraph_reasoner.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.config import GpuGateConfig  # noqa: E402
from conceptpoc.progress import iter_bar, log_epoch, log_stage  # noqa: E402

from .data import (  # noqa: E402
    CachedParagraphCorpus,
    ParagraphSequenceDataset,
    collate_paragraph_batch,
    load_and_build,
)
from .reasoner import (  # noqa: E402
    RankingReasoner,
    RankingReasonerConfig,
    ranking_ce_hard,
    ranking_loss,
)


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_m3(
    model: RankingReasoner,
    dataset: ParagraphSequenceDataset,
    indices: list[int],
    device: torch.device,
    n_candidates: int = 100,
    max_examples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out M3: top-1% and MRR over 100 in-doc candidates (true next @0)."""
    model.eval()
    rng = np.random.default_rng(seed)
    if len(indices) > max_examples:
        pick = rng.choice(len(indices), size=max_examples, replace=False)
        indices = [indices[int(i)] for i in pick]

    ranks: list[int] = []
    for idx in indices:
        item = dataset[idx]
        ctx = item["context"].unsqueeze(0).to(device)
        # Rebuild a fresh 100-candidate set with true next at index 0
        doc_id = int(item["doc_id"])
        local_next = int(item["para_idx"])
        a, b = dataset.corpus.doc_slices()[doc_id]
        n = b - a
        true_g = a + local_next
        # Context span for banned set
        T = item["context"].shape[0]
        ctx_start_local = local_next - T
        banned = set(range(ctx_start_local, local_next + 1))
        pool = [i for i in range(n) if i not in banned]
        need = n_candidates - 1
        if not pool:
            continue
        if len(pool) >= need:
            dist = rng.choice(pool, size=need, replace=False).tolist()
        else:
            dist = rng.choice(pool, size=need, replace=True).tolist()
        cand_locals = [local_next] + [int(x) for x in dist]
        cand_g = [a + i for i in cand_locals]
        candidates = dataset.vectors[cand_g].unsqueeze(0).to(device)
        query = model(ctx)
        logits = model.score(query, candidates)  # (1, K)
        order = torch.argsort(logits[0], descending=True).cpu().tolist()
        rank = order.index(0) + 1  # 1-based rank of true next
        ranks.append(rank)

    if not ranks:
        return {"m3_top1": float("nan"), "m3_mrr": float("nan"), "n": 0.0}
    top1 = sum(1 for r in ranks if r == 1) / len(ranks)
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    return {"m3_top1": top1, "m3_mrr": mrr, "n": float(len(ranks))}


def train(
    corpus: CachedParagraphCorpus,
    *,
    window: int = 8,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "auto",
    out: str | Path = "artifacts/paragraph_reasoner.pt",
    n_candidates: int = 100,
    n_following_pos: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    db_path: str | Path | None = "artifacts/paragraph_reasoner_ckpts.sqlite",
    run_id: int = 0,
    use_gpu_gate: bool = True,
    progress: bool = True,
    hard_ce: bool = False,
) -> tuple[RankingReasoner, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device(device)
    log_stage(f"device={dev}", enabled=progress)

    full = ParagraphSequenceDataset(
        corpus,
        window=window,
        n_candidates=n_candidates,
        n_following_pos=n_following_pos,
        seed=seed,
    )
    if len(full) == 0:
        raise ValueError("empty paragraph dataset; need docs with >=2 paragraphs")

    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full), generator=gen).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_ds = Subset(full, train_idx)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_paragraph_batch,
        drop_last=len(train_ds) >= batch_size,
    )

    cfg = RankingReasonerConfig(max_seq_len=max(32, window + 2))
    model = RankingReasoner(full.dim, cfg).to(dev)
    log_stage(
        f"RankingReasoner params={model.num_parameters():,}  "
        f"train={len(train_idx)} val={len(val_idx)} paragraphs={corpus.n_paragraphs}",
        enabled=progress,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, epochs * max(1, len(train_dl)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    gate = None
    if use_gpu_gate and dev.type == "cuda":
        from conceptpoc.gpu_gate import GpuGate

        gate = GpuGate(GpuGateConfig(enabled=True), dev, progress=progress)
        if not gate.active:
            gate = None
        else:
            log_stage("gpu-gate active", enabled=progress)

    db = None
    db_run = None
    start_epoch = 0
    if db_path is not None:
        from conceptpoc.ckpt_db import CheckpointDB

        db = CheckpointDB(Path(db_path))
        db_run = run_id or db.latest_run()
        if db_run is None:
            db_run = db.start_run(
                "paragraph_concepts",
                "ranking",
                seed,
                full.dim,
                window,
                {
                    "reasoner": asdict(cfg),
                    "group_size": corpus.group_size,
                    "n_candidates": n_candidates,
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                },
            )
        loaded = db.load_latest_checkpoint(db_run, str(dev))
        if loaded is not None:
            try:
                model.load_state_dict(loaded[0])
                opt.load_state_dict(loaded[1])
                start_epoch = loaded[2] + 1
                log_stage(f"resumed from epoch {start_epoch}", enabled=progress)
            except Exception as e:  # noqa: BLE001
                log_stage(f"checkpoint mismatch ({e}); start fresh", enabled=progress)

    history: dict = {
        "train_loss": [],
        "val_m3_top1": [],
        "val_m3_mrr": [],
        "best_epoch": -1,
        "best_m3_top1": -1.0,
    }
    best_state: dict | None = None
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        eloss = 0.0
        n_batches = 0
        for batch in iter_bar(
            train_dl,
            desc=f"epoch {epoch + 1}/{epochs}",
            total=len(train_dl),
            enabled=progress,
            unit="batch",
        ):
            context, pad_mask, _targets, candidates, n_pos, _, _ = batch
            context = context.to(dev)
            pad_mask = pad_mask.to(dev)
            candidates = candidates.to(dev)
            n_pos = n_pos.to(dev)
            opt.zero_grad()
            query = model(context, pad_mask)
            if hard_ce:
                loss = ranking_ce_hard(query, candidates, model.cfg.temperature)
            else:
                loss = ranking_loss(query, candidates, n_pos, model.cfg.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                try:
                    sched.step()
                except Exception:  # noqa: BLE001
                    pass
            if gate is not None:
                gate.step(model, opt)
            eloss += float(loss.item())
            n_batches += 1
        train_loss = eloss / max(1, n_batches)

        m3 = evaluate_m3(model, full, val_idx, dev, n_candidates=n_candidates, seed=seed + epoch)
        history["train_loss"].append(train_loss)
        history["val_m3_top1"].append(m3["m3_top1"])
        history["val_m3_mrr"].append(m3["m3_mrr"])

        if m3["m3_top1"] > history["best_m3_top1"]:
            history["best_m3_top1"] = m3["m3_top1"]
            history["best_epoch"] = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        log_epoch(
            "para-rank",
            epoch + 1,
            epochs,
            {
                "train_loss": train_loss,
                "m3_top1%": 100.0 * m3["m3_top1"],
                "m3_mrr": m3["m3_mrr"],
                "n_eval": int(m3["n"]),
            },
            enabled=progress,
        )
        if db is not None and db_run is not None:
            db.record_epoch(
                db_run,
                epoch,
                {
                    "train_loss": train_loss,
                    "m3_top1": m3["m3_top1"],
                    "m3_mrr": m3["m3_mrr"],
                },
            )
            db.save_checkpoint(db_run, epoch, model, opt)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final held-out eval on best weights
    final = evaluate_m3(
        model, full, val_idx, dev, n_candidates=n_candidates, max_examples=5000, seed=seed + 999
    )
    history["final_m3_top1"] = final["m3_top1"]
    history["final_m3_mrr"] = final["m3_mrr"]
    history["final_n"] = final["n"]
    history["elapsed_s"] = time.time() - t0
    history["group_size"] = corpus.group_size
    history["n_params"] = model.num_parameters()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "concept_dim": full.dim,
            "group_size": corpus.group_size,
            "window": window,
            "history": history,
        },
        out,
    )
    log_stage(f"saved reasoner -> {out}", enabled=progress)

    _print_eval_table(history, corpus.group_size)

    if db is not None:
        db.close()
    return model, history


def _print_eval_table(history: dict, group_size: int) -> None:
    print("\n" + "=" * 64)
    print("PARAGRAPH-SCALE RANKING REASONER — HELD-OUT M3 EVAL")
    print("=" * 64)
    print(f"{'metric':<28} {'value':>12}")
    print("-" * 42)
    top1 = history.get("final_m3_top1", float("nan"))
    mrr = history.get("final_m3_mrr", float("nan"))
    print(f"{'group size G':<28} {group_size:>12d}")
    print(f"{'params':<28} {history.get('n_params', 0):>12,}")
    print(f"{'best epoch':<28} {history.get('best_epoch', -1):>12d}")
    print(f"{'best val M3 top-1 %':<28} {100.0 * history.get('best_m3_top1', 0):>12.2f}")
    print(f"{'final held-out M3 top-1 %':<28} {100.0 * top1:>12.2f}")
    print(f"{'final held-out M3 MRR':<28} {mrr:>12.4f}")
    print(f"{'n eval examples':<28} {int(history.get('final_n', 0)):>12d}")
    if history.get("train_loss"):
        print(f"{'final train loss':<28} {history['train_loss'][-1]:>12.4f}")
    print("-" * 42)
    print("Coarser-prediction note:")
    print("  Sentence-scale (G=1) M3 top-1 was ~5-11% in offline probes.")
    print(
        f"  Paragraph-scale (G={group_size}) trained ranking reasoner "
        f"M3 top-1 = {100.0 * top1:.2f}%."
    )
    if top1 >= 0.25:
        print("  Gate PASS: meaningfully above sentence-scale baseline (target 25-45%+).")
    elif top1 >= 0.15:
        print("  Gate SOFT-PASS: above sentence-scale, below ideal 25% band.")
    else:
        print("  Gate FAIL: not clearly above sentence-scale baseline.")
    print("=" * 64 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train paragraph ranking reasoner")
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dev", type=str, default="auto")
    p.add_argument("--out", type=str, default="artifacts/paragraph_reasoner.pt")
    p.add_argument("--corpus", type=str, default="data/corpus_1m.txt")
    p.add_argument(
        "--jepa",
        type=str,
        default="artifacts/jepa_v3_128t/jepa/joint_robust.pt",
    )
    p.add_argument("--max-docs", type=int, default=2000)
    p.add_argument("--n-candidates", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache-dir",
        type=str,
        default="artifacts/paragraph_concepts_cache",
    )
    p.add_argument(
        "--db-path",
        type=str,
        default="artifacts/paragraph_reasoner_ckpts.sqlite",
    )
    p.add_argument("--no-gpu-gate", action="store_true")
    p.add_argument("--hard-ce", action="store_true", help="Use single-positive CE")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    corpus = load_and_build(
        corpus_path=args.corpus,
        jepa_ckpt=args.jepa,
        group_size=args.group_size,
        max_docs=args.max_docs,
        device=args.dev,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    _, history = train(
        corpus,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.dev,
        out=args.out,
        n_candidates=args.n_candidates,
        seed=args.seed,
        db_path=args.db_path,
        use_gpu_gate=not args.no_gpu_gate,
        hard_ce=args.hard_ce,
    )
    # Also dump a JSON sidecar for the deliverable
    side = Path(args.out).with_suffix(".metrics.json")
    side.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


if __name__ == "__main__":
    main()
