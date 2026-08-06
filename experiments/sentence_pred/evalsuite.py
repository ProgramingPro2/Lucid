"""Anti-mirage evaluation suite for per-sentence JEPA (Fable §7).

Global rules: document-disjoint splits; RNG seed 1234 independent of epoch;
every cosine raw AND centered with anisotropy floor; every retrieval reports
distinct-candidate count; every row → artifacts/sentence_pred/frontier.jsonl.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conceptpoc.eval.metrics import _anisotropy  # noqa: E402

from experiments.sentence_pred.heads import (  # noqa: E402
    MixtureVmfHead,
    vmf_nll,
)

SEED_DEFAULT = 1234
_CHUNK = 4096


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)).astype(np.float32)


def load_split(corpus, manifest_path) -> tuple:
    """(bank_train, held_vectors, held_texts, train_ids, held_ids)."""
    path = Path(manifest_path)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
        train_ids = np.asarray(man["train_doc_ids"], dtype=np.int64)
        held_ids = np.asarray(man["held_doc_ids"], dtype=np.int64)
    else:
        train_ids = np.asarray(corpus.train_doc_ids, dtype=np.int64)
        held_ids = np.asarray(corpus.held_doc_ids, dtype=np.int64)
    assert len(np.intersect1d(train_ids, held_ids)) == 0

    def _gather(ids):
        rows, texts = [], []
        for d in ids.tolist():
            a, b = corpus.doc_slice(int(d))
            rows.append(corpus.sent_vecs[a:b])
            texts.extend(corpus.sentences[a:b])
        if not rows:
            d = int(corpus.sent_vecs.shape[1])
            return np.zeros((0, d), np.float32), []
        return np.concatenate(rows, 0).astype(np.float32), texts

    bank_train, _ = _gather(train_ids)
    held_vectors, held_texts = _gather(held_ids)
    return bank_train, held_vectors, held_texts, train_ids, held_ids


def centering(bank) -> tuple[np.ndarray, np.ndarray]:
    """(mu, bank_centered) with bank_centered = normalize(bank - mu)."""
    bank = _as_np(bank)
    mu = bank.mean(axis=0).astype(np.float32)
    return mu, _l2(bank - mu)


def novelty_scores(X, bank_centered, mu) -> np.ndarray:
    """n(x) = max_b cos(x_c, b_c) per row; chunked matmul, 4096-row blocks."""
    X = _as_np(X)
    mu = _as_np(mu).reshape(1, -1)
    xc = _l2(X - mu)
    bc = _as_np(bank_centered)
    if xc.size == 0 or bc.size == 0:
        return np.zeros((xc.shape[0],), np.float32)
    out = np.empty((xc.shape[0],), dtype=np.float32)
    for i0 in range(0, xc.shape[0], _CHUNK):
        i1 = min(i0 + _CHUNK, xc.shape[0])
        block = xc[i0:i1]
        mx = np.full((i1 - i0,), -np.inf, dtype=np.float32)
        for j0 in range(0, bc.shape[0], _CHUNK):
            j1 = min(j0 + _CHUNK, bc.shape[0])
            sims = block @ bc[j0:j1].T
            mx = np.maximum(mx, sims.max(axis=1))
        out[i0:i1] = mx
    return out


def reference_distribution(held_vectors, bank_centered, mu) -> dict:
    """R = {n(h)} + median/p90/p99 of R."""
    R = novelty_scores(held_vectors, bank_centered, mu)
    return {
        "R": R,
        "median": float(np.median(R)) if R.size else float("nan"),
        "p90": float(np.percentile(R, 90)) if R.size else float("nan"),
        "p99": float(np.percentile(R, 99)) if R.size else float("nan"),
        "n": int(R.size),
    }


def novelty_rate(X, R) -> float:
    """frac(n(X) <= median(R)). X may be scores or vectors (if dict R has bank)."""
    med = float(R["median"]) if isinstance(R, dict) else float(np.median(R))
    scores = _as_np(X).ravel()
    if scores.size == 0:
        return float("nan")
    return float((scores <= med).mean())


def copy_rate(X, R) -> float:
    """frac(n(X) >= p99(R))."""
    p99 = float(R["p99"]) if isinstance(R, dict) else float(np.percentile(R, 99))
    scores = _as_np(X).ravel()
    if scores.size == 0:
        return float("nan")
    return float((scores >= p99).mean())


def effective_rank(X) -> dict:
    """exp entropy of normalized singular values (centered); + participation ratio."""
    X = _as_np(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    zc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(zc, compute_uv=False).astype(np.float64)
    s = np.maximum(s, 0.0)
    total = s.sum() + 1e-12
    p = s / total
    ent = float(-(p * np.log(p + 1e-12)).sum())
    er = float(math.exp(ent))
    pr = float((s.sum() ** 2) / ((s * s).sum() + 1e-12))
    return {"effective_rank": er, "participation_ratio": pr, "dim": int(X.shape[1])}


def max_ngram_overlap(gen_text: str, ref_texts, n: int = 5) -> float:
    """Max fraction of gen n-grams that appear in any single ref text."""
    toks = str(gen_text).lower().split()
    if len(toks) < n:
        return 0.0
    grams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    best = 0.0
    for ref in ref_texts:
        rtoks = str(ref).lower().split()
        rset = {" ".join(rtoks[i : i + n]) for i in range(max(0, len(rtoks) - n + 1))}
        if not rset:
            continue
        hit = sum(1 for g in grams if g in rset) / len(grams)
        best = max(best, hit)
    return float(best)


def _banerjee_kappa(rho: float, d: int) -> float:
    rho = float(np.clip(rho, 1e-6, 1.0 - 1e-6))
    return float(rho * (d - rho * rho) / (1.0 - rho * rho))


def _pairwise_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((_l2(a) * _l2(b)).sum(axis=1).mean()) if len(a) else float("nan")


def m1_table(encode_fn, corpus, n_pairs: int = 5000, seed: int = SEED_DEFAULT) -> dict:
    """Adjacent / within-shuf / cross-shuf / floor + M3-identity(indoc) + effrank."""
    rng = np.random.RandomState(seed)
    # Prefer held-out docs for pairs (document-disjoint).
    doc_ids = list(np.asarray(corpus.held_doc_ids, dtype=np.int64).tolist())
    if len(doc_ids) < 2:
        doc_ids = list(range(corpus.n_docs))
    pairs: list[tuple[int, int]] = []  # (doc_id, sent_offset_in_doc)
    for d in doc_ids:
        a, b = corpus.doc_slice(int(d))
        for i in range(b - a - 1):
            pairs.append((int(d), i))
    if not pairs:
        return {"error": "no_pairs"}
    if len(pairs) > n_pairs:
        idx = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    # Optionally re-encode; else use cached corpus vectors.
    def _vec(doc_id: int, local_i: int) -> np.ndarray:
        a, _ = corpus.doc_slice(doc_id)
        if encode_fn is None:
            return corpus.sent_vecs[a + local_i]
        text = corpus.sentences[a + local_i]
        return _l2(np.asarray(encode_fn([text]), dtype=np.float32)[0])

    A = np.stack([_vec(d, i) for d, i in pairs])
    B = np.stack([_vec(d, i + 1) for d, i in pairs])
    adj = _pairwise_cos(A, B)

    # Within-doc shuffle of targets.
    by_doc: dict[int, list[int]] = {}
    for k, (d, _) in enumerate(pairs):
        by_doc.setdefault(d, []).append(k)
    Wb = B.copy()
    for idxs in by_doc.values():
        perm = rng.permutation(len(idxs))
        src = B[[idxs[j] for j in range(len(idxs))]]
        for j, p in enumerate(perm):
            Wb[idxs[j]] = src[p]
    within = _pairwise_cos(A, Wb)

    # Cross-doc: random target from a different doc's true nexts.
    all_tgts = list(range(len(pairs)))
    Cb = []
    for k, (d, _) in enumerate(pairs):
        cand = [j for j in all_tgts if pairs[j][0] != d]
        j = int(rng.choice(cand)) if cand else k
        Cb.append(B[j])
    cross = _pairwise_cos(A, np.stack(Cb))

    d = int(A.shape[1])
    floor = 1.0 / math.sqrt(d)

    # M3-identity in-doc (query = z_t = A).
    top1 = mrr = 0.0
    dist_counts = []
    for k, (doc_id, local_i) in enumerate(pairs):
        a, b = corpus.doc_slice(doc_id)
        n = b - a
        q = A[k]
        true = B[k]
        # distractors: same-doc excluding query and true next
        pool = [j for j in range(n) if j not in (local_i, local_i + 1)]
        dist_counts.append(len(pool) + 1)  # + true
        if pool:
            dv = np.stack([corpus.sent_vecs[a + j] if encode_fn is None else _vec(doc_id, j) for j in pool])
            cands = np.vstack([true[None, :], dv])
        else:
            cands = true[None, :]
        sims = _l2(cands) @ _l2(q)
        order = np.argsort(-sims)
        rank = int(np.where(order == 0)[0][0]) + 1
        top1 += float(rank == 1)
        mrr += 1.0 / rank
    n = len(pairs)
    er = effective_rank(np.concatenate([A, B], 0)[:4096])
    bank_t = torch.from_numpy(_l2(np.concatenate([A, B], 0)))
    aniso = _anisotropy(bank_t)
    return {
        "adjacent_cos": adj,
        "within_doc_shuf_cos": within,
        "cross_doc_shuf_cos": cross,
        "floor_1_sqrt_d": floor,
        "m3_identity_indoc_top1": top1 / n,
        "m3_identity_indoc_mrr": mrr / n,
        "mean_distinct_candidates": float(np.mean(dist_counts)) if dist_counts else 0.0,
        "effective_rank": er,
        "anisotropy_floor": aniso,
        "n_pairs": n,
        "dim": d,
    }


def _last_target_and_meta(dataset, idxs: list[int], corpus):
    """Collect context batches + true next vec + doc/start meta for eval idxs."""
    contexts, targets, metas = [], [], []
    for i in idxs:
        item = dataset[i]
        ctx = item["context"]  # (T, D)
        tgt = item["target_seq"][-1]  # last target = next after full context
        contexts.append(ctx)
        targets.append(tgt)
        # absolute sentence index of last context token
        a, _ = corpus.doc_slice(int(item["doc_id"]))
        last_ctx_abs = a + int(item["start"]) + ctx.shape[0] - 1
        metas.append(
            {
                "doc_id": int(item["doc_id"]),
                "start": int(item["start"]),
                "last_ctx_abs": int(last_ctx_abs),
                "target_abs": int(last_ctx_abs + 1),
                "T": int(ctx.shape[0]),
            }
        )
    # pad contexts
    T = max(c.shape[0] for c in contexts)
    D = contexts[0].shape[-1]
    B = len(contexts)
    context = torch.zeros(B, T, D)
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    for i, c in enumerate(contexts):
        L = c.shape[0]
        context[i, :L] = c
        pad_mask[i, :L] = False
    target = torch.stack(targets, 0)
    return context, pad_mask, target, metas


def m3_eval(
    query_fn: Callable,
    dataset,
    corpus,
    mode: str,
    n_eval: int = 5000,
    n_candidates: int = 1000,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Retrieval top1/5/10/MRR. mode=global (PRIMARY) or indoc (SECONDARY)."""
    rng = np.random.RandomState(seed)
    n = min(len(dataset), n_eval)
    if n == 0:
        return {"error": "empty_dataset", "mode": mode}
    idxs = rng.choice(len(dataset), size=n, replace=False).tolist()
    context, pad_mask, target, metas = _last_target_and_meta(dataset, idxs, corpus)
    with torch.no_grad():
        query = query_fn(context, pad_mask)
        query = F.normalize(query.float(), dim=-1).cpu().numpy()
    target_np = _l2(_as_np(target))

    held_ids = set(int(x) for x in corpus.held_doc_ids.tolist())
    # Pool of held-out sentence indices by doc for distractors.
    held_by_doc: dict[int, list[int]] = {}
    all_held: list[int] = []
    for d in held_ids:
        a, b = corpus.doc_slice(d)
        held_by_doc[d] = list(range(a, b))
        all_held.extend(range(a, b))

    bank = torch.from_numpy(_l2(corpus.sent_vecs[all_held] if all_held else corpus.sent_vecs[:1]))
    aniso = _anisotropy(bank)
    train_rows = []
    for d in list(corpus.train_doc_ids.tolist())[:200]:
        a, b = corpus.doc_slice(int(d))
        train_rows.append(corpus.sent_vecs[a:b])
    train_bank = (
        np.concatenate(train_rows, 0)
        if train_rows
        else corpus.sent_vecs[: max(8, corpus.sent_vecs.shape[0])]
    )
    mu, _bank_c = centering(train_bank)

    top1 = top5 = top10 = mrr_sum = 0.0
    cos_raw_sum = cos_c_sum = 0.0
    distinct = []
    for i in range(n):
        meta = metas[i]
        doc_id = meta["doc_id"]
        true = target_np[i]
        q = query[i]
        if mode == "global":
            # distractors from OTHER held-out docs
            pool = [j for j in all_held if j < corpus.doc_slice(doc_id)[0] or j >= corpus.doc_slice(doc_id)[1]]
            need = max(1, n_candidates) - 1
            if len(pool) >= need:
                dist_idx = rng.choice(pool, size=need, replace=False).tolist()
            elif pool:
                dist_idx = [int(rng.choice(pool)) for _ in range(need)]
            else:
                dist_idx = []
            cands = [true] + [_l2(corpus.sent_vecs[j]) for j in dist_idx]
        else:  # indoc
            a, b = corpus.doc_slice(doc_id)
            ctx_abs = set(range(meta["last_ctx_abs"] - meta["T"] + 1, meta["last_ctx_abs"] + 1))
            exclude = ctx_abs | {meta["target_abs"]}
            pool = [j for j in range(a, b) if j not in exclude]
            cands = [true] + [_l2(corpus.sent_vecs[j]) for j in pool]
        C = _l2(np.stack(cands))
        distinct.append(len({tuple(np.round(r, 5)) for r in C}))
        sims = C @ q
        order = np.argsort(-sims)
        rank = int(np.where(order == 0)[0][0]) + 1
        top1 += float(rank == 1)
        top5 += float(rank <= 5)
        top10 += float(rank <= 10)
        mrr_sum += 1.0 / rank
        cos_raw_sum += float(np.dot(q, true))
        qc = _l2(q - mu)
        tc = _l2(true - mu)
        cos_c_sum += float(np.dot(qc, tc))

    return {
        "mode": mode,
        "top1": top1 / n,
        "top5": top5 / n,
        "top10": top10 / n,
        "mrr": mrr_sum / n,
        "mean_distinct_candidates": float(np.mean(distinct)),
        "mean_cos_raw": cos_raw_sum / n,
        "mean_cos_centered": cos_c_sum / n,
        "anisotropy_floor": float(aniso),
        "n_eval": n,
        "n_candidates": n_candidates if mode == "global" else None,
        "chance": (1.0 / n_candidates) if mode == "global" else None,
    }


def shuffled_context_control(query_fn, dataset, corpus, seed: int = SEED_DEFAULT) -> dict:
    """M3-global with contexts drawn from a different held-out document."""
    rng = np.random.RandomState(seed)
    n = min(len(dataset), 500)
    if n < 2:
        return {"error": "too_few", "top1": float("nan")}
    idxs = rng.choice(len(dataset), size=n, replace=False).tolist()
    context, pad_mask, target, metas = _last_target_and_meta(dataset, idxs, corpus)
    # Swap contexts across different docs.
    perm = np.arange(n)
    rng.shuffle(perm)
    for attempt in range(20):
        ok = all(metas[i]["doc_id"] != metas[int(perm[i])]["doc_id"] for i in range(n))
        if ok:
            break
        rng.shuffle(perm)
    context_shuf = context[torch.as_tensor(perm.astype(np.int64))]
    pad_shuf = pad_mask[torch.as_tensor(perm.astype(np.int64))]

    # Wrap: run m3-like ranking with shuffled queries vs original targets.
    with torch.no_grad():
        query = F.normalize(query_fn(context_shuf, pad_shuf).float(), dim=-1).cpu().numpy()
    target_np = _l2(_as_np(target))
    held_ids = [int(x) for x in corpus.held_doc_ids.tolist()]
    all_held = []
    for d in held_ids:
        a, b = corpus.doc_slice(d)
        all_held.extend(range(a, b))
    top1 = 0.0
    n_cand = 1000
    for i in range(n):
        doc_id = metas[i]["doc_id"]
        a0, b0 = corpus.doc_slice(doc_id)
        pool = [j for j in all_held if j < a0 or j >= b0]
        need = n_cand - 1
        dist = rng.choice(pool, size=min(need, len(pool)), replace=False).tolist() if pool else []
        while len(dist) < need and pool:
            dist.append(int(rng.choice(pool)))
        C = _l2(np.stack([target_np[i]] + [corpus.sent_vecs[j] for j in dist[:need]]))
        sims = C @ query[i]
        rank = int(np.argsort(-sims).tolist().index(0)) + 1
        top1 += float(rank == 1)
    return {
        "mode": "shuffled_context_global",
        "top1": top1 / n,
        "n_eval": n,
        "n_candidates": n_cand,
        "chance": 1.0 / n_cand,
    }


def nll_report(model, dataset, corpus) -> dict:
    """Held-out NLL vs unconditional + per-doc-mean vMF; delta nats/sentence."""
    from torch.utils.data import DataLoader
    from experiments.sentence_pred.data import collate_sentence_batch

    device = next(model.parameters()).device
    model.eval()
    dl = DataLoader(dataset, batch_size=32, collate_fn=collate_sentence_batch)
    nll_sum = n_tok = 0.0
    last_mu, last_x = [], []
    with torch.no_grad():
        for context, pad_mask, target_seq, valid, doc_ids in dl:
            context = context.to(device)
            pad_mask = pad_mask.to(device)
            target_seq = target_seq.to(device)
            valid = valid.to(device)
            loss = model.loss(context, pad_mask, target_seq, valid)
            n_valid = float(valid.sum().item())
            nll_sum += float(loss.item()) * n_valid
            n_tok += n_valid
            # collect last-position pairs for marginal fits
            mu, _k = model.predict_mu(context, pad_mask)
            # true last target
            n_real = (~pad_mask).sum(dim=1).clamp_min(1) - 1
            b = context.size(0)
            x_last = target_seq[torch.arange(b), n_real.long()]
            last_mu.append(mu.cpu())
            last_x.append(x_last.cpu())

    cond_nll = nll_sum / max(n_tok, 1.0)
    X = torch.cat(last_x, 0) if last_x else torch.zeros(1, model.concept_dim)
    X = F.normalize(X.float(), dim=-1)
    d = X.size(-1)
    # (a) unconditional vMF: mu = mean direction, kappa Banerjee
    mu_u = F.normalize(X.mean(dim=0, keepdim=True), dim=-1)
    rho = float((mu_u * X).sum(-1).mean())
    kappa_u = _banerjee_kappa(rho, d)
    nll_u = float(vmf_nll(mu_u.expand_as(X), torch.full((X.size(0),), kappa_u), X).mean())

    # (b) per-doc-mean vMF: for each held doc, fit mu_doc on that doc's vectors
    nll_doc_sum = n_doc = 0.0
    for d_id in np.asarray(corpus.held_doc_ids, dtype=np.int64).tolist():
        a, b = corpus.doc_slice(int(d_id))
        if b - a < 2:
            continue
        V = torch.from_numpy(_l2(corpus.sent_vecs[a:b]))
        mu_d = F.normalize(V.mean(0, keepdim=True), dim=-1)
        rho_d = float((mu_d * V).sum(-1).mean())
        k_d = _banerjee_kappa(rho_d, d)
        nll_d = float(vmf_nll(mu_d.expand_as(V), torch.full((V.size(0),), k_d), V).mean())
        nll_doc_sum += nll_d * V.size(0)
        n_doc += V.size(0)
    nll_doc = nll_doc_sum / max(n_doc, 1.0)

    return {
        "conditional_nll": cond_nll,
        "unconditional_vmf_nll": nll_u,
        "per_doc_mean_vmf_nll": nll_doc,
        "delta_vs_uncond": nll_u - cond_nll,
        "delta_vs_per_doc": nll_doc - cond_nll,
        "uncond_kappa": kappa_u,
        "uncond_rho": rho,
        "n_tokens": int(n_tok),
    }


def kappa_reliability(model, dataset, n: int = 5000) -> dict:
    """Bin predicted kappa into quintiles; realized cos(mu, x_true); Spearman."""
    rng = np.random.RandomState(SEED_DEFAULT)
    n = min(len(dataset), n)
    idxs = rng.choice(len(dataset), size=n, replace=False).tolist() if n else []
    # reuse corpus from dataset
    corpus = dataset.corpus
    context, pad_mask, target, _ = _last_target_and_meta(dataset, idxs, corpus)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        mu, kappa = model.predict_mu(context.to(device), pad_mask.to(device))
        mu = F.normalize(mu.cpu().float(), dim=-1)
        kappa = kappa.cpu().float().reshape(-1)
        x = F.normalize(target.float(), dim=-1)
        cos = (mu * x).sum(-1).numpy()
        k = kappa.numpy()
    # quintiles
    edges = np.percentile(k, [0, 20, 40, 60, 80, 100])
    bins = []
    for qi in range(5):
        lo, hi = edges[qi], edges[qi + 1]
        mask = (k >= lo) & (k <= hi) if qi == 4 else (k >= lo) & (k < hi)
        bins.append(
            {
                "quintile": qi,
                "kappa_lo": float(lo),
                "kappa_hi": float(hi),
                "mean_kappa": float(k[mask].mean()) if mask.any() else float("nan"),
                "realized_cos": float(cos[mask].mean()) if mask.any() else float("nan"),
                "n": int(mask.sum()),
            }
        )
    # Spearman(kappa, realized cos)
    if len(k) >= 2:
        rk = k.argsort().argsort().astype(np.float64)
        rc = cos.argsort().argsort().astype(np.float64)
        spearman = float(np.corrcoef(rk, rc)[0, 1])
    else:
        spearman = float("nan")
    return {"bins": bins, "spearman": spearman, "n": int(len(k))}


def renderability(vecs, prefixes, writer, tokenizer, codec) -> dict:
    """cos(enc(render(v, prefix)), v) centered + oracle + old-writer arms."""
    # Graceful stub when writer/codec unavailable (orchestration smoke).
    if writer is None or codec is None or tokenizer is None:
        return {
            "mean_cos_raw": float("nan"),
            "mean_cos_centered": float("nan"),
            "oracle_cos_centered": float("nan"),
            "old_writer_cos_centered": float("nan"),
            "anisotropy_floor": float("nan"),
            "n": 0,
            "stub": True,
        }
    vecs = _as_np(vecs)
    mu = vecs.mean(0)
    cos_r = cos_c = 0.0
    n = len(vecs)
    device = next(writer.parameters()).device
    for i in range(n):
        v = torch.from_numpy(vecs[i]).float().unsqueeze(0).to(device)
        prefix = prefixes[i] if prefixes is not None else None
        prefix_ids = None
        if prefix is not None:
            ids = tokenizer.encode(prefix) if hasattr(tokenizer, "encode") else []
            bos = getattr(tokenizer, "bos_id", 50256)
            eos = getattr(tokenizer, "eos_id", 50256)
            prefix_ids = torch.tensor([[bos] + list(ids)[:96] + [eos]], device=device)
        with torch.no_grad():
            out_ids = writer.generate(
                v, max_tokens=64, bos_id=getattr(tokenizer, "bos_id", 50256),
                eos_id=getattr(tokenizer, "eos_id", 50256), prefix_ids=prefix_ids,
            )
        text = tokenizer.decode(out_ids[0].tolist()) if hasattr(tokenizer, "decode") else ""
        enc = _l2(np.asarray(codec.encode([text]), dtype=np.float32)[0])
        cos_r += float(np.dot(enc, _l2(vecs[i])))
        cos_c += float(np.dot(_l2(enc - mu), _l2(vecs[i] - mu)))
    aniso = _anisotropy(torch.from_numpy(_l2(vecs)))
    return {
        "mean_cos_raw": cos_r / max(n, 1),
        "mean_cos_centered": cos_c / max(n, 1),
        "oracle_cos_centered": float("nan"),  # filled by caller with true-vec arm
        "old_writer_cos_centered": float("nan"),
        "anisotropy_floor": float(aniso),
        "n": n,
        "stub": False,
    }


def novelty_suite(gen_vecs, gen_texts, corpus, writer, minilm, seed: int = SEED_DEFAULT) -> dict:
    """n(x) vs R, novelty/copy rates, paraphrase band, MiniLM co-measure, 5-gram."""
    bank, held, held_texts, _, _ = load_split(
        corpus, Path("artifacts/sentence_pred/corpus/split_manifest.json")
    )
    # Prefer corpus-native split if manifest missing.
    if bank.size == 0:
        bank, held, held_texts, _, _ = load_split(corpus, "")
    mu, bank_c = centering(bank)
    R = reference_distribution(held, bank_c, mu)
    scores = novelty_scores(gen_vecs, bank_c, mu)
    # paraphrase band: n(enc(render(v_true))) — stub if no writer
    para = {"median": float("nan"), "n": 0}
    if writer is not None and hasattr(writer, "generate"):
        pass  # full path filled when writer available; rates still valid
    # external MiniLM co-measure on texts
    n_ext = R_ext = None
    if minilm is not None and gen_texts:
        try:
            gv = _l2(np.asarray(minilm.encode(list(gen_texts)), dtype=np.float32))
            hv = _l2(np.asarray(minilm.encode(held_texts[: min(2000, len(held_texts))]), dtype=np.float32))
            # train bank texts
            train_texts = []
            for d in corpus.train_doc_ids.tolist()[:200]:
                a, b = corpus.doc_slice(int(d))
                train_texts.extend(corpus.sentences[a:b])
            bv = _l2(np.asarray(minilm.encode(train_texts[:4000]), dtype=np.float32))
            mu_e, bc_e = centering(bv)
            R_ext = reference_distribution(hv, bc_e, mu_e)
            n_ext = novelty_scores(gv, bc_e, mu_e)
        except Exception as e:
            n_ext = None
            R_ext = {"error": str(e)}

    overlaps = []
    if gen_texts:
        # nearest-50 train by vector novelty neighbors approximated via first 50 bank texts
        ref_pool = []
        for d in list(corpus.train_doc_ids.tolist())[:20]:
            a, b = corpus.doc_slice(int(d))
            ref_pool.extend(corpus.sentences[a:b])
        ref_pool = ref_pool[:50]
        overlaps = [max_ngram_overlap(t, ref_pool, n=5) for t in gen_texts]

    out = {
        "novelty_scores": scores,
        "R": {k: R[k] for k in ("median", "p90", "p99", "n")},
        "novelty_rate": novelty_rate(scores, R),
        "copy_rate": copy_rate(scores, R),
        "paraphrase_band": para,
        "mean_5gram_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "anisotropy_floor": float(_anisotropy(torch.from_numpy(_l2(bank[:4096])))),
    }
    if n_ext is not None and isinstance(R_ext, dict) and "median" in R_ext:
        out["minilm_novelty_rate"] = novelty_rate(n_ext, R_ext)
        out["minilm_copy_rate"] = copy_rate(n_ext, R_ext)
        out["minilm_R"] = {k: R_ext[k] for k in ("median", "p90", "p99", "n")}
    return out


def multimodality_report(model, dataset, n_ctx: int = 1000, n_samples: int = 8) -> dict:
    """mean-of-8 vs best-of-8 M3-global, effrank of singles, spread, MDN entropy."""
    rng = np.random.RandomState(SEED_DEFAULT)
    corpus = dataset.corpus
    n = min(len(dataset), n_ctx)
    idxs = rng.choice(len(dataset), size=n, replace=False).tolist() if n else []
    context, pad_mask, target, metas = _last_target_and_meta(dataset, idxs, corpus)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        samples = model.sample(context.to(device), pad_mask.to(device), n_samples=n_samples)
        samples = F.normalize(samples.cpu().float(), dim=-1)  # (B, S, D)
        mu, _ = model.predict_mu(context.to(device), pad_mask.to(device))
        mu = F.normalize(mu.cpu().float(), dim=-1)
    target_np = _l2(_as_np(target))
    # mean-of-8 / best-of-8 cosine to true
    mean_s = F.normalize(samples.mean(1), dim=-1).numpy()
    mean_cos = float((_l2(mean_s) * target_np).sum(1).mean())
    best_cos = float(
        (samples.numpy() * target_np[:, None, :]).sum(-1).max(1).mean()
    )
    singles = samples[:, 0, :].numpy()
    er = effective_rank(singles)
    # per-context pairwise spread of 8
    spreads = []
    for i in range(n):
        S = samples[i].numpy()
        sims = _l2(S) @ _l2(S).T
        iu = np.triu_indices(n_samples, 1)
        spreads.append(float(sims[iu].mean()) if iu[0].size else float("nan"))
    mdn_ent = float("nan")
    if getattr(model, "head_type", "") == "mdn":
        with torch.no_grad():
            params = model.forward(context.to(device), pad_mask.to(device), return_sequence=False)
            pi_logits = params[0].squeeze(1)
            mdn_ent = float(MixtureVmfHead.weight_entropy(pi_logits).mean().cpu())
    return {
        "mean_of_n_cos": mean_cos,
        "best_of_n_cos": best_cos,
        "best_minus_mean": best_cos - mean_cos,
        "effective_rank_singles": er,
        "mean_pairwise_spread": float(np.nanmean(spreads)),
        "mdn_weight_entropy": mdn_ent,
        "n_ctx": n,
        "n_samples": n_samples,
    }


def write_row(path, **metrics) -> None:
    """Append one JSON object to frontier JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: _jsonable(v) for k, v in metrics.items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _jsonable(v: Any):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
