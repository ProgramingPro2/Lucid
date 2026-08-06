#!/usr/bin/env python3
"""D5 ablation probe: does the Phase-2 FiLM writer respond to concept conditioning?

For each of ~8 corpus sentences, generate text from:
  (a) the true encoded concept vector
  (b) a random unit vector (shuffled concept dims)
  (c) a different sentence's concept vector

Report lexical overlap (token Jaccard + char 3-gram Jaccard) of T_true vs
T_shuf / T_other. Low mean overlap => strong concept conditioning; high =>
writer behaves like a near-standalone LM.

CPU-only. Run from repo root:
  source .venv/bin/activate && PYTHONPATH=src python scripts/probe_writer_ablation.py
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path


def _force_cpu_only() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repo_bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _overlap(true: str, other: str) -> dict[str, float]:
    tok_j = _jaccard(_tokenize(true), _tokenize(other))
    char_j = _jaccard(_char_ngrams(true, 3), _char_ngrams(other, 3))
    return {
        "token_jaccard": tok_j,
        "char3_jaccard": char_j,
        "mean": 0.5 * (tok_j + char_j),
    }


def _pick_sentences(corpus_path: Path, n: int = 8, seed: int = 0) -> list[str]:
    from conceptpoc.data.corpus import load_documents

    docs = load_documents(corpus_path)
    pool: list[str] = []
    for d in docs:
        for s in d.sentences:
            s = s.strip()
            if 40 <= len(s) <= 220:
                pool.append(s)
    if len(pool) < n:
        raise ValueError(f"Need >= {n} varied sentences; found {len(pool)}")
    rng = random.Random(seed)
    # Prefer length diversity: sample from quartiles then fill randomly.
    pool_sorted = sorted(pool, key=len)
    quartiles = []
    for q in range(4):
        lo = q * len(pool_sorted) // 4
        hi = (q + 1) * len(pool_sorted) // 4
        chunk = pool_sorted[lo:hi] or pool_sorted
        quartiles.append(rng.choice(chunk))
    chosen = list(dict.fromkeys(quartiles))  # unique, preserve order
    while len(chosen) < n:
        cand = rng.choice(pool)
        if cand not in chosen:
            chosen.append(cand)
    return chosen[:n]


def _unit(v):
    import numpy as np

    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def _shuffle_unit(v, rng: random.Random):
    import numpy as np

    v = np.asarray(v, dtype=np.float32).reshape(-1).copy()
    rng.shuffle(v)
    return _unit(v)


def _generate_text(writer, tok, concept_np, max_tokens: int = 128) -> str:
    import torch

    concept = torch.from_numpy(_unit(concept_np).astype("float32")).unsqueeze(0)
    token_ids = writer.generate(
        concept,
        max_tokens=max_tokens,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.3,
    )[0].tolist()
    return tok.decode(token_ids)


def main() -> int:
    _force_cpu_only()
    root = _repo_bootstrap()
    t0 = time.time()

    print("== D5 writer ablation probe (CPU-only) ==")
    print(f"cwd={root}")

    try:
        import torch
        import numpy as np
        from conceptpoc.persistence import load_joint, load_writer
        from conceptpoc.codec.jepa_codec import JepaCodec
    except Exception as e:
        print(f"IMPORT FAILED: {type(e).__name__}: {e}")
        return 1

    jepa_path = "artifacts/jepa_v3_128t/jepa/joint_robust.pt"
    writer_path = "artifacts/writer_p2_film/writer_cond.pt"
    corpus_path = Path("data/corpus_1m.txt")

    try:
        print(f"Loading JEPA: {jepa_path}")
        model, tok = load_joint(jepa_path, device="cpu")
        codec = JepaCodec(model, tok, 128, device="cpu")
        print(f"  concept_dim={codec.dim}")
    except Exception as e:
        print(f"JEPA LOAD FAILED: {type(e).__name__}: {e}")
        return 1

    try:
        print(f"Loading writer: {writer_path}")
        writer = load_writer(writer_path, device="cpu")
        writer.eval()
        print(f"  writer params={writer.num_parameters():,}")
    except Exception as e:
        print(f"WRITER LOAD FAILED: {type(e).__name__}: {e}")
        return 1

    try:
        sentences = _pick_sentences(corpus_path, n=8, seed=42)
    except Exception as e:
        print(f"CORPUS LOAD FAILED: {type(e).__name__}: {e}")
        return 1

    print(f"\nSelected {len(sentences)} probe sentences:\n")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s[:100]}{'...' if len(s) > 100 else ''}")

    # Encode all concepts once.
    try:
        concepts = codec.encode(sentences)
    except Exception as e:
        print(f"ENCODE FAILED: {type(e).__name__}: {e}")
        return 1

    rng = random.Random(42)
    rows: list[dict] = []

    print("\n-- Per-sentence generations --")
    for i, sent in enumerate(sentences):
        z_true = concepts[i]
        z_shuf = _shuffle_unit(z_true, rng)
        j_other = (i + 1 + rng.randrange(len(sentences) - 1)) % len(sentences)
        z_other = concepts[j_other]

        try:
            t_true = _generate_text(writer, tok, z_true)
            t_shuf = _generate_text(writer, tok, z_shuf)
            t_other = _generate_text(writer, tok, z_other)
        except Exception as e:
            print(f"\n[{i}] GENERATE FAILED: {type(e).__name__}: {e}")
            continue

        ov_shuf = _overlap(t_true, t_shuf)
        ov_other = _overlap(t_true, t_other)
        rows.append(
            {
                "i": i,
                "other_i": j_other,
                "source": sent,
                "t_true": t_true,
                "t_shuf": t_shuf,
                "t_other": t_other,
                "ov_shuf": ov_shuf,
                "ov_other": ov_other,
            }
        )

        print(f"\n[{i}] SOURCE: {sent}")
        print(f"     T_true : {t_true[:160]}{'...' if len(t_true) > 160 else ''}")
        print(f"     T_shuf : {t_shuf[:160]}{'...' if len(t_shuf) > 160 else ''}")
        print(
            f"     T_other (from [{j_other}]): "
            f"{t_other[:160]}{'...' if len(t_other) > 160 else ''}"
        )
        print(
            f"     overlap(true,shuf)  tok={ov_shuf['token_jaccard']:.3f} "
            f"char3={ov_shuf['char3_jaccard']:.3f} mean={ov_shuf['mean']:.3f}"
        )
        print(
            f"     overlap(true,other) tok={ov_other['token_jaccard']:.3f} "
            f"char3={ov_other['char3_jaccard']:.3f} mean={ov_other['mean']:.3f}"
        )

    if not rows:
        print("\nNo successful generations; aborting summary.")
        return 1

    mean_shuf = float(np.mean([r["ov_shuf"]["mean"] for r in rows]))
    mean_other = float(np.mean([r["ov_other"]["mean"] for r in rows]))
    mean_shuf_tok = float(np.mean([r["ov_shuf"]["token_jaccard"] for r in rows]))
    mean_other_tok = float(np.mean([r["ov_other"]["token_jaccard"] for r in rows]))
    mean_shuf_c3 = float(np.mean([r["ov_shuf"]["char3_jaccard"] for r in rows]))
    mean_other_c3 = float(np.mean([r["ov_other"]["char3_jaccard"] for r in rows]))

    print("\n== Summary table ==")
    print(f"{'metric':<28} {'true vs shuffle':>16} {'true vs other':>16}")
    print("-" * 62)
    print(f"{'token Jaccard':<28} {mean_shuf_tok:16.4f} {mean_other_tok:16.4f}")
    print(f"{'char-3gram Jaccard':<28} {mean_shuf_c3:16.4f} {mean_other_c3:16.4f}")
    print(f"{'mean overlap':<28} {mean_shuf:16.4f} {mean_other:16.4f}")

    print("\nMETRIC SUMMARY: "
          f"mean overlap(true, shuffle)={mean_shuf:.4f} | "
          f"mean overlap(true, other)={mean_other:.4f}")
    print(
        "INTERPRETATION: low overlap => writer strongly conditioned by concept; "
        "high overlap => writer mostly ignores concept (near standalone LM)."
    )
    print(f"\nDone in {time.time() - t0:.1f}s (CPU-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
