# Sharpen the Reasoner: Fixing Topic Drift in the Hybrid Pipeline

> **For Hermes:** Work task-by-task. Each change gets a real measurement before/after. Do NOT delete existing artifacts — every stage uses the SQLite resumable checkpoints.

**Goal:** Stop the text from drifting off-topic. The pipeline now *writes fluently* (grammar fixed); the remaining defect is that the **reasoner's next-sentence prediction is too fuzzy (cosine 0.048) to pin the topic, and the writer's single concept token gets diluted among 256 language tokens.**

**Architecture:** Two independent levers, diagnosed first:
1. **Is next-sentence meaning inherently hard to predict?** (measure the ceiling)
2. **Does the writer actually *use* the concept vector strongly enough?** (Likely NO — it's 1 token in 257.)

**Tech Stack:** Lucid codebase @ `~/Documents/Lucid`, RTX 3090, PyTorch 2.6, existing JEPA+hybrid pipeline, SQLite checkpointing.

---

## Current evidence (why this plan)

| Signal | Value | Meaning |
|--------|-------|---------|
| Predictor val_sample_cosine | 0.048 | Reasoner's next-guess is weak |
| Hybrid seq cosine | 0.172 | Inflated — includes seed prompt |
| Judge score | ~0.55 | Grammar/coherence fine; not the blocker |
| Observed | scientist→whales→beer→gods | Topic lost after 1-3 sentences |

**Two candidate root causes, tested before we invest hours:**
- **(A) Ceiling:** adjacent sentences in real text don't share much meaning → 0.048 might be *near the best possible*. If so, chasing higher predictor cosine is useless.
- **(B) Dilution (most likely):** the writer gets the topic as **ONE concept token** at position 0 of 257. After 12 layers of self-attention, that single token's signal is swamped. The topic is technically "in" but effectively gone.

The plan measures (A) cheaply, then fixes (B) regardless — because (B) is a real, concrete, fixable weakness.

---

## Task 1: Measure the prediction ceiling (diagnostic — 2 min, CPU)

**Objective:** Find the average cosine between *ground-truth adjacent* sentence embeddings. If it's ~0.05, the predictor is already near ceiling and we stop chasing cosine; if it's ~0.3+, there's real headroom.

**Files:**
- Create: `src/conceptpoc/eval/ceiling.py`
- Run against `artifacts/jepa_v3_128t/jepa/joint_robust.pt`

**Step 1: Write the diagnostic**

```python
"""Measure: how predictable are adjacent sentences in concept space?"""
from __future__ import annotations
import sys, numpy as np, torch
sys.path.insert(0, "src")
from src.conceptpoc.data.corpus import load_documents
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.persistence import load_joint

def main():
    model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt")
    codec = JepaCodec(model, tok, max_tokens=128, device="cpu")
    docs = load_documents("data/corpus_1m.txt")
    # Sample ~2000 consecutive sentence pairs from ~200 docs.
    cosines = []
    n = 0
    for d in docs[:200]:
        sents = d.sentences[:50]
        for i in range(1, len(sents)):
            if n >= 2000: break
            v0 = codec.encode([sents[i-1]])[0]
            v1 = codec.encode([sents[i]])[0]
            cosines.append(float((v0 @ v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))))
            n += 1
    a = np.array(cosines)
    print(f"adjacent-sentence cosine: mean={a.mean():.4f} median={np.median(a):.4f} std={a.std():.4f}")

if __name__ == "__main__":
    main()
```

**Step 2: Run it**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -m src.conceptpoc.eval.ceiling`
Expected: prints mean cosine (~0.02–0.3). **Record the number.**

**Step 3: Interpret (decision gate)**

- If mean < 0.10 → adjacent sentences ARE inherently near-orthogonal in this concept space. **Predictor cosine 0.048 is near the ceiling.** → Skip Task 3 (don't waste hours on more predictor training); the whole "predict the next sentence meaning" objective is the weak link.
- If mean > 0.15 → there's real headroom; pursue Task 3 too.

Either way proceed to **Task 2** (fixing the writer's concept dilution), because that's an independent defect.

**Step 4: Commit**

```bash
git add src/conceptpoc/eval/ceiling.py && git commit -m "diag: adjacent-sentence embedding cosine ceiling probe"
```

---

## Task 2: Make the writer actually USE the concept (THE main fix)

**Objective:** Stop the topic signal from being washed out. Currently the concept is ONE token at position 0. Fix: inject the concept **with FiLM (scale+shift) into every decoder layer**, and **repeat it across several slots**, so the topic stays active throughout generation — the same FiLM trick that helped the predictor.

**Files:**
- Modify: `src/conceptpoc/writer/model.py` (Block + FluWriter)
- Test: extend `tests/test_writer.py`

**Step 1: Add FiLM conditioning to each Block**

Add a `concept_dim` to `Block` and a learned FiLM transform. Inject after the FFN (like the predictor's DenoiseHead):

```python
class Block(nn.Module):
    def __init__(self, dim, n_heads, rope_len, ff_mult=4, dropout=0.0, film_dim=0):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, rope_len, dropout)
        self.ffn = SwiGLU(dim, dim * ff_mult)
        self.norm2 = RMSNorm(dim)
        self.film = None
        if film_dim:
            # Learns a per-layer scale+shift from the concept vector.
            self.film = nn.Linear(film_dim, 2 * dim, bias=False)

    def forward(self, x, concept=None):
        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))
        if self.film is not None and concept is not None:
            gamma, beta = self.film(concept).unsqueeze(1).chunk(2, dim=-1)
            x = x * gamma + beta
        return x
```

**Step 2: Wire FiLM through FluWriter**

Pass `concept` into every block; keep the (now optional) prepended concept token for attention, but rely on FiLM for strong injection:

```python
class FluWriter(nn.Module):
    def __init__(self, ..., use_concept=True, use_film=True, concept_dim=512):
        ...
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, max_seq_len+8, ff_mult, dropout,
                  film_dim=concept_dim if (use_concept and use_film) else 0)
            for _ in range(n_layers)
        ])
        ...
    def forward(self, input_ids, concept=None, labels=None):
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        h = self.tok_embed(input_ids)
        if self.use_concept and concept is not None:
            c = self.concept_proj(concept).unsqueeze(1)
            h = torch.cat([c, h], dim=1)
        for blk in self.blocks:
            h = blk(h, concept=concept if self.use_film else None)
        h = self.norm_out(h)
        logits = self.lm_head(h[:, 1:])
        ...
```

**Step 3: Add a FiLM test (concept must change output strongly)**

```python
def test_writer_film_uses_concept_strongly():
    vocab = 2048
    w = FluWriter(vocab, dim=256, n_layers=4, n_heads=4, max_seq_len=64,
                  concept_dim=128, use_concept=True, use_film=True)
    ids = torch.randint(0, vocab, (2, 16))
    cA = torch.zeros(2, 128); cA[:, 0] = 5.0
    cB = torch.zeros(2, 128); cB[:, 1] = 5.0
    la, lb = w(ids, cA, labels=ids[:,1:].clone()), w(ids, cB, labels=ids[:,1:].clone())
    # Different concepts -> measurably different loss/logits
    assert not torch.allclose(w(ids,cA), w(ids,cB), atol=1e-2)
```

**Step 4: Run tests**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -m pytest tests/test_writer.py -v`
Expected: all pass (including new FiLM test).

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add per-layer FiLM concept injection to writer (fixes topic dilution)"
```

---

## Task 3: (Conditional) Improve the predictor IF ceiling test shows headroom

**Objective:** Only if Task 1 showed adjacent-sentence cosine > 0.15. Train the diffusion predictor longer / larger so its 0.048 guess improves.

**Files:**
- Modify: `configs/jepa_v3_128t.json` (bump epochs 15→25, n_layers 8→12, head_hidden 1024→1536)

**Step 1: Make a v4 config (don't touch v3)**

Copy to `configs/jepa_v4_pred.json`; bump predictor capacity + epochs.

**Step 2: Train + measure**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli --config configs/jepa_v4_pred.json train-diffusion
```
Watch val_sample_cosine in the DB per epoch (resumable). Target: > 0.06.

**Step 3: Commit**

```bash
git add configs/jepa_v4_pred.json && git commit -m "feat: higher-capacity predictor v4 (conditional on ceiling test)"
```

---

## Task 4: Retrain Phase-2 writer with FiLM + re-evaluate

**Objective:** With FiLM in the writer, retrain Phase-2 conditioning so the writer learns to obey the concept vector strongly. Re-run the hybrid eval and compare topic-tracking.

**Files:**
- Modify: `configs/writer_phase2.json` (add `use_film: true`)
- Run: `writer-condition`

**Step 1: Update Phase-2 config to enable FiLM**

Add `"use_film": true` to the writer block. (The writer model now supports it.)

**Step 2: Retrain Phase-2**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli --config configs/writer_phase2.json writer-condition
```
(Epochs ~8, resumable via DB. A fresh run_id or clear the P2 DB so it doesn't resume the no-FiLM checkpoint.)

**Step 3: Hybrid eval**

```bash
python -m conceptpoc.cli --config configs/hybrid_eval.json generate-hybrid --prompt "The scientist discovered that" --steps 6 --writer artifacts/writer_p2/writer_cond.pt
python -m conceptpoc.cli --config configs/hybrid_eval.json eval-harness --out eval_hybrid_film
```

**Step 4: Compare (acceptance)**

Record **seq cosine** (Generated-vs-prompt context tracking) and judge score. Success = seq cosine notably above 0.172 AND the generated samples visibly stay on-topic (scientist continues about science, not whales/beer).

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: retrain Phase-2 with FiLM; eval_hybrid_film results"
```

---

## What we are NOT doing (honest scope)

- **Not** endlessly training a bigger JEPA encoder — the encoder is good (0.85 recon).
- **Not** chasing predictor cosine upward if the ceiling test says adjacent sentences are near-orthogonal — that's physics, not a bug.
- **Not** buying hardware. Everything fits the 3090.

## Decision gates

- **Task 1 gate:** cosine < 0.10 → skip Task 3 (predictor is at ceiling); cosine > 0.15 → also do Task 3.
- **Task 4 gate:** if seq cosine + topic-stickiness don't improve after FiLM, the root cause is the *objective itself* (next-sentence embedding prediction is a weak generative goal). Then the honest pivot is a **token-level autoregressive continuation conditioned on concept** (the GPT-2-from-scratch path) rather than sentence-level reconstruction.

## Risks

| Risk | Mitigation |
|------|-----------|
| FiLM changes writer's learned weights / breaks fluency | `strict=False` load; only the 3-5 top layers + FiLM are trainable in Phase-2, base layers frozen |
| Retraining Phase-2 resumes a stale no-FiLM checkpoint | assign a **fresh run_id** (or clear P2 DB) before retrain |
| FiLM with random-init = NaN early | clip gradient; small lr; sanity-check loss finite at step 1 |

## Files changed

| File | Change |
|------|--------|
| `src/conceptpoc/eval/ceiling.py` | New diagnostic |
| `src/conceptpoc/writer/model.py` | FiLM per-layer concept injection |
| `tests/test_writer.py` | FiLM tests |
| `configs/writer_phase2.json` | `use_film: true` |
| `configs/jepa_v4_pred.json` | Conditional: bigger predictor |
| `src/conceptpoc/writer/train_phase2.py` | Optional: pass use_film to model build |

---

## Execution order

```
Task 1 (2 min, cheap) → read ceiling
  ├─ regardless → Task 2 (FiLM writer) + tests
  │    └─ Task 4 (retrain Phase-2 + eval) ← THE main experiment
  └─ if ceiling high → Task 3 (bigger predictor) in parallel, then re-run Task 4
```
