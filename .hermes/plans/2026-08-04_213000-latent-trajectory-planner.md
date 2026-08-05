# Latent Trajectory Planner: JEPA Plans Ahead in Latent Space, Writer Executes

> **For Hermes:** Work task-by-task with a real measurement per stage. Reuse existing resumable SQLite checkpointing. Do NOT delete old artifacts — each run gets a fresh dir.

**Goal:** Stop topic drift by making the model **plan a multi-sentence trajectory in latent space *before* writing any text**, then have the (already-fluent) writer realize that plan. The one-step "predict next sentence embedding" had no headroom (ceiling test: 0.054); a **trajectory-level plan** is a coarser, learnable target that can actually steer a paragraph.

**Architecture:** Three collaborating modules.
1. **Planner (JEPA + latent trajectory)**: given context, produces a *sequence of K future concept vectors* (the outline of where the next 4-6 sentences go), by iterating a recurrent latent block — **Coconut-style continuous thought** (Hao et al. 2412.06769), not text tokens.
2. **Writer (existing `FluWriter`)**: realizes each planned concept into fluent sentences (already built, FiLM'd, judge 0.567).
3. **Anchor**: training stays grounded in real text (the JEPA target = real next-k concept vectors), so the loop can't collapse into self-consistent nonsense.

**Status vs prior work (honest):** individual pieces exist — Coconut (latent reasoning), hierarchical Text-JEPA (multi-scale prediction), plan-then-write (2017/2019), LLM-JEPA (reasoning booster). The **unpublished combination** is: *latent trajectory planning that is used to drive fluent text generation, planner+writer trained jointly*. That's the claim of this project.

**Tech Stack:** Lucid @ `~/Documents/Lucid`, RTX 3090, PyTorch 2.6, existing JEPA codec + `FluWriter` + SQLite checkpoints.

---

## The problem this attacks

Our ceiling test showed adjacent-sentence concept cosine ≈ **0.054** — one-step prediction is ill-posed (a million valid next sentences). But predicting *a coarse direction over multiple sentences* ("this passage is heading toward X") is far more constrained and learnable. A planner that emits K future concepts gives the writer a *target trajectory* instead of a single fuzzy guess. Coconut proved latent-space planning captures structure that token-space can't (implicit search). We apply that to *generation* instead of *reasoning benchmarks*.

---

## Task 1: Add K-step trajectory data to the planner (CPU, ~30 min)

**Objective:** Build a dataset where each example is `(context_vectors, next_K_concept_vectors)` — the *target* the planner must learn to hit.

**Files:**
- Create: `src/conceptpoc/trajectory/data.py`
- Reuse: `JepaCodec.encode` (frozen reasoner from `joint_robust.pt`)

**Step 1: Write the dataset**

```python
"""K-step latent trajectory dataset: context -> next-K concept vectors."""
from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.data.corpus import load_documents
from src.conceptpoc.persistence import load_joint

class TrajectoryDataset(Dataset):
    """Each item: (context_window, target_next_K_vectors).
    context_window  : (C, dim) previous C sentence vectors
    target_next_K   : (K, dim) the next K sentence vectors (the plan to hit)
    """
    def __init__(self, corpus_path, codec: JepaCodec, C=8, K=5, max_items=None):
        docs = load_documents(corpus_path)
        self.items = []
        for d in docs[: (max_items or len(docs))]:
            sents = d.sentences
            if len(sents) < C + K:
                continue
            vecs = codec.encode(sents)  # (N, dim)
            for i in range(0, len(vecs) - C - K + 1):
                ctx = vecs[i:i+C]
                tgt = vecs[i+C:i+C+K]
                self.items.append((ctx, tgt))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    ctxs = torch.stack([torch.from_numpy(b[0]) for b in batch])
    tgts = torch.stack([torch.from_numpy(b[1]) for b in batch])
    return ctxs, tgts
```

**Step 2: Quick unit test** (`tests/test_trajectory_data.py`): on `austen_corpus.txt`, assert shapes `(B, C, dim)`, `(B, K, dim)`, all finite.

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: K-step latent trajectory dataset"
```

---

## Task 2: Build the latent trajectory Planner (GPU code, ~1 hr)

**Objective:** A recurrent latent block (Coconut-style) that, given context vectors, iterates in latent space to emit a planned trajectory of K concept vectors — *without* writing text.

**Files:**
- Create: `src/conceptpoc/trajectory/planner.py`

**Step 1: The Planner module**

```python
"""Latent trajectory planner: iterated latent thinking, emits K future concepts."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class TrajectoryPlanner(nn.Module):
    def __init__(self, dim=512, n_layers=4, ctx_len=8, K=5, iters_per_step=2):
        super().__init__()
        self.dim, self.K = dim, K
        # Map context -> initial latent state
        self.ctx_proj = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=4*dim, batch_first=True)
            for _ in range(n_layers)
        ])
        self.latent_recur = nn.GRUCell(dim, dim)   # Coconut-style latent recurrence
        self.out_proj = nn.Linear(dim, K*dim)

    def forward(self, context):  # context (B, ctx_len, dim)
        h0 = self.ctx_proj(context.mean(dim=1))
        hid = h0  # (B, dim) latent "continuous thought"
        # Iterate latent space (Coconut-style) to refine the plan
        for _ in range(self.ctx_len):   # a few latent steps
            hid = self.latent_recur(hid, hid)
            hid = hid + h0
        traj = self.out_proj(hid).view(-1, self.K, self.dim)
        return traj  # (B, K, dim) planned future concepts
```

**Step 2: Sanity test** — `tests/test_planner.py`: random context `(2,8,512)` → output `(2,5,512)`; different context changes trajectory.

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: latent trajectory planner (recurrent continuous thought)"
```

---

## Task 3: Train the Planner against real trajectories (GPU, ~1-2 hrs on 3090)

**Objective:** Train planner to predict the *actual* next-K concept vectors (ground truth from real text). Cosine loss per step, weighted toward early steps (nearer = more predictable).

**Files:**
- Create: `src/conceptpoc/trajectory/train_planner.py`
- Create: `configs/trajectory_planner.json`

**Step 1: Trainer** — MSE/cosine loss on predicted vs target trajectories, AdamW, cosine LR, SQLite checkpointing every epoch, GPU gate. Freeze the JEPA encoder (targets are precomputed vectors; planner only learns to predict them).

**Step 2: Config** — `corpus_1m`, C=8, K=5, dim=512, 4 layers, batch 64, 5 epochs, lr 3e-4, `db_path` set, `gpu_gate` on.

**Step 3: Smoke then full run**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli evaluate nothing  # (verify import)
python -m src.conceptpoc.trajectory.train_planner --config configs/trajectory_planner.json
```

**Step 4: Measure** — record **trajectory prediction cosine** per epoch from the DB. **Acceptance:** reaching cosine notably above the one-step 0.054 ceiling (e.g. >0.10 on early steps) proves the multi-step objective is more learnable — the whole premise.

**Step 5: Commit + push**

```bash
git add -A && git commit -m "feat: train latent trajectory planner; measure multi-step prediction"
```

---

## Task 4: Decode the plan through the Writer (integration, ~1 hr)

**Objective:** At generation, planner emits K concepts → writer realizes each into text → feed realized text back. This is the closed loop that the published work lacks.

**Files:**
- Modify: `src/conceptpoc/cli.py` (new `generate-trajectory`)

**Step 1: The closed-loop generate**

```python
# 1. Encode seed prompt -> context vectors (JepaCodec)
# 2. planner(context) -> planned_vectors (K, dim)
# 3. for each planned vector v in planned_vectors:
#      writer.generate(v) -> sentence text   (existing FluWriter, FiLM, rep-penalty)
#      encode(sentence) -> append to context  (refresh for next)
# 4. return all sentences
```

**Step 2: Smoke test on 2 prompts**: "The scientist discovered that", "The morning light moved across the valley as". **Read:** do the sentences now stick to *one coherent direction* (follow the plan) instead of jumping topics? This is the deliverable — plural sentences tracking a single planned trajectory.

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: closed-loop trajectory generation (plan then write)"
```

---

## Task 5: Evaluate — the honest test

**Objective:** Show whether planning ahead beats one-step on *coherence* (the metric the prior hierarchical-JEPA work never measured).

**Files:**
- Reuse: `cmd_eval_harness` on `configs/hybrid_eval.json`

**Step 1: Compare vs one-step hybrid**

- one-step hybrid (already have): seq cosine 0.153, judge 0.567
- trajectory hybrid (new): same harness

**Step 2: The key metric** — does the **trajectory** version hold a topic across all K sentences? Look at: (a) `seq` cosine, (b) manual read — are the K sentences actually about the same thing? The planner only "wins" if plural sentences track one direction. (c) Optionally add a sentence-level topic-consistency score (embed consecutive generated sentences, check inter-sentence cosine).

**Step 3: Commit results**

```bash
git add -A && git commit -m "eval: trajectory vs one-step generation comparison"
```

---

## Decision gates

- **Task 3 gate:** if trajectory cosine realistically exceeds 0.054, the multi-step objective is learnable → proceed. If it's also ~0.05, the concept space itself is the limit → the real lever is *retraining the JEPA encoder to build a space where forward prediction is possible* (the VICReg-uniformity-vs-predictability tension we identified).
- **Task 5 gate:** if planned sentences hold a topic better than one-step → the architecture is novel and working; then scale it (more data / rented compute is now justified).
- If planning does NOT beat one-step on coherence → honest negative result: hierarchy helps representation (per prior work) but not generation, and the fundamental fix is a token-level generative baseline.

## Risks

| Risk | Mitigation |
|------|-----------|
| Planner trivially copies last context vector (mode collapse) | Early-step loss weighting + variance regularization on planned vectors + EMA target; check trajectory diversity |
| FiLM writer already "good" so plan changes little | Measure inter-sentence topic consistency, not just judge score |
| Planned vectors off-manifold (writer garbles) | Decoder-robust already trained; reuse robust decoder for decode |
| Recurrent latent (Coconut-style) unstable | LayerNorm on latent state, residual connections, init to identity |

## Files changed

| File | Change |
|------|--------|
| `src/conceptpoc/trajectory/__init__.py` | new package |
| `src/conceptpoc/trajectory/data.py` | K-step trajectory dataset |
| `src/conceptpoc/trajectory/planner.py` | latent recurrent planner |
| `src/conceptpoc/trajectory/train_planner.py` | training with DB checkpoints |
| `configs/trajectory_planner.json` | new config |
| `src/conceptpoc/cli.py` | `generate-trajectory` command |
| `tests/test_trajectory_data.py`, `tests/test_planner.py` | tests |

## Execution order

```
Task 1 (data) -> Task 2 (planner) -> Task 3 (train + gate) ->
Task 4 (closed-loop generate) -> Task 5 (evaluate)
```
