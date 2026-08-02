# Lucid JEPA+Diffusion Improvements — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix the three critical bottlenecks in the Lucid JEPA+diffusion text generation system: weak context conditioning, decoder manifold gap, and tiny context window. Make the general model produce on-topic text instead of word salad.

**Architecture:** Add classifier-free guidance and FiLM conditioning to the diffusion predictor so it actually pays attention to the prompt. Bump context window from 8→32 sentences and token budget from 48→128. Strengthen decoder robustness training with more epochs and higher diffusion mix ratio.

**Tech Stack:** PyTorch 2.6, existing Lucid codebase at `~/Documents/Lucid/`

---

## Current State (what we know)

- **JEPA encoder/decoder:** 512-dim, 4-layer encoder, 4-layer decoder, GPT-2 tokenizer, jointly trained with VICReg anti-collapse
- **Diffusion predictor:** 6-layer context transformer, 3-layer MLP denoise head, DDIM sampler, 100 train timesteps, 30 sampling steps
- **Collapse solved:** seq_cos near 0 (was 0.849 with regression), distinct-2 near 1.0, zero verbatim repeats
- **Remaining problems:**
  1. Context conditioning nearly dead — val_sample_cosine = 0.025 (predictions barely correlate with prompt)
  2. Decoder manifold gap — bank cosine = 0.34 (predicted vectors are off-manifold, decoder produces garbled text)
  3. Context window = 8 sentences → model has almost no memory
  4. Sentence token budget = 48 → sentences get cut off

## Available data

| Corpus | Size | Sentences | Notes |
|--------|------|-----------|-------|
| `data/corpus_1m.txt` | 130MB, ~24M words | ~1M | Current general corpus |
| `data/large_corpus.txt` | 39MB | ~300K | Alternative larger corpus |
| `data/corpus_100k.txt` | 12MB | ~100K | Small corpus for fast experiments |
| `data/austen_corpus.txt` | 737KB | ~3.6K | Jane Austen (already works well) |

---

## Task 1: Add CFG config fields to DiffusionConfig

**Objective:** Add classifier-free guidance parameters to the config dataclass.

**Files:**
- Modify: `src/conceptpoc/config.py:55-70`

**Step 1: Add fields to DiffusionConfig**

```python
@dataclass
class DiffusionConfig:
    """Concept-space conditional diffusion head (replaces point regression)."""
    num_train_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    predict_type: str = "epsilon"
    sampler: str = "ddim"
    sampling_steps: int = 30
    head_hidden: int = 1024
    head_layers: int = 3
    time_dim: int = 128
    sample_jitter: float = 0.0
    # Classifier-free guidance: probability of dropping context during training.
    # 0.1 = 10% of training steps train unconditionally.
    cfg_drop_prob: float = 0.1
    # Guidance scale at inference. >1 amplifies conditioning; 1 = normal.
    # Standard values: 3.0 (balanced), 5.0 (strong prompt adherence), 7.5+ (aggressive).
    cfg_scale: float = 3.0
    # FiLM conditioning: inject context as scale+shift at every denoising layer
    # instead of only concatenating at the input. Much stronger conditioning.
    use_film: bool = False
```

**Step 2: Verify**

Run: `python -c "from src.conceptpoc.config import DiffusionConfig; c = DiffusionConfig(); print(c.cfg_drop_prob, c.cfg_scale, c.use_film)"`
Expected: `0.1 3.0 False`

**Step 3: Commit**

```bash
cd ~/Documents/Lucid
git add src/conceptpoc/config.py
git commit -m "feat: add cfg_drop_prob, cfg_scale, use_film to DiffusionConfig"
```

---

## Task 2: Add FiLM conditioning to DenoiseHead

**Objective:** Replace the simple context concatenation with FiLM (Feature-wise Linear Modulation) — the context embedding generates scale+shift parameters injected at every layer. This forces the model to attend to the context.

**Files:**
- Modify: `src/conceptpoc/predictor/diffusion.py:45-71`

**Step 1: Rewrite DenoiseHead with FiLM**

Replace the entire `DenoiseHead` class:

```python
class DenoiseHead(nn.Module):
    """MLP that predicts diffusion target from (x_noisy, context, timestep).

    With ``use_film=True``, context is injected as scale+shift at every layer
    instead of only concatenated at the input. This is a much stronger
    conditioning signal and directly addresses the weak-context problem.
    """

    def __init__(
        self,
        concept_dim: int,
        context_dim: int,
        cfg: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.use_film = getattr(cfg, "use_film", False)
        self.time_emb = SinusoidalTimestepEmbedding(cfg.time_dim)
        
        if self.use_film:
            # Input: [x_noisy, time_emb]
            in_dim = concept_dim + cfg.time_dim
            self.film_generators = nn.ModuleList()
        else:
            # Input: [x_noisy, context, time_emb] (original behavior)
            in_dim = concept_dim + context_dim + cfg.time_dim
        
        layers: list[nn.Module] = []
        hidden = cfg.head_hidden
        d = in_dim
        for i in range(cfg.head_layers):
            layers.extend([nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)])
            if self.use_film:
                # FiLM: generate scale+shift from context for each layer
                self.film_generators.append(
                    nn.Sequential(
                        nn.Linear(context_dim, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, hidden * 2),  # scale + shift
                    )
                )
            d = hidden
        layers.append(nn.Linear(d, concept_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x_noisy: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        te = self.time_emb(t)
        if self.use_film:
            h = torch.cat([x_noisy, te], dim=-1)
            # Apply each layer manually so we can inject FiLM in between
            layer_idx = 0
            film_idx = 0
            for module in self.net:
                h = module(h)
                # After each LayerNorm (every 3rd submodule: Linear, GELU, LayerNorm)
                if isinstance(module, nn.LayerNorm) and film_idx < len(self.film_generators):
                    film = self.film_generators[film_idx](context)
                    scale, shift = film.chunk(2, dim=-1)
                    h = h * (1.0 + scale) + shift
                    film_idx += 1
                layer_idx += 1
            return h
        else:
            h = torch.cat([x_noisy, context, te], dim=-1)
            return self.net(h)
```

**Step 2: Verify syntax**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.predictor.diffusion import DenoiseHead; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/conceptpoc/predictor/diffusion.py
git commit -m "feat: add FiLM conditioning to DenoiseHead for stronger context signal"
```

---

## Task 3: Add CFG to DiffusionConceptPredictor

**Objective:** Add classifier-free guidance training (random context dropout) and inference (conditional + unconditional combination with guidance scale).

**Files:**
- Modify: `src/conceptpoc/predictor/diffusion.py` — the `DiffusionConceptPredictor` class

**Step 1: Add `_model_output` method that handles CFG null context**

Add/modify these methods inside `DiffusionConceptPredictor`:

```python
def _null_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Zero context embedding for unconditional CFG passes."""
    return torch.zeros(batch_size, self.cfg.d_model, device=device)

def _model_output(
    self,
    x_noisy: torch.Tensor,
    t: torch.Tensor,
    context_emb: torch.Tensor,
) -> torch.Tensor:
    return self.head(x_noisy, t, context_emb)
```

**Step 2: Modify `diffusion_loss` to randomly drop context for CFG**

Replace the method (lines 175-201):

```python
def diffusion_loss(
    self,
    context: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    *,
    segment_ids: torch.Tensor | None = None,
    theme_vec: torch.Tensor | None = None,
) -> DiffusionLossOutput:
    """Denoising MSE with optional CFG context dropout."""
    c = self.encode_context(
        context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec
    )
    # CFG: randomly drop context during training
    if self.diff_cfg.cfg_drop_prob > 0 and self.training:
        drop_mask = (
            torch.rand(c.size(0), 1, device=c.device) < self.diff_cfg.cfg_drop_prob
        ).float()
        null_c = self._null_context(c.size(0), c.device)
        c = drop_mask * null_c + (1.0 - drop_mask) * c
    
    x0 = F.normalize(target, dim=-1)
    b = x0.size(0)
    t = torch.randint(
        0, self.schedule.num_train_timesteps, (b,), device=x0.device, dtype=torch.long
    )
    noise = torch.randn_like(x0)
    x_t = self.schedule.q_sample(x0, t, noise)
    pred = self._model_output(x_t, t, c)
    if self.diff_cfg.predict_type == "x0":
        target_out = x0
    else:
        target_out = noise
    mse = F.mse_loss(pred, target_out)
    return DiffusionLossOutput(total=mse, mse=mse)
```

**Step 3: Modify `sample` method to use CFG at inference**

Replace the `sample` method (lines 223-276):

```python
@torch.no_grad()
def sample(
    self,
    context: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    *,
    segment_ids: torch.Tensor | None = None,
    theme_vec: torch.Tensor | None = None,
    sampling_steps: int | None = None,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Reverse diffusion with classifier-free guidance."""
    c_cond = self.encode_context(
        context, pad_mask, segment_ids=segment_ids, theme_vec=theme_vec
    )
    b, device = c_cond.size(0), c_cond.device
    steps = sampling_steps or self.diff_cfg.sampling_steps
    
    # Null context for CFG unconditional pass
    c_uncond = self._null_context(b, device)
    use_cfg = self.diff_cfg.cfg_scale != 1.0 and self.diff_cfg.cfg_drop_prob > 0
    
    x = torch.randn(b, self.concept_dim, device=device)

    if self.diff_cfg.sampler == "ddim":
        timesteps = torch.linspace(
            self.schedule.num_train_timesteps - 1, 0, steps, device=device
        ).long()
        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val.item(), device=device, dtype=torch.long)
            t_prev = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.zeros((), device=device, dtype=torch.long)
            )
            t_prev_b = torch.full((b,), t_prev.item(), device=device, dtype=torch.long)
            
            if use_cfg:
                # CFG: out = uncond + scale * (cond - uncond)
                out_cond = self._model_output(x, t, c_cond)
                out_uncond = self._model_output(x, t, c_uncond)
                out = out_uncond + self.diff_cfg.cfg_scale * (out_cond - out_uncond)
            else:
                out = self._model_output(x, t, c_cond)
            
            x = self.schedule.ddim_step(x, t, t_prev_b, out, self.diff_cfg.predict_type)
    else:
        for t_val in reversed(range(self.schedule.num_train_timesteps)):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            
            if use_cfg:
                out_cond = self._model_output(x, t, c_cond)
                out_uncond = self._model_output(x, t, c_uncond)
                out = out_uncond + self.diff_cfg.cfg_scale * (out_cond - out_uncond)
            else:
                out = self._model_output(x, t, c_cond)
            
            x0 = self.schedule.predict_x0_from_output(x, t, out, self.diff_cfg.predict_type)
            if t_val > 0:
                noise = torch.randn_like(x)
                var = self.schedule.posterior_variance[t]
                x = x0 + torch.sqrt(var) * noise
            else:
                x = x0

    jitter = temperature if temperature > 0 else self.diff_cfg.sample_jitter
    if jitter > 0:
        x = x + torch.randn_like(x) * jitter
    return F.normalize(x, dim=-1)
```

**Step 4: Verify**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.predictor.diffusion import DiffusionConceptPredictor; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add src/conceptpoc/predictor/diffusion.py
git commit -m "feat: add classifier-free guidance to diffusion predictor (train dropout + inference CFG)"
```

---

## Task 4: Add CFG to rollout (pass cfg_scale through to model)

**Objective:** The generation rollout needs to pass the CFG scale through. Currently the diffusion model reads `cfg_scale` from its own config, so this should work automatically. But let's verify and add a `cfg_scale` override parameter.

**Files:**
- Modify: `src/conceptpoc/generate/rollout.py:39-59` — the `_predict_next` function

**Step 1: Add cfg_scale parameter to _predict_next**

```python
def _predict_next(
    model,
    ctx: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    theme_vec: torch.Tensor | None = None,
    temperature: float = 0.0,
    cfg_scale: float | None = None,
) -> torch.Tensor:
    """One next-vector step; dispatches regression vs diffusion sampling."""
    if _is_diffusion_model(model):
        # Override cfg_scale if provided
        if cfg_scale is not None:
            orig_scale = model.diff_cfg.cfg_scale
            model.diff_cfg.cfg_scale = cfg_scale
        result = model.sample(
            ctx,
            segment_ids=segment_ids,
            theme_vec=theme_vec,
            temperature=temperature,
        )
        if cfg_scale is not None:
            model.diff_cfg.cfg_scale = orig_scale
        return result
    pred = model(ctx, segment_ids=segment_ids, theme_vec=theme_vec)
    if temperature > 0:
        pred = pred + torch.randn_like(pred) * temperature
        pred = torch.nn.functional.normalize(pred, dim=-1)
    return pred
```

**Step 2: Verify**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.generate.rollout import generate_rollout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/conceptpoc/generate/rollout.py
git commit -m "feat: add cfg_scale override to rollout _predict_next"
```

---

## Task 5: Create improved config with all fixes enabled

**Objective:** Create a new config that:
- Context window: 8 → 32 (4x bigger memory)
- Token budget: 48 → 128 (sentences no longer cut off)
- CFG enabled (drop_prob=0.1, scale=3.0)
- FiLM conditioning enabled
- Bigger decoder robustness pass
- Larger predictor (8 layers instead of 6)
- More diffusion timesteps (200 instead of 100)
- Train on `large_corpus.txt` (39MB) for faster iteration

**Files:**
- Create: `configs/jepa_large_v2.json`

**Step 1: Write the config**

```json
{
  "artifacts_dir": "artifacts/jepa_large_v2",
  "codec": {
    "kind": "jepa",
    "device": "auto",
    "jepa_checkpoint": "artifacts/jepa_large_unif/jepa/joint.pt"
  },
  "data": {
    "corpus_path": "data/large_corpus.txt",
    "context_len": 32
  },
  "predictor": {
    "head": "diffusion",
    "d_model": 512,
    "n_layers": 8,
    "n_heads": 8,
    "max_seq_len": 64,
    "dropout": 0.1,
    "diffusion": {
      "num_train_timesteps": 200,
      "beta_start": 0.0001,
      "beta_end": 0.02,
      "predict_type": "epsilon",
      "sampler": "ddim",
      "sampling_steps": 30,
      "head_hidden": 1024,
      "head_layers": 4,
      "time_dim": 128,
      "sample_jitter": 0.0,
      "cfg_drop_prob": 0.1,
      "cfg_scale": 3.0,
      "use_film": true
    }
  },
  "train": {
    "batch_size": 16,
    "lr": 0.0003,
    "weight_decay": 0.01,
    "epochs": 15,
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "val_fraction": 0.05,
    "seed": 42
  },
  "eval": {
    "eval_path": "data/eval",
    "rollout_steps": 6,
    "output_dir": "eval_v2",
    "compute_latent_diagnostics": true
  },
  "gpu_gate": {
    "enabled": false
  },
  "jepa": {
    "dim": 512,
    "tokenizer": "gpt2",
    "max_tokens": 128,
    "context_len": 32
  },
  "decoder_robust": {
    "epochs": 6,
    "batch_size": 8,
    "lr": 0.0001,
    "perturb_sigma": 0.12,
    "scheduled_sampling_start": 0.0,
    "scheduled_sampling_end": 0.6,
    "sampling_steps": 15
  }
}
```

Note: This reuses the existing `jepa_large_unif` joint checkpoint. If that doesn't exist, we'll need to train it first (Task 5b).

**Step 2: Verify the config loads**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.config import Config; c = Config.load('configs/jepa_large_v2.json'); print(f'context_len={c.data.context_len}, max_tokens={c.jepa.max_tokens}, cfg={c.predictor.diffusion.cfg_scale}, film={c.predictor.diffusion.use_film}')"`
Expected: `context_len=32, max_tokens=128, cfg=3.0, film=True`

**Step 3: Commit**

```bash
git add configs/jepa_large_v2.json
git commit -m "feat: add jepa_large_v2 config with CFG, FiLM, 32-context, 128-tokens"
```

---

## Task 6: Verify existing JEPA checkpoint exists

**Objective:** Check if `artifacts/jepa_large_unif/jepa/joint.pt` exists. If not, we need to train the joint JEPA on large_corpus first before the diffusion predictor.

**Files:**
- Check: `artifacts/jepa_large_unif/jepa/joint.pt`

**Step 1: Check**

Run: `ls -la artifacts/jepa_large_unif/jepa/joint.pt 2>/dev/null && echo "EXISTS" || echo "MISSING"`
If missing: we need to train the JEPA encoder/decoder first.

**Step 2a: If checkpoint exists**

Skip to Task 7.

**Step 2b: If checkpoint missing — create config and train JEPA first**

Create `configs/jepa_large_v2_jepa.json` (JEPA-only pretraining config, no diffusion) and run `python -m conceptpoc.cli --config configs/jepa_large_v2_jepa.json train-joint`. This takes ~a few hours and must complete before Task 7.

---

## Task 7: Train the improved diffusion predictor

**Objective:** Run the diffusion training with all improvements enabled.

**Command:**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli --config configs/jepa_large_v2.json train-diffusion
```

Expected: Training runs for 15 epochs on `large_corpus.txt`, printing per-epoch val_loss and val_sample_cosine. We want to see val_sample_cosine rise above 0.05 (currently 0.025 with old config — CFG+FiLM should push this up).

**Validation:** Monitor for:
- val_sample_cosine increasing (means context conditioning is working)
- val_loss decreasing
- No CUDA OOM (batch_size=16 with 512-dim should fit easily on 24GB)

---

## Task 8: Run decoder-robust training with higher mix ratio

**Objective:** After diffusion training, strengthen the decoder to handle off-manifold diffusion samples better.

**Command:**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli --config configs/jepa_large_v2.json train-decoder-robust
```

Expected: 6 epochs, mixing in up to 60% diffusion-predicted vectors. recon_acc should stay >0.5.

---

## Task 9: Evaluate with generation

**Objective:** Generate text with the improved model and check if it stays on topic.

**Command:**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
# Try multiple prompts with different CFG scales
python -m conceptpoc.cli --config configs/jepa_large_v2.json generate --prompt "The scientist discovered that" --steps 6
python -m conceptpoc.cli --config configs/jepa_large_v2.json generate --prompt "The morning light moved across" --steps 6
python -m conceptpoc.cli --config configs/jepa_large_v2.json generate --prompt "In the year 2045, humanity" --steps 6
```

**Success criteria:**
- Output is grammatically structured (should be — diffusion already solved this)
- Output is **somewhat related to the prompt** (this is what CFG+FiLM should fix)
- Output doesn't repeat itself (shouldn't — collapse is solved)

---

## Risks and fallbacks

| Risk | Mitigation |
|------|-----------|
| CFG causes unstable training | Drop cfg_drop_prob to 0.05, lower cfg_scale to 2.0 |
| FiLM increases memory too much | Fall back to use_film=false, rely on CFG alone |
| large_corpus.txt JEPA checkpoint missing | Train JEPA on large_corpus first (~hours) |
| val_sample_cosine still < 0.05 after training | Increase cfg_scale to 5.0, head_layers to 6 |
| Decoder still garbles text after robust training | Increase perturb_sigma to 0.15, epochs to 10 |
| CUDA OOM on 24GB | Lower batch_size from 16 to 8, use gradient accumulation |

## What changed and why (plain English)

| Change | What it does | Why |
|--------|-------------|-----|
| **CFG** | Randomly trains the model to work WITHOUT the prompt sometimes, then at generation time amplifies the "with prompt" signal | Forces the model to actually use the prompt instead of ignoring it |
| **FiLM** | Injects the context/prompt into EVERY layer of the denoiser, not just the input | The context signal gets stronger the deeper it goes — single injection at input gets diluted |
| **context_len 32** | Model sees 32 previous sentences instead of 8 | 8 sentences is ~200 words of memory — not enough to track what a conversation is about |
| **max_tokens 128** | Sentences can be up to 128 tokens instead of 48 | Current 48 cuts most sentences in half, breaking meaning |
| **decoder_robust 6 epochs, 60% mix** | Train the decoder on more predicted vectors, with stronger perturbations | Current decoder was trained on clean vectors but gets off-manifold diffusion output |

## Execution order

```
Task 1 (config fields) → Task 2 (FiLM head) → Task 3 (CFG in predictor) → Task 4 (rollout) → Task 5 (new config) → Task 6 (check checkpoint) → Task 7 (train diffusion) → Task 8 (train decoder robust) → Task 9 (evaluate)
```
