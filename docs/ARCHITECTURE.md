# Architecture

## Overview

The system predicts ideas (sentence-level concept vectors) instead of tokens.
Three stages, with the encoder/decoder frozen and only the predictor trained:

```
                 frozen                     trained                  frozen
   text  ----------------->  concept    ------------->  next      ----------->  text
 (sentences)   encoder       vectors      predictor    concept       decoder
                                                        vector
```

A reward loop wraps generation: decoded text is scored by an LLM-as-judge and by
human raters; the scores drive reward-ranked fine-tuning of the predictor.

## Components

### Codec (`src/conceptpoc/codec`)
- `ConceptCodec` (ABC): `encode(texts) -> (n, dim)` and `decode(vectors) -> texts`.
  Vectors are L2-normalized so dot products are cosine similarities.
- `SentenceTransformerCodec`: frozen sentence-transformer encoder; retrieval
  decoder (`RetrievalDecoder`) snaps a vector to the nearest sentence in a bank.
- `SonarCodec`: optional; SONAR encoder + generative decoder (LCM-style).
- `HashingCodec`: deterministic, dependency-free, for tests.
- `JepaCodec`: wraps a jointly-trained encoder/decoder you train yourself (see
  below). Decoding is generative (no bank), and it carries the jointly-trained
  predictor.

### Joint JEPA codec (`src/conceptpoc/jepa`) - train your own concept space

Instead of borrowing a frozen codec, this trains the encoder, decoder and
predictor together, self-supervised, on the corpus. The entire design exists to
defeat representation collapse (the trivial solution where the encoder maps
everything to one constant vector and prediction becomes meaninglessly easy).

Three defenses, used together:

1. Reconstruction anchor: `TextDecoder` must regenerate each sentence from its
   own concept vector (`losses.reconstruction_loss`), so a constant vector is
   impossible.
2. EMA stop-grad target (`ema.EmaEncoder`): prediction targets come from a slow
   momentum copy of the encoder with gradients cut, so the encoder cannot make
   targets trivial.
3. VICReg variance + covariance (`losses.variance_loss`, `covariance_loss`) on
   the *unnormalized* projection, a direct anti-collapse push.

Combined loss: `L = L_recon + alpha*L_pred + var_weight*L_var + cov_weight*L_cov`.
`L_recon` wants `z` to keep all sentence detail; `L_pred` wants `z` predictable;
`alpha` balances them and is ramped in via curriculum:

```
phase A (recon_warmup): alpha = 0   -> establish a usable, non-collapsed space
phase B (alpha_warmup): alpha 0 -> alpha_max while predictor + EMA target train
```

`train_joint.py` runs the curriculum, updates the EMA encoder each step, and
logs a collapse alarm (mean per-dim std of the unnormalized embedding). The
trained `JointJepa` is wrapped by `JepaCodec` so generation, reward, and the
self-improvement loop work unchanged. Modules: `tokenizer.py` (BPE or offline
char), `encoder.py`, `decoder.py` (autoregressive), `ema.py`, `losses.py`,
`joint_model.py`, `dataset.py` (windowed sentences), `train_joint.py`.

### Data (`src/conceptpoc/data`)
- `segment.py`: regex sentence segmentation (no heavy NLP deps).
- `corpus.py`: load `.txt` files; blank lines delimit documents; sentence order
  is preserved within a document.
- `cache.py`: encode once, cache vectors + sentences + document offsets (npz).
- `dataset.py`: fixed-length concept-sequence windows that never cross document
  boundaries; right-padded collate with a key-padding mask.

### Predictor (`src/conceptpoc/predictor`)
- `model.py`: causal `TransformerEncoder` over concept vectors. Reads the last
  real position and projects to a normalized next-vector prediction.
- `train.py`: AdamW + OneCycle; loss is `mse_weight * MSE + cosine_weight * (1 - cos)`.

### Generation (`src/conceptpoc/generate`)
- `rollout.py`: autoregressive rollout in concept space with optional Gaussian
  noise (temperature) for sampling diverse candidates; decode via the codec.

### Reward (`src/conceptpoc/reward`)
- `judge.py`: `HeuristicJudge` (offline default), `LemonadeJudge` (local LLM
  via Lemonade Server), and `OpenAIJudge` (cloud LLM). Select with
  `JUDGE_BACKEND=heuristic|lemonade|openai`.
- `quality_control.py`: flags constant / too-fast / consensus-violating raters.
- `preference.py`: builds (chosen, rejected) pairs from scored rollouts.
- `reward_model.py`: Bradley-Terry reward model over concept trajectories.

### Loop (`src/conceptpoc/loop`)
- `self_improve.py`: reward-ranked fine-tuning. Sample N rollouts per prompt,
  keep the top fraction by reward, fine-tune the predictor on them, repeat.

### Experts (`src/conceptpoc/experts`)
- `router.py`: `GatingRouter` produces a soft distribution over experts;
  `MixtureOfPredictors` blends expert next-vector predictions. Prototype of the
  "experts that form bonds" idea.

### Rating app (`rating-app/`)
- FastAPI + SQLite. `items` and `ratings` tables; HTML rating UI; JSON APIs for
  ingest (`/api/items`), export (`/api/export`), and quality control (`/api/quality`).

## Why concept-level, why frozen codec
- Sequences of sentences are short, so the predictor stays small and trains on a
  single GPU.
- A frozen pretrained codec gives a usable concept space and a decoder for free,
  which is what makes the language version feasible without lab-scale compute.

## Data flow for the self-improvement loop

```
prompts -> rollouts (xN) -> decode -> judge/human -> reward
   ^                                                   |
   |                   keep top-k by reward            v
   +-------------------- fine-tune predictor <---------+
```
