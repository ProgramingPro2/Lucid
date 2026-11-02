# Concept Predictor POC

A proof-of-concept "predict ideas, not tokens" language model with a
self-improvement loop. Text is encoded into per-sentence **concept vectors** by
a frozen codec, a small transformer predicts the **next concept vector**, and
predicted vectors are decoded back to text. An LLM-as-judge plus a human rating
app provide the reward signal that drives reward-ranked fine-tuning.

This is the language analogue of JEPA-style representation prediction, modeled
on Meta's Large Concept Model (LCM) and SONAR. It runs on a single GPU (or CPU
for tests). It is a research prototype, **not** an attempt at human-level AI.
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for an honest account of scope.

## How it works

```
text --[frozen encoder]--> concept vectors --[predictor]--> next concept vector --[frozen decoder]--> text
                                                                       |
                                              LLM judge + human ratings --> reward-ranked fine-tuning
```

- **Codec** (`src/conceptpoc/codec`): frozen encoder + decoder. Default encodes
  with a sentence-transformer and decodes by snapping to the nearest sentence in
  a bank (always runs). Optional SONAR backend gives generative decoding. Or
  train your own jointly (see "Train your own codec" below).
- **Predictor** (`src/conceptpoc/predictor`): a causal transformer over concept
  vectors, trained to predict the next vector (MSE + cosine loss).
- **Generation** (`src/conceptpoc/generate`): autoregressive rollout in concept
  space, then decode.
- **Reward** (`src/conceptpoc/reward`): heuristic, Lemonade, or OpenAI judge;
  human-rating quality control, preference pairs, and a reward model.
- **Loop** (`src/conceptpoc/loop`): reward-ranked fine-tuning (RAFT/RWR).
- **Experts** (`src/conceptpoc/experts`): stretch prototype of a gating router
  over multiple predictor experts.
- **Rating app** (`rating-app/`): FastAPI + SQLite UI for human raters.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# For a GPU machine, install the CUDA build of torch first (see pytorch.org).
pip install -r requirements.txt
pip install -e .
```

Optional generative decoding via SONAR: `pip install -r requirements-sonar.txt`
(stricter Python support; the project runs without it).

Optional LLM judge: copy `.env.example` to `.env` and set `JUDGE_BACKEND` to
`lemonade` or `openai`. Without an explicit backend, a deterministic offline
heuristic judge is used.

#### Local judge via Lemonade (recommended for private/offline scoring)

[Lemonade Server](https://lemonade-server.ai/) exposes an OpenAI-compatible API
for local models. Lucid uses [Flow-Judge-v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1-GGUF),
a 3.8B model purpose-built for LLM-as-judge evaluation.

```bash
# Install Lemonade Server, then pull and run the judge model
lemonade status
lemonade pull flowaicom/Flow-Judge-v0.1-GGUF:Q4_K_M --yes
lemonade list   # copy the registered model id
lemonade run <model-id>

# Configure Lucid (.env)
# JUDGE_BACKEND=lemonade
# LEMONADE_JUDGE_MODEL=<model-id from lemonade list>
```

#### Cloud judge via OpenAI

Set `JUDGE_BACKEND=openai` and `OPENAI_API_KEY` in `.env`.

## Usage

```bash
python -m conceptpoc.cli prepare    # encode corpus, cache vectors, fit decoder bank
python -m conceptpoc.cli train      # train the concept predictor
python -m conceptpoc.cli eval       # reconstruction + next-vector metrics
python -m conceptpoc.cli generate --prompt "The morning light moved across the valley." --steps 4
python -m conceptpoc.cli improve    # run the self-improvement loop
```

### Progress output

Long-running commands show tqdm progress bars and per-epoch metric lines on stderr.
Use `--quiet` / `-q` to suppress them (useful in scripts and CI):

```bash
python -m conceptpoc.cli --quiet train
```

Example training output:

```
encoding 36465 sentences...
encoding: 100%|████████| 143/143
epoch 1/40 train: 100%|████████| 512/512
[train] epoch 1/40  train_loss=0.156  val_loss=0.148  val_cosine=0.712
...
trained predictor (4,123,456 params); final val cosine=0.891
```

Joint JEPA training prints `std_z` and `recon_acc` each epoch so you can watch
for representation collapse during the run (see [docs/LIMITATIONS.md](docs/LIMITATIONS.md)).

### Train your own codec (joint JEPA)

Instead of a frozen codec, train your own encoder + decoder + predictor jointly
and self-supervised (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
anti-collapse design). Set `codec.kind` to `"jepa"` in a config, then:

```bash
python -m conceptpoc.cli --config my.json train-joint   # trains + saves artifacts/jepa/joint.pt
python -m conceptpoc.cli --config my.json eval
python -m conceptpoc.cli --config my.json generate --prompt "..." --steps 4
python -m conceptpoc.cli --config my.json improve
```

Watch the collapse alarm (`std_z`) printed by `train-joint` and stored in
`artifacts/jepa/metrics.json`. Use `jepa.tokenizer="char"` for a quick offline
run, or `"gpt2"` for real training.

Point at your own data by editing `data.corpus_path` (a `.txt` file or a
directory of `.txt` files) via a JSON config passed with `--config`, or by
replacing `data/sample_corpus.txt`.

### Rating app

```bash
uvicorn app:app --app-dir rating-app
```

Open the page, enter a name, and rate continuations. Push generated items in via
`POST /api/items`; export ratings via `GET /api/export`; check rater quality via
`GET /api/quality`.

### Model visualizer

An interactive React app explores sentence-embedding space in 3D and diagrams the
model architecture / data flow. See [rating-app/viz/README.md](rating-app/viz/README.md).

```bash
cd rating-app/viz && npm install && npm run dev          # dev server
cd rating-app/viz && npm run build:fastapi                 # build for /viz mount
uvicorn app:app --app-dir rating-app                       # then open /viz/
python tools/export_embeddings.py                          # regenerate embeddings.json
```

## Tests

```bash
pytest
```

Tests use a deterministic hashing codec and need no downloads or network.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - design and data flow
- [docs/PLAN.md](docs/PLAN.md) - scope and key decisions
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) - what this does and does not do

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
