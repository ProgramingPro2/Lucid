# Concept Model Visualizer

Interactive web UI for exploring sentence-embedding space and the concept-model architecture.

**Stack:** React + Vite + TypeScript, Tailwind CSS, `@react-three/fiber` + `@react-three/drei` (3D), React Flow (architecture diagram).

## Views

### Idea Space

3D explorer of PCA-projected sentence embeddings (`public/embeddings.json`):

- Instanced point cloud (18k–100k points) colored by KMeans cluster
- Orbit / zoom / pan controls
- Hover tooltip, click for detail panel with 10 nearest neighbors (Euclidean distance in 3D — approximate)
- Search filter/highlights matching sentences
- Cluster legend with show/hide toggles
- Optional convex hull overlay per cluster
- Banner noting the lossy PCA projection from 512-d vectors

### Architecture & Data Flow

Interactive React Flow diagram of the JEPA / frozen codec training graph:

- Clickable nodes with descriptions and tensor shapes
- JEPA vs Frozen codec toggle (changes Encoder/Decoder descriptions)
- Animated token dot along edges (“Play training flow”)
- Generate (inference) sub-panel simulating encode → predict → decode rollout
- Static training metrics: reconstruction_cosine ~0.99, next_vector_cosine ~0.58, std_z ~0.62

## Prerequisites

- **Node.js 20+** and npm
- **Python 3.10+** with project venv (only needed to regenerate embeddings from real caches)

## Development

```bash
cd rating-app/viz
npm install
npm run dev
```

Open http://localhost:5173 — both views work with the bundled sample `public/embeddings.json`.

## Regenerating embeddings

### From real ML cache (recommended)

After encoding the corpus (or running `eval`, which encodes and caches vectors):

```bash
source .venv/bin/activate
pip install -e .

# JEPA: train first, then eval creates the cache under artifacts_dir/cache/
python -m conceptpoc.cli --config configs/jepa_austen.json train-joint
python -m conceptpoc.cli --config configs/jepa_austen.json eval
python tools/export_embeddings.py --config configs/jepa_austen.json

# Frozen codec (no joint checkpoint needed):
python -m conceptpoc.cli --config configs/frozen.json prepare
python tools/export_embeddings.py --config configs/frozen.json
```

Without `--config`, export uses the **newest** `artifacts/**/cache/corpus_*.npz` by file mtime.
With `--config`, export uses that config's `artifacts_dir/cache/`.

Each entry: `{ id, text, x, y, z, cluster }`.

### Dev fallback (no cache)

When no `.npz` cache exists yet:

```bash
python tools/generate_dev_embeddings.py --limit 20000
```

Uses sentences from `data/books_corpus.txt` with random 512-d vectors — layout is for UI dev only, not semantic structure.

## Production build (standalone)

```bash
npm run build
npm run preview
```

Output: `dist/`

## FastAPI integration

Build for serving under `/viz` on the rating app:

```bash
npm run build:fastapi   # outputs to rating-app/static/viz/
uvicorn app:app --app-dir rating-app
```

- Visualizer: http://localhost:8000/viz/
- Embeddings API: http://localhost:8000/api/embeddings
- Rating UI nav includes a link to the visualizer

The FastAPI mount is skipped until `rating-app/static/viz/` exists (after first `build:fastapi`).

## Performance notes

- Points render via a single `InstancedMesh` with per-instance colors for 20k+ smooth interaction
- Convex hulls are computed on demand with `quickhull3d` (can be slow on very large clusters — toggle off if needed)
- Nearest-neighbor lookup is O(n) brute force in 3D; adequate for ~100k points on click

## Data source

Embeddings originate from the Python project's embedding cache ([`src/conceptpoc/data/cache.py`](../../src/conceptpoc/data/cache.py)):

| NPZ key        | Description                          |
|----------------|--------------------------------------|
| `vectors`      | float32 `(N, D)` sentence embeddings |
| `sentences`    | object array of sentence strings     |
| `doc_offsets`  | document boundary indices            |

Export script: [`tools/export_embeddings.py`](../../tools/export_embeddings.py)
