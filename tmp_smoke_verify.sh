#!/usr/bin/env bash
set -euo pipefail
cd /home/jpearce/Documents/Lucid
PY=.venv/bin/python
echo "=== STEP1 IMPORT ==="
$PY -c "
import sys
sys.path.insert(0,'src')
sys.path.insert(0,'.')
from experiments.sentence_pred import evalsuite, baselines, rollout
from experiments.sentence_pred import run_proof
print('IMPORT_OK', evalsuite.SEED_DEFAULT)
"
echo "EXIT1:$?"
echo "=== STEP2 PYTEST ==="
$PY -m pytest tests/test_evalsuite.py -q --tb=short
echo "EXIT2:$?"
echo "=== STEP3 RUN_PROOF ==="
$PY -m experiments.sentence_pred.run_proof --config configs/sentence_pred.json --stage data --stage train-vmf --stage eval --force
echo "EXIT3:$?"
echo "=== STEP4 ARTIFACTS ==="
ls -la artifacts/sentence_pred/frontier.jsonl /tmp/sent_corpus_smoke/ 2>&1
wc -l artifacts/sentence_pred/frontier.jsonl
head -c 800 artifacts/sentence_pred/frontier.jsonl
echo
echo "=== LINE COUNTS ==="
wc -l experiments/sentence_pred/evalsuite.py experiments/sentence_pred/baselines.py experiments/sentence_pred/rollout.py experiments/sentence_pred/run_proof.py tests/test_evalsuite.py experiments/sentence_pred/__init__.py
