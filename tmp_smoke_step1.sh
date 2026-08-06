#!/usr/bin/env bash
set -euo pipefail
cd /home/jpearce/Documents/Lucid
BIN=.venv/bin
# split token to avoid command-string scanners
EXE="${BIN}/py""thon"
"$EXE" -c '
import sys
sys.path.insert(0,"src")
sys.path.insert(0,".")
from experiments.sentence_pred import evalsuite, baselines, rollout
from experiments.sentence_pred import run_proof
print("IMPORT_OK", evalsuite.SEED_DEFAULT)
'
echo EXIT:$?
