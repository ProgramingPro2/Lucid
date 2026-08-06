#!/usr/bin/env bash
set -o pipefail
cd /home/jpearce/Documents/Lucid
.venv/bin/python -m experiments.sentence_pred.run_proof --config configs/sentence_pred.json --stage data --stage train-vmf --stage eval --force
echo EXIT3:$?
