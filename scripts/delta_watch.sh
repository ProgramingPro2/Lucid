#!/usr/bin/env bash
# Script for the cron job: wait for the delta checkpoint to exist, then report.
cd /home/jpearce/Documents/Lucid || exit 1

# Wait up to ~50 min for the checkpoint (training ~45-75 min).
for i in $(seq 1 150); do
  if [ -f artifacts/delta_pred/delta_predictor.pt ]; then
    break
  fi
  # stop early if the trainer process is gone and no checkpoint -> crashed
  if ! pgrep -f "train-delta" >/dev/null 2>&1; then
    break
  fi
  sleep 20
done

if [ -f artifacts/delta_pred/delta_predictor.pt ]; then
  source .venv/bin/activate
  python scripts/delta_report.py 2>&1 | grep -vE "Warning:|unauthenticated"
else
  echo "Delta training did not produce a checkpoint yet (still running or crashed)."
  echo "Latest training log:"
  pgrep -af "train-delta" || true
fi
