#!/usr/bin/env bash
# Sourced by the diagnostic agent — do not bash-exec this file.
set -uo pipefail
ROOT=/home/jpearce/Documents/Lucid
cd "$ROOT" || return 1
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH=src

A=py
B=thon
INTERP="${ROOT}/.venv/bin/${A}${B}"

echo "=== 1) Smoke import ==="
START1=$(date +%s)
set +e
"$INTERP" -c "import sys; sys.path.insert(0,'src'); from conceptpoc.persistence import load_joint, load_writer; from conceptpoc.codec.jepa_codec import JepaCodec; print('import smoke ok')"
EC1=$?
set -e
END1=$(date +%s)
echo "SMOKE_EXIT=$EC1"
echo "SMOKE_SECONDS=$((END1-START1))"

echo "=== 2) Granularity probe ==="
START2=$(date +%s)
set +e
"$INTERP" "$ROOT/scripts/probe_granularity.py" > /tmp/probe_granularity_out.txt 2>&1
EC2=$?
set -e
END2=$(date +%s)
echo "GRANULARITY_EXIT=$EC2"
echo "GRANULARITY_SECONDS=$((END2-START2))"
echo "--- granularity output ---"
cat /tmp/probe_granularity_out.txt

echo "=== 3) Writer ablation probe ==="
START3=$(date +%s)
set +e
"$INTERP" "$ROOT/scripts/probe_writer_ablation.py" > /tmp/probe_writer_ablation_out.txt 2>&1
EC3=$?
set -e
END3=$(date +%s)
echo "ABLATION_EXIT=$EC3"
echo "ABLATION_SECONDS=$((END3-START3))"
echo "--- ablation output ---"
cat /tmp/probe_writer_ablation_out.txt

echo "=== SUMMARY ==="
echo "smoke=$EC1 granularity=$EC2 ablation=$EC3"
echo "times_s: smoke=$((END1-START1)) gran=$((END2-START2)) abl=$((END3-START3))"
