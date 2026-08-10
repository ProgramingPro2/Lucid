#!/usr/bin/env bash
# Stage-B orchestrator: baseline → B0 rollouts → winners → distill → gate → optional GRPO.
# Designed to run DETACHED; resumes via JSONL / SQLite.
set -euo pipefail
ROOT="/home/jpearce/Documents/Lucid"
cd "$ROOT"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Keep lemonade unloaded
bash scripts/lemon_keeper.sh 200000 20 >> /tmp/lemon_keeper_stageB.log 2>&1 &

CFG="${CFG:-configs/pivot_stage_b.json}"
LOG="${LOG:-artifacts/pivot_stageB/orchestrator.log}"
mkdir -p artifacts/pivot_stageB/b0 artifacts/pivot_stageB/grpo
exec >>"$LOG" 2>&1

echo "===== Stage-B orchestrator start $(date -Is) ====="

echo "----- 0) Phase-A baseline freeze -----"
python -u experiments/pivot/gate_stage_b.py --config "$CFG" --freeze-baseline --probe-n 128

echo "----- 1) B0 rollouts -----"
python -u experiments/pivot/rollout_stage_b.py --config "$CFG"

echo "----- 2) B0 select winners -----"
python -u experiments/pivot/raft_stage_b.py --config "$CFG" --select-only

echo "----- 3) B0 distill -----"
python -u experiments/pivot/raft_stage_b.py --config "$CFG" --distill-only

echo "----- 4) B0 gate eval -----"
python -u experiments/pivot/gate_stage_b.py --config "$CFG" \
  --ckpt artifacts/pivot_stageB/b0/ckpt.pt --label b0 --n-seeds 5

# Branch: if B0 passes, stop; else GRPO
python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/pivot_stageB/gate_b0.json")
if not p.exists():
    print("NO_GATE_FILE")
    raise SystemExit(2)
g = json.loads(p.read_text())
print("B0 overall_pass=", g.get("overall_pass"), "hard_stop=", g.get("hard_stop_reward_hacking"))
Path("artifacts/pivot_stageB/b0_branch.txt").write_text(
    "PASS" if g.get("overall_pass") else "GRPO"
)
if g.get("overall_pass"):
    raise SystemExit(0)
raise SystemExit(10)
PY
BRANCH=$?
if [ "$BRANCH" -eq 0 ]; then
  echo "B0 PASSED gates — GRPO optional, stopping."
  echo "===== Stage-B done (B0 pass) $(date -Is) ====="
  exit 0
fi

echo "----- 5) GRPO from B0 -----"
python -u experiments/pivot/train_stage_b_grpo.py --config "$CFG" \
  --init-ckpt artifacts/pivot_stageB/b0/ckpt.pt \
  --ref-ckpt artifacts/pivot_stageA/checkpoint_epoch01.pt

echo "----- 6) Final gate -----"
python -u experiments/pivot/gate_stage_b.py --config "$CFG" \
  --ckpt artifacts/pivot_stageB/grpo/checkpoint_latest.pt --label final --n-seeds 20

echo "===== Stage-B done $(date -Is) ====="
