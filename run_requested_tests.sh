#!/usr/bin/env bash
set -euo pipefail
cd /home/jpearce/Documents/Lucid
source .venv/bin/activate
python -c "from src.conceptpoc.jepa.joint_model import JointJepa; from src.conceptpoc.config import JepaConfig, PredictorConfig; m=JointJepa(100,0,1,JepaConfig(dim=16,d_model=32),PredictorConfig()); print('ok')"
python -m pytest tests/test_losses_masked.py tests/test_vmf.py -q --tb=short 2>&1
