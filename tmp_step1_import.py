import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from experiments.sentence_pred import evalsuite, baselines, rollout
from experiments.sentence_pred import run_proof
print("IMPORT_OK", evalsuite.SEED_DEFAULT)
