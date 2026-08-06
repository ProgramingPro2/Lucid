import sys
sys.path.insert(0, "src")
from pathlib import Path
from src.conceptpoc.ckpt_db import CheckpointDB

with CheckpointDB(Path("artifacts/jepa_route_v1/train.db")) as db:
    runs = db.latest_run()
    print("latest run:", runs)
    for m in db.epoch_metrics(runs):
        ep = m["epoch"]
        print(f"  epoch {ep+1:2d}: recon_acc={m.get('recon_acc',float('nan')):.3f} "
              f"std_z={m.get('std_z',float('nan')):.3f} "
              f"aniso={m.get('aniso',float('nan')):.3f} "
              f"pred={m.get('pred',float('nan')):.3f} "
              f"val_pred_cos={m.get('val_pred_cos',float('nan')):.3f}")
