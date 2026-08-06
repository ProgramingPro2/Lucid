"""Post-training report for the delta predictor.

Runs once training completes:
  1. Prints the gate (val_cosine from the DB — the authoritative held-out number).
  2. Generates closed-loop delta continuations on standard prompts via the CLI.
  3. Prints a comparison summary vs exact-next baselines.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "artifacts/delta_pred/delta_predictor.pt")


def read_val_cosine():
    """Pull best/final val_cosine from the SQLite checkpoint DB."""
    try:
        sys.path.insert(0, ROOT)  # so `src.conceptpoc` is importable
        from src.conceptpoc.ckpt_db import CheckpointDB
        db = CheckpointDB(os.path.join(ROOT, "artifacts/delta_pred/train.db"))
        run = db.latest_run()
        if run is None:
            return None
        rows = db.epoch_metrics(run)
        if not rows:
            return None
        best = max((r.get("val_cosine", 0.0) for r in rows), default=None)
        last = rows[-1].get("val_cosine")
        db.close()
        return best, last, len(rows)
    except Exception as e:  # noqa: BLE001
        return f"db-error: {e}"


def main():
    if not os.path.exists(CHECKPOINT):
        print("GATE: delta predictor checkpoint not ready yet.")
        return 0

    vc = read_val_cosine()
    print("=" * 60)
    print("DELTA PREDICTOR POST-TRAINING REPORT")
    print("=" * 60)
    if isinstance(vc, tuple) and vc[0] is not None:
        best, last, nep = vc
        print(f"\n[GATE] trained {nep} epochs; best val_cosine = {best:.4f} (final {last:.4f})")
        print(f"       baselines: exact-next wall ~0.05 | persistence/reversal +/-0.49")
        print(f"       random ~0 | ridge-upper ~0.44")
        verdict = "PASS" if best and best > 0.3 else ("CHECK" if best and best > 0.1 else "FAIL")
        print(f"       => {verdict}: delta predictor {'clearly' if verdict=='PASS' else ''} above the exact-next wall")
    else:
        print(f"\n[GATE] {vc}")

    print("\n[GENERATION] closed-loop delta continuation (predict change -> writer):")
    for prompt in ["The scientist discovered that",
                   "The morning light moved across the valley as"]:
        print(f"\n  PROMPT: {prompt}")
        r = subprocess.run(
            [sys.executable, "-m", "conceptpoc.cli", "--config", "configs/delta_pred.json",
             "generate-delta", "--prompt", prompt, "--steps", "5",
             "--delta", CHECKPOINT,
             "--writer", os.path.join(ROOT, "artifacts/writer_p2_film/writer_cond.pt")],
            capture_output=True, text=True,
            cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"},
        )
        out = (r.stdout or "").strip()
        # strip progress bars
        lines = [l for l in out.splitlines() if not l.strip().startswith(("\r", "epoch", "val", "encoding"))]
        print("\n".join(lines[-12:]))
        if r.returncode != 0 and not out:
            print("  [err]", (r.stderr or "")[-400:])

    print("\n[NOTE] For a full eval-harness comparison run manually; the key signal is the")
    print("       gate (val_cosine >> 0.05) + whether continuations visually stay on-topic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
