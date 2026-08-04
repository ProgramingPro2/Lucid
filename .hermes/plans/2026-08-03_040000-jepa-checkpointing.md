# JEPA Training Checkpointing + Recovery Plan

> **For Hermes:** Test each change as you go. Do NOT delete existing artifacts or restart training until checkpointing is verified.

**Goal:** Add epoch-level disk checkpointing to `train_joint.py` so that a 14-epoch training run survives crashes. Instead of losing all progress, it saves `checkpoint_epoch_N.pt` after each epoch and can resume from the latest.

**Architecture:** Modify `train_joint()` to accept an optional `checkpoint_dir` path. After each epoch, save model state + optimizer state + training history to disk. On restart, load the latest checkpoint and continue from where it left off.

**Tech Stack:** PyTorch, existing Lucid codebase at `~/Documents/Lucid/`

---

## Current state

- `train_joint.py` already tracks best_state in memory (lines 121, 173-181, 213-221) but never writes to disk mid-training
- `cli.py:cmd_train_joint` calls `save_joint()` only at the very end (line 193)
- The OOM crash at epoch 14/15 was from process PID 2413743 grabbing 7.08 GiB — need to identify it
- Current artifacts at `artifacts/jepa_1m_128t/` are empty — nothing was saved

---

## Task 1: Add `_alpha_for_epoch` to persistence (save/load JointHistory + optimizer)

**Objective:** Extend `persistence.py` with functions to save/load JEPA training checkpoints (model + optimizer + history + epoch number).

**Files:**
- Modify: `src/conceptpoc/persistence.py`

**Step 1: Add `save_joint_checkpoint` function**

Append after line 162 in `persistence.py`:

```python
def save_joint_checkpoint(
    path: str | Path,
    model,
    optimizer,
    history: dict,
    epoch: int,
    tokenizer_name: str,
    seed: int,
) -> None:
    """Save a resumable training checkpoint (model + optimizer + history)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_size": model.decoder.vocab_size,
            "pad_id": model.pad_id,
            "bos_id": model.bos_id,
            "jepa_cfg": model.cfg.__dict__,
            "predictor_cfg": asdict(model.predictor.cfg),
            "tokenizer_name": tokenizer_name,
            "epoch": epoch,
            "history": history,
            "seed": seed,
        },
        path,
    )


def load_joint_checkpoint(
    path: str | Path, device: str = "cpu"
) -> dict:
    """Load a resumable training checkpoint. Returns a dict with all components."""
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    return blob
```

**Step 2: Verify syntax**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.persistence import save_joint_checkpoint, load_joint_checkpoint; print('OK')"`  
Expected: `OK`

**Step 3: Commit**

```bash
cd ~/Documents/Lucid && git add src/conceptpoc/persistence.py && git commit -m "feat: add save/load joint checkpoint with optimizer state"
```

---

## Task 2: Modify `train_joint()` to checkpoint after each epoch

**Objective:** After each epoch, save a checkpoint to disk. Accept `checkpoint_dir` parameter. Load latest checkpoint on restart.

**Files:**
- Modify: `src/conceptpoc/jepa/train_joint.py`

**Step 1: Add `checkpoint_dir` parameter to `train_joint()`**

Change the function signature (line 77-84):

```python
def train_joint(
    docs: list[Document],
    cfg: JepaConfig,
    predictor_cfg: PredictorConfig,
    device: str = "auto",
    tokenizer: Tokenizer | None = None,
    progress: bool = True,
    gate: GpuGate | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[JointJepa, Tokenizer, JointHistory]:
```

**Step 2: Add resume logic after model/optimizer creation (after line 118)**

Insert after `opt = torch.optim.AdamW(...)`:

```python
    start_epoch = 0
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            from ..persistence import load_joint_checkpoint
            blob = load_joint_checkpoint(latest, dev)
            model.load_state_dict(blob["state_dict"])
            opt.load_state_dict(blob["optimizer_state"])
            start_epoch = blob["epoch"] + 1
            # Restore history so we don't lose previous metrics
            saved_history = blob.get("history", {})
            for key in ["total", "recon", "pred", "recon_acc", "std_z", "aniso", "alpha", "val_pred_cos"]:
                if key in saved_history:
                    getattr(history, key).extend(saved_history[key])
            history.best_epoch = saved_history.get("best_epoch", -1)
            history.best_val_cos = saved_history.get("best_val_cos", float("-inf"))
            if progress:
                log_stage(
                    f"resumed from {latest.name} (epoch {start_epoch}/{cfg.epochs})",
                    enabled=True,
                )
```

**Step 3: Save checkpoint after each epoch (after the log_epoch call, around line 197)**

Insert after the collapse warning check:

```python
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
            from ..persistence import save_joint_checkpoint
            save_joint_checkpoint(
                ckpt_path,
                model,
                opt,
                {
                    "total": history.total,
                    "recon": history.recon,
                    "pred": history.pred,
                    "recon_acc": history.recon_acc,
                    "std_z": history.std_z,
                    "aniso": history.aniso,
                    "alpha": history.alpha,
                    "val_pred_cos": history.val_pred_cos,
                    "best_epoch": history.best_epoch,
                    "best_val_cos": history.best_val_cos,
                },
                epoch,
                tokenizer.name,
                cfg.seed,
            )
```

**Step 4: Change the for-loop to start from `start_epoch`**

```python
    for epoch in range(start_epoch, cfg.epochs):
```

**Step 5: Verify syntax and logic**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.jepa.train_joint import train_joint; print('OK')"`  
Expected: `OK`

**Step 6: Commit**

```bash
cd ~/Documents/Lucid && git add src/conceptpoc/jepa/train_joint.py && git commit -m "feat: add epoch-level checkpointing + resume to train_joint"
```

---

## Task 3: Update CLI to pass checkpoint_dir

**Objective:** Make `cmd_train_joint` pass the artifacts dir as the checkpoint directory.

**Files:**
- Modify: `src/conceptpoc/cli.py:182-219` (the `cmd_train_joint` function)

**Step 1: Pass checkpoint_dir to train_joint**

In `cmd_train_joint`, change the `train_joint()` call:

```python
    model, _, history = train_joint(
        docs, cfg.jepa, cfg.predictor, progress=progress,
        gate=_build_gate(cfg, progress=progress),
        checkpoint_dir=cfg.artifacts_dir / "checkpoints",
    )
```

**Step 2: Verify**

Run: `cd ~/Documents/Lucid && source .venv/bin/activate && python -c "from src.conceptpoc.cli import cmd_train_joint; print('OK')"`  
Expected: `OK`

**Step 3: Commit**

```bash
cd ~/Documents/Lucid && git add src/conceptpoc/cli.py && git commit -m "feat: pass checkpoint_dir from CLI to train_joint"
```

---

## Task 4: Test checkpointing with a quick 2-epoch dry run

**Objective:** Verify that checkpoints are saved and can be resumed.

**Step 1: Create a tiny test config**

Copy `configs/jepa_1m_128t.json` to create `configs/jepa_test_ckpt.json` but with:
- `corpus_path`: `data/austen_corpus.txt` (tiny, fast)
- `epochs`: 3
- `batch_size`: 4
- `artifacts_dir`: `artifacts/jepa_test_ckpt`

**Step 2: Run 1 epoch, interrupt, resume**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
python -m conceptpoc.cli --config configs/jepa_test_ckpt.json train-joint
# Let it finish epoch 1, then Ctrl+C
# Check that checkpoint_epoch_0000.pt exists
ls artifacts/jepa_test_ckpt/checkpoints/
# Run again — should say "resumed from checkpoint_epoch_0000.pt (epoch 1/3)"
python -m conceptpoc.cli --config configs/jepa_test_ckpt.json train-joint
```

Expected: Second run prints "resumed from checkpoint_epoch_0000.pt" and continues from epoch 1.

---

## Task 5: Identify the crashing process (PID 2413743)

**Objective:** Find what program grabbed 7GB and killed training at epoch 14.

**Step 1: Check if PID 2413743 is still alive**

```bash
ps -p 2413743 -o pid,comm,args 2>&1 || echo "Process no longer running"
```

**Step 2: Check system logs for OOM killer events**

```bash
dmesg | grep -i "out of memory\|oom\|killed" | tail -20
journalctl -k --since "2 hours ago" | grep -i oom | tail -10
```

**Step 3: Check nvidia-smi for current GPU users**

```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```

Based on the memory footprint (7.08 GiB), likely candidates:
- Lemonade auto-loading a model (4-7GB for Q4 quantized models)
- A Python process loading a model
- StockBot3.0 (we saw it using 342 MiB earlier)

---

## Task 6: Restart JEPA training with checkpointing enabled

**Objective:** Clear old partial artifacts and restart full 15-epoch training on `corpus_1m.txt` with checkpointing.

**Prerequisites:** Tasks 1-4 complete and verified.

**Step 1: Unload any GPU hogs**

```bash
for model in "Huihui-gemma-4-26B-A4B-it-abliterated-GGUF-UD-Q4_K_M" "Devstral-Small-2-24B-Instruct-2512-GGUF-Q4_K_M" "Gemma-4-12B-it-GGUF" "Gemma-4-26B-A4B-it-GGUF" "Qwen3.6-27B-MTP-GGUF-Q4_K_M" "Gemma-4-E4B-it-GGUF"; do
  curl -s http://localhost:11434/api/generate -d "{\"model\": \"$model\", \"keep_alive\": 0}" > /dev/null 2>&1
done
```

**Step 2: Clear old artifacts (user must approve)**

```bash
rm -rf ~/Documents/Lucid/artifacts/jepa_1m_128t
```

**Step 3: Restart training**

```bash
cd ~/Documents/Lucid && source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m conceptpoc.cli --config configs/jepa_1m_128t.json train-joint
```

Note: The config already has `gpu_gate.enabled: true` which will pause training if another process uses the GPU (PID 2413743). Combined with checkpointing, a crash just means losing <1 epoch instead of 14.

**Step 4: Verify checkpoints are being created**

```bash
watch -n 30 "ls -lh ~/Documents/Lucid/artifacts/jepa_1m_128t/checkpoints/ 2>/dev/null | tail -5"
```

---

## Task 7: After JEPA completes — full pipeline

**Objective:** Train diffusion predictor → decoder-robust → evaluate.

**Step 1: Update `jepa_large_v2.json` to point at new checkpoint**

Change `jepa_checkpoint` to `artifacts/jepa_1m_128t/jepa/joint.pt`

**Step 2: Train diffusion predictor**

```bash
python -m conceptpoc.cli --config configs/jepa_large_v2.json train-diffusion
```

**Step 3: Train decoder-robust**

```bash
python -m conceptpoc.cli --config configs/jepa_large_v2.json train-decoder-robust
```

**Step 4: Generate and evaluate**

```bash
python -m conceptpoc.cli --config configs/jepa_large_v2.json generate --prompt "The scientist discovered that" --steps 6
```

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Checkpoint files are huge (model+optimizer ~2GB each) | Delete old checkpoints, keep only last 3 |
| Resume restores wrong epoch | The `start_epoch` is saved in the checkpoint; verify with log message |
| GPU gate doesn't catch the 7GB process | The crashing process might load during a forward/backward pass. Checkpoint means we only lose one epoch |
| OOM from other process mid-epoch | Checkpointing saves at epoch boundaries. Worst case: lose current epoch, resume from previous |

## Files changed

| File | Change |
|------|--------|
| `src/conceptpoc/persistence.py` | Add `save_joint_checkpoint` and `load_joint_checkpoint` |
| `src/conceptpoc/jepa/train_joint.py` | Add checkpoint_dir param, resume logic, per-epoch saves |
| `src/conceptpoc/cli.py` | Pass checkpoint_dir to train_joint |
