"""SQLite-backed checkpoint store for long training runs.

Stores training state (model weights, optimizer state, per-epoch metrics) so a
crash mid-run costs at most one epoch instead of the whole run. Benefits over
flat files:

- Per-epoch metrics are queryable (no need to load a JSON blob to see history).
- Model + optimizer snapshots live in one transactional DB, not a pile of .pt
  files that can be half-written by a crash.
- ``keep_last`` bounds total size by pruning old weight snapshots while keeping
  all metrics forever.

Design:
  runs        : one row per training run (config, corpus, tokenizer, seed).
  epochs      : one row per epoch (metrics JSON), append-only, queryable.
  checkpoints : one row per saved weight snapshot (model + optimizer BLOBs). The
                ``latest_epoch`` row links the most recent snapshot.

Weights are serialized with torch.save into bytes (BLOB). Storing them in the
DB keeps crash-consistency via SQLite transactions (WAL mode).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import torch


class CheckpointDB:
    def __init__(self, path: str | Path, keep_last: int = 3) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                corpus TEXT,
                tokenizer TEXT,
                seed INTEGER,
                max_tokens INTEGER,
                context_len INTEGER,
                config_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS epochs (
                run_id INTEGER,
                epoch INTEGER,
                metrics_json TEXT,
                saved_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (run_id, epoch)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id INTEGER,
                epoch INTEGER,
                file_path TEXT,
                saved_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (run_id, epoch)
            );
            """
        )
        self._conn.commit()

    # ---- runs ----
    def start_run(
        self,
        corpus: str,
        tokenizer: str,
        seed: int,
        max_tokens: int,
        context_len: int,
        config_json: dict,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (corpus, tokenizer, seed, max_tokens, context_len, config_json)"
            " VALUES (?,?,?,?,?,?)",
            (
                corpus,
                tokenizer,
                seed,
                max_tokens,
                context_len,
                json.dumps(config_json),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_run_config(self, run_id: int, config_json: dict) -> None:
        self._conn.execute(
            "UPDATE runs SET config_json=? WHERE id=?",
            (json.dumps(config_json), run_id),
        )
        self._conn.commit()

    def latest_run(self) -> int | None:
        row = self._conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def get_run(self, run_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ---- epochs (metrics) ----
    def record_epoch(self, run_id: int, epoch: int, metrics: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO epochs (run_id, epoch, metrics_json) VALUES (?,?,?)",
            (run_id, epoch, json.dumps(metrics)),
        )
        self._conn.commit()

    def epoch_metrics(self, run_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT epoch, metrics_json FROM epochs WHERE run_id=? ORDER BY epoch",
            (run_id,),
        ).fetchall()
        return [
            {"epoch": r["epoch"], **json.loads(r["metrics_json"])} for r in rows
        ]

    def highest_epoch(self, run_id: int) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(epoch) AS m FROM epochs WHERE run_id=? AND metrics_json IS NOT NULL",
            (run_id,),
        ).fetchone()
        return row["m"] if row and row["m"] is not None else None

    # ---- checkpoints (weight snapshots) ----
    def save_checkpoint(
        self,
        run_id: int,
        epoch: int,
        model,
        optimizer,
        *,
        device: str = "cpu",
    ) -> None:
        """Atomically write model+optimizer to a .pt file and track it in the DB.

        Weight snapshots live in files (they can exceed SQLite's ~1GB blob
        limit once AdamW momentum accumulates), while the DB stores metrics and
        the path to the latest snapshot. Writes are atomic (tmp + rename) so a
        crash never leaves a half-written snapshot.
        """
        ckpt_dir = self.path.parent / f"runs" / f"run_{run_id}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        tmp_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt.tmp"

        # Move weights to CPU before serializing to keep VRAM headroom.
        state = {k: v.detach().to(device) for k, v in model.state_dict().items()}
        opt_state = {
            k: (
                v.to(device)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in optimizer.state_dict().items()
        }
        torch.save({"model": state, "optimizer": opt_state}, tmp_path)
        tmp_path.replace(final_path)

        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (run_id, epoch, file_path)"
            " VALUES (?,?,?)",
            (run_id, epoch, str(final_path)),
        )
        self._conn.commit()
        self._prune(run_id)

    def latest_checkpoint_epoch(self, run_id: int) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(epoch) AS m FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()
        return row["m"] if row and row["m"] is not None else None

    def load_latest_checkpoint(
        self, run_id: int, device: str = "cpu"
    ) -> tuple[dict, dict, int] | None:
        """Return (model_state, optimizer_state, epoch)."""
        epoch = self.latest_checkpoint_epoch(run_id)
        if epoch is None:
            return None
        row = self._conn.execute(
            "SELECT file_path FROM checkpoints WHERE run_id=? AND epoch=?",
            (run_id, epoch),
        ).fetchone()
        path = Path(row["file_path"])
        if not path.exists():
            return None
        blob = torch.load(path, map_location=device, weights_only=False)
        return blob["model"], blob.get("optimizer", {}), epoch

    def _prune(self, run_id: int) -> None:
        # Keep the newest keep_last checkpoints, drop older weight snapshots
        # (both the DB row and the on-disk file).
        rows = self._conn.execute(
            "SELECT epoch, file_path FROM checkpoints WHERE run_id=? ORDER BY epoch DESC",
            (run_id,),
        ).fetchall()
        if len(rows) <= self.keep_last:
            return
        for r in rows[self.keep_last :]:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE run_id=? AND epoch=?", (run_id, r["epoch"])
            )
            try:
                Path(r["file_path"]).unlink(missing_ok=True)
            except OSError:
                pass
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
