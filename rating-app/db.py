"""SQLite storage for the human rating app.

Two tables: ``items`` (a prompt + one generated continuation, with the optional
concept vectors that produced it) and ``ratings`` (one row per human judgement).
Vectors are stored as base64-encoded .npy so the export can rebuild the concept
trajectories needed for preference learning.
"""
from __future__ import annotations

import base64
import io
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "ratings.sqlite3"


def _resolve(db_path: Path | str | None) -> Path | str:
    return db_path if db_path is not None else DB_PATH


def _conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_resolve(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    with _conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                text TEXT NOT NULL,
                vectors_b64 TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                rater_id TEXT NOT NULL,
                score REAL NOT NULL,
                seconds_spent REAL,
                created_at REAL NOT NULL,
                FOREIGN KEY (item_id) REFERENCES items (id)
            );
            CREATE INDEX IF NOT EXISTS idx_ratings_item ON ratings (item_id);
            """
        )


def encode_vectors(vectors: np.ndarray | None) -> str | None:
    if vectors is None:
        return None
    buf = io.BytesIO()
    np.save(buf, np.asarray(vectors, dtype=np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_vectors(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    return np.load(io.BytesIO(base64.b64decode(b64)))


def add_item(
    prompt: str, text: str, vectors: np.ndarray | None = None, db_path=None
) -> str:
    item_id = uuid.uuid4().hex
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO items (id, prompt, text, vectors_b64, created_at) VALUES (?,?,?,?,?)",
            (item_id, prompt, text, encode_vectors(vectors), time.time()),
        )
    return item_id


def next_unrated_item(rater_id: str, db_path=None) -> sqlite3.Row | None:
    with _conn(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM items
            WHERE id NOT IN (
                SELECT item_id FROM ratings WHERE rater_id = ?
            )
            ORDER BY (SELECT COUNT(*) FROM ratings r WHERE r.item_id = items.id) ASC,
                     created_at ASC
            LIMIT 1
            """,
            (rater_id,),
        ).fetchone()


def add_rating(
    item_id: str, rater_id: str, score: float, seconds_spent: float | None, db_path=None
) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO ratings (item_id, rater_id, score, seconds_spent, created_at) VALUES (?,?,?,?,?)",
            (item_id, rater_id, float(score), seconds_spent, time.time()),
        )


def all_ratings(db_path=None) -> list[sqlite3.Row]:
    with _conn(db_path) as conn:
        return conn.execute("SELECT * FROM ratings").fetchall()


def all_items(db_path=None) -> list[sqlite3.Row]:
    with _conn(db_path) as conn:
        return conn.execute("SELECT * FROM items").fetchall()


def counts(db_path=None) -> dict[str, int]:
    with _conn(db_path) as conn:
        items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        ratings = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    return {"items": items, "ratings": ratings}
