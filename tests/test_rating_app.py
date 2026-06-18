"""Integration test for the human rating app.

The app lives in ``rating-app/`` as top-level modules (not a package), so we add
that directory to the path and point the DB at a temp file before importing.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "rating-app"


@pytest.fixture
def client(tmp_path):
    sys.path.insert(0, str(APP_DIR))
    import db

    db.DB_PATH = tmp_path / "ratings.sqlite3"
    app_module = importlib.import_module("app")
    importlib.reload(app_module)
    from starlette.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c, db
    sys.path.remove(str(APP_DIR))


def test_full_rating_flow(client):
    c, db = client
    assert c.get("/health").json()["items"] == 0

    iid = c.post(
        "/api/items",
        json={"prompt": "p", "text": "The river moved slowly. Birds rose up."},
    ).json()["id"]

    home = c.get("/?rater=alex")
    assert home.status_code == 200 and iid in home.text

    redirect = c.post(
        "/rate",
        data={"rater": "alex", "item_id": iid, "score": 4, "seconds_spent": 6.0},
        follow_redirects=False,
    )
    assert redirect.status_code == 303

    export = c.get("/api/export").json()
    assert len(export["ratings"]) == 1
    assert abs(export["ratings"][0]["score"] - 0.8) < 1e-6  # 4/5

    assert c.get("/api/quality").status_code == 200
    assert "Start rating" in c.get("/").text
