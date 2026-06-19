"""FastAPI human rating app.

Raters open the page, enter a name, and score generated continuations. An LLM
(or the offline heuristic quality-control checks) can later flag low-effort or
adversarial raters via /api/quality. Generated items are pushed in by the
pipeline through POST /api/items.

Run: uvicorn app:app --app-dir rating-app --reload
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import db

# Make the conceptpoc package importable for the quality-control pass.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

APP_DIR = Path(__file__).parent
VIZ_DIR = APP_DIR / "static" / "viz"
EMBEDDINGS = APP_DIR / "viz" / "public" / "embeddings.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Concept POC Rating App", lifespan=lifespan)
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request, rater: str = "") -> HTMLResponse:
    if not rater:
        return templates.TemplateResponse(request, "login.html")
    item = db.next_unrated_item(rater)
    return templates.TemplateResponse(
        request,
        "rate.html",
        {"rater": rater, "item": item, "counts": db.counts()},
    )


@app.post("/rate")
def rate(
    rater: str = Form(...),
    item_id: str = Form(...),
    score: float = Form(...),
    seconds_spent: float = Form(0.0),
) -> RedirectResponse:
    db.add_rating(item_id, rater, score / 5.0, seconds_spent)
    return RedirectResponse(url=f"/?rater={rater}", status_code=303)


class ItemIn(BaseModel):
    prompt: str
    text: str
    vectors: list[list[float]] | None = None


@app.post("/api/items")
def api_add_item(item: ItemIn) -> JSONResponse:
    import numpy as np

    vecs = np.asarray(item.vectors, dtype=np.float32) if item.vectors else None
    item_id = db.add_item(item.prompt, item.text, vecs)
    return JSONResponse({"id": item_id})


@app.get("/api/export")
def api_export() -> JSONResponse:
    items = {r["id"]: dict(r) for r in db.all_items()}
    ratings = [dict(r) for r in db.all_ratings()]
    for it in items.values():
        it.pop("vectors_b64", None)
    return JSONResponse({"items": list(items.values()), "ratings": ratings})


@app.get("/api/quality")
def api_quality() -> JSONResponse:
    from conceptpoc.reward.quality_control import HumanRating, QualityControl

    rows = db.all_ratings()
    ratings = [
        HumanRating(
            rater_id=r["rater_id"],
            item_id=r["item_id"],
            score=r["score"],
            seconds_spent=r["seconds_spent"],
        )
        for r in rows
    ]
    verdicts = QualityControl().assess(ratings)
    return JSONResponse(
        {rid: vars(v) for rid, v in verdicts.items()}
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **db.counts()}


@app.get("/api/embeddings")
def api_embeddings() -> JSONResponse:
    if not EMBEDDINGS.is_file():
        return JSONResponse({"error": "embeddings.json not found"}, status_code=404)
    return JSONResponse(json.loads(EMBEDDINGS.read_text(encoding="utf-8")))


if VIZ_DIR.is_dir():
    app.mount("/viz", StaticFiles(directory=str(VIZ_DIR), html=True), name="viz")
