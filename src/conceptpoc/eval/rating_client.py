"""Optional rating-app integration for the eval harness.

Uses stdlib ``urllib`` (no extra dependency) to push generated items into the
human rating app (``POST /api/items``) and to merge human scores back
(``GET /api/export``). Items are joined to exported items by ``(prompt, text)``
so a merge works even in a later, separate invocation.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .harness import ItemResult


def _post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def push_to_rating_app(
    results: list[ItemResult], base_url: str, timeout: float = 10.0
) -> int:
    """POST each result to ``/api/items``; record the returned id. Returns count."""
    url = base_url.rstrip("/") + "/api/items"
    pushed = 0
    for r in results:
        payload = {"prompt": r.prompt, "text": r.text, "vectors": None}
        try:
            resp = _post_json(url, payload, timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"failed to push to rating app at {url}: {exc}") from exc
        r.rating_app_id = str(resp.get("id")) if resp.get("id") is not None else None
        pushed += 1
    return pushed


def merge_human_scores(
    results: list[ItemResult], base_url: str, timeout: float = 10.0
) -> int:
    """Pull ``/api/export`` and attach mean human score per item by (prompt, text).

    Returns the number of results that received a human score.
    """
    url = base_url.rstrip("/") + "/api/export"
    try:
        export = _get_json(url, timeout=timeout)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"failed to read rating app at {url}: {exc}") from exc

    items = export.get("items", [])
    ratings = export.get("ratings", [])

    scores_by_item: dict[str, list[float]] = {}
    for r in ratings:
        item_id = str(r.get("item_id"))
        try:
            scores_by_item.setdefault(item_id, []).append(float(r["score"]))
        except (KeyError, TypeError, ValueError):
            continue

    mean_by_key: dict[tuple[str, str], float] = {}
    for it in items:
        item_id = str(it.get("id"))
        vals = scores_by_item.get(item_id)
        if not vals:
            continue
        key = (str(it.get("prompt", "")), str(it.get("text", "")))
        mean_by_key[key] = sum(vals) / len(vals)

    merged = 0
    for r in results:
        key = (r.prompt, r.text)
        if key in mean_by_key:
            r.human_score = mean_by_key[key]
            merged += 1
    return merged
