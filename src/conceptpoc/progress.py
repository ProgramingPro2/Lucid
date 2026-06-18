"""Shared terminal progress helpers for the conceptpoc pipeline."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def iter_bar(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
    unit: str = "it",
) -> Iterator[T]:
    """Wrap *iterable* with a tqdm bar; no-op when *enabled* is False."""
    yield from tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=not enabled,
        unit=unit,
    )


def log_stage(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def log_epoch(
    stage: str,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float | str],
    *,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    parts = [f"[{stage}] epoch {epoch}/{total_epochs}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    print("  ".join(parts), flush=True)
