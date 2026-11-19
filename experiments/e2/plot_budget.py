"""Render the E2 CoT budget curve as a dependency-free SVG.

matplotlib is not a project dependency, and the figure is two small panels; a
few hundred lines of SVG is cheaper than adding a plotting stack. The script
reads the two ``budget_*.json`` artifacts and writes one SVG.

Usage:
    PYTHONPATH=src .venv/bin/python -m experiments.e2.plot_budget \
        --inputs artifacts/e2/budget_1.7b.json artifacts/e2/budget_8b.json \
        --out docs/figures/e2_budget_curve.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

W, H = 980, 540
ML, MR, MT, MB = 64, 24, 56, 104
GAP = 52
PANEL_W = (W - ML - MR - GAP) / 2
PANEL_H = 380  # fixed, so the shared legend always has room below the axis label
COLOURS = {"f1": "#1f77b4", "f2": "#d62728", "f3": "#2ca02c"}
LABELS = {"f1": "F1 arithmetic", "f2": "F2 relational", "f3": "F3 state tracking"}


def _y(value: float, top: float) -> float:
    return top + PANEL_H * (1.0 - value / 1.0)


def _panel(data: dict, x0: float, title: str, budgets: list[int]) -> list[str]:
    parts: list[str] = []
    top = MT
    parts.append(
        f'<text x="{x0 + PANEL_W / 2:.0f}" y="{top - 22}" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="#222">{title}</text>'
    )
    # Axes and horizontal grid.
    parts.append(
        f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + PANEL_H}" '
        f'stroke="#444" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{top + PANEL_H}" x2="{x0 + PANEL_W}" '
        f'y2="{top + PANEL_H}" stroke="#444" stroke-width="1"/>'
    )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _y(tick, top)
        parts.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + PANEL_W}" y2="{y:.1f}" '
            f'stroke="#e5e5e5" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#555">{tick:.2f}</text>'
        )
    # X positions: equally spaced by budget order.
    xs = {
        b: x0 + 12 + (PANEL_W - 24) * i / (len(budgets) - 1)
        for i, b in enumerate(budgets)
    }
    for b, x in xs.items():
        parts.append(
            f'<text x="{x:.1f}" y="{top + PANEL_H + 20}" text-anchor="middle" '
            f'font-size="11" fill="#555">{b}</text>'
        )
    parts.append(
        f'<text x="{x0 + PANEL_W / 2:.0f}" y="{top + PANEL_H + 42}" '
        f'text-anchor="middle" font-size="12" fill="#333">CoT budget (new tokens)</text>'
    )
    parts.append(
        f'<text x="{x0 - 46}" y="{top + PANEL_H / 2:.0f}" text-anchor="middle" '
        f'font-size="12" fill="#333" transform="rotate(-90 {x0 - 46} '
        f'{top + PANEL_H / 2:.0f})">exact-match accuracy</text>'
    )

    for family in ("f1", "f2", "f3"):
        entry = data[family]
        colour = COLOURS[family]
        direct = entry["direct"]["accuracy"]
        # Direct baseline (same colour, dashed, no marker).
        yd = _y(direct, top)
        parts.append(
            f'<line x1="{x0}" y1="{yd:.1f}" x2="{x0 + PANEL_W}" y2="{yd:.1f}" '
            f'stroke="{colour}" stroke-width="1.2" stroke-dasharray="4 3" '
            f'opacity="0.55"/>'
        )
        points = []
        for b in budgets:
            value = entry["cot"].get(str(b))
            if value is None:
                continue
            points.append((xs[b], _y(value["accuracy"], top), value["accuracy"]))
        if len(points) >= 2:
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                for i, (x, y, _) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.2"/>'
            )
        for x, y, value in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{colour}"/>'
            )
        # Label at the right end.
        if points:
            x, y, value = points[-1]
            parts.append(
                f'<text x="{x + 6:.1f}" y="{y + 4:.1f}" font-size="11" '
                f'fill="{colour}">{LABELS[family].split()[0]}</text>'
            )
    return parts


def render(inputs: list[Path], out: Path) -> None:
    panels = []
    for path in inputs:
        data = json.loads(path.read_text())["summary"]
        model = json.loads(path.read_text())["model"]
        budgets = sorted(
            {int(b) for fam in data.values() for b in fam["cot"]}
        )
        panels.append((data, model, budgets))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    for i, (data, model, budgets) in enumerate(panels[:2]):
        x0 = ML + i * (PANEL_W + GAP)
        short = model.split("/")[-1]
        parts.extend(_panel(data, x0, f"{short} (instruct)", budgets))

    # Shared legend.
    lx, ly = ML, H - 16
    for i, family in enumerate(("f1", "f2", "f3")):
        x = lx + i * 190
        parts.append(
            f'<line x1="{x}" y1="{ly}" x2="{x + 22}" y2="{ly}" '
            f'stroke="{COLOURS[family]}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<circle cx="{x + 11}" cy="{ly}" r="3.2" fill="{COLOURS[family]}"/>'
        )
        parts.append(
            f'<text x="{x + 30}" y="{ly + 4}" font-size="12" fill="#333">'
            f'{LABELS[family]}</text>'
        )
    parts.append(
        f'<text x="{W - MR}" y="{ly + 4}" text-anchor="end" font-size="11" '
        f'fill="#777">dashed = direct answer (no working)</text>'
    )
    parts.append("</svg>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default="docs/figures/e2_budget_curve.svg")
    args = p.parse_args(argv)
    render([Path(i) for i in args.inputs], Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
