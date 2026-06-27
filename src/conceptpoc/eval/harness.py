"""Per-capability generation eval harness.

For each held-out prompt: encode -> greedy latent rollout -> decode, then score
the joined continuation with the configured judge (``build_judge``) and compute
generation diagnostics (distinct-2, verbatim sentence-repeat rate, sentence
count). Aggregates a mean judge score per capability and overall, and writes a
``report.json`` plus a readable text summary.

The core (:func:`run_eval`) takes already-built codec/model/judge objects so it
is fully offline and deterministic under tests; :func:`run_harness` is the
config-driven entry point used by the CLI.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..codec.base import ConceptCodec
from ..data.segment import segment_sentences
from ..generate.rollout import (
    generate_conditioned,
    generate_rollout,
    rollout_to_text,
)
from ..predictor.model import ConceptPredictor
from ..reward.judge import Judge

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Capabilities whose prompt is an instruction: in conditioned mode these are
# generated with ``generate_conditioned`` (instruction prefix + optional theme).
# Other capabilities use plain continuation rollout, so the legacy behavior and
# tests are unchanged.
INSTRUCTION_CAPABILITIES = frozenset({"instruction_following", "themed_writing"})


@dataclass
class EvalItem:
    id: str
    capability: str
    prompt: str
    reference: str | None = None
    theme: str | None = None


@dataclass
class ItemResult:
    id: str
    capability: str
    prompt: str
    text: str
    score: float
    rationale: str
    diagnostics: dict
    rating_app_id: str | None = None
    human_score: float | None = None


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Load eval items from a ``.jsonl`` file or every ``.jsonl`` in a dir."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    items: list[EvalItem] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            items.append(
                EvalItem(
                    id=str(raw["id"]),
                    capability=str(raw["capability"]),
                    prompt=str(raw["prompt"]),
                    reference=raw.get("reference"),
                    theme=raw.get("theme"),
                )
            )
    return items


def distinct_2(text: str) -> float:
    """Fraction of distinct word bigrams (lexical diversity proxy)."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words[:-1], words[1:]))
    return len(set(bigrams)) / len(bigrams)


def verbatim_repeat_rate(sentences: list[str]) -> float:
    """Fraction of sentences that duplicate an earlier (normalized) sentence."""
    if not sentences:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for s in sentences:
        key = s.strip().lower()
        if key in seen:
            repeats += 1
        else:
            seen.add(key)
    return repeats / len(sentences)


def _generate(
    codec: ConceptCodec,
    model: ConceptPredictor,
    prompt: str,
    steps: int,
    device: str,
) -> tuple[str, list[str], np.ndarray]:
    seed_sents = segment_sentences(prompt) or [prompt]
    seed_vecs = codec.encode(seed_sents)
    roll = generate_rollout(
        model, seed_vecs, steps=steps, temperature=0.0, device=device
    )
    lines = rollout_to_text(roll, codec)
    text = " ".join(lines)
    return text, lines, roll.generated_vectors


def _generate_conditioned(
    codec: ConceptCodec,
    model: ConceptPredictor,
    instruction: str,
    theme: str | None,
    steps: int,
    device: str,
) -> tuple[str, list[str], np.ndarray]:
    instr_sents = segment_sentences(instruction) or [instruction]
    instr_vecs = codec.encode(instr_sents)
    theme_vec = codec.encode([theme])[0] if theme else None
    roll = generate_conditioned(
        model,
        instr_vecs,
        steps=steps,
        theme_vec=theme_vec,
        temperature=0.0,
        device=device,
    )
    lines = rollout_to_text(roll, codec)
    text = " ".join(lines)
    return text, lines, roll.generated_vectors


def _diagnostics(text: str, lines: list[str]) -> dict:
    sentences = lines or [s for s in _SENT_RE.split(text.strip()) if s]
    return {
        "distinct_2": round(distinct_2(text), 4),
        "verbatim_repeat_rate": round(verbatim_repeat_rate(sentences), 4),
        "sentence_count": len(sentences),
    }


def _model_device(model: ConceptPredictor) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def run_eval(
    items: list[EvalItem],
    codec: ConceptCodec,
    model: ConceptPredictor,
    judge: Judge,
    steps: int = 6,
    device: str | None = None,
    conditioned: bool = False,
) -> list[ItemResult]:
    """Generate + score every item. Pure and deterministic for greedy rollout.

    When ``conditioned`` is True, items in :data:`INSTRUCTION_CAPABILITIES` are
    generated via ``generate_conditioned`` (prompt = instruction, ``item.theme``
    = optional theme); all other items keep plain continuation rollout, so the
    default ``conditioned=False`` path is byte-for-byte the legacy behavior.
    """
    dev = device or _model_device(model)
    results: list[ItemResult] = []
    for item in items:
        if conditioned and item.capability in INSTRUCTION_CAPABILITIES:
            text, lines, gen_vecs = _generate_conditioned(
                codec, model, item.prompt, item.theme, steps, dev
            )
        else:
            text, lines, gen_vecs = _generate(codec, model, item.prompt, steps, dev)
        verdict = judge.score(item.prompt, text, gen_vecs)
        results.append(
            ItemResult(
                id=item.id,
                capability=item.capability,
                prompt=item.prompt,
                text=text,
                score=float(verdict.score),
                rationale=verdict.rationale,
                diagnostics=_diagnostics(text, lines),
            )
        )
    return results


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def aggregate(results: list[ItemResult]) -> dict:
    """Mean score + mean diagnostics per capability and overall."""
    by_cap: dict[str, list[ItemResult]] = {}
    for r in results:
        by_cap.setdefault(r.capability, []).append(r)

    def _summary(group: list[ItemResult]) -> dict:
        humans = [r.human_score for r in group if r.human_score is not None]
        summary = {
            "n": len(group),
            "mean_score": round(_mean([r.score for r in group]), 4),
            "mean_distinct_2": round(
                _mean([r.diagnostics["distinct_2"] for r in group]), 4
            ),
            "mean_verbatim_repeat_rate": round(
                _mean([r.diagnostics["verbatim_repeat_rate"] for r in group]), 4
            ),
            "mean_sentence_count": round(
                _mean([float(r.diagnostics["sentence_count"]) for r in group]), 4
            ),
        }
        if humans:
            summary["mean_human_score"] = round(_mean(humans), 4)
        return summary

    return {
        "per_capability": {cap: _summary(g) for cap, g in sorted(by_cap.items())},
        "overall": _summary(results),
    }


def build_report(
    results: list[ItemResult], *, steps: int, judge_backend: str, config: str | None
) -> dict:
    agg = aggregate(results)
    return {
        "config": config,
        "judge_backend": judge_backend,
        "steps": steps,
        "num_items": len(results),
        "per_capability": agg["per_capability"],
        "overall": agg["overall"],
        "items": [asdict(r) for r in results],
    }


def format_summary(report: dict) -> str:
    lines = ["Eval harness report"]
    if report.get("config"):
        lines.append(f"  config: {report['config']}")
    lines.append(f"  judge_backend: {report.get('judge_backend')}")
    lines.append(f"  steps: {report.get('steps')}  items: {report.get('num_items')}")
    lines.append("")
    header = f"  {'capability':<22}{'n':>4}{'score':>9}{'distinct2':>11}{'repeat':>9}{'sents':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    rows = list(report["per_capability"].items()) + [("OVERALL", report["overall"])]
    for cap, s in rows:
        human = ""
        if "mean_human_score" in s:
            human = f"  human={s['mean_human_score']:.3f}"
        lines.append(
            f"  {cap:<22}{s['n']:>4}{s['mean_score']:>9.3f}"
            f"{s['mean_distinct_2']:>11.3f}{s['mean_verbatim_repeat_rate']:>9.3f}"
            f"{s['mean_sentence_count']:>8.2f}{human}"
        )
    return "\n".join(lines)


def write_report(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    txt_path = out / "report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(format_summary(report) + "\n", encoding="utf-8")
    return json_path, txt_path


def run_harness(
    cfg,
    *,
    eval_path: str | Path,
    steps: int,
    judge: Judge,
    judge_backend: str,
    config_label: str | None = None,
    progress: bool = True,
    conditioned: bool = False,
) -> tuple[list[ItemResult], dict]:
    """Config-driven entry point: load codec+predictor, eval, build report.

    With ``conditioned=True`` the segment/theme-conditioned predictor (saved by
    ``train-conditioned``) is loaded and instruction-capability items are
    generated with conditioning; otherwise the model's continuation predictor is
    used exactly as before.
    """
    from ..loader import codec_and_conditioned_predictor, codec_and_predictor

    items = load_eval_set(eval_path)
    if not items:
        raise ValueError(f"no eval items found at {eval_path}")
    # Fit texts only matter for retrieval codecs; the jepa codec ignores them.
    if conditioned:
        codec, model = codec_and_conditioned_predictor(cfg, None)
    else:
        codec, model = codec_and_predictor(cfg, None)
    results = run_eval(items, codec, model, judge, steps=steps, conditioned=conditioned)
    report = build_report(
        results, steps=steps, judge_backend=judge_backend, config=config_label
    )
    return results, report
