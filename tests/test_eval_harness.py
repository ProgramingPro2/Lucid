from __future__ import annotations

import json

import numpy as np

from conceptpoc.config import PredictorConfig
from conceptpoc.eval.harness import (
    EvalItem,
    aggregate,
    build_report,
    distinct_2,
    load_eval_set,
    run_eval,
    verbatim_repeat_rate,
    write_report,
)
from conceptpoc.predictor.model import ConceptPredictor
from conceptpoc.reward.judge import HeuristicJudge


def _model(dim: int) -> ConceptPredictor:
    return ConceptPredictor(
        concept_dim=dim,
        cfg=PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16),
    )


def _conditioned_model(dim: int) -> ConceptPredictor:
    return ConceptPredictor(
        concept_dim=dim,
        cfg=PredictorConfig(
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=16,
            use_segment_embedding=True,
            use_theme=True,
        ),
    )


def _items() -> list[EvalItem]:
    return [
        EvalItem("a1", "continuation", "the river moved slowly across the valley"),
        EvalItem("a2", "continuation", "a small town sat at the edge of the water"),
        EvalItem("b1", "coherence", "learning begins with attention and practice"),
        EvalItem("b2", "coherence", "the old library stood quietly on the hill"),
    ]


def test_diagnostics_helpers():
    assert distinct_2("the cat the cat") < 1.0
    assert distinct_2("one two three four") == 1.0
    assert verbatim_repeat_rate(["a.", "a.", "b."]) == 1 / 3
    assert verbatim_repeat_rate([]) == 0.0


def test_run_eval_structure_and_determinism(fitted_codec):
    model = _model(fitted_codec.dim)
    judge = HeuristicJudge()
    items = _items()

    first = run_eval(items, fitted_codec, model, judge, steps=4, device="cpu")
    second = run_eval(items, fitted_codec, model, judge, steps=4, device="cpu")

    assert len(first) == len(items)
    assert [r.score for r in first] == [r.score for r in second]
    assert [r.text for r in first] == [r.text for r in second]

    for r in first:
        assert 0.0 <= r.score <= 1.0
        assert set(r.diagnostics) == {
            "distinct_2",
            "verbatim_repeat_rate",
            "sentence_count",
        }


def test_aggregate_per_capability(fitted_codec):
    model = _model(fitted_codec.dim)
    results = run_eval(_items(), fitted_codec, model, HeuristicJudge(), steps=4)
    agg = aggregate(results)

    assert set(agg["per_capability"]) == {"continuation", "coherence"}
    assert agg["per_capability"]["continuation"]["n"] == 2
    assert agg["per_capability"]["coherence"]["n"] == 2
    assert agg["overall"]["n"] == 4

    cont = [r.score for r in results if r.capability == "continuation"]
    assert agg["per_capability"]["continuation"]["mean_score"] == round(
        float(np.mean(cont)), 4
    )


def test_build_and_write_report(fitted_codec, tmp_path):
    model = _model(fitted_codec.dim)
    results = run_eval(_items(), fitted_codec, model, HeuristicJudge(), steps=3)
    report = build_report(
        results, steps=3, judge_backend="heuristic", config="tiny.json"
    )

    assert report["num_items"] == 4
    assert report["judge_backend"] == "heuristic"
    assert len(report["items"]) == 4
    assert "overall" in report and "per_capability" in report

    json_path, txt_path = write_report(report, tmp_path / "out")
    assert json_path.is_file() and txt_path.is_file()
    loaded = json.loads(json_path.read_text())
    assert loaded["overall"] == report["overall"]


def test_conditioned_eval_runs_instruction_and_theme(fitted_codec):
    # In conditioned mode, instruction/themed items use generate_conditioned;
    # continuation items still use plain rollout. Deterministic under greedy.
    model = _conditioned_model(fitted_codec.dim)
    judge = HeuristicJudge()
    items = [
        EvalItem("c1", "continuation", "the river moved slowly across the valley"),
        EvalItem("i1", "instruction_following", "describe a calm morning"),
        EvalItem("t1", "themed_writing", "write about the water", theme="the sea"),
    ]
    first = run_eval(items, fitted_codec, model, judge, steps=3, conditioned=True)
    second = run_eval(items, fitted_codec, model, judge, steps=3, conditioned=True)

    assert len(first) == 3
    assert [r.text for r in first] == [r.text for r in second]
    for r in first:
        assert 0.0 <= r.score <= 1.0
        assert r.diagnostics["sentence_count"] >= 1


def test_conditioned_flag_off_matches_continuation(fitted_codec):
    # An instruction item with conditioned=False must take the continuation path,
    # identical to the legacy behavior (backward compatibility guarantee).
    model = _conditioned_model(fitted_codec.dim)
    judge = HeuristicJudge()
    item = EvalItem("i1", "instruction_following", "describe a calm morning")
    off = run_eval([item], fitted_codec, model, judge, steps=3, conditioned=False)
    cont = run_eval(
        [EvalItem("i1", "continuation", "describe a calm morning")],
        fitted_codec,
        model,
        judge,
        steps=3,
        conditioned=False,
    )
    assert off[0].text == cont[0].text


def test_load_eval_set_from_file(tmp_path):
    f = tmp_path / "tiny.jsonl"
    f.write_text(
        '{"id": "x1", "capability": "continuation", "prompt": "hello world"}\n'
        '{"id": "x2", "capability": "coherence", "prompt": "explain why"}\n',
        encoding="utf-8",
    )
    items = load_eval_set(f)
    assert len(items) == 2
    assert items[0].id == "x1"
    assert items[1].capability == "coherence"
