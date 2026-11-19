"""Tests for the reproducibility and qualification infrastructure.

Covers ``experiments.protocol.manifest`` (docs/EXPERIMENTS.md §9, §11) and
``experiments.protocol.qualify`` (docs/PROTOCOL.md §10, docs/EXPERIMENTS.md §8).
The point is the gate: an incomplete manifest must not enter analysis, and a
failed qualification must return VOID with reasons rather than a result.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.protocol.manifest import (
    REQUIRED_FIELDS,
    RunManifest,
    capture_git_commit,
    hash_config,
    partition_manifests,
    spawn_keys_for_seed,
    validate_completeness,
)
from experiments.protocol.qualify import (
    NanCounter,
    QualificationReport,
    assert_token_streams_identical,
    check_step0_identity,
    from_json,
    report_to_json,
)
from experiments.protocol.seeds import STREAMS, spawn_key

SEED = 17


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _good_manifest(**overrides) -> RunManifest:
    base = dict(
        run_id="e1/A/s17",
        git_commit="0" * 40,
        config_sha256=hash_config({"cell": "A", "seed": SEED}),
        corpus_sha256="ab" * 32,
        spawn_keys=spawn_keys_for_seed(SEED),
        token_budget=10_000_000,
        final_checkpoint="/runs/e1/A/s17/final.pt",
        wall_clock_s=3_600.0,
        peak_vram_gb=20.5,
        nan_skips=0,
        arms=["A", "B"],
        notes="pilot",
    )
    base.update(overrides)
    return RunManifest(**base)


def test_spawn_keys_for_seed_is_the_six_protocol_streams():
    keys = spawn_keys_for_seed(SEED)
    assert set(keys) == set(STREAMS)
    assert keys == {name: spawn_key(SEED, name) for name in STREAMS}


def test_manifest_round_trip_preserves_every_field(tmp_path):
    manifest = _good_manifest()
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    loaded = RunManifest.from_json(path)

    assert loaded == manifest
    for name in REQUIRED_FIELDS:
        assert getattr(loaded, name) == getattr(manifest, name)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["spawn_keys"] == spawn_keys_for_seed(SEED)


def test_from_json_names_missing_fields(tmp_path):
    manifest = _good_manifest()
    data = manifest.to_dict()
    del data["nan_skips"]
    del data["peak_vram_gb"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        RunManifest.from_json(path)

    message = str(excinfo.value)
    assert "nan_skips" in message
    assert "peak_vram_gb" in message


def test_hash_config_is_order_insensitive_and_value_sensitive():
    a = {"seed": 17, "cell": "A", "nested": {"lr": 1e-4, "depth": 3}}
    b = {"nested": {"depth": 3, "lr": 1e-4}, "cell": "A", "seed": 17}
    assert hash_config(a) == hash_config(b)
    assert len(hash_config(a)) == 64

    changed = copy.deepcopy(a)
    changed["nested"]["depth"] = 4
    assert hash_config(changed) != hash_config(a)


def test_validate_completeness_passes_a_good_manifest():
    assert validate_completeness(_good_manifest()) == []


def test_validate_completeness_flags_bad_spawn_keys():
    keys = spawn_keys_for_seed(SEED)
    del keys["decode"]
    problems = validate_completeness(_good_manifest(spawn_keys=keys))
    assert any("decode" in problem for problem in problems)


def test_validate_completeness_flags_bad_corpus_hash():
    problems = validate_completeness(_good_manifest(corpus_sha256="not-a-sha256"))
    assert any("corpus_sha256" in problem for problem in problems)


def test_validate_completeness_flags_negative_nan_skips():
    problems = validate_completeness(_good_manifest(nan_skips=-1))
    assert any("nan_skips" in problem for problem in problems)


def test_validate_completeness_flags_zero_budget_and_empty_arms():
    problems = validate_completeness(_good_manifest(token_budget=0, arms=[]))
    assert any("token_budget" in problem for problem in problems)
    assert any("arms" in problem for problem in problems)


def test_partition_manifests_separates_good_from_bad():
    good = _good_manifest()
    bad_nan = replace(good, run_id="e1/B/s17", nan_skips=-2)
    bad_corpus = replace(good, run_id="e1/E/s17", corpus_sha256="deadbeef")

    included, excluded = partition_manifests([good, bad_nan, bad_corpus])

    assert included == [good]
    assert [manifest.run_id for manifest, _ in excluded] == [
        "e1/B/s17",
        "e1/E/s17",
    ]
    assert all(reasons for _, reasons in excluded)


def test_capture_git_commit_never_raises(tmp_path):
    root = Path(__file__).resolve().parents[1]
    commit = capture_git_commit(root)
    assert commit == "unknown" or (
        len(commit) == 40
        and all(char in "0123456789abcdef" for char in commit)
    )

    missing = tmp_path / "does-not-exist"
    assert capture_git_commit(missing) == "unknown"


# --------------------------------------------------------------------------
# NanCounter
# --------------------------------------------------------------------------


def test_nan_counter_counts_and_reports_a_rate():
    counter = NanCounter()
    for _ in range(3):
        counter.record()

    assert counter.total == 3
    assert counter.rate(500) == pytest.approx(3 / 500)

    with pytest.raises(ValueError, match="steps"):
        counter.rate(0)


def test_nan_counter_is_thread_safe_enough():
    counter = NanCounter()

    def record_many() -> None:
        for _ in range(500):
            counter.record()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: record_many(), range(8)))

    assert counter.total == 4_000


def test_nan_counter_context_managers():
    counter = NanCounter()
    with counter as active:
        assert active is counter
        active.record()

    with counter.sanitised():
        pass

    assert counter.total == 2


# --------------------------------------------------------------------------
# Step-0 identity
# --------------------------------------------------------------------------


def _constant_linear() -> nn.Sequential:
    layer = nn.Linear(4, 3)
    with torch.no_grad():
        layer.weight.fill_(0.5)
        layer.bias.fill_(0.25)
    return nn.Sequential(layer)


def test_check_step0_identity_true_for_identical_modules():
    x = torch.full((5, 4), 0.5)
    a = _constant_linear()
    b = copy.deepcopy(a)

    result = check_step0_identity({"A": a, "B": b}, x)

    assert result["A|B"] is True
    assert result["identical"] is True


def test_check_step0_identity_false_for_nonzero_extra_bias():
    x = torch.full((5, 4), 0.5)
    a = _constant_linear()
    shifted = copy.deepcopy(a)
    with torch.no_grad():
        shifted[0].bias.add_(1e-3)

    result = check_step0_identity({"A": a, "E": shifted}, x)

    assert result["A|E"] is False
    assert result["identical"] is False


def test_check_step0_identity_uses_exact_equality_not_allclose():
    """A sub-tolerance difference still voids step-0 bit-identity."""
    x = torch.full((5, 4), 0.5)
    a = _constant_linear()
    almost = copy.deepcopy(a)
    with torch.no_grad():
        almost[0].bias.add_(1e-6)

    with torch.no_grad():
        out_a = a(x)
        out_almost = almost(x)
    assert torch.allclose(out_a, out_almost)
    assert not torch.equal(out_a, out_almost)

    result = check_step0_identity({"A": a, "E": almost}, x)
    assert result["identical"] is False


def test_check_step0_identity_forces_eval_mode_and_restores_it():
    torch.manual_seed(0)
    a = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
    b = copy.deepcopy(a)
    a.train()
    b.train()
    x = torch.randn(6, 4)

    result = check_step0_identity({"A": a, "B": b}, x)

    assert result["identical"] is True
    assert a.training is True
    assert b.training is True


# --------------------------------------------------------------------------
# Token-stream identity
# --------------------------------------------------------------------------


def test_assert_token_streams_identical_passes_on_equal_hashes():
    hashes = {f"e1/{arm}/s17": "a" * 64 for arm in ("A", "B", "E")}
    assert_token_streams_identical(hashes)


def test_assert_token_streams_identical_names_the_odd_one_out():
    hashes = {
        "e1/A/s17": "a" * 64,
        "e1/B/s17": "a" * 64,
        "e1/E/s17": "b" * 64,
    }
    with pytest.raises(ValueError) as excinfo:
        assert_token_streams_identical(hashes)

    message = str(excinfo.value)
    assert "e1/E/s17" in message
    assert "e1/B/s17" not in message


def test_assert_token_streams_identical_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="no token-stream hashes"):
        assert_token_streams_identical({})


# --------------------------------------------------------------------------
# QualificationReport
# --------------------------------------------------------------------------


def _good_report(**overrides) -> QualificationReport:
    base = dict(
        nan_skips=0,
        steps=500,
        throughput_tok_s=2_500.0,
        micro_batch=4,
        step0_identical=True,
        token_hash_identical=True,
        eval_finite=True,
        arm_nan_spread_pp=0.7,
    )
    base.update(overrides)
    return QualificationReport(**base)


def test_qualification_report_all_good_passes():
    verdict, reasons = _good_report().verdict()
    assert verdict == "PASS"
    assert reasons == []


def test_qualification_report_single_nan_skip_is_void():
    verdict, reasons = _good_report(nan_skips=1).verdict()
    assert verdict == "VOID"
    assert any("sanitis" in reason.lower() for reason in reasons)


def test_qualification_report_throughput_1999_is_void():
    verdict, reasons = _good_report(throughput_tok_s=1_999.0).verdict()
    assert verdict == "VOID"
    assert any("throughput" in reason for reason in reasons)


def test_qualification_report_spread_2_5_is_void():
    verdict, reasons = _good_report(arm_nan_spread_pp=2.5).verdict()
    assert verdict == "VOID"
    assert any("spread" in reason for reason in reasons)


@pytest.mark.parametrize(
    "overrides",
    [
        {"micro_batch": 2},
        {"step0_identical": False},
        {"token_hash_identical": False},
        {"eval_finite": False},
    ],
)
def test_qualification_report_each_remaining_gate_is_void(overrides):
    verdict, reasons = _good_report(**overrides).verdict()
    assert verdict == "VOID"
    assert reasons


def test_qualification_report_boundaries_pass():
    verdict, _ = _good_report(
        throughput_tok_s=2_000.0, micro_batch=4, arm_nan_spread_pp=2.0
    ).verdict()
    assert verdict == "PASS"


def test_qualification_report_json_round_trip(tmp_path):
    report = _good_report()
    path = tmp_path / "qualification.json"
    report_to_json(report, path)

    loaded = from_json(path)
    assert loaded == report
    assert loaded.verdict() == ("PASS", [])


def test_qualify_main_reads_stdin_and_sets_exit_status():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(root / "src"), env.get("PYTHONPATH")])
    )
    command = [sys.executable, "-m", "experiments.protocol.qualify"]

    good = subprocess.run(
        command,
        input=json.dumps(asdict(_good_report())),
        text=True,
        capture_output=True,
        cwd=root,
        env=env,
        timeout=120,
    )
    assert good.returncode == 0, good.stderr
    assert "PASS" in good.stdout

    bad = subprocess.run(
        command,
        input=json.dumps(asdict(_good_report(nan_skips=1))),
        text=True,
        capture_output=True,
        cwd=root,
        env=env,
        timeout=120,
    )
    assert bad.returncode == 1, bad.stderr
    assert "VOID" in bad.stdout
    assert "sanitis" in bad.stdout.lower()
