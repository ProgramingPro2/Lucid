from __future__ import annotations

import torch

from conceptpoc.config import GpuGateConfig
from conceptpoc.gpu_gate import GpuGate, GpuReading, _move_optimizer_state


def _model_and_opt():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    # Populate optimizer state so _move_optimizer_state has tensors to move.
    out = model(torch.randn(2, 4)).sum()
    out.backward()
    opt.step()
    return model, opt


def test_disabled_gate_is_noop():
    gate = GpuGate(GpuGateConfig(enabled=False), torch.device("cpu"))
    model, opt = _model_and_opt()
    gate.step(model, opt)  # should do nothing, never read the GPU
    assert gate.active is False


def test_gate_inactive_on_cpu_even_when_enabled():
    gate = GpuGate(GpuGateConfig(enabled=True), torch.device("cpu"))
    assert gate.active is False


def test_idle_parked_model_does_not_pause():
    # High VRAM but ~0% SM (idle parked LLM): must NOT trigger a pause.
    def fake_read(pid, idx):
        return GpuReading(foreign_sm_util=0.0, foreign_mem_mb=9000.0, ok=True)

    cfg = GpuGateConfig(enabled=True, foreign_util_pct=40.0, mem_trigger_mb=0.0)
    gate = GpuGate(cfg, torch.device("cpu"), read_fn=fake_read)
    gate.active = True
    assert gate._is_busy(fake_read(0, 0)) is False


def test_active_compute_triggers_pause():
    def fake_read(pid, idx):
        return GpuReading(foreign_sm_util=85.0, foreign_mem_mb=0.0, ok=True)

    cfg = GpuGateConfig(enabled=True, foreign_util_pct=40.0)
    gate = GpuGate(cfg, torch.device("cpu"), read_fn=fake_read)
    gate.active = True
    assert gate._is_busy(fake_read(0, 0)) is True


def test_guard_pauses_until_clear():
    readings = iter(
        [
            GpuReading(85.0, 9000.0, ok=True),  # active compute -> trigger pause
            GpuReading(85.0, 9000.0, ok=True),  # still busy
            GpuReading(5.0, 9000.0, ok=True),  # clear 1 (parked VRAM ignored)
            GpuReading(5.0, 9000.0, ok=True),  # clear 2
            GpuReading(5.0, 9000.0, ok=True),  # clear 3 -> resume
        ]
    )
    calls = []

    def fake_read(pid, idx):
        r = next(readings)
        calls.append(r)
        return r

    cfg = GpuGateConfig(
        enabled=True,
        foreign_util_pct=40.0,
        resume_util_pct=20.0,
        poll_interval_s=0.0,
        clear_polls=3,
        release_memory=False,
    )
    gate = GpuGate(cfg, torch.device("cpu"), read_fn=fake_read)
    gate.active = True  # force-activate for the CPU test

    model, opt = _model_and_opt()
    gate.guard(model, opt)
    # 1 initial busy read + 4 poll reads consumed before resuming.
    assert len(calls) == 5


def test_guard_resumes_when_reading_unavailable():
    readings = iter([GpuReading(85.0, 0.0, ok=True), GpuReading(0.0, 0.0, ok=False)])

    def fake_read(pid, idx):
        return next(readings)

    cfg = GpuGateConfig(enabled=True, foreign_util_pct=40.0, poll_interval_s=0.0)
    gate = GpuGate(cfg, torch.device("cpu"), read_fn=fake_read)
    gate.active = True
    model, opt = _model_and_opt()
    gate.guard(model, opt)  # must not deadlock when the GPU can't be read


def test_move_optimizer_state_roundtrips():
    _model, opt = _model_and_opt()
    _move_optimizer_state(opt, "cpu")
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                assert v.device.type == "cpu"
