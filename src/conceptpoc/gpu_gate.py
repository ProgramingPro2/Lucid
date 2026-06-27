"""Polite GPU scheduling: yield the GPU to other *actively computing* processes.

The trigger is per-process **compute (SM) utilization**, not VRAM. A model that is
merely parked in VRAM but idle (e.g. an LLM server like lemonade/Ollama between
requests) reports ~0% SM, so we ignore it and keep training. When another process
actually does GPU work (a game, or that LLM server while generating), its SM% rises
and we yield: the model + optimizer are offloaded to CPU, CUDA memory is released,
and we poll until the foreign compute load drops before resuming.

Detection uses ``nvidia-smi pmon`` (per-process sm%), which is available on
consumer GPUs. Our own PID is excluded, so the signal is "how much GPU compute are
*other* processes using right now".
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

import torch

from .config import GpuGateConfig
from .progress import log_stage


@dataclass
class GpuReading:
    foreign_sm_util: float  # summed SM% of other processes (active compute)
    foreign_mem_mb: float  # summed VRAM of other processes (optional trigger)
    ok: bool


def _read_pmon(own_pid: int, device_index: int, samples: int) -> tuple[float, bool]:
    """Average summed foreign SM% over `samples` one-second pmon snapshots."""
    proc = subprocess.run(
        ["nvidia-smi", "pmon", "-c", str(max(1, samples)), "-s", "u"],
        capture_output=True,
        text=True,
        timeout=10 + 2 * samples,
    )
    if proc.returncode != 0:
        return 0.0, False
    total_sm = 0.0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 4:
            continue
        try:
            gpu = int(cols[0])
            pid = int(cols[1])
        except ValueError:
            continue
        if gpu != device_index or pid == own_pid:
            continue
        sm_token = cols[3]
        try:
            total_sm += float(sm_token) if sm_token != "-" else 0.0
        except ValueError:
            continue
    return total_sm / max(1, samples), True


def _read_compute_apps_mem(own_pid: int, device_index: int) -> float:
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        return 0.0
    foreign = 0.0
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mem = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid != own_pid:
            foreign += mem
    return foreign


def make_reader(samples: int, want_mem: bool) -> Callable[[int, int], GpuReading]:
    def reader(own_pid: int, device_index: int) -> GpuReading:
        try:
            sm, ok = _read_pmon(own_pid, device_index, samples)
        except Exception:
            return GpuReading(0.0, 0.0, ok=False)
        mem = 0.0
        if want_mem:
            try:
                mem = _read_compute_apps_mem(own_pid, device_index)
            except Exception:
                mem = 0.0
        return GpuReading(foreign_sm_util=sm, foreign_mem_mb=mem, ok=ok)

    return reader


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class GpuGate:
    """Pauses training while other processes are actively computing on the GPU."""

    def __init__(
        self,
        cfg: GpuGateConfig,
        device: torch.device,
        *,
        read_fn: Callable[[int, int], GpuReading] | None = None,
        progress: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.own_pid = os.getpid()
        self.progress = progress
        self._read_fn = read_fn or make_reader(cfg.pmon_samples, cfg.mem_trigger_mb > 0)
        self._last_check = 0.0
        idx = cfg.device_index
        if idx is None and device.type == "cuda":
            idx = device.index if device.index is not None else torch.cuda.current_device()
        self._device_index = idx or 0
        self.active = cfg.enabled and device.type == "cuda"

    def _read(self) -> GpuReading:
        return self._read_fn(self.own_pid, self._device_index)

    def _log(self, msg: str) -> None:
        log_stage(f"[gpu-gate] {msg}", enabled=self.progress)

    def _is_busy(self, r: GpuReading) -> bool:
        if not r.ok:
            return False
        if r.foreign_sm_util >= self.cfg.foreign_util_pct:
            return True
        if self.cfg.mem_trigger_mb > 0 and r.foreign_mem_mb >= self.cfg.mem_trigger_mb:
            return True
        return False

    def step(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Call once per training batch; probes at most every `check_interval_s`."""
        if not self.active:
            return
        now = time.monotonic()
        if now - self._last_check < self.cfg.check_interval_s:
            return
        self._last_check = now
        self.guard(model, optimizer)

    def guard(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Block (offloading to CPU) while foreign GPU compute exceeds the threshold."""
        if not self.active:
            return
        reading = self._read()
        if not self._is_busy(reading):
            return
        self._yield(model, optimizer, reading)

    def _yield(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        reading: GpuReading,
    ) -> None:
        self._log(
            f"other processes using {reading.foreign_sm_util:.0f}% GPU compute "
            f"(threshold {self.cfg.foreign_util_pct:.0f}%); yielding GPU."
        )
        if self.cfg.release_memory:
            model.to("cpu")
            _move_optimizer_state(optimizer, "cpu")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        clear = 0
        while True:
            time.sleep(self.cfg.poll_interval_s)
            reading = self._read()
            if not reading.ok:
                self._log("could not read GPU state; resuming to avoid a deadlock.")
                break
            if reading.foreign_sm_util < self.cfg.resume_util_pct:
                clear += 1
                if clear >= self.cfg.clear_polls:
                    break
            else:
                clear = 0
        if self.cfg.release_memory:
            model.to(self.device)
            _move_optimizer_state(optimizer, self.device)
        self._log("foreign GPU compute idle; resuming training.")
