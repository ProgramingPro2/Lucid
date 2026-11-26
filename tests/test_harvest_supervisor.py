"""Schedule tests for scripts/harvest_supervisor.py.

A fake clock and a fake child. No network, no harvest process, no real sleep.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ZONE = ZoneInfo("America/New_York")
QUIET_START = dt_time(23, 30)
QUIET_END = dt_time(6, 0)


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "harvest_supervisor.py"
    spec = importlib.util.spec_from_file_location("harvest_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["harvest_supervisor"] = module
    spec.loader.exec_module(module)
    return module


hs = _load()


def at(hour: int, minute: int, *, day: int = 26, second: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, second, tzinfo=ZONE)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self._now = now
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float, *, interruptible: bool = True) -> None:
        del interruptible
        seconds = float(seconds)
        if seconds < 0:
            raise AssertionError(f"negative sleep {seconds}")
        self.slept.append(seconds)
        self._now += timedelta(seconds=seconds)


class FakeChild:
    def __init__(self, *, exit_code: int = 0, polls_before_exit: int | None = None, clock: FakeClock) -> None:
        self.exit_code = exit_code
        self.polls_before_exit = polls_before_exit
        self.clock = clock
        self.polls = 0
        self.returncode: int | None = None
        self.term_count = 0
        self.terminated_at: datetime | None = None
        self.killed = False

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        self.polls += 1
        if (
            self.polls_before_exit is not None
            and self.polls >= self.polls_before_exit
        ):
            self.returncode = self.exit_code
        return self.returncode

    def terminate(self) -> None:
        self.term_count += 1
        self.terminated_at = self.clock.now()
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _log_to(lines: list[str]):
    def log(message: str) -> None:
        lines.append(message)
    return log


def test_launch_at_2200_starts_child_immediately():
    clock = FakeClock(at(22, 0))
    child = FakeChild(clock=clock, polls_before_exit=None)
    order: list[tuple[str, datetime]] = []
    real_sleep = clock.sleep

    def sleep(seconds: float, *, interruptible: bool = True) -> None:
        order.append(("sleep", clock.now()))
        real_sleep(seconds, interruptible=interruptible)

    clock.sleep = sleep  # type: ignore[method-assign]

    def spawn():
        order.append(("spawn", clock.now()))
        return child

    rc = hs.supervise(
        quiet_start=QUIET_START,
        quiet_end=QUIET_END,
        spawn=spawn,
        clock=clock,
        log=_log_to([]),
        poll_seconds=15.0,
    )

    assert rc == 0
    assert order[0] == ("spawn", at(22, 0))
    assert [kind for kind, _ in order].count("spawn") == 1
    assert child.term_count == 1
    assert child.killed is False
    assert child.terminated_at == at(23, 30)


def test_launch_at_0200_sleeps_until_0600_and_starts_nothing():
    spawned: list[datetime] = []
    lines: list[str] = []

    class StopWhenAllowed(FakeClock):
        def sleep(self, seconds: float, *, interruptible: bool = True) -> None:
            super().sleep(seconds, interruptible=interruptible)
            if not hs.in_quiet(self.now(), QUIET_START, QUIET_END):
                raise RuntimeError("window opened")

    clock = StopWhenAllowed(at(2, 0))

    def spawn():
        spawned.append(clock.now())
        raise AssertionError("started a child during the quiet window")

    decision = hs.describe_decision(at(2, 0), QUIET_START, QUIET_END)
    assert "start nothing" in decision
    assert "06:00" in decision

    with pytest.raises(RuntimeError, match="window opened"):
        hs.supervise(
            quiet_start=QUIET_START,
            quiet_end=QUIET_END,
            spawn=spawn,
            clock=clock,
            log=_log_to(lines),
        )

    assert spawned == []
    assert clock.slept == [4 * 3600]
    assert clock.now() == at(6, 0)
    assert any("starting nothing" in line for line in lines)


def test_child_running_at_2329_is_terminated_at_2330():
    clock = FakeClock(at(23, 29))
    child = FakeChild(clock=clock, polls_before_exit=None)
    lines: list[str] = []

    def spawn():
        assert clock.now() == at(23, 29)
        return child

    rc = hs.supervise(
        quiet_start=QUIET_START,
        quiet_end=QUIET_END,
        spawn=spawn,
        clock=clock,
        log=_log_to(lines),
        poll_seconds=30.0,
    )

    assert rc == 0
    assert child.term_count == 1
    assert child.terminated_at == at(23, 30)
    assert child.killed is False
    assert clock.now() == at(23, 30)
    text = "\n".join(lines)
    assert "SIGTERM" in text
    assert "no SIGKILL" in text
    decision = hs.describe_decision(
        at(23, 29), QUIET_START, QUIET_END, child_running=True
    )
    assert "SIGTERM at 23:30" in decision
    assert "no SIGKILL" in decision


def test_nonzero_exit_restarts_after_backoff_and_cap():
    clock = FakeClock(at(12, 0))
    spawned: list[datetime] = []

    def spawn():
        spawned.append(clock.now())
        return FakeChild(clock=clock, exit_code=1, polls_before_exit=1)

    rc = hs.supervise(
        quiet_start=QUIET_START,
        quiet_end=QUIET_END,
        spawn=spawn,
        clock=clock,
        log=_log_to([]),
        max_restarts=2,
        backoff=10.0,
        backoff_cap=100.0,
        poll_seconds=30.0,
    )

    assert rc == 1
    assert spawned == [at(12, 0), at(12, 0, second=10), at(12, 0, second=30)]
    assert clock.slept == [10.0, 20.0]
    assert hs.backoff_delay(1, 30.0, 60.0) == 30.0
    assert hs.backoff_delay(2, 30.0, 60.0) == 60.0
    assert hs.backoff_delay(8, 30.0, 60.0) == 60.0


def test_quiet_window_wraps_midnight():
    duration = hs.quiet_duration(QUIET_START, QUIET_END)
    assert duration == timedelta(hours=6, minutes=30)
    assert duration > timedelta(0)
    allowed = timedelta(hours=24) - duration
    assert allowed == timedelta(hours=17, minutes=30)

    inside = [(23, 30), (23, 59), (0, 0), (2, 0), (5, 59)]
    outside = [(6, 0), (6, 1), (12, 0), (22, 0), (23, 29)]
    for hour, minute in inside:
        assert hs.in_quiet(at(hour, minute), QUIET_START, QUIET_END), (hour, minute)
    for hour, minute in outside:
        assert not hs.in_quiet(at(hour, minute), QUIET_START, QUIET_END), (hour, minute)

    assert hs.next_quiet_end(at(23, 30), QUIET_END) == at(6, 0, day=27)
    assert hs.next_quiet_end(at(2, 0), QUIET_END) == at(6, 0)
    assert hs.next_quiet_start(at(22, 0), QUIET_START, QUIET_END) == at(23, 30)
    assert hs.next_quiet_start(at(6, 0), QUIET_START, QUIET_END) == at(23, 30)

    # A same-day window must not be rewritten into the midnight wrap.
    midday_start, midday_end = dt_time(12, 0), dt_time(13, 0)
    assert hs.quiet_duration(midday_start, midday_end) == timedelta(hours=1)
    assert hs.in_quiet(at(12, 30), midday_start, midday_end)
    assert not hs.in_quiet(at(2, 0), midday_start, midday_end)
    assert not hs.in_quiet(at(23, 30), midday_start, midday_end)

    # Equal endpoints are empty, which is how 23:30–06:00 must not be treated.
    assert hs.quiet_duration(dt_time(6, 0), dt_time(6, 0)) == timedelta(0)
    assert not hs.in_quiet(at(6, 0), dt_time(6, 0), dt_time(6, 0))
    assert not hs.in_quiet(at(2, 0), dt_time(6, 0), dt_time(6, 0))


def test_dry_run_prints_schedule(capsys):
    rc = hs.main(
        [
            "--dry-run",
            "--at", "22:00",
            "--at", "02:00",
            "--quiet-start", "23:30",
            "--quiet-end", "06:00",
            "--cutoff", "2025-06-01",
            "--max-docs", "10",
            "--delay", "3",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "would start the harvester immediately" in captured.out
    assert "start nothing" in captured.out
    assert "--resume" in captured.out
    assert "6.5 h" in captured.out
    assert "17.5 h" in captured.out
    assert captured.err == ""


def test_command_passes_resume_and_overrides(tmp_path):
    args = hs.build_parser().parse_args(
        [
            "--cutoff", "2025-06-01",
            "--until", "2026-11-26",
            "--delay", "4.5",
            "--max-docs", "20",
            "--out", str(tmp_path / "biorxiv_2026.jsonl"),
            "--server", "medrxiv",
        ]
    )
    command = hs.build_harvest_command(args, Path(__file__).resolve().parents[1])
    assert "--resume" in command
    assert command[command.index("--delay") + 1] == "4.5"
    assert command[command.index("--max-docs") + 1] == "20"
    assert command[command.index("--until") + 1] == "2026-11-26"
    assert command[command.index("--cutoff") + 1] == "2025-06-01"
    assert command[command.index("--server") + 1] == "medrxiv"
    assert "--no-resume" not in command
