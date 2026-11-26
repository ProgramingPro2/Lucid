#!/usr/bin/env python3
"""Keep the corpus harvest inside the hours this machine has wifi.

Wifi is powered off from 23:30 to 06:00 America/New_York every night. A
harvest that is still on the network at 23:30 dies mid-request. This process
is the thing that remembers that. It is standard library only.

Schedule. The quiet window is ``--quiet-start`` to ``--quiet-end`` on the
local clock (default 23:30 to 06:00). That interval wraps midnight: it is
6.5 hours, and the allowed window is the other 17.5. A launch inside the
quiet window sleeps until the window ends, then starts the harvester, so a
start at 02:00 is safe. A launch inside the allowed window starts the child
immediately. When the quiet window begins while the child is still running,
the supervisor sends SIGTERM, waits for the child to exit, and then exits.
It never sends SIGKILL. The next morning's timer starts a fresh supervisor.

Restarts. Inside the allowed window, any child exit (crash, abort, or a
clean finish) is followed by a bounded exponential backoff and a new
process with ``--resume``. Resume is per document id and per page cursor, so
a restart does not lose or duplicate a document that was fully written. After
``--max-restarts`` restarts the supervisor exits 1, so a broken run is
visible instead of looping until morning.

SIGTERM and the files (read from ``experiments/corpus/harvest.py``; this
script does not modify it):

- ``harvest.py`` does not import ``signal`` and installs no handler. SIGTERM
  uses the default terminate action. The in-flight ``write`` syscall finishes;
  the process is then killed at the next userspace instruction.
- The JSONL is append-only. Each document is one ``write`` of a JSON line
  plus a newline, then ``flush``. Completed lines are on disk.
- ``TextIOWrapper.flush`` goes out through a buffer of
  ``io.DEFAULT_BUFFER_SIZE`` (8192 bytes on this interpreter). A full-text
  document is much larger than that, so one flush is several ``write``
  calls. SIGTERM can land between those calls and leave a truncated final
  line with no newline. The next append would glue onto that partial line.
  Waiting for the child's own exit, and not following SIGTERM with SIGKILL,
  is the whole of what the supervisor can do about that. ``load_documents``
  skips a malformed final line and still raises ``JSONDecodeError`` on a
  malformed line earlier in the file. ``_load_done`` skips any line that is
  not valid JSON, so resume continues.
- The cursor sidecar is written to ``*.cursors.json.tmp`` and moved into
  place with ``Path.replace`` (atomic rename on the same filesystem). A kill
  during the temp write leaves the previous cursor file intact. The cursor
  advances only after every row of that page has been pulled by the
  consumer, which is after those rows have been flushed.

Dry run. ``--dry-run --at HH:MM`` prints the decision for a simulated local
clock and does not start a child, sleep, or open the network.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Callable, Protocol, TextIO


REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUIET_START = "23:30"
DEFAULT_QUIET_END = "06:00"
DEFAULT_OUT = "artifacts/corpus/biorxiv_2026.jsonl"
DEFAULT_LOG = "artifacts/corpus/supervisor.log"
# Operative harvest cutoff. Not the window start: with a 30-day grace the
# harvester opens on 2025-07-02, the first day the date gate keeps.
DEFAULT_CUTOFF = "2025-06-01"


class Clock(Protocol):
    def now(self) -> datetime: ...

    def sleep(self, seconds: float, *, interruptible: bool = True) -> None: ...


class Child(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class WallClock:
    """Local wall clock. Long sleeps return early once ``stop`` is set.

    ``interruptible=False`` is the wait after SIGTERM: that wait must not
    spin just because a stop was already requested, and it must not be cut
    off by a second signal. The caller still never sends SIGKILL.
    """

    def __init__(
        self,
        stop: Callable[[], bool] | None = None,
        slice_seconds: float = 5.0,
    ) -> None:
        self._stop = stop
        self._slice = slice_seconds

    def now(self) -> datetime:
        return datetime.now().astimezone()

    def sleep(self, seconds: float, *, interruptible: bool = True) -> None:
        end = time.monotonic() + max(0.0, float(seconds))
        while True:
            if interruptible and self._stop is not None and self._stop():
                return
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(self._slice, remaining))


class Log:
    def __init__(self, path: Path | None, clock: Clock, *, echo: bool = True) -> None:
        self.path = path
        self.clock = clock
        self.echo = echo
        self._fh: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode is O_APPEND, so a line from the child and a line
            # from this process do not overwrite each other.
            self._fh = path.open("a", buffering=1, encoding="utf-8")

    def __call__(self, message: str) -> None:
        stamp = self.clock.now().isoformat(timespec="seconds")
        line = f"{stamp} {message}\n"
        if self._fh is not None:
            self._fh.write(line)
            self._fh.flush()
        if self.echo:
            sys.stdout.write(line)
            sys.stdout.flush()

    def stream(self) -> TextIO | int:
        if self._fh is None:
            return subprocess.DEVNULL
        return self._fh

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


class PopenChild:
    """A harvest subprocess. ``terminate`` is SIGTERM. There is no kill method."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def terminate(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


@dataclass(frozen=True)
class Outcome:
    kind: str  # "exit" or "stop"
    rc: int | None


def parse_hhmm(text: str) -> dt_time:
    parts = text.strip().split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"expected HH:MM, got {text!r}"
        )
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected HH:MM, got {text!r}"
        ) from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise argparse.ArgumentTypeError(
            f"clock time out of range: {text!r}"
        )
    return dt_time(hour, minute, second)


def _face(now: datetime) -> dt_time:
    """Local clock face, without a timezone attached to the ``time`` object."""
    return dt_time(now.hour, now.minute, now.second, now.microsecond)


def in_quiet(now: datetime, quiet_start: dt_time, quiet_end: dt_time) -> bool:
    """True when ``now`` is inside the quiet window.

    A window with ``start > end`` wraps midnight (23:30–06:00). A window with
    ``start < end`` is the same calendar day. Equal endpoints are an empty
    window: the harvester is always allowed to run.
    """
    if quiet_start == quiet_end:
        return False
    current = _face(now)
    if quiet_start < quiet_end:
        return quiet_start <= current < quiet_end
    return current >= quiet_start or current < quiet_end


def quiet_duration(quiet_start: dt_time, quiet_end: dt_time) -> timedelta:
    """Length of the quiet window. 23:30–06:00 is 6.5 hours, not zero."""
    if quiet_start == quiet_end:
        return timedelta(0)
    start_m = quiet_start.hour * 60 + quiet_start.minute
    end_m = quiet_end.hour * 60 + quiet_end.minute
    if quiet_start < quiet_end:
        minutes = end_m - start_m
    else:
        minutes = (24 * 60 - start_m) + end_m
    return timedelta(minutes=minutes)


def _at_clock(now: datetime, clock_time: dt_time) -> datetime:
    return now.replace(
        hour=clock_time.hour,
        minute=clock_time.minute,
        second=clock_time.second,
        microsecond=0,
    )


def next_quiet_end(now: datetime, quiet_end: dt_time) -> datetime:
    """The next instant the quiet window ends, strictly after ``now`` if ``now`` is that instant."""
    candidate = _at_clock(now, quiet_end)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def next_quiet_start(
    now: datetime, quiet_start: dt_time, quiet_end: dt_time
) -> datetime:
    """When the next quiet window begins. ``now`` itself if it is already quiet."""
    if in_quiet(now, quiet_start, quiet_end):
        return now
    candidate = _at_clock(now, quiet_start)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """1-based restart count to a bounded exponential delay, with no jitter.

    Jitter would make the schedule tests depend on a seed. The harvester has
    its own jitter for HTTP retries; this delay is only between processes.
    """
    if attempt <= 0 or base <= 0 or cap <= 0:
        return 0.0
    return float(min(cap, base * (2 ** (attempt - 1))))


def describe_decision(
    now: datetime,
    quiet_start: dt_time,
    quiet_end: dt_time,
    *,
    child_running: bool = False,
) -> str:
    """One line a human can check without waiting for the real clock."""
    stamp = now.isoformat(timespec="minutes")
    if in_quiet(now, quiet_start, quiet_end):
        wake = next_quiet_end(now, quiet_end)
        return (
            f"{stamp} quiet; would sleep until {wake.isoformat(timespec='minutes')} "
            "and start nothing"
        )
    quiet_at = next_quiet_start(now, quiet_start, quiet_end)
    when = quiet_at.strftime("%H:%M")
    if child_running:
        return (
            f"{stamp} allowed; child is running; would send SIGTERM at {when} "
            "and wait for the child to exit (no SIGKILL)"
        )
    return (
        f"{stamp} allowed; would start the harvester immediately; "
        f"quiet begins at {when}"
    )


def _stopped(stop_flag: Callable[[], bool] | None) -> bool:
    return bool(stop_flag and stop_flag())


def stop_child(
    child: Child,
    log: Callable[[str], None],
    clock: Clock,
    poll_seconds: float,
) -> int | None:
    """SIGTERM, then wait until the process exits. Never SIGKILL.

    A hard kill is the case that can cut a JSONL line between buffer chunks.
    The default SIGTERM disposition already stops ``harvest.py``; the wait is
    so a flush that is inside a single ``write`` syscall can finish, and so
    we do not race a second signal into the next chunk.
    """
    log(
        "sending SIGTERM; waiting for the child to exit on its own (no SIGKILL)"
    )
    if child.poll() is None:
        child.terminate()
    waited = 0.0
    next_notice = 30.0
    while child.poll() is None:
        step = min(max(poll_seconds, 0.1), 5.0)
        clock.sleep(step, interruptible=False)
        waited += step
        if waited >= next_notice:
            log(
                f"still waiting for the child {waited:.0f}s after SIGTERM; "
                "not sending SIGKILL"
            )
            next_notice += 30.0
    rc = child.poll()
    log(f"child exited rc={rc}")
    return rc


def _monitor(
    child: Child,
    clock: Clock,
    log: Callable[[str], None],
    quiet_start: dt_time,
    quiet_end: dt_time,
    poll_seconds: float,
    stop_flag: Callable[[], bool] | None,
) -> Outcome:
    while True:
        if _stopped(stop_flag):
            stop_child(child, log, clock, poll_seconds)
            return Outcome("stop", child.poll())
        now = clock.now()
        if in_quiet(now, quiet_start, quiet_end):
            log("quiet window began while the harvester was running")
            stop_child(child, log, clock, poll_seconds)
            return Outcome("stop", child.poll())
        rc = child.poll()
        if rc is not None:
            return Outcome("exit", rc)
        quiet_at = next_quiet_start(now, quiet_start, quiet_end)
        remaining = (quiet_at - now).total_seconds()
        if remaining <= 0:
            log("quiet window began while the harvester was running")
            stop_child(child, log, clock, poll_seconds)
            return Outcome("stop", child.poll())
        clock.sleep(min(poll_seconds, remaining))


def supervise(
    *,
    quiet_start: dt_time,
    quiet_end: dt_time,
    spawn: Callable[[], Child],
    clock: Clock,
    log: Callable[[str], None],
    max_restarts: int = 8,
    backoff: float = 30.0,
    backoff_cap: float = 600.0,
    poll_seconds: float = 15.0,
    stop_flag: Callable[[], bool] | None = None,
) -> int:
    """Run the harvest inside the allowed window. See the module docstring."""
    if max_restarts < 0:
        raise ValueError("max_restarts must be >= 0")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be > 0")
    restarts = 0
    while True:
        if _stopped(stop_flag):
            log("stop requested; exiting")
            return 0
        now = clock.now()
        if in_quiet(now, quiet_start, quiet_end):
            wake = next_quiet_end(now, quiet_end)
            log(
                "inside quiet window; sleeping until "
                f"{wake.isoformat(timespec='seconds')} and starting nothing"
            )
            clock.sleep((wake - now).total_seconds())
            if _stopped(stop_flag):
                log("stop requested during the quiet window; exiting without starting")
                return 0
            continue

        log("starting harvester")
        child = spawn()
        outcome = _monitor(
            child,
            clock,
            log,
            quiet_start,
            quiet_end,
            poll_seconds,
            stop_flag,
        )
        if outcome.kind == "stop":
            log("harvester stopped for the quiet window; supervisor exiting")
            return 0

        rc = outcome.rc
        restarts += 1
        if restarts > max_restarts:
            log(
                f"restart cap {max_restarts} reached after rc={rc}; exiting 1"
            )
            return 1
        delay = backoff_delay(restarts, backoff, backoff_cap)
        now = clock.now()
        quiet_at = next_quiet_start(now, quiet_start, quiet_end)
        until_quiet = (quiet_at - now).total_seconds()
        if until_quiet <= delay:
            log(
                f"harvester exited rc={rc}; backoff of {delay:.0f}s would run into "
                "the quiet window; stopping for the night"
            )
            if until_quiet > 0:
                clock.sleep(until_quiet)
            return 0
        log(
            f"harvester exited rc={rc}; restart {restarts}/{max_restarts} "
            f"after {delay:.0f}s"
        )
        clock.sleep(delay)


def build_harvest_command(args: argparse.Namespace, repo: Path) -> list[str]:
    python = args.python
    if not python:
        venv_python = repo / ".venv" / "bin" / "python"
        python = str(venv_python if venv_python.exists() else Path(sys.executable))
    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    cmd = [
        python,
        "-m",
        "experiments.corpus.harvest",
        "--server",
        args.server,
        "--cutoff",
        args.cutoff,
        "--out",
        str(out),
        "--max-docs",
        str(args.max_docs),
        "--delay",
        str(args.delay),
        "--resume",
    ]
    if args.until:
        cmd.extend(["--until", args.until])
    if args.window_retries is not None:
        cmd.extend(["--window-retries", str(args.window_retries)])
    if args.cooldown is not None:
        cmd.extend(["--cooldown", str(args.cooldown)])
    return cmd


def child_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    parts = [str(repo), str(repo / "src")]
    prior = env.get("PYTHONPATH")
    if prior:
        parts.append(prior)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def acquire_lock(path: Path) -> TextIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def foreign_harvest_pid(out: Path) -> int | None:
    """Pid of a harvest already writing ``out``, or None.

    Read-only. Used so a morning start does not append into a JSONL that
    another process already has open. This does not signal that process.
    """
    needles = {str(out), out.name}
    try:
        needles.add(str(out.resolve()))
    except OSError:
        pass
    me = os.getpid()
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        text = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        if "experiments.corpus.harvest" not in text and "corpus/harvest.py" not in text:
            continue
        if any(needle and needle in text for needle in needles):
            return pid
    return None


def _resolve_at(text: str, today: datetime) -> datetime:
    stripped = text.strip()
    if "T" in stripped or len(stripped) > 8:
        parsed = datetime.fromisoformat(stripped)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=today.tzinfo)
        return parsed.astimezone(today.tzinfo)
    clock_time = parse_hhmm(stripped)
    return today.replace(
        hour=clock_time.hour,
        minute=clock_time.minute,
        second=clock_time.second,
        microsecond=0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the corpus harvest only while this machine has wifi, "
            "and restart it with --resume when it exits."
        )
    )
    parser.add_argument("--quiet-start", type=parse_hhmm, default=DEFAULT_QUIET_START)
    parser.add_argument("--quiet-end", type=parse_hhmm, default=DEFAULT_QUIET_END)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between harvest requests")
    parser.add_argument("--max-docs", type=int, default=1_000_000)
    parser.add_argument(
        "--until",
        default=None,
        help="ISO date passed to harvest; omit to let harvest use today at each start",
    )
    parser.add_argument(
        "--cutoff",
        default=DEFAULT_CUTOFF,
        help=(
            "ISO date or cutoff JSON, passed through to harvest. "
            "Windows open the day after this date plus 30 days, the first "
            "day the date gate keeps (2025-06-01 opens on 2025-07-02). "
            "Do not move this date to chase the window."
        ),
    )
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--out", default=DEFAULT_OUT, help="harvest JSONL path")
    parser.add_argument("--server", choices=["biorxiv", "medrxiv"], default="biorxiv")
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument("--backoff", type=float, default=30.0, help="initial restart backoff, seconds")
    parser.add_argument("--backoff-cap", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--python", default=None, help="interpreter used to launch harvest")
    parser.add_argument("--window-retries", type=int, default=None)
    parser.add_argument("--cooldown", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print schedule decisions for --at and do not start a child",
    )
    parser.add_argument(
        "--at",
        action="append",
        default=None,
        help="simulated local time HH:MM or ISO datetime; repeatable; used by --dry-run",
    )
    parser.add_argument(
        "--child-running",
        action="store_true",
        help="dry-run: assume the harvester is already running at each --at",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        now = datetime.now().astimezone()
        moments = [_resolve_at(item, now) for item in (args.at or [])] or [now]
        duration = quiet_duration(args.quiet_start, args.quiet_end)
        hours = duration.total_seconds() / 3600.0
        print(
            f"quiet {args.quiet_start.strftime('%H:%M')}-{args.quiet_end.strftime('%H:%M')} "
            f"local is {hours:.1f} h; allowed window is {24.0 - hours:.1f} h"
        )
        command = build_harvest_command(args, REPO)
        print("would run: " + " ".join(command))
        for moment in moments:
            print(
                describe_decision(
                    moment,
                    args.quiet_start,
                    args.quiet_end,
                    child_running=args.child_running,
                )
            )
        return 0

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = REPO / log_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path

    state = {"stop": False}

    def stop_flag() -> bool:
        return state["stop"]

    def handle(signum: int, _frame: object) -> None:
        state["stop"] = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    clock = WallClock(stop=stop_flag)
    log = Log(log_path, clock, echo=True)
    lock = acquire_lock(log_path.with_suffix(log_path.suffix + ".lock"))
    if lock is None:
        log("another supervisor holds the lock; exiting")
        log.close()
        return 0

    existing = foreign_harvest_pid(out_path)
    if existing is not None:
        log(
            f"harvest pid {existing} is already writing {out_path}; "
            "not starting a second writer"
        )
        log.close()
        return 0

    def spawn() -> PopenChild:
        command = build_harvest_command(args, REPO)
        log("command: " + " ".join(command))
        proc = subprocess.Popen(
            command,
            cwd=str(REPO),
            env=child_env(REPO),
            stdout=log.stream(),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log(f"harvester pid={proc.pid}")
        return PopenChild(proc)

    log(
        "supervisor up; quiet "
        f"{args.quiet_start.strftime('%H:%M')}-{args.quiet_end.strftime('%H:%M')} "
        f"local; out={out_path}"
    )
    try:
        return supervise(
            quiet_start=args.quiet_start,
            quiet_end=args.quiet_end,
            spawn=spawn,
            clock=clock,
            log=log,
            max_restarts=args.max_restarts,
            backoff=args.backoff,
            backoff_cap=args.backoff_cap,
            poll_seconds=args.poll_seconds,
            stop_flag=stop_flag,
        )
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
