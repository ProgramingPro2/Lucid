#!/usr/bin/env python3
"""Serialize GPU jobs on the single shared card.

The lock is an ``flock`` on ``artifacts/.locks/gpu.lock``. The kernel drops
that lock when the holder exits, including a crash or SIGKILL, so the next
acquire recovers it. Status reads ``/proc/locks`` and does not call ``flock``,
so a status check does not take or release a live holder's lock.

Stdlib only. No sudo. A training command wraps itself:

    python scripts/gpu_lock.py --require-free-mb 14000 --wait --timeout 3600 \\
        --record -- python -m experiments.e1.train ...

``--require-free-mb`` asks nvidia-smi for free memory after the flock is held.
Below the bar the wrapper unlocks and either refuses or, with ``--wait``,
keeps the place in the queue and retries until ``--timeout``. A failed
nvidia-smi query is treated as "free memory unknown" and refuses: a job that
cannot see the card does not start.

``--record`` appends a start row and an end row (with peak process MiB) to
``artifacts/.locks/gpu_journal.csv``.
"""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_DIR = REPO / "artifacts" / ".locks"
LOCK_NAME = "gpu.lock"
HOLDER_NAME = "gpu.holder"
JOURNAL_NAME = "gpu_journal.csv"
QUEUE_NAME = "queue"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REFUSED = 2

JOURNAL_FIELDS = (
    "timestamp",
    "event",
    "pid",
    "peak_mb",
    "free_mb",
    "exit_code",
    "command",
)

TABLE_WIDTH = 16


@dataclass
class GpuSnapshot:
    total_mb: int
    used_mb: int
    free_mb: int


@dataclass
class Hold:
    fd: int
    lock_path: Path
    holder_path: Path
    queue_file: Path | None

    def release(self) -> None:
        if self.holder_path.exists():
            try:
                self.holder_path.unlink()
            except OSError:
                pass
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = -1
        if self.queue_file is not None:
            try:
                self.queue_file.unlink()
            except OSError:
                pass
            self.queue_file = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        stat = (proc / "stat").read_text().split()
    except OSError:
        return False
    # state is the third field. Z means the pid has not been reaped.
    return len(stat) < 3 or stat[2] != "Z"


def _proc_start(pid: int) -> float | None:
    try:
        return Path(f"/proc/{pid}").stat().st_mtime
    except OSError:
        return None


def _format_age(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _dev_token(path: Path) -> str:
    st = path.stat()
    return f"{os.major(st.st_dev):x}:{os.minor(st.st_dev):02x}:{st.st_ino}"


def flock_holder_pid(lock_path: Path) -> int | None:
    """Pid that currently holds the flock, or None.

    Reads ``/proc/locks`` only. Does not open the lock with ``LOCK_EX``.
    """
    if not lock_path.exists():
        return None
    try:
        token = _dev_token(lock_path)
        text = Path("/proc/locks").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        if parts[1] != "FLOCK" or parts[3] != "WRITE":
            continue
        if parts[5] != token:
            continue
        try:
            return int(parts[4])
        except ValueError:
            return None
    return None


def _read_holder(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_holder(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def query_gpu(nvidia_smi: str) -> GpuSnapshot | None:
    cmd = [
        nvidia_smi,
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    cells = [cell.strip() for cell in lines[0].split(",")]
    if len(cells) < 3:
        return None
    try:
        total, used, free = (int(cells[0]), int(cells[1]), int(cells[2]))
    except ValueError:
        return None
    return GpuSnapshot(total_mb=total, used_mb=used, free_mb=free)


def query_process_memory(nvidia_smi: str) -> dict[int, int] | None:
    cmd = [
        nvidia_smi,
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    found: dict[int, int] = {}
    for line in proc.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2 or not cells[0]:
            continue
        try:
            found[int(cells[0])] = int(cells[1])
        except ValueError:
            continue
    return found


def _descendant_pids(root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return {root}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "status").read_text()
        except OSError:
            continue
        ppid = None
        for line in text.splitlines():
            if line.startswith("PPid:"):
                try:
                    ppid = int(line.split()[1])
                except (IndexError, ValueError):
                    ppid = None
                break
        if ppid is None:
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def peak_of(nvidia_smi: str, root_pid: int) -> int | None:
    usage = query_process_memory(nvidia_smi)
    if usage is None:
        return None
    pids = _descendant_pids(root_pid)
    return sum(usage.get(pid, 0) for pid in pids)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _queue_path(queue_dir: Path) -> Path:
    _ensure_dir(queue_dir)
    return queue_dir / f"{time.time_ns():020d}-{os.getpid()}"


def _queue_position(queue_dir: Path, self_name: str) -> int:
    names: list[str] = []
    try:
        entries = list(queue_dir.iterdir())
    except OSError:
        return 1
    for entry in entries:
        parts = entry.name.rsplit("-", 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if entry.name != self_name and not _pid_alive(pid):
            continue
        names.append(entry.name)
    names.sort()
    if self_name not in names:
        return len(names) + 1
    return names.index(self_name) + 1


def _open_lock(lock_path: Path) -> int:
    _ensure_dir(lock_path.parent)
    return os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)


def _memory_message(free: GpuSnapshot | None, require_mb: int, *, waiting: bool) -> str:
    if free is None:
        observed = "unknown"
    else:
        observed = f"{free.free_mb} MiB"
    verb = "waiting" if waiting else "refusing to start"
    return (
        f"gpu_lock: {verb}; free memory is {observed} "
        f"and this job requires {require_mb} MiB"
    )


def _timed_out(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline



def acquire_or_exit(
    *,
    lock_dir: Path,
    nvidia_smi: str,
    require_free_mb: int | None,
    wait: bool,
    timeout: float | None,
    poll_seconds: float,
    command: list[str],
) -> Hold | int:
    """Take the flock, then the free-memory bar.

    The flock is held across the memory check, so two waiters cannot both
    observe the same free memory and start. A refusal unlocks before
    returning. The returned :class:`Hold` owns the fd until ``release``.
    """
    lock_path = lock_dir / LOCK_NAME
    holder_path = lock_dir / HOLDER_NAME
    queue_file = _queue_path(lock_dir / QUEUE_NAME)
    queue_file.write_text(f"{os.getpid()}\n")
    fd = _open_lock(lock_path)
    handed_off = False
    deadline = None if timeout is None else time.monotonic() + timeout
    announced: int | None = None
    try:
        while True:
            position = _queue_position(lock_dir / QUEUE_NAME, queue_file.name)
            holder = flock_holder_pid(lock_path)
            if position != announced:
                free_now = query_gpu(nvidia_smi)
                holder_text = "none" if holder is None else str(holder)
                free_text = "unknown" if free_now is None else f"{free_now.free_mb} MiB"
                need_text = "n/a" if require_free_mb is None else f"{require_free_mb} MiB"
                print(
                    f"gpu_lock: queue position {position}; "
                    f"holder pid {holder_text}; free {free_text}; need {need_text}",
                    flush=True,
                )
                announced = position
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if not wait:
                    holder_text = "unknown" if holder is None else str(holder)
                    print(
                        f"gpu_lock: refusing to start; lock held by pid {holder_text}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return EXIT_REFUSED
                if _timed_out(deadline):
                    print(
                        f"gpu_lock: timed out after {timeout}s waiting for the lock",
                        file=sys.stderr,
                        flush=True,
                    )
                    return EXIT_REFUSED
                time.sleep(poll_seconds)
                continue

            if require_free_mb is not None:
                free = query_gpu(nvidia_smi)
                short = free is None or free.free_mb < require_free_mb
                if short:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    print(
                        _memory_message(free, require_free_mb, waiting=wait),
                        file=sys.stderr,
                        flush=True,
                    )
                    if not wait:
                        return EXIT_REFUSED
                    if _timed_out(deadline):
                        print(
                            f"gpu_lock: timed out after {timeout}s waiting for free memory",
                            file=sys.stderr,
                            flush=True,
                        )
                        return EXIT_REFUSED
                    time.sleep(poll_seconds)
                    continue

            _write_holder(
                holder_path,
                {
                    "pid": os.getpid(),
                    "child_pid": None,
                    "started_unix": time.time(),
                    "command": command,
                    "state": "running",
                },
            )
            try:
                queue_file.unlink()
            except OSError:
                pass
            queue_file = None
            free = query_gpu(nvidia_smi)
            free_text = "unknown" if free is None else f"{free.free_mb} MiB"
            print(
                f"gpu_lock: acquired; holder pid {os.getpid()}; free {free_text}",
                flush=True,
            )
            handed_off = True
            return Hold(
                fd=fd,
                lock_path=lock_path,
                holder_path=holder_path,
                queue_file=None,
            )
    finally:
        if not handed_off:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            if queue_file is not None:
                try:
                    queue_file.unlink()
                except OSError:
                    pass


def _append_journal(path: Path, row: dict[str, object]) -> None:
    _ensure_dir(path.parent)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOURNAL_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in JOURNAL_FIELDS})


def _status_rows(lock_dir: Path, nvidia_smi: str) -> list[tuple[str, str]]:
    lock_path = lock_dir / LOCK_NAME
    holder_path = lock_dir / HOLDER_NAME
    pid = flock_holder_pid(lock_path) if lock_path.exists() else None
    meta = _read_holder(holder_path) if holder_path.exists() else {}
    meta_pid = meta.get("pid")
    live_meta = isinstance(meta_pid, int) and pid is not None and meta_pid == pid
    if pid is None:
        age = "-"
        child = "none"
        command = "-"
        holder = "none"
    else:
        holder = str(pid)
        started = meta.get("started_unix") if live_meta else None
        if not isinstance(started, (int, float)):
            started = _proc_start(pid)
        age = "-" if started is None else _format_age(time.time() - float(started))
        child_pid = meta.get("child_pid") if live_meta else None
        child = str(child_pid) if isinstance(child_pid, int) and child_pid > 0 else "none"
        command_value = meta.get("command") if live_meta else None
        if isinstance(command_value, list):
            command = " ".join(str(part) for part in command_value) or "-"
        else:
            command = "-"
    free = query_gpu(nvidia_smi)
    return [
        ("holder_pid", holder),
        ("holder_age", age),
        ("child_pid", child),
        ("free_mb", "unknown" if free is None else str(free.free_mb)),
        ("total_mb", "unknown" if free is None else str(free.total_mb)),
        ("used_mb", "unknown" if free is None else str(free.used_mb)),
        ("command", command),
    ]


def format_status(rows: list[tuple[str, str]]) -> str:
    lines = ["gpu_lock status", f"{'field':<{TABLE_WIDTH}}value"]
    for key, value in rows:
        lines.append(f"{key:<{TABLE_WIDTH}}{value}")
    return "\n".join(lines) + "\n"


def run_command(
    hold: Hold,
    command: list[str],
    *,
    nvidia_smi: str,
    poll_seconds: float,
    journal: Path | None,
) -> int:
    child_box: dict[str, subprocess.Popen[bytes] | None] = {"proc": None}

    def _forward(signum: int, _frame: object) -> None:
        proc = child_box["proc"]
        if proc is not None and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except OSError:
                pass

    previous_term = signal.signal(signal.SIGTERM, _forward)
    previous_int = signal.signal(signal.SIGINT, _forward)
    free_at_start = query_gpu(nvidia_smi)
    start_row_free = "" if free_at_start is None else free_at_start.free_mb
    proc = subprocess.Popen(command)
    child_box["proc"] = proc
    meta = _read_holder(hold.holder_path)
    meta["child_pid"] = proc.pid
    meta["state"] = "running"
    _write_holder(hold.holder_path, meta)
    if journal is not None:
        _append_journal(
            journal,
            {
                "timestamp": _now_iso(),
                "event": "start",
                "pid": proc.pid,
                "peak_mb": "",
                "free_mb": start_row_free,
                "exit_code": "",
                "command": " ".join(command),
            },
        )
    peak: int | None = None
    saw_sample = False
    while True:
        sample = peak_of(nvidia_smi, proc.pid)
        if sample is not None:
            saw_sample = True
            peak = sample if peak is None else max(peak, sample)
        if proc.poll() is not None:
            break
        time.sleep(poll_seconds)
    rc = proc.returncode
    assert rc is not None
    free_at_end = query_gpu(nvidia_smi)
    if journal is not None:
        _append_journal(
            journal,
            {
                "timestamp": _now_iso(),
                "event": "end",
                "pid": proc.pid,
                "peak_mb": "" if not saw_sample or peak is None else peak,
                "free_mb": "" if free_at_end is None else free_at_end.free_mb,
                "exit_code": rc,
                "command": " ".join(command),
            },
        )
    signal.signal(signal.SIGTERM, previous_term)
    signal.signal(signal.SIGINT, previous_int)
    if rc < 0:
        return 128 + abs(rc)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hold the shared GPU lock around one training command."
    )
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=DEFAULT_LOCK_DIR,
        help="directory for the flock file (default: artifacts/.locks)",
    )
    parser.add_argument(
        "--require-free-mb",
        type=int,
        default=None,
        help="refuse or wait unless nvidia-smi reports at least this much free",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="wait for the lock and for free memory instead of refusing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds --wait may block before refusing (default: wait without a limit)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.5,
        help="seconds between lock, memory, and peak probes",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print holder pid, age, and free memory, then exit",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="append start/end/peak-MB rows to the CSV journal",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="journal CSV path (default: <lock-dir>/gpu_journal.csv)",
    )
    parser.add_argument(
        "--nvidia-smi",
        default="nvidia-smi",
        help="nvidia-smi executable, or a fake that prints the same CSV",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run once the lock is held; pass it after --",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if args.require_free_mb is not None and args.require_free_mb < 0:
        print("gpu_lock: --require-free-mb must be >= 0", file=sys.stderr)
        return EXIT_USAGE
    if args.timeout is not None and args.timeout < 0:
        print("gpu_lock: --timeout must be >= 0", file=sys.stderr)
        return EXIT_USAGE
    if args.poll_seconds <= 0:
        print("gpu_lock: --poll-seconds must be > 0", file=sys.stderr)
        return EXIT_USAGE
    if args.timeout is not None and not args.wait:
        print("gpu_lock: --timeout requires --wait", file=sys.stderr)
        return EXIT_USAGE
    if args.status:
        if command or args.record:
            print("gpu_lock: --status takes no command and no --record", file=sys.stderr)
            return EXIT_USAGE
        print(
            format_status(_status_rows(args.lock_dir, args.nvidia_smi)),
            end="",
        )
        return EXIT_OK
    if not command:
        print("gpu_lock: pass a command after --", file=sys.stderr)
        return EXIT_USAGE

    acquired = acquire_or_exit(
        lock_dir=args.lock_dir,
        nvidia_smi=args.nvidia_smi,
        require_free_mb=args.require_free_mb,
        wait=args.wait,
        timeout=args.timeout,
        poll_seconds=args.poll_seconds,
        command=command,
    )
    if isinstance(acquired, int):
        return acquired
    journal = None
    if args.record:
        journal = args.journal or (args.lock_dir / JOURNAL_NAME)
    try:
        return run_command(
            acquired,
            command,
            nvidia_smi=args.nvidia_smi,
            poll_seconds=args.poll_seconds,
            journal=journal,
        )
    finally:
        acquired.release()


if __name__ == "__main__":
    sys.exit(main())
