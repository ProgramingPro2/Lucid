"""Flock-backed tests for scripts/gpu_lock.py. No GPU and no sudo."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gpu_lock.py"
TABLE_WIDTH = 16

FAKE_SMI = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

spec = Path(os.environ["GPU_LOCK_FAKE_SPEC"]).read_text().splitlines()
joined = " ".join(sys.argv)
if "compute-apps" in joined:
    if len(spec) > 1 and spec[1].strip():
        print(spec[1].strip())
    raise SystemExit(0)
if os.environ.get("GPU_LOCK_FAKE_FAIL") == "1":
    raise SystemExit(1)
print(spec[0].strip())
"""


def _parse_status(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) < TABLE_WIDTH:
            continue
        key = line[:TABLE_WIDTH].strip()
        value = line[TABLE_WIDTH:].strip()
        if key and key != "field":
            rows[key] = value
    return rows


def _write_fake(directory: Path, spec: str) -> Path:
    path = directory / "fake-nvidia-smi.py"
    path.write_text(FAKE_SMI)
    path.chmod(0o755)
    spec_path = directory / "smi-spec.txt"
    spec_path.write_text(spec if spec.endswith("\n") else spec + "\n")
    return path


def _env(spec_path: Path, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GPU_LOCK_FAKE_SPEC"] = str(spec_path)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(extra)
    return env


def _status(lock_dir: Path, fake: Path, spec_path: Path) -> dict[str, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--status",
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(fake),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(spec_path),
    )
    assert proc.returncode == 0, proc.stderr
    return _parse_status(proc.stdout)


def _wait_holder(lock_dir: Path, fake: Path, spec_path: Path, timeout: float = 5.0) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--status",
                "--lock-dir",
                str(lock_dir),
                "--nvidia-smi",
                str(fake),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_env(spec_path),
        )
        last = proc.stdout + proc.stderr
        rows = _parse_status(proc.stdout)
        if rows.get("holder_pid") not in (None, "none"):
            return rows
        time.sleep(0.05)
    raise AssertionError(f"no holder appeared\n{last}")


def test_two_acquires_serialize_and_report_queue_position(tmp_path: Path) -> None:
    fake = _write_fake(tmp_path, "24576, 8000, 16576")
    spec = tmp_path / "smi-spec.txt"
    lock_dir = tmp_path / "locks"
    log = tmp_path / "order.txt"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys, time, pathlib\n"
        "path, token, delay = sys.argv[1:]\n"
        "handle = pathlib.Path(path).open('a')\n"
        "handle.write(token + '1\\n')\n"
        "handle.flush()\n"
        "time.sleep(float(delay))\n"
        "handle.write(token + '2\\n')\n"
        "handle.flush()\n"
        "handle.close()\n"
    )
    common = [
        sys.executable,
        str(SCRIPT),
        "--lock-dir",
        str(lock_dir),
        "--nvidia-smi",
        str(fake),
        "--poll-seconds",
        "0.05",
        "--require-free-mb",
        "1000",
        "--wait",
        "--timeout",
        "10",
        "--",
        sys.executable,
        str(worker),
        str(log),
    ]
    first = subprocess.Popen(
        common + ["A", "0.8"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(spec),
    )
    held = _wait_holder(lock_dir, fake, spec)
    assert held["holder_pid"] == str(first.pid)
    assert held["free_mb"] == "16576"
    second = subprocess.Popen(
        common + ["B", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(spec),
    )
    first_out, first_err = first.communicate(timeout=10)
    second_out, second_err = second.communicate(timeout=10)
    assert first.returncode == 0, first_err
    assert second.returncode == 0, second_err
    order = [line for line in log.read_text().splitlines() if line]
    assert order == ["A1", "A2", "B1", "B2"], order
    assert "queue position 1" in second_out
    assert f"holder pid {first.pid}" in second_out
    assert "acquired" in first_out


def test_crashed_holder_lock_is_recoverable(tmp_path: Path) -> None:
    fake = _write_fake(tmp_path, "24576, 8000, 16576")
    spec = tmp_path / "smi-spec.txt"
    lock_dir = tmp_path / "locks"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(fake),
            "--poll-seconds",
            "0.05",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(spec),
        start_new_session=True,
    )
    rows = _wait_holder(lock_dir, fake, spec)
    assert rows["holder_pid"] == str(holder.pid)
    os.killpg(holder.pid, signal.SIGKILL)
    holder.communicate(timeout=5)
    after = _status(lock_dir, fake, spec)
    assert after["holder_pid"] == "none"
    assert after["holder_age"] == "-"
    # The killed process left its holder file behind. Status must not report it.
    assert (lock_dir / "gpu.holder").exists()
    sentinel = tmp_path / "recovered"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(fake),
            "--poll-seconds",
            "0.05",
            "--",
            sys.executable,
            "-c",
            f"open(r'{sentinel}', 'w').write('ok')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(spec),
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert sentinel.read_text() == "ok"
    released = _status(lock_dir, fake, spec)
    assert released["holder_pid"] == "none"


def test_free_memory_guard_refuses_when_the_card_is_too_full(tmp_path: Path) -> None:
    fake = _write_fake(tmp_path, "24576, 23576, 1000")
    spec = tmp_path / "smi-spec.txt"
    lock_dir = tmp_path / "locks"
    sentinel = tmp_path / "should-not-run"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(fake),
            "--require-free-mb",
            "8000",
            "--",
            sys.executable,
            "-c",
            f"open(r'{sentinel}', 'w').write('ran')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(spec),
        timeout=10,
    )
    assert proc.returncode == 2
    assert "refusing to start" in proc.stderr
    assert "1000" in proc.stderr
    assert "8000" in proc.stderr
    assert not sentinel.exists()
    rows = _status(lock_dir, fake, spec)
    assert rows["holder_pid"] == "none"
    assert rows["free_mb"] == "1000"

    missing = tmp_path / "missing-smi"
    blinded = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(missing),
            "--require-free-mb",
            "1",
            "--",
            sys.executable,
            "-c",
            f"open(r'{sentinel}', 'w').write('ran')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(spec),
        timeout=10,
    )
    assert blinded.returncode == 2
    assert "unknown" in blinded.stderr
    assert not sentinel.exists()


def test_status_reports_no_holder_when_the_holder_file_is_stale(tmp_path: Path) -> None:
    fake = _write_fake(tmp_path, "24576, 7309, 16807")
    spec = tmp_path / "smi-spec.txt"
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    (lock_dir / "gpu.holder").write_text(
        json.dumps(
            {
                "pid": 2147483646,
                "child_pid": 2147483645,
                "started_unix": time.time() - 3600,
                "command": ["phantom-train"],
                "state": "running",
            }
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--status",
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(fake),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(spec),
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    rows = _parse_status(proc.stdout)
    assert rows["holder_pid"] == "none"
    assert rows["holder_age"] == "-"
    assert rows["child_pid"] == "none"
    assert rows["free_mb"] == "16807"
    assert rows["total_mb"] == "24576"
    assert rows["used_mb"] == "7309"
    assert "phantom-train" not in proc.stdout
    # Status is read-only: the stale file is still there, and no lock was taken.
    assert (lock_dir / "gpu.holder").exists()
    assert _status(lock_dir, fake, spec)["holder_pid"] == "none"


def test_record_appends_start_and_end_with_peak(tmp_path: Path) -> None:
    fake = _write_fake(tmp_path, "24576, 8000, 16576\n")
    spec = tmp_path / "smi-spec.txt"
    lock_dir = tmp_path / "locks"
    journal = lock_dir / "gpu_journal.csv"
    sentinel = tmp_path / "ran"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock-dir",
            str(lock_dir),
            "--nvidia-smi",
            str(fake),
            "--poll-seconds",
            "0.05",
            "--record",
            "--require-free-mb",
            "1000",
            "--",
            sys.executable,
            "-c",
            f"open(r'{sentinel}', 'w').write('ok')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(spec),
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert sentinel.read_text() == "ok"
    rows = list(__import__("csv").DictReader(journal.open()))
    assert [row["event"] for row in rows] == ["start", "end"]
    assert rows[0]["peak_mb"] == ""
    assert rows[1]["peak_mb"] == "0"
    assert rows[1]["exit_code"] == "0"
    assert rows[1]["free_mb"] == "16576"
    assert rows[0]["pid"] == rows[1]["pid"]
    assert "open" in rows[1]["command"]
