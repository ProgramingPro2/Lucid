"""Per-run reproducibility manifests (docs/EXPERIMENTS.md §9 and §11).

A run that cannot be reproduced from its ID and seed does not count. The
manifest is the artefact that makes that checkable: it pins the code
(``git_commit``), the configuration (``config_sha256``), the data order
(``corpus_sha256``), all six spawn keys, the token budget, and the operational
record (final checkpoint, wall clock, peak VRAM, sanitised microbatches).

Two rules from the docs drive the design:

- Only the **final** checkpoint is analysed, so ``final_checkpoint`` is a
  required field rather than a directory listing (§9).
- A run missing **any** required field is excluded from analysis and reported
  as excluded (§11). :func:`partition_manifests` implements that split;
  :meth:`RunManifest.from_json` refuses to construct a manifest in the first
  place when a field is absent, so incompleteness is caught at ingest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .seeds import STREAMS, spawn_key

#: Fields that must be present for a manifest to enter the analysis set.
#: Order is the order used in error messages and the reproducibility record.
REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "git_commit",
    "config_sha256",
    "corpus_sha256",
    "spawn_keys",
    "token_budget",
    "final_checkpoint",
    "wall_clock_s",
    "peak_vram_gb",
    "nan_skips",
    "arms",
    "notes",
    "git_dirty",
    "git_diff_sha256",
    "determinism",
)

#: Keys :func:`experiments.e1.train.configure_determinism` records. A manifest
#: that omits one of them does not say which kernels the run actually used.
DETERMINISM_FIELDS: tuple[str, ...] = (
    "enabled",
    "deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "matmul_allow_tf32",
    "cudnn_allow_tf32",
    "cublas_workspace_config",
    "float32_matmul_precision",
)

_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class RunManifest:
    """The reproducibility record for one run (docs/EXPERIMENTS.md §11).

    ``spawn_keys`` maps each of the six protocol streams to the 64-bit key
    derived from the run's seed. It is stored rather than recomputed so that a
    manifest remains self-describing even if the derivation code later changes:
    a changed key is then a visible protocol amendment, not a silent rerun.
    """

    run_id: str
    git_commit: str
    config_sha256: str
    corpus_sha256: str
    spawn_keys: dict[str, int]
    token_budget: int
    final_checkpoint: str
    wall_clock_s: float
    peak_vram_gb: float
    nan_skips: int
    arms: list[str]
    notes: str = ""
    # ``git_commit`` alone names a tree the run may not have executed. A dirty
    # worktree is excluded: the diff is not stored, so ``git_diff_sha256``
    # cannot be turned back into the code that ran.
    git_dirty: bool = False
    git_diff_sha256: str = ""
    determinism: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as plain JSON-compatible data."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write a stable, human-diffable representation to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        target.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        """Build a manifest, rejecting any record that is missing a field.

        The error lists every missing name at once so an operator can repair
        one record rather than rediscovering the schema one field per attempt.
        """
        missing = [name for name in REQUIRED_FIELDS if name not in data]
        if missing:
            raise ValueError(
                f"{cls.__name__} is missing required field(s): {', '.join(missing)}"
            )
        return cls(**{name: data[name] for name in REQUIRED_FIELDS})

    @classmethod
    def from_json(cls, path: str | Path) -> "RunManifest":
        """Read a manifest written by :meth:`to_json`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"{cls.__name__} JSON must be an object, got {type(raw).__name__}"
            )
        return cls.from_dict(raw)


def spawn_keys_for_seed(seed: int) -> dict[str, int]:
    """Return the six spawn keys a run with ``seed`` must record.

    This is the producer-side helper. It exists so no training script derives
    its streams locally: the manifest and the trainer call the same function.
    """
    return {name: spawn_key(seed, name) for name in STREAMS}


@dataclass(frozen=True)
class GitState:
    """What ``git_commit`` does and does not describe.

    ``diff_sha256`` is the SHA-256 of the raw bytes of ``git diff HEAD``
    (staged and unstaged changes to tracked files). Untracked files do not
    appear in that diff; they still set ``dirty`` via ``git status
    --porcelain``. A clean tree hashes the empty diff.
    """

    commit: str
    dirty: bool
    diff_sha256: str


def capture_git_state(repo_root: str | Path = ".") -> GitState:
    """Record HEAD, whether the worktree differs from it, and the diff hash.

    A failure (not a repo, git missing, timeout) is recorded as dirty with a
    non-hash ``diff_sha256`` of ``"unknown"``. That manifest fails completeness
    and is excluded, which is the safe side: an unreadable tree is not a
    reproducible one. This function does not raise.
    """
    commit = capture_git_commit(repo_root)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            timeout=30,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return GitState(commit, True, "unknown")
    if status.returncode != 0 or diff.returncode != 0:
        return GitState(commit, True, "unknown")
    dirty = bool(status.stdout.strip()) or bool(diff.stdout)
    return GitState(commit, dirty, hashlib.sha256(diff.stdout).hexdigest())


def capture_git_commit(repo_root: str | Path = ".") -> str:
    """Return ``git rev-parse HEAD`` for ``repo_root``, or ``"unknown"``.

    Recording a commit must never be able to kill a run, so every failure mode
    (not a repo, missing git, timeout) degrades to ``"unknown"``. A manifest
    carrying ``"unknown"`` is still parseable; the exclusion decision is left
    to the operator rather than hidden in an exception here.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    commit = proc.stdout.strip()
    return commit or "unknown"


def hash_config(obj: Any) -> str:
    """SHA-256 of canonical JSON for a config object.

    Sorting keys and stripping whitespace makes the hash order-insensitive, so
    reordering a YAML file does not look like a new analysis. ``allow_nan`` is
    off because ``NaN`` has no canonical JSON spelling; a config containing one
    is a bug worth failing on.
    """
    canonical = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_DIGITS for ch in value)
    )


def validate_completeness(manifest: RunManifest) -> list[str]:
    """Return the list of completeness problems with ``manifest``.

    An empty list means the run is eligible for analysis. The checks cover the
    reproducibility fields §11 calls out, not statistical content: this module
    has no opinion on whether a result is good, only on whether the record is
    trustworthy enough to analyse.
    """
    problems: list[str] = []

    if not manifest.run_id:
        problems.append("run_id is empty")

    if not _is_sha256_hex(manifest.config_sha256):
        problems.append(
            "config_sha256 is not a 64-character lowercase hex digest"
        )
    if not _is_sha256_hex(manifest.corpus_sha256):
        problems.append(
            "corpus_sha256 is not a 64-character lowercase hex digest"
        )

    keys = manifest.spawn_keys
    if not isinstance(keys, Mapping):
        problems.append("spawn_keys is not a mapping of stream -> int")
    else:
        missing = [name for name in STREAMS if name not in keys]
        if missing:
            problems.append(
                "spawn_keys is missing stream(s): " + ", ".join(missing)
            )
        unknown = sorted(str(name) for name in keys if name not in STREAMS)
        if unknown:
            problems.append(
                "spawn_keys contains unknown stream(s): " + ", ".join(unknown)
            )
        for name, value in keys.items():
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"spawn_keys[{name!r}] is not an integer")
            elif value < 0 or value >= 1 << 64:
                problems.append(
                    f"spawn_keys[{name!r}] is outside the 64-bit range: {value}"
                )

    if manifest.token_budget <= 0:
        problems.append(f"token_budget must be positive, got {manifest.token_budget}")

    if not manifest.final_checkpoint:
        problems.append("final_checkpoint is empty")

    if manifest.wall_clock_s < 0:
        problems.append(f"wall_clock_s must be non-negative, got {manifest.wall_clock_s}")

    if manifest.peak_vram_gb < 0:
        problems.append(f"peak_vram_gb must be non-negative, got {manifest.peak_vram_gb}")

    if manifest.nan_skips < 0:
        problems.append(f"nan_skips must be non-negative, got {manifest.nan_skips}")

    if not manifest.arms:
        problems.append("arms is empty; a run must name at least one arm")

    if not isinstance(manifest.git_dirty, bool):
        problems.append("git_dirty is not a boolean")
    elif manifest.git_dirty:
        # The commit names a tree the process did not run. The diff hash
        # identifies the patch and does not contain it, so the run cannot be
        # reproduced from the manifest.
        problems.append(
            "git working tree was dirty; git_commit does not describe the "
            "code and git_diff_sha256 cannot be reproduced from the commit"
        )
    if not _is_sha256_hex(manifest.git_diff_sha256):
        problems.append(
            "git_diff_sha256 is not a 64-character lowercase hex digest"
        )

    flags = manifest.determinism
    if not isinstance(flags, Mapping):
        problems.append("determinism flags are missing")
    else:
        missing_flags = [name for name in DETERMINISM_FIELDS if name not in flags]
        if missing_flags:
            problems.append(
                "determinism is missing flag(s): " + ", ".join(missing_flags)
            )

    return problems


def partition_manifests(
    manifests: Iterable[RunManifest],
) -> tuple[list[RunManifest], list[tuple[RunManifest, list[str]]]]:
    """Split manifests into ``(included, excluded_with_reasons)``.

    Implements docs/EXPERIMENTS.md §11: an incomplete run is not dropped
    silently, it is carried into the report *with its reasons* so the exclusion
    itself is auditable. Missing fields are rejected earlier, at
    :meth:`RunManifest.from_json`.
    """
    included: list[RunManifest] = []
    excluded: list[tuple[RunManifest, list[str]]] = []
    for manifest in manifests:
        problems = validate_completeness(manifest)
        if problems:
            excluded.append((manifest, problems))
        else:
            included.append(manifest)
    return included, excluded
