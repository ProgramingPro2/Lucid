"""Pipeline qualification gate (docs/PROTOCOL.md §10, docs/EXPERIMENTS.md §8).

Qualification is a **VOID gate**, not an aspiration. If any check below fails,
no confirmatory answer exists for that run, and the failure is recorded as a
void rather than folded into a result. The checks are deliberately absolute
(zero sanitised microbatches, bit-identical step 0) because the protocol's
failure mode is an arm-dependent harness intervention that is invisible in the
arm means.

The module is import-safe without a GPU: torch CUDA is never touched, and the
step-0 check runs on whatever device the supplied modules and input already
live on.
"""

from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn

#: Thresholds fixed by docs/PROTOCOL.md §10. They are exported so the run
#: scripts and the report cannot drift apart.
MIN_NAN_SKIPS = 0
MIN_THROUGHPUT_TOK_S = 2000.0
MIN_MICRO_BATCH = 4
MAX_ARM_NAN_SPREAD_PP = 2.0


class NanCounter:
    """Counts microbatches discarded because their gradients were sanitised.

    The protocol requires zero, and the count must be logged *per arm*: a
    harness that skips 10 % of one arm's steps has changed that arm's training
    distribution, not merely its speed. A lock is enough here because the
    counter is incremented once per step at most; contention is irrelevant and
    correctness across dataloader threads is not.

    Use either explicitly::

        counter.record()
        counter.rate(500)

    or as a context manager that records one sanitised microbatch::

        with counter.sanitised():
            ...  # the discarded step
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0

    def record(self) -> None:
        """Record one sanitised (discarded) microbatch."""
        with self._lock:
            self._total += 1

    @property
    def total(self) -> int:
        """Number of sanitised microbatches recorded so far."""
        with self._lock:
            return self._total

    def rate(self, steps: int) -> float:
        """Sanitised microbatches per step over ``steps`` qualification steps."""
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        return self.total / steps

    def reset(self) -> None:
        """Zero the counter between qualification runs."""
        with self._lock:
            self._total = 0

    def __enter__(self) -> "NanCounter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    @contextmanager
    def sanitised(self) -> Iterator[None]:
        """Record exactly one sanitised microbatch for the enclosed block."""
        try:
            yield
        finally:
            self.record()


def check_step0_identity(
    modules: Mapping[str, nn.Module], x: torch.Tensor
) -> dict[str, bool]:
    """Compare every pair of modules' forward output on one input.

    The protocol requires **bit-identical** step-0 behaviour across A / B / E
    after zero-init, because that identity is the only proof that the inserted
    block is the sole difference between arms. ``torch.equal`` is used rather
    than ``torch.allclose``: a difference below ``allclose``'s tolerance would
    still mean the arms started from different functions and could grow into a
    treatment effect.

    Modules are temporarily switched to eval mode so dropout masks cannot make
    two identical modules look different; their original mode is restored.
    Returns one boolean per pair under ``"<a>|<b>"`` plus an overall
    ``"identical"`` flag.
    """
    if len(modules) < 2:
        raise ValueError("need at least two modules to compare step-0 identity")

    names = list(modules)
    outputs: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name, module in modules.items():
            was_training = module.training
            module.eval()
            try:
                out = module(x)
            finally:
                module.train(was_training)
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"module {name!r} returned {type(out).__name__}, expected a Tensor"
                )
            outputs[name] = out.detach()

    results: dict[str, bool] = {}
    identical = True
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            same = bool(torch.equal(outputs[a], outputs[b]))
            results[f"{a}|{b}"] = same
            identical = identical and same
    results["identical"] = identical
    return results


def assert_token_streams_identical(hashes: Mapping[str, str]) -> None:
    """Raise if the packed token-stream hash differs across arms.

    Token order is the one input every arm must share exactly; a mismatch means
    the arms saw different data and the contrast is void regardless of the
    outcome (docs/PROTOCOL.md §10).

    The first entry is the reference. An empty mapping is rejected rather than
    treated as a pass: a gate with nothing to check has not been satisfied.
    """
    if not hashes:
        raise ValueError("no token-stream hashes supplied; cannot certify identity")

    items = list(hashes.items())
    ref_id, ref_hash = items[0]
    mismatched = [(run_id, digest) for run_id, digest in items[1:] if digest != ref_hash]
    if mismatched:
        detail = ", ".join(f"{run_id}={digest}" for run_id, digest in mismatched)
        raise ValueError(
            "token-stream SHA-256 mismatch: reference "
            f"{ref_id}={ref_hash}; mismatching run(s): {detail}"
        )


@dataclass
class QualificationReport:
    """The measured outcome of one pipeline-qualification pass.

    ``steps`` is recorded for auditing (the protocol qualifies over 500 steps)
    but does not itself gate the verdict; the seven thresholds in
    :meth:`verdict` are the pre-registered ones.
    """

    nan_skips: int
    steps: int
    throughput_tok_s: float
    micro_batch: int
    step0_identical: bool
    token_hash_identical: bool
    eval_finite: bool
    arm_nan_spread_pp: float

    def verdict(self) -> tuple[str, list[str]]:
        """Return ``("PASS", [])`` or ``("VOID", reasons)``.

        Thresholds are exactly those in docs/PROTOCOL.md §10. Every failed
        check is reported, not just the first: an operator needs the whole
        repair list before re-running qualification.
        """
        reasons: list[str] = []

        if self.nan_skips != MIN_NAN_SKIPS:
            reasons.append(
                f"nan_skips={self.nan_skips}: sanitised microbatches must be "
                f"{MIN_NAN_SKIPS} (an arm-dependent sanitisation rate confounds "
                "the treatment with the harness)"
            )
        if not self.throughput_tok_s >= MIN_THROUGHPUT_TOK_S:
            reasons.append(
                f"throughput {self.throughput_tok_s:g} tok/s is below "
                f"{MIN_THROUGHPUT_TOK_S:g} tok/s (a power requirement)"
            )
        if self.micro_batch < MIN_MICRO_BATCH:
            reasons.append(
                f"micro_batch {self.micro_batch} is below {MIN_MICRO_BATCH} "
                "(micro-batch-1 gradient variance inflates sigma and n)"
            )
        if not self.step0_identical:
            reasons.append(
                "step-0 outputs are not bit-identical across arms after zero-init"
            )
        if not self.token_hash_identical:
            reasons.append("token-stream SHA-256 differs across arms")
        if not self.eval_finite:
            reasons.append("an eval metric was NaN or Inf")
        if not self.arm_nan_spread_pp <= MAX_ARM_NAN_SPREAD_PP:
            reasons.append(
                f"arm sanitise-rate spread {self.arm_nan_spread_pp:g} pp exceeds "
                f"{MAX_ARM_NAN_SPREAD_PP:g} pp"
            )

        if reasons:
            return "VOID", reasons
        return "PASS", []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REPORT_FIELDS = tuple(field.name for field in fields(QualificationReport))


def _report_from_dict(data: Mapping[str, Any]) -> QualificationReport:
    missing = [name for name in _REPORT_FIELDS if name not in data]
    if missing:
        raise ValueError(
            "QualificationReport is missing required field(s): " + ", ".join(missing)
        )
    return QualificationReport(**{name: data[name] for name in _REPORT_FIELDS})


def report_to_json(report: QualificationReport, path: str | Path) -> None:
    """Write a qualification report for the run log and the shell gate."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    target.write_text(text + "\n", encoding="utf-8")


def from_json(path: str | Path) -> QualificationReport:
    """Read a report written by :func:`report_to_json`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "QualificationReport JSON must be an object, got "
            f"{type(raw).__name__}"
        )
    return _report_from_dict(raw)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the verdict for a report file or for stdin.

    Exit status is 0 for PASS and 1 for VOID, so a shell gate can simply use
    ``python -m experiments.protocol.qualify report.json || exit 1``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print("usage: python -m experiments.protocol.qualify [report.json]", file=sys.stderr)
        return 2

    if args:
        report = from_json(args[0])
    else:
        report = _report_from_dict(json.load(sys.stdin))

    verdict, reasons = report.verdict()
    print(verdict)
    for reason in reasons:
        print(f"  - {reason}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
