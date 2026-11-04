"""Device-independent record/audit helpers extracted from the reference host."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

def anchor() -> dict[str, int]:
    before = time.monotonic_ns()
    utc = time.time_ns()
    after = time.monotonic_ns()
    return {"utc_ns": utc, "monotonic_ns": (before + after) // 2, "sampling_span_ns": after - before}


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_loads(data: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, data: bytes, *, readonly: bool = False) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if readonly:
        path.chmod(0o444)


def freeze_file(source: Path, directory: Path, label: str) -> dict[str, str]:
    data = source.read_bytes()
    target = directory / (label + "-" + digest(data)[:16] + source.suffix)
    if not target.exists():
        write_new(target, data, readonly=True)
    elif target.read_bytes() != data:
        raise ValueError("frozen dependency collision")
    return {"original_path": str(source.resolve()), "frozen_path": str(target), "sha256": digest(data)}


class Audit:
    def __init__(self, path: Path, run_id: str):
        self.run_id = run_id
        self.stream = path.open("x", encoding="utf-8")
        self._lock = threading.RLock()
        self.progress_sink = None

    def __call__(self, kind: str, **fields: Any) -> None:
        with self._lock:
            self.telemetry(kind, **fields)
            os.fsync(self.stream.fileno())

    def telemetry(self, kind: str, **fields: Any) -> None:
        """Flush timing observations; the next durable event commits them too.

        Native replay still fsyncs its target/feedback record before every raw
        send. Repeated fsyncs of scheduling and returned-call observations can
        otherwise consume the entire original ~33ms sample interval. A power
        loss may lose the last uncommitted observation; it never proves arrival.
        Ordinary callers retain the synchronous __call__ contract.
        """
        with self._lock:
            self.stream.write(encoded({"type": kind, "run_id": self.run_id, "pid": os.getpid(),
                                       "anchor": anchor(), **fields}).decode() + "\n")
            self.stream.flush()
            if self.progress_sink is not None:
                self.progress_sink(kind, fields)

    def close(self):
        if not self.stream.closed:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            finally:
                self.stream.close()

