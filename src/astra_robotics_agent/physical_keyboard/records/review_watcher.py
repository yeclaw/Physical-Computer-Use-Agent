"""Media-only left-wrist export for closed episodes; imports no robot SDK."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

from astra_robotics_agent.physical_keyboard.records.episode_review import export_episode


def owner_identity(pid, proc=Path("/proc")):
    try:
        # Field 22 is process starttime. A recycled PID must not keep us alive.
        text = (proc / str(pid) / "stat").read_text()
        return text.rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class ReviewScanner:
    def __init__(self, episodes_root, output_root, exporter=export_episode):
        self.episodes_root = Path(episodes_root).resolve(strict=True)
        self.output_root = Path(output_root).resolve()
        self.exporter = exporter
        self.seen = {}
        self.results = {}

    def scan(self, emit):
        attempted = 0
        for summary in sorted(self.episodes_root.rglob("keyboard-demo.json")):
            episode = summary.parent
            data = episode / "data.json"
            try:
                if summary.is_symlink() or episode.is_symlink():
                    raise ValueError("source symlink refused")
                stat = data.stat()
                signature = (hashlib.sha256(summary.read_bytes()).hexdigest(), stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                continue  # the old writer has not yet created/finalized its data
            if self.seen.get(str(episode)) == signature:
                continue
            attempted += 1
            self.seen[str(episode)] = signature
            try:
                result = self.exporter(episode, self.output_root, camera="left_wrist_camera")
            except Exception as exc:
                result = {"status": "failed", "episode": str(episode), "error": repr(exc)}
            self.results[str(episode)] = result
            emit(result)
        return attempted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll interval must be 1..60 seconds")
    identity = owner_identity(args.owner_pid)
    if identity is None:
        parser.error("collector owner is not running")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    scanner = ReviewScanner(args.episodes_root, args.output_root)
    with (args.log_dir / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def event(row):
            with (args.log_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"utc_ns": time.time_ns(), **row}, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        def status(state):
            rows = list(scanner.results.values())
            value = {"utc_ns": time.time_ns(), "pid": os.getpid(), "state": state,
                     "owner_pid": args.owner_pid, "owner_starttime": identity,
                     "episodes_root": str(scanner.episodes_root), "output_root": str(scanner.output_root),
                     "exported_or_existing": sum(r["status"] in ("exported", "existing") for r in rows),
                     "failed": sum(r["status"] == "failed" for r in rows), "results": rows,
                     "robot_commands": 0, "camera_commands": 0}
            temp = args.log_dir / "status.json.tmp"
            temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            temp.replace(args.log_dir / "status.json")
        event({"type": "worker_started", "pid": os.getpid(), "owner_pid": args.owner_pid,
               "owner_starttime": identity, "robot_commands": 0})
        status("encoding_initial_batch")
        while not stop.is_set():
            attempts = scanner.scan(event)
            if attempts or not (args.log_dir / "initial-pass-complete.json").exists():
                status("watching")
                (args.log_dir / "initial-pass-complete.json").write_text(json.dumps({
                    "utc_ns": time.time_ns(), "processed": len(scanner.results), "attempts": attempts}) + "\n")
            if owner_identity(args.owner_pid) != identity:
                event({"type": "owner_exited"})
                break
            stop.wait(args.poll_seconds)
        status("stopped")
        event({"type": "worker_stopped"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
