"""Post-save review workflow; no SDK imports or robot commands.

The original writer closes every JPEG/data.json before this hook runs. A fixed
episode path is handed to the encoder, never the writer's mutable episode_dir.
Only an explicit operator X disposition can delete a closed episode.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from queue import Queue
import shutil
import threading
import time

LOG = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_episode(episode, episodes_root):
    """Refuse roots, paths outside this collector, or any symlink contents."""
    raw, root = Path(episode).absolute(), Path(episodes_root).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents, root, *root.parents)):
        raise ValueError("episode/root path contains a symlink")
    path, root = raw.resolve(strict=True), root.resolve(strict=True)
    if path == root or root not in path.parents or not path.is_dir():
        raise ValueError("episode must be a child directory of episodes_root")
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("episode contents contain a symlink")
    for name in ("keyboard-demo.json", "keyboard-events.jsonl", "data.json"):
        if not (path / name).is_file():
            raise ValueError("episode is not closed: " + name)
    return path


class ReviewJobs:
    def __init__(self, episodes_root, review_root, exporter=None):
        self.episodes_root = Path(episodes_root).resolve()
        self.review_root = Path(review_root).resolve()
        self.review_root.mkdir(parents=True, exist_ok=True)
        if self.episodes_root == self.review_root or self.episodes_root in self.review_root.parents:
            raise ValueError("review artifacts must live outside episodes_root")
        if exporter is None:
            from astra_robotics_agent.physical_keyboard.records.episode_review import export_episode
            exporter = export_episode
        self.exporter = exporter
        self.queue = Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="keyboard-review-video", daemon=True)
        self.thread.start()

    def event(self, kind, **fields):
        row = {"type": kind, "utc_ns": time.time_ns(), **fields}
        with self.lock, (self.review_root / "review-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def submit(self, episode, discard=False):
        path = checked_episode(episode, self.episodes_root)
        if discard:
            # Discards do not wait behind video encodes. The original JPEG
            # writer has already completed before the hook calls submit().
            self._discard(path)
        else:
            summary = json.loads((path / "keyboard-demo.json").read_text())
            if summary.get("operator_invalid"):
                self.event("invalid_preserved", episode=str(path), reason=summary.get("reason"))
                return
            self.event("video_queued", episode=str(path))
            self.queue.put(path)

    def _discard(self, path):
        summary = json.loads((path / "keyboard-demo.json").read_text())
        if summary.get("operator_invalid") is not True or summary.get("reason") != "operator_discard":
            raise ValueError("deletion requires explicit operator_discard summary")
        events = path / "keyboard-events.jsonl"
        if sha256(events) != summary.get("events_sha256"):
            raise ValueError("discard events checksum mismatch")
        data = json.loads((path / "data.json").read_text())
        if len(data["data"]) != summary["sample_count"]:
            raise ValueError("discard writer not completely saved")
        files = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size,
                  "sha256": sha256(p)} for p in sorted(path.rglob("*")) if p.is_file()]
        receipt = {"protocol": "keyboard-operator-discard-v1", "episode": str(path),
                   "authorization": "DEC-064 explicit operator X", "utc_ns": time.time_ns(),
                   "summary": summary, "files": files,
                   "file_count": len(files), "bytes": sum(p["bytes"] for p in files)}
        receipts = self.review_root / "deletion-receipts"
        receipts.mkdir(exist_ok=True)
        target = receipts / (path.name + ".json")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.event("discard_requested", episode=str(path), receipt=str(target))
        # Recheck boundaries immediately before deletion. No normal/fault/q
        # episode is eligible, and no background writer retains this path.
        checked_episode(path, self.episodes_root)
        shutil.rmtree(path)
        self.event("discard_deleted", episode=str(path), receipt=str(target), exists=path.exists())
        LOG.info("X discarded episode deleted: %s", path)

    def _run(self):
        while True:
            path = self.queue.get()
            try:
                if path is None:
                    return
                try:
                    result = self.exporter(path, self.review_root / "videos")
                    self.event("video_finished", episode=str(path), result=result)
                    LOG.info("Review video ready: %s", result.get("video"))
                except Exception as exc:
                    try:
                        self.event("video_failed", episode=str(path), error=repr(exc), raw_preserved=True)
                    except Exception:
                        LOG.exception("Review failure audit could not be written")
                    LOG.exception("Review video failed; raw episode preserved: %s", path)
            finally:
                self.queue.task_done()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.queue.join()
        self.thread.join()


def review_writer_class(base):
    """Wrap the pinned headless writer without changing its media serializer."""
    class ReviewEpisodeWriter(base):
        def __init__(self, task_dir, *args, review_root=None, **kwargs):
            self._review_disposition = {}
            self.review_jobs = ReviewJobs(task_dir, review_root or Path(task_dir).resolve().parent / "review")
            try:
                super().__init__(task_dir, *args, **kwargs)
            except BaseException:
                self.review_jobs.close()
                raise

        def save_episode(self, *, discard=False):
            path = str(Path(self.episode_dir).resolve())
            if path not in self._review_disposition:
                self._review_disposition[path] = bool(discard)
            else:
                # The upstream close() repeats save_episode() while the queue
                # is draining; preserve an already explicit X disposition.
                if discard and not self._review_disposition[path]:
                    raise ValueError("episode save disposition cannot be changed")
                return
            super().save_episode()

        def _save_episode(self):
            # Snapshot before super marks the writer available to create the
            # next episode. submit never reads recorder.episode_dir afterwards.
            path = Path(self.episode_dir).resolve()
            discard = self._review_disposition.get(str(path), False)
            super()._save_episode()
            self._review_disposition.pop(str(path), None)
            try:
                self.review_jobs.submit(path, discard=discard)
            except Exception as exc:
                try:
                    self.review_jobs.event("post_save_failed", episode=str(path), discard=discard,
                                           error=repr(exc), raw_preserved=path.exists())
                except Exception:
                    LOG.exception("Post-save failure audit could not be written")
                LOG.exception("Post-save review failed; inspect review-events.jsonl")

        def close(self):
            try:
                super().close()
            finally:
                self.review_jobs.close()

    return ReviewEpisodeWriter
