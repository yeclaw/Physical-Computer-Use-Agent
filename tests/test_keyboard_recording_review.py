"""Save/next-episode/X-delete integration, without a robot or camera."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard import recording_review as review


def episode(root, name="one", invalid=False, reason="operator_stop"):
    path = root / name
    path.mkdir(parents=True)
    events = path / "keyboard-events.jsonl"
    events.write_text('{"type":"end"}\n')
    (path / "keyboard-demo.json").write_text(json.dumps({
        "operator_invalid": invalid, "reason": reason, "sample_count": 1,
        "events_sha256": hashlib.sha256(events.read_bytes()).hexdigest()}))
    (path / "data.json").write_text(json.dumps({"data": [{"idx": 0}], "summary": {}}))
    (path / "colors").mkdir()
    (path / "colors/000000_right_wrist_camera.jpg").write_bytes(b"test-only-image")
    return path


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "episodes"
        self.root.mkdir()
        self.out = Path(self.tmp.name).resolve() / "review"

    def jobs(self, exporter=lambda p, out: {"video": str(out / p.name)}):
        worker = review.ReviewJobs(self.root, self.out, exporter)
        self.addCleanup(worker.close)
        return worker

    def test_x_deletes_only_closed_explicit_discard_with_external_receipt(self):
        worker = self.jobs()
        path = episode(self.root, invalid=True, reason="operator_discard")
        original_sha = review.sha256(path / "keyboard-events.jsonl")
        worker.submit(path, discard=True)
        self.assertFalse(path.exists())
        receipt = json.loads((self.out / "deletion-receipts/one.json").read_text())
        self.assertEqual(receipt["summary"]["events_sha256"], original_sha)
        self.assertEqual(receipt["file_count"], 4)
        rows = [json.loads(s) for s in (self.out / "review-events.jsonl").read_text().splitlines()]
        self.assertEqual([r["type"] for r in rows], ["discard_requested", "discard_deleted"])

    def test_normal_exit_fault_and_nonclosed_cannot_be_deleted(self):
        worker = self.jobs()
        for index, (invalid, reason) in enumerate(((False, "operator_stop"),
                          (True, "collector_exit"), (True, "sampling_failed"))):
            path = episode(self.root, str(index), invalid, reason)
            with self.assertRaisesRegex(ValueError, "explicit"):
                worker.submit(path, discard=True)
            self.assertTrue(path.exists())
        path = episode(self.root, "unclosed", True, "operator_discard")
        (path / "data.json").unlink()
        with self.assertRaisesRegex(ValueError, "not closed"):
            worker.submit(path, discard=True)
        self.assertTrue(path.exists())

    def test_path_root_external_and_symlinks_refused(self):
        worker = self.jobs()
        outside = episode(Path(self.tmp.name).resolve(), "outside", True, "operator_discard")
        for path in (outside, self.root):
            with self.assertRaises(ValueError): worker.submit(path, discard=True)
        link = self.root / "linked"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.submit(link, discard=True)
        path = episode(self.root, invalid=True, reason="operator_discard")
        (path / "linked-file").symlink_to(outside / "data.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.submit(path, discard=True)
        self.assertTrue(outside.exists())

    def test_bad_hash_or_incomplete_data_preserves_x(self):
        worker = self.jobs()
        path = episode(self.root, invalid=True, reason="operator_discard")
        (path / "keyboard-events.jsonl").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            worker.submit(path, discard=True)
        self.assertTrue(path.exists())
        other = episode(self.root, "other", True, "operator_discard")
        (other / "data.json").write_text('{"data":[]}')
        with self.assertRaisesRegex(ValueError, "completely saved"):
            worker.submit(other, discard=True)
        self.assertTrue(other.exists())

    def test_video_is_background_and_failed_encode_keeps_raw_and_logs(self):
        entered, release = threading.Event(), threading.Event()
        def encode(path, out):
            entered.set()
            release.wait(2)
            raise RuntimeError("synthetic encoder error")
        worker = self.jobs(encode)
        path = episode(self.root)
        worker.submit(path)
        self.assertTrue(entered.wait(2))
        # X removal must not wait behind a slow encode of the previous B.
        discard = episode(self.root, "discard", True, "operator_discard")
        worker.submit(discard, discard=True)
        self.assertFalse(discard.exists())
        release.set()
        worker.queue.join()
        self.assertTrue(path.exists())
        rows = [json.loads(s) for s in (self.out / "review-events.jsonl").read_text().splitlines()]
        self.assertIn("video_failed", [r["type"] for r in rows])

    def test_q_invalid_preserved_without_encoding(self):
        calls = []
        worker = self.jobs(lambda *args: calls.append(args))
        path = episode(self.root, invalid=True, reason="collector_exit")
        worker.submit(path)
        worker.queue.join()
        self.assertTrue(path.exists())
        self.assertEqual(calls, [])

    def test_audit_failure_does_not_kill_encoder_or_hang_close(self):
        calls = []
        worker = self.jobs(lambda path, out: calls.append(path) or {"video": "test"})
        original = worker.event
        def broken(kind, **fields):
            if kind in ("video_finished", "video_failed"):
                raise OSError("synthetic disk full")
            return original(kind, **fields)
        worker.event = broken
        with self.assertLogs(review.LOG, level="ERROR"):
            worker.submit(episode(self.root, "first"))
            worker.submit(episode(self.root, "second"))
            closer = threading.Thread(target=worker.close)
            closer.start()
            closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(len(calls), 2)
        worker.close()  # no second sentinel or hang

    def test_writer_save_completion_snapshot_survives_next_episode(self):
        first = episode(self.root, "first", True, "operator_discard")
        second = episode(self.root, "second")
        calls = []
        class Base:
            def __init__(self, task_dir): self.episode_dir = str(first); self.calls = 0
            def save_episode(self): self.calls += 1
            def _save_episode(self):
                # Emulate is_available=True then a new main-loop create_episode
                # before the wrapper regains control, as the upstream permits.
                self.episode_dir = str(second)
            def close(self): pass
        def exporter(path, out):
            calls.append(path)
            return {"video": "test"}
        with patch.object(review.ReviewJobs, "__init__", autospec=True) as init:
            init.side_effect = lambda obj, root, out: None
            writer = review.review_writer_class(Base)(self.root)
        real_jobs = self.jobs(exporter)
        writer.review_jobs = real_jobs
        writer.save_episode(discard=True)
        writer.save_episode()  # original close repeats save while not ready
        self.assertEqual(writer.calls, 1)
        self.assertTrue(first.exists())  # save() is asynchronous, no early deletion
        writer._save_episode()
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        writer.save_episode()
        writer._save_episode()
        real_jobs.queue.join()
        self.assertEqual(calls, [second])


if __name__ == "__main__":
    unittest.main()
