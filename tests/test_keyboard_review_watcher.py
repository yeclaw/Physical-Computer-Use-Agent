"""Closed-episode/lifecycle media scanning, no real hardware or polling."""
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.review_watcher import ReviewScanner, owner_identity


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.episodes = self.root / "episodes"
        self.episodes.mkdir()

    def episode(self, name, invalid=False):
        path = self.episodes / name
        path.mkdir()
        (path / "data.json").write_text('{"summary":{},"data":[]}')
        (path / "keyboard-demo.json").write_text(json.dumps({"operator_invalid": invalid}))
        return path

    def test_exports_left_only_once_and_ignores_unclosed_recording(self):
        path = self.episode("complete")
        open_path = self.episodes / "active"
        open_path.mkdir()
        (open_path / "keyboard-events.jsonl").write_text("begin")
        calls, events = [], []
        def exporter(episode, out, *, camera):
            calls.append((episode, out, camera))
            return {"status": "exported", "episode": str(episode)}
        scanner = ReviewScanner(self.episodes, self.root / "left", exporter)
        self.assertEqual(scanner.scan(events.append), 1)
        self.assertEqual(scanner.scan(events.append), 0)
        self.assertEqual(calls, [(path.resolve(), (self.root / "left").resolve(), "left_wrist_camera")])
        self.assertTrue(open_path.exists())

    def test_encoder_error_retries_only_after_metadata_change(self):
        path = self.episode("one")
        calls, events = [], []
        def exporter(*args, **kwargs): calls.append(args); raise RuntimeError("synthetic incomplete writer")
        scanner = ReviewScanner(self.episodes, self.root / "left", exporter)
        scanner.scan(events.append)
        scanner.scan(events.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events[0]["status"], "failed")
        (path / "data.json").write_text('{"summary":{},"data":[{"idx":0}]}')
        scanner.scan(events.append)
        self.assertEqual(len(calls), 2)

    def test_new_closed_episode_and_invalid_are_distinct(self):
        self.episode("first")
        calls, events = [], []
        def exporter(path, *args, **kwargs):
            calls.append(path.name)
            invalid = json.loads((path / "keyboard-demo.json").read_text())["operator_invalid"]
            return {"status": "skipped_invalid" if invalid else "exported", "episode": str(path)}
        scanner = ReviewScanner(self.episodes, self.root / "left", exporter)
        scanner.scan(events.append)
        self.episode("second")
        self.episode("discard", True)
        scanner.scan(events.append)
        self.assertEqual(calls, ["first", "discard", "second"])
        self.assertEqual([e["status"] for e in events], ["exported", "skipped_invalid", "exported"])

    def test_owner_starttime_detects_pid_reuse_and_exit(self):
        proc = self.root / "proc"
        owner = proc / "42"
        owner.mkdir(parents=True)
        fields = ['S'] + ['0'] * 18 + ['12345']
        stat = owner / "stat"
        stat.write_text('42 (collector name) ' + ' '.join(fields))
        old = owner_identity(42, proc)
        self.assertEqual(old, '12345')
        fields[-1] = '12346'
        stat.write_text('42 (collector name) ' + ' '.join(fields))
        self.assertNotEqual(owner_identity(42, proc), old)
        stat.unlink()
        self.assertIsNone(owner_identity(42, proc))


if __name__ == "__main__":
    unittest.main()
