"""Media/export contract tests; no robot, SDK or camera device."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard import episode_review as review


class EpisodeReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode = self.root / "raw" / "demo"
        (self.episode / "colors").mkdir(parents=True)
        self.output = self.root / "reviews"
        self.times = [5_000_000_000, 5_060_000_000, 5_170_000_000, 5_410_000_000]
        self.make_episode()

    def make_episode(self):
        items, events = [], [{"type": "begin"}]
        for index, stamp in enumerate(self.times):
            name = f"colors/{index:06d}_right_wrist_camera.jpg"
            # Validation-only tests need no decoder. Real media tests replace these.
            (self.episode / name).write_bytes(b"not-a-decoded-image")
            events.append({"type": "sample", "item_index": index,
                           "loop_anchor": {"monotonic_ns": stamp, "utc_ns": 100_000 + stamp}})
            items.append({"idx": index, "colors": {"right_wrist_camera": name}})
        events.append({"type": "end", "sample_count": len(items), "operator_invalid": False})
        (self.episode / "keyboard-events.jsonl").write_text("".join(json.dumps(event)+"\n" for event in events))
        self.summary = {"operator_invalid": False, "sample_count": len(items)}
        self.refresh_sha()
        (self.episode / "data.json").write_text(json.dumps({"data": items, "summary": {}, "info": {"image": {"fps": 30}}}))

    def refresh_sha(self):
        self.summary["events_sha256"] = hashlib.sha256((self.episode/"keyboard-events.jsonl").read_bytes()).hexdigest()
        (self.episode / "keyboard-demo.json").write_text(json.dumps(self.summary))

    def change_events(self, mutate):
        path = self.episode/"keyboard-events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(events)
        path.write_text("".join(json.dumps(event)+"\n" for event in events))
        self.refresh_sha()

    def add_left_images(self, content=b"left-image"):
        data_path = self.episode / "data.json"
        data = json.loads(data_path.read_text())
        for item in data["data"]:
            relative = f"colors/{item['idx']:06d}_left_wrist_camera.jpg"
            (self.episode / relative).write_bytes(content)
            item["colors"]["left_wrist_camera"] = relative
        data_path.write_text(json.dumps(data))

    def test_camera_selection_preserves_default_contract_and_timeline(self):
        self.add_left_images()
        right = review.inspect_episode(self.episode)
        explicit_right = review.inspect_episode(self.episode, camera="right_wrist_camera")
        left = review.inspect_episode(self.episode, camera="left_wrist_camera")
        self.assertEqual(right, explicit_right)
        for key in ("source_frame_count", "sample_elapsed_s", "last_frame_duration_s", "expected_duration_s"):
            self.assertEqual(left[key], right[key])
        self.assertEqual([f["monotonic_ns"] for f in left["frames"]], [f["monotonic_ns"] for f in right["frames"]])
        self.assertNotEqual(left["frames"][0]["sha256"], right["frames"][0]["sha256"])
        protocol, config, filename = review.camera_settings("right_wrist_camera")
        self.assertEqual(protocol, review.PROTOCOL)
        self.assertEqual(config, review.CONFIG)
        self.assertEqual(filename, "right_wrist.webm")
        self.assertIsNot(config, review.CONFIG)
        protocol, config, filename = review.camera_settings("left_wrist_camera")
        self.assertEqual(protocol, "keyboard-left-wrist-review-v1")
        self.assertEqual(config["camera"], "left_wrist_camera")
        self.assertEqual(filename, "left_wrist.webm")
        self.assertEqual(review.CONFIG["camera"], "right_wrist_camera")

    def test_unknown_camera_is_rejected_before_creating_output(self):
        for camera in ("head_camera", "left_wrist", "../left", None):
            with self.assertRaisesRegex(ValueError, "unsupported review camera"):
                review.export_episode(self.episode, self.output, camera=camera)
        self.assertFalse(self.output.exists())

    def test_actual_time_and_all_input_hashes(self):
        source = review.inspect_episode(self.episode)
        self.assertEqual(source["source_frame_count"], 4)
        self.assertAlmostEqual(source["sample_elapsed_s"], .41)
        self.assertAlmostEqual(source["last_frame_duration_s"], .11)
        self.assertAlmostEqual(source["expected_duration_s"], .52)
        self.assertEqual(len(source["inputs"]), 3)
        self.assertTrue(all(len(frame["sha256"]) == 64 for frame in source["frames"]))

    def test_causal_resampling_sequence_uses_constant_memory_iterator(self):
        source = review.inspect_episode(self.episode)
        frames = review.playback_frames(source)
        self.assertIs(iter(frames), frames)
        self.assertEqual(review.playback_frame_count(source), 16)
        self.assertEqual([f["index"] for f in frames],
                         [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3])

    def test_invalid_is_explicit_skip_without_export_or_deletion(self):
        self.summary["operator_invalid"] = True
        self.refresh_sha()
        result = review.export_episode(self.episode, self.output)
        self.assertEqual(result["status"], "skipped_invalid")
        self.assertIsNone(result["video"])
        self.assertFalse(self.output.exists())
        self.assertTrue(self.episode.exists())

    def test_missing_image_refuses_export(self):
        (self.episode / "colors/000001_right_wrist_camera.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing source"):
            review.export_episode(self.episode, self.output)
        self.assertFalse(self.output.exists())

    def test_wrong_event_sha_refuses_export(self):
        with (self.episode/"keyboard-events.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            review.export_episode(self.episode, self.output)

    def test_duplicate_time_refuses_export(self):
        self.change_events(lambda e: e[2]["loop_anchor"].update(monotonic_ns=self.times[0]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            review.export_episode(self.episode, self.output)

    def test_missing_end_refuses_export(self):
        self.change_events(lambda e: e.pop())
        with self.assertRaisesRegex(ValueError, "not closed"):
            review.export_episode(self.episode, self.output)

    def test_index_count_and_writer_finalization_refuse_export(self):
        data_path = self.episode/"data.json"
        original = data_path.read_text()
        for change, message in ((lambda d: d["data"][1].update(idx=0), "index"),
                                (lambda d: d.pop("summary"), "not finalized"),
                                (lambda d: d["data"].pop(), "count mismatch")):
            data = json.loads(original); change(data); data_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, message):
                review.export_episode(self.episode, self.output)

    def test_relative_escape_absolute_path_and_symlink_refused(self):
        data_path = self.episode/"data.json"
        original = data_path.read_text()
        for name in ("../outside.jpg", "/tmp/outside.jpg", "colors/a\nb.jpg"):
            data = json.loads(original); data["data"][0]["colors"]["right_wrist_camera"] = name
            data_path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                review.export_episode(self.episode, self.output)
        data_path.write_text(original)
        target = self.root/"outside.jpg";target.write_bytes(b"outside")
        image = self.episode/"colors/000000_right_wrist_camera.jpg"
        image.unlink();image.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            review.export_episode(self.episode, self.output)

    def test_no_outputs_inside_raw(self):
        with self.assertRaisesRegex(ValueError, "outside the raw"):
            review.export_episode(self.episode, self.episode/"review")

    def test_export_lock_and_incomplete_output_are_not_overwritten(self):
        self.output.mkdir()
        lock = self.output/".demo.export-lock";lock.mkdir()
        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            review.export_episode(self.episode, self.output)
        lock.rmdir()
        (self.output/"demo").mkdir()
        with self.assertRaises(ValueError):
            review.export_episode(self.episode, self.output)
        self.assertTrue((self.output/"demo").is_dir())
        self.assertFalse(lock.exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg tools required")
    def test_real_media_duration_zero_timeline_idempotence_and_raw_integrity(self):
        base = self.root/"source.jpg"
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=red:size=64x48", "-frames:v", "1", str(base)], check=True)
        for path in (self.episode/"colors").glob("*.jpg"):
            path.write_bytes(base.read_bytes())
        original = {str(p): p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}
        result = review.export_episode(self.episode, self.output)
        self.assertEqual(result["status"], "exported")
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["media"]["pixel_format"], "yuv420p")
        self.assertEqual(manifest["media"]["codec"], "vp8")
        self.assertEqual(Path(result["video"]).suffix, ".webm")
        self.assertEqual(manifest["media"]["start_time_s"], 0)
        self.assertLessEqual(abs(manifest["media"]["duration_s"]-.52), 1/30+.002)
        self.assertEqual(manifest["media"]["full_decode"], "passed")
        self.assertEqual(manifest["source"]["source_frame_count"], 4)
        self.assertEqual(manifest["media"]["encoded_frame_count"], 16)
        self.assertEqual(manifest["command"][manifest["command"].index("-r")+1], "30")
        self.assertNotIn("-t", manifest["command"])
        self.assertEqual({str(p): p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}, original)
        self.assertEqual(review.export_episode(self.episode,self.output)["status"], "existing")
        # Changed raw and modified media both fail the existing-output contract.
        image = next((self.episode/"colors").glob("*.jpg")); image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"refusing overwrite"):
            review.export_episode(self.episode,self.output)
        image.write_bytes(base.read_bytes())
        with Path(result["video"]).open("ab") as stream:stream.write(b"changed")
        with self.assertRaisesRegex(ValueError,"refusing overwrite"):
            review.export_episode(self.episode,self.output)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg tools required")
    def test_singleton_exports_one_frame(self):
        self.times = [self.times[0]];self.make_episode()
        image = self.episode/"colors/000000_right_wrist_camera.jpg"
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=blue:size=64x48", "-frames:v", "1", "-y", str(image)],check=True)
        result = review.export_episode(self.episode,self.output)
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["media"]["encoded_frame_count"],1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg tools required")
    def test_left_media_matches_right_timing_preserves_existing_right_and_raw(self):
        right_jpeg, left_jpeg = self.root / "right.jpg", self.root / "left.jpg"
        for color, target in (("red", right_jpeg), ("blue", left_jpeg)):
            subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color="+color+":size=64x48", "-frames:v", "1", str(target)], check=True)
        for path in (self.episode / "colors").glob("*.jpg"):
            path.write_bytes(right_jpeg.read_bytes())
        self.add_left_images(left_jpeg.read_bytes())
        raw_before = {str(p): p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}
        right = review.export_episode(self.episode, self.output)
        right_before = {str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        left_output = self.root / "reviews-left"
        left = review.export_episode(self.episode, left_output, camera="left_wrist_camera")
        self.assertEqual(Path(left["video"]).name, "left_wrist.webm")
        left_manifest = json.loads(Path(left["manifest"]).read_text())
        right_manifest = json.loads(Path(right["manifest"]).read_text())
        for field in ("encoded_frame_count", "duration_s", "start_time_s", "codec", "pixel_format"):
            self.assertEqual(left_manifest["media"][field], right_manifest["media"][field])
        self.assertEqual(left_manifest["media"]["full_decode"], "passed")
        self.assertNotEqual(left_manifest["video_sha256"], right_manifest["video_sha256"])
        self.assertEqual(review.export_episode(self.episode, self.output)["status"], "existing")
        self.assertEqual(review.export_episode(self.episode, left_output, camera="left_wrist_camera")["status"], "existing")
        self.assertEqual({str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}, right_before)
        self.assertEqual({str(p): p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}, raw_before)
        # A left-only missing JPEG does not invalidate any right source/hash.
        (self.episode / "colors/000000_left_wrist_camera.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing source"):
            review.export_episode(self.episode, left_output, camera="left_wrist_camera")
        self.assertEqual(review.export_episode(self.episode, self.output)["status"], "existing")
        self.assertEqual({str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}, right_before)


if __name__ == "__main__":
    unittest.main()
