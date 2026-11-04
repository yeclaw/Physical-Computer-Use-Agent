"""Literal task priors and real-device evidence fixtures; no actuation or models."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.hints import (
    ContextError, freeze_context, load_context,
)


HINTS = b'''{
  "protocol": "physical-keyboard-task-hints-v1",
  "id": "keyboard-view-start",
  "version": 1,
  "scope": {"task_id":"physical_keyboard_view", "device_id":"shanghai_startouch", "layout_id":"keyboard_20261022_a"},
  "origin": "human_provided",
  "hints": [
    {"id":"roles", "text":"Left gripper petal inputs; right wrist observes the monitor.", "source":"User configuration, 2026-10-22"},
    {"id":"visibility", "text":"If keys are not visible, compare images after a small permitted view adjustment.", "source":"User suggestion; not an autonomous discovery"}
  ]
}
'''

EXPERIENCE = b'''{
  "protocol": "physical-keyboard-scoped-experience-v1",
  "id": "exp015-view-once",
  "scope": {"task_id":"physical_keyboard_view", "device_id":"shanghai_startouch", "layout_id":"keyboard_20261022_a"},
  "status": "observed_once_not_promoted",
  "source_experiment": "EXP-015",
  "source_manifest_sha256": "ca5d459d4ae49f831cdb01f5d24cb07d572319c4db94dc0f73930d5718e3c6cc",
  "observations": ["One +3 degree base-Y probe made additional local letter keys visible."],
  "limitations": ["Full keyboard rows and petal clearance were not verified.", "No keypress or stable reusable skill was demonstrated."],
  "automatic_replay": false
}
'''


class HintsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="keyboard-hints-")
        self.addCleanup(self.temporary.cleanup)
        # macOS /var and /tmp are system symlinks, not canonical input paths.
        self.root = Path(self.temporary.name).resolve()
        self.hints = self.root / "hints.json"
        self.experience = self.root / "experience.json"
        self.hints.write_bytes(HINTS)
        self.experience.write_bytes(EXPERIENCE)
        self.scope = {"task_id": "physical_keyboard_view", "device_id": "shanghai_startouch",
                      "layout_id": "keyboard_20261022_a"}

    def load(self, experience=False, **scope):
        return load_context(self.hints, self.experience if experience else None,
                            **(scope or self.scope))

    def replace(self, path, mutate):
        data = json.loads(HINTS if path == self.hints else EXPERIENCE)
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_hints_only_is_read_only_and_frozen(self):
        context = self.load()
        self.assertIn("HUMAN-PROVIDED TASK HINTS", context.text)
        self.assertNotIn("OBSERVED-ONCE DEVICE EXPERIENCE", context.text)
        self.assertIn("not host permissions", context.text)
        self.assertIn("Current visual and robot observations take priority", context.text)
        self.assertEqual(context.sources, (("hints", str(self.hints), HINTS),))
        self.assertEqual(self.hints.read_bytes(), HINTS)
        with self.assertRaises(FrozenInstanceError):
            context.text = "replace"

    def test_experience_has_separate_label_scope_and_provenance(self):
        context = self.load(experience=True)
        self.assertIn("OBSERVED-ONCE DEVICE EXPERIENCE — REVALIDATE", context.text)
        self.assertIn("No keypress or stable reusable skill", context.text)
        self.assertIn('"automatic_replay": false', context.text)
        self.assertEqual(context.metadata["scope"], self.scope)
        self.assertEqual([source["origin"] for source in context.metadata["inputs"]],
                         ["human_provided", "observed_once_not_promoted"])
        self.assertEqual(context.metadata["inputs"][1]["source_experiment"], "EXP-015")
        self.assertEqual(context.metadata["inputs"][1]["sha256"], hashlib.sha256(EXPERIENCE).hexdigest())

    def test_task_device_and_layout_mismatches_are_rejected_for_each_kind(self):
        for path in (self.hints, self.experience):
            for key in self.scope:
                with self.subTest(path=path.name, key=key):
                    self.replace(path, lambda data: data["scope"].update({key: "other"}))
                    with self.assertRaises(ContextError):
                        self.load(experience=True)
                    path.write_bytes(HINTS if path == self.hints else EXPERIENCE)
        with self.assertRaises(ContextError):
            self.load(task_id=" ", device_id="d", layout_id="l")

    def test_hints_exact_schema_and_text_boundaries(self):
        mutations = [
            lambda d: d.update(profile={"speed": 1}),
            lambda d: d.pop("origin"),
            lambda d: d.update(origin="observed_once_not_promoted"),
            lambda d: d.update(protocol="physical-keyboard-scoped-experience-v1"),
            lambda d: d.update(version=True),
            lambda d: d.update(version=0),
            lambda d: d.update(id=" "),
            lambda d: d.update(hints=[]),
            lambda d: d["hints"].append(dict(d["hints"][0])),
            lambda d: d["hints"][0].update(text=" "),
            lambda d: d["hints"][0].update(text="x" * 4097),
            lambda d: d["hints"][0].update(safety_override=True),
            lambda d: d["hints"][0].update(source=7),
            lambda d: d["scope"].update(extra="unexpected"),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                self.replace(self.hints, mutation)
                with self.assertRaises(ContextError):
                    self.load()

    def test_experience_cannot_be_promoted_or_confused_with_human_prior(self):
        mutations = [
            lambda d: d.update(status="stable"),
            lambda d: d.update(origin="human_provided"),
            lambda d: d.update(protocol="physical-keyboard-task-hints-v1"),
            lambda d: d.update(automatic_replay=True),
            lambda d: d.update(automatic_replay=0),
            lambda d: d.update(source_manifest_sha256="z" * 64),
            lambda d: d.update(source_experiment=""),
            lambda d: d.update(observations=[]),
            lambda d: d.update(limitations=[False]),
            lambda d: d.pop("limitations"),
            lambda d: d.update(host={"disable_guard": True}),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                self.replace(self.experience, mutation)
                with self.assertRaises(ContextError):
                    self.load(experience=True)

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        corruptions = [
            HINTS.replace(b'"version": 1', b'"version": 1, "version": 2'),
            HINTS.replace(b'"text":"Left', b'"text":"old", "text":"Left'),
            HINTS.replace(b'"version": 1', b'"version": NaN'),
            HINTS.replace(b'"version": 1', b'"version": Infinity'),
            HINTS.replace(b'"version": 1', b'"version": -Infinity'),
            HINTS.replace(b'"version": 1', b'"version": 1e9999'),
        ]
        for index, raw in enumerate(corruptions):
            with self.subTest(case=index):
                self.hints.write_bytes(raw)
                with self.assertRaises(ContextError):
                    self.load()
        self.hints.write_bytes(HINTS)
        self.experience.write_bytes(EXPERIENCE.replace(b'"automatic_replay": false',
                                                     b'"automatic_replay": false, "automatic_replay": false'))
        with self.assertRaises(ContextError):
            self.load(experience=True)

    def test_source_size_encoding_and_complete_document_are_checked(self):
        padded = HINTS + b" " * (32768 - len(HINTS))
        self.hints.write_bytes(padded)
        self.assertEqual(self.load().sources[0][2], padded)
        for raw in (padded + b" ", b"", b"\xff", HINTS + b"{}", b"[]"):
            with self.subTest(length=len(raw)):
                self.hints.write_bytes(raw)
                with self.assertRaises(ContextError):
                    self.load()

    def test_input_symlinks_and_nonregular_files_are_not_followed(self):
        link = self.root / "link.json"
        link.symlink_to(self.hints)
        with self.assertRaises(ContextError):
            load_context(link, **self.scope)
        directory_link = self.root / "alias"
        directory_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ContextError):
            load_context(directory_link / "hints.json", **self.scope)
        with self.assertRaises(ContextError):
            load_context(self.root, **self.scope)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ContextError):
            load_context(fifo, **self.scope)

    def test_freeze_preserves_exact_original_bytes_and_hashes(self):
        context = self.load(experience=True)
        output = self.root / "frozen"
        result = freeze_context(context, output)
        self.assertEqual((output / "hints.json").read_bytes(), HINTS)
        self.assertEqual((output / "experience.json").read_bytes(), EXPERIENCE)
        self.assertEqual((output / "model-context.txt").read_text(), context.text)
        self.assertEqual(json.loads((output / "manifest.json").read_bytes()), result)
        self.assertEqual(set(result["files"]), {"hints.json", "experience.json", "model-context.txt"})
        for name, info in result["files"].items():
            raw = (output / name).read_bytes()
            self.assertEqual(info, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        self.assertEqual(result["context"]["inputs"][0]["sha256"], hashlib.sha256(HINTS).hexdigest())

    def test_snapshot_survives_input_edits_and_deletion(self):
        context = self.load(experience=True)
        text = context.text
        self.hints.write_text("later unrelated content")
        self.experience.unlink()
        output = self.root / "snapshot"
        freeze_context(context, output)
        self.assertEqual((output / "hints.json").read_bytes(), HINTS)
        self.assertEqual((output / "experience.json").read_bytes(), EXPERIENCE)
        self.assertEqual((output / "model-context.txt").read_text(), text)
        self.assertEqual(self.hints.read_text(), "later unrelated content")

    def test_output_must_be_new_and_does_not_overwrite(self):
        output = self.root / "existing"
        output.mkdir()
        original = output / "hints.json"
        original.write_bytes(b"keep")
        with self.assertRaises(ContextError):
            freeze_context(self.load(), output)
        self.assertEqual(original.read_bytes(), b"keep")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ContextError):
            freeze_context(self.load(), empty)

    def test_import_has_no_sdk_network_or_process_side_effects(self):
        code = '''
import builtins, sys
original = builtins.__import__
forbidden = ('socket', 'urllib', 'http', 'requests', 'easydp', 'startouch', 'torch')
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def audit(event, args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        raise AssertionError('unexpected side effect: ' + event)
sys.addaudithook(audit)
from astra_robotics_agent.physical_keyboard import hints
assert hints.HINTS_PROTOCOL == 'physical-keyboard-task-hints-v1'
print('import-only: no SDK/network/process calls')
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no SDK/network/process calls", result.stdout)


if __name__ == "__main__":
    unittest.main()
