"""New Runtime evidence remains useful for generic, historical source learning."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.adapters.skill_sources.learning_sources import LearningSources
from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
from astra_robotics_agent.physical_keyboard.agent.learning import ContinualLearning
from astra_robotics_agent.physical_keyboard.records.learning_contracts import LearningContractError
from astra_robotics_agent.physical_keyboard.runtime.device_adapter import AdapterDescription, Observation
from astra_robotics_agent.physical_keyboard.runtime.mock_device import MockDevice
from astra_robotics_agent.physical_keyboard.runtime.program_runtime import ProgramRuntime
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aVj8AAAAASUVORK5CYII=")
SCOPE = dict(task_id="type", device_id="screen_fixture", layout_id="generic")


class ScreenDevice(MockDevice):
    def describe(self):
        return AdapterDescription("screen_fixture", "generic_base", ("screen",), "synthetic")

    def observe(self):
        return Observation(self._state(), [{"type": "inputText", "text": "Synthetic observation"},
            {"type": "inputImage", "view_id": "screen", "imageUrl":
             "data:image/png;base64," + base64.b64encode(PNG).decode()}])


class GenericLearningEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = ProgramRuntime(ScreenDevice(), self.root / "runtime", run_id="source_audit")
        self.addCleanup(self.runtime.close)
        self.store = KnowledgeStore(self.root / "knowledge", SCOPE, "source_audit")
        self.sources = LearningSources(SCOPE)

    def observation(self, mutate=None):
        observation = self.runtime.observe()
        files = observation["evidence_files"]
        if mutate:
            path = next(p for p in files if p.name == "record.json")
            value = json.loads(path.read_text())
            mutate(value)
            path.chmod(0o600)
            path.write_text(json.dumps(value))
        self.store.register_evidence(observation["evidence_id"], files)
        return observation

    def test_single_arbitrary_view_can_be_recalled_as_historical_image(self):
        observation = self.observation()
        source = self.sources.register_evidence(self.store, observation["evidence_id"])
        self.assertEqual(source["metadata"]["historical_images"], [{"index": 0, "views": ["screen"]}])
        got = self.sources.inspect(source["source_id"], dict(fields=["files"], image_indices=[0], camera="screen"))
        self.assertTrue(got["provenance"]["historical"])
        self.assertTrue(got["provenance"]["current_observation_unchanged"])
        self.assertEqual(got["images"][0]["sha256"], hashlib.sha256(PNG).hexdigest())
        self.assertEqual(Path(got["images"][0]["path"]).read_bytes(), PNG)
        self.assertNotIn("data:image/", json.dumps(got["data"]))
        self.assertTrue(got["data"]["files"]["record.json"]["content_items"][1]["encoding_omitted"])

    def test_default_image_view_uses_registered_source_not_a_fixed_wrist(self):
        observation = self.observation()
        source = self.sources.register_evidence(self.store, observation["evidence_id"])
        got = self.sources.inspect(source["source_id"], dict(fields=["files"], image_indices=[0]))
        self.assertEqual(got["images"][0]["camera"], "screen")
        self.assertEqual(got["images"][0]["status"], "available")

    def test_source_json_projection_does_not_change_current_device_state(self):
        observation = self.observation()
        source = self.sources.register_evidence(self.store, observation["evidence_id"])
        before = self.runtime.adapter.observe().state
        data = self.sources.analysis_data(source["source_id"], json_pointer="/files/record.json/state")
        self.assertEqual(data["observed_text"], "")
        self.assertEqual(self.runtime.adapter.observe().state, before)

    def test_learning_directory_discovers_new_runtime_observation(self):
        observation = self.observation()
        learning = ContinualLearning(DevelopmentActions(self.store), object(), output=self.root / "learning")
        listing = learning._source_directory(dict(origin="evidence"))
        items = listing["registered_evidence"]["items"]
        self.assertEqual([item["evidence_id"] for item in items], [observation["evidence_id"]])
        result, extra = learning._source_inspect(dict(source_id="evidence:" + observation["evidence_id"],
            fields=["files"], image_indices=[0], camera="screen"))
        self.assertEqual(result["images"][0]["status"], "available")
        self.assertTrue(any(item["type"] == "inputImage" for item in extra))

    def test_source_reads_frozen_pixels_when_original_capture_is_later_changed(self):
        observation = self.observation()
        source = self.sources.register_evidence(self.store, observation["evidence_id"])
        path = next(p for p in observation["evidence_files"] if p.suffix == ".png")
        path.chmod(0o600)
        path.write_bytes(b"later capture cannot replace the source object")
        image = self.sources.inspect(source["source_id"], dict(fields=["files"], image_indices=[0]))["images"][0]
        self.assertEqual(Path(image["path"]).read_bytes(), PNG)

    def test_declared_manifest_paths_cannot_escape_registered_siblings(self):
        observation = self.observation(lambda value: value["images"][0].update(path="../image-00.png"))
        with self.assertRaises(LearningContractError):
            self.sources.register_evidence(self.store, observation["evidence_id"])

    def test_manifest_pin_and_inline_pixels_must_agree(self):
        observation = self.observation(lambda value: value["images"][0].update(sha256="0" * 64))
        with self.assertRaises(LearningContractError):
            self.sources.register_evidence(self.store, observation["evidence_id"])

    def test_observation_with_audit_only_truth_is_not_a_learning_source(self):
        observation = self.observation(lambda value: value["state"].update(hidden_case_id="withheld"))
        with self.assertRaises(LearningContractError):
            self.sources.register_evidence(self.store, observation["evidence_id"])

    def test_generic_execution_source_pins_its_program_and_profile(self):
        executed = self.runtime.execute(dict(label="observe", profile={},
            source="def main(robot, profile):\n    return robot.observe()\n"))
        self.store.register_evidence(executed["evidence_id"], executed["evidence_files"])
        source = self.sources.register_evidence(self.store, executed["evidence_id"])
        data = self.sources.analysis_data(source["source_id"])
        self.assertEqual(data["files"]["record.json"]["protocol"], "physical-agent-execution-v1")
        self.assertFalse(source["metadata"]["current_state"])
        self.assertIn("def main", data["files"]["skill.py"])

    def test_sources_do_not_expose_keyboard_catalog_or_playback_methods(self):
        self.assertFalse(hasattr(self.sources, "register_catalog"))
        self.assertFalse(hasattr(self.sources, "_raw_mapping"))
        source = self.sources.register("provided_example", kind="human_demo",
            data={"observation": {"tool_tip_offset_m": [0, 0, -.02]}, "effect": "candidate observation"})
        self.assertEqual(source["kind"], "human_demo")
        self.assertNotIn("play", json.dumps(self.sources.describe()))

    def test_unrelated_registered_record_does_not_break_learning_directory(self):
        folder = self.root / 'unrelated'
        folder.mkdir()
        path = folder / 'record.json'
        path.write_text(json.dumps(dict(protocol='application-audit-v1', independent_score=1)))
        self.store.register_evidence('unrelated', [path])
        learning = ContinualLearning(DevelopmentActions(self.store), object(), output=self.root / 'learning')
        got = learning._source_directory(dict(origin='evidence'))
        self.assertEqual(got['registered_evidence']['items'], [])
        with self.assertRaises(LearningContractError):
            self.sources.register_evidence(self.store, 'unrelated')


if __name__ == "__main__":
    unittest.main()
