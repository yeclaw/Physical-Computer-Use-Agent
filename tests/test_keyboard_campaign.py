"""Frozen-goal lifecycle tests, including real scoped storage; no actuation."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from astra_robotics_agent.physical_keyboard.campaign import (
    CampaignLifecycle, COMPLETION_REQUIREMENTS, load_task_contract,
)
from astra_robotics_agent.physical_keyboard.knowledge import KnowledgeStore


SCOPE = {"task_id": "physical_keyboard_view", "device_id": "shanghai_startouch",
         "layout_id": "keyboard_20261022_a"}
DOCUMENT = {
    "protocol": "physical-keyboard-task-contract-v1", "id": "english-typing",
    "version": 1, "scope": SCOPE,
    "objective": "Learn physical English typing and leave exactly hello visible.",
    "target_text": "hello", "completion_requirements": list(COMPLETION_REQUIREMENTS),
}


class ContractFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="keyboard-campaign-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.path = self.base / "contract.json"
        self.path.write_text(json.dumps(DOCUMENT), encoding="utf-8")

    def load(self):
        return load_task_contract(self.path, dict(SCOPE))


class ContractTests(ContractFixture, unittest.TestCase):
    def test_load_preserves_source_and_provenance_without_mutable_aliases(self):
        raw = self.path.read_bytes()
        contract = self.load()
        self.assertEqual(contract.raw, raw)
        self.assertEqual(contract.sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(contract.document, DOCUMENT)
        self.assertEqual(contract.metadata["source_path"], str(self.path))
        self.assertEqual(contract.metadata["bytes"], len(raw))
        self.assertIn("not an independent", contract.text)
        self.assertIn("partial", contract.text)
        contract.document["scope"]["layout_id"] = "changed"
        contract.metadata["scope"]["device_id"] = "changed"
        self.assertEqual(contract.document, DOCUMENT)
        self.assertEqual(self.path.read_bytes(), raw)
        with self.assertRaises(FrozenInstanceError):
            contract.raw = b"changed"

    def test_contract_requires_exact_schema_and_versions(self):
        for field, value in (
            ("extra", True), ("protocol", "another-protocol"), ("id", ""),
            ("id", "中文"), ("id", "a" * 65), ("id", "bad id"),
            ("version", True), ("version", 0), ("version", 1.0),
            ("objective", " "), ("objective", "a" * 4097),
            ("target_text", ""), ("target_text", " " * 4),
            ("target_text", "Hello"), ("target_text", "hi\n"),
            ("target_text", "a" * 257), ("completion_requirements", []),
            ("completion_requirements", list(reversed(COMPLETION_REQUIREMENTS))),
        ):
            data = json.loads(json.dumps(DOCUMENT))
            data[field] = value
            self.path.write_text(json.dumps(data))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load()
        for key in DOCUMENT:
            data = dict(DOCUMENT)
            del data[key]
            self.path.write_text(json.dumps(data))
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.load()

    def test_expected_and_document_scopes_are_exact(self):
        for key in SCOPE:
            data = json.loads(json.dumps(DOCUMENT))
            data["scope"][key] = "other"
            self.path.write_text(json.dumps(data))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        self.path.write_text(json.dumps(DOCUMENT))
        for expected in ({}, dict(SCOPE, extra="a"), dict(SCOPE, task_id=" ")):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                load_task_contract(self.path, expected)

    def test_duplicate_members_and_nonfinite_json_are_rejected(self):
        valid = json.dumps(DOCUMENT)
        invalid = (
            valid.replace('"version": 1', '"version": 1, "version": 2'),
            valid.replace('"task_id": "physical_keyboard_view"',
                          '"task_id":"physical_keyboard_view","task_id":"other"'),
            valid.replace('"version": 1', '"version": NaN'),
            valid.replace('"version": 1', '"version": Infinity'),
            valid.replace('"version": 1', '"version": 1e99999'),
            "[]", "{}", "{", "null",
        )
        for raw in invalid:
            self.path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.load()
        self.path.write_bytes(b"\xff")
        with self.assertRaises(ValueError):
            self.load()

    def test_oversized_and_nonregular_inputs_are_rejected(self):
        self.path.write_bytes(b" " * (32 * 1024 + 1))
        with self.assertRaises(ValueError):
            self.load()
        self.path.unlink()
        self.path.mkdir()
        with self.assertRaises(ValueError):
            self.load()

    def test_symlinked_file_and_parent_are_rejected(self):
        alias = self.base / "alias.json"
        alias.symlink_to(self.path)
        with self.assertRaises(ValueError):
            load_task_contract(alias, SCOPE)
        directory = self.base / "linked-dir"
        directory.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            load_task_contract(directory / self.path.name, SCOPE)


class LifecycleTests(ContractFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.lifecycle = CampaignLifecycle(self.load())
        self.completed = {
            "status": "completed", "observed_text": "hello",
            "reflection": {"summary": "Model claims the target is visible.",
                           "limitations": ["Visual model report only."]},
        }
        self.args = {
            "review_observation_evidence_id": "obs-review-1", "observed_text": "hello",
            "released_and_input_stable": True, "skill_candidate_ids": ["skill-example"],
        }
        self.skill = {
            "candidate_id": "skill-example", "label": "example",
            "source": "def main(robot, profile):\n    return robot.observe()\n",
            "profile": {},
        }

    def begin(self):
        return self.lifecycle.begin_completion_review(self.completed, "obs-review-1")

    def confirm(self, arguments=None, observation_id="obs-review-1", get_skill=None):
        return self.lifecycle.confirm_completion(
            self.args if arguments is None else arguments, observation_id,
            get_skill or (lambda identifier: dict(self.skill)),
        )

    def test_all_partial_finishes_are_nonterminal_checkpoints(self):
        result = {"status": "partial", "reflection": {"summary": "first phase"}}
        for number, status in enumerate(
                ("partial", "blocked", "view_ready", "calibrated_partial", "partial"), 1):
            result["status"] = status
            feedback = self.lifecycle.checkpoint(result, requested_action="finish")
            self.assertFalse(feedback["terminal"])
            self.assertEqual(feedback["checkpoint_number"], number)
            self.assertEqual(feedback["result"], result)
            self.assertEqual(feedback["task_contract"], DOCUMENT)
            self.assertEqual(self.lifecycle.checkpoint_count, number)
            self.assertIsNone(self.lifecycle.pending_completion)
            feedback["result"]["reflection"]["summary"] = "mutated"
            self.assertEqual(result["reflection"]["summary"], "first phase")

    def test_nonterminal_checkpoint_does_not_clear_pending_review(self):
        self.begin()
        before = self.lifecycle.pending_completion
        self.lifecycle.checkpoint({"status": "partial"})
        self.assertEqual(self.lifecycle.pending_completion, before)
        for status in ("completed", "invented", None, []):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.lifecycle.checkpoint({"status": status})
        self.assertEqual(self.lifecycle.checkpoint_count, 1)

    def test_completion_must_first_request_a_completed_review(self):
        callback = Mock(return_value=self.skill)
        with self.assertRaises(ValueError):
            self.confirm(get_skill=callback)
        callback.assert_not_called()
        for result in ({"status": "partial"}, {"status": "blocked"}, {}, None):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.lifecycle.begin_completion_review(result, "obs-review-1")
        feedback = self.begin()
        self.assertFalse(feedback["terminal"])
        self.assertEqual(feedback["review_observation_evidence_id"], "obs-review-1")
        self.completed["reflection"]["summary"] = "changed outside"
        feedback["result"]["reflection"]["summary"] = "changed feedback"
        copy = self.lifecycle.pending_completion
        copy["result"]["reflection"]["summary"] = "changed copy"
        self.assertEqual(self.lifecycle.pending_completion["result"]["reflection"]["summary"],
                         "Model claims the target is visible.")

    def test_completion_carries_explicit_model_basis_without_promoting_skill(self):
        self.begin()
        callback = Mock(return_value=self.skill)
        result = self.confirm(get_skill=callback)
        callback.assert_called_once_with("skill-example")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["reflection"], self.completed["reflection"])
        review = result["completion_review"]
        self.assertEqual(review["basis"], "model_visual_confirmation")
        self.assertIs(review["independently_verified"], False)
        self.assertEqual(review["skill_candidate_ids"], ["skill-example"])
        self.assertNotIn("stable_promoted", result)
        self.assertIsNone(self.lifecycle.pending_completion)

    def test_wrong_text_stale_observation_and_nontrue_release_do_not_complete(self):
        self.begin()
        before = self.lifecycle.pending_completion
        callback = Mock(return_value=self.skill)
        for field, value in (
            ("observed_text", "Hello"), ("observed_text", "helloo"),
            ("observed_text", "hello "), ("observed_text", None),
            ("review_observation_evidence_id", "old-observation"),
            ("released_and_input_stable", False), ("released_and_input_stable", 1),
            ("released_and_input_stable", "true"),
        ):
            args = dict(self.args, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.confirm(args, get_skill=callback)
            self.assertEqual(self.lifecycle.pending_completion, before)
        with self.assertRaises(ValueError):
            self.confirm(observation_id="newer-observation", get_skill=callback)
        callback.assert_not_called()

    def test_confirmation_schema_and_skill_ids_are_strict(self):
        self.begin()
        callback = Mock(return_value=self.skill)
        for args in (
            dict(self.args, extra=True),
            {key: value for key, value in self.args.items() if key != "observed_text"},
            dict(self.args, skill_candidate_ids=[]),
            dict(self.args, skill_candidate_ids=["skill-example", "skill-example"]),
            dict(self.args, skill_candidate_ids=[True]),
            dict(self.args, skill_candidate_ids="skill-example"),
            dict(self.args, skill_candidate_ids=[" "]),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.confirm(args, get_skill=callback)
        callback.assert_not_called()
        self.assertIsNotNone(self.lifecycle.pending_completion)

    def test_lookup_failure_or_forged_candidate_scope_retains_pending(self):
        self.begin()
        before = self.lifecycle.pending_completion
        for skill in (
            None, {}, dict(self.skill, candidate_id="other"),
            dict(self.skill, profile=None), dict(self.skill, source=" "),
            dict(self.skill, kind="lesson"),
            dict(self.skill, scope=dict(SCOPE, device_id="other")),
            dict(self.skill, profile={"bad": float("nan")}),
        ):
            with self.subTest(skill=skill), self.assertRaises(ValueError):
                self.confirm(get_skill=lambda identifier, result=skill: result)
            self.assertEqual(self.lifecycle.pending_completion, before)
        with self.assertRaises(ValueError):
            self.confirm(get_skill=Mock(side_effect=ValueError("not registered")))
        self.assertEqual(self.lifecycle.pending_completion, before)

    def test_new_observation_or_motion_invalidation_disallows_confirmation(self):
        self.begin()
        self.lifecycle.invalidate_completion_review()
        self.assertIsNone(self.lifecycle.pending_completion)
        with self.assertRaises(ValueError):
            self.confirm()
        self.begin()
        self.assertEqual(self.confirm()["status"], "completed")

    def test_real_store_allows_historical_scoped_candidate_without_execution_or_promotion(self):
        receipt = self.base / "evidence.json"
        receipt.write_text('{"diagnostic":true}')
        store = KnowledgeStore(self.base / "knowledge", SCOPE, "previous-run")
        store.register_evidence("obs-previous", [receipt])
        saved = store.propose("skill", {
            "id": "previous-observe", "source_lines": self.skill["source"].splitlines(),
            "profile": {}, "applicability": ["Same rig only."],
            "limitations": ["Candidate is not a verified typing skill."],
            "evidence_ids": ["obs-previous"],
        })
        before = (store.namespace / "index.jsonl").read_bytes()
        self.begin()
        args = dict(self.args, skill_candidate_ids=[saved["candidate_id"]])
        with patch("astra_robotics_agent.physical_keyboard.skill_language.run_skill",
                   side_effect=AssertionError("must not execute")):
            result = self.confirm(args, get_skill=store.get_skill)
        self.assertEqual(result["completion_review"]["skill_candidate_ids"], [saved["candidate_id"]])
        self.assertEqual((store.namespace / "index.jsonl").read_bytes(), before)
        self.assertEqual(saved["status"], "candidate_not_promoted")
        other = KnowledgeStore(self.base / "knowledge", dict(SCOPE, layout_id="other"), "next-run")
        self.begin()
        with self.assertRaises(ValueError):
            self.confirm(args, get_skill=other.get_skill)
        self.assertIsNotNone(self.lifecycle.pending_completion)

    def test_assistance_requires_explicit_external_category_current_view_and_human_action(self):
        request = {"kind": "operator_action", "reason": "Keyboard fixture moved.",
                   "required_human_action": "Refix the keyboard before motion.",
                   "observation_evidence_id": "obs-current"}
        for kind in ("physical_safety", "hardware_fault", "operator_action", "missing_external_capability"):
            response = self.lifecycle.validate_assistance(dict(request, kind=kind), "obs-current")
            self.assertEqual(response["status"], "assistance_required")
            self.assertIs(response["terminal"], True)
        for change in (
            {"kind": "ordinary_uncertainty"}, {"kind": "blocked"},
            {"kind": []},
            {"reason": " "}, {"required_human_action": ""},
            {"observation_evidence_id": "old-view"}, {"extra": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.lifecycle.validate_assistance(dict(request, **change), "obs-current")
        with self.assertRaises(ValueError):
            self.lifecycle.validate_assistance({"kind": "physical_safety"}, "obs-current")


if __name__ == "__main__":
    unittest.main()
