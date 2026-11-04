"""Candidate storage with literal receipt fixtures; no source is executed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.knowledge import KnowledgeError, KnowledgeStore


SCOPE = {"task_id": "physical_keyboard_view", "device_id": "shanghai_startouch",
         "layout_id": "keyboard_20261022_a"}
RECEIPT = b'{"status":"ok","actual_pitch_delta_deg":3.563,"xyz_drift_mm":2.223}\n'
SOURCE = ["def main(robot, profile):", "    state = robot.observe()", "    return state"]


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="keyboard-knowledge-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.receipt = self.base / "actual-receipt.json"
        self.receipt.write_bytes(RECEIPT)
        self.store = KnowledgeStore(self.base / "store", dict(SCOPE), "EXP-016-a01")
        self.evidence = self.store.register_evidence("view-probe-1", [self.receipt])

    def lesson(self, **updates):
        result = {"id": "improve-view", "claim": "A small view probe exposed more local letter keys.",
                  "applicability": ["Current keyboard and rig only; inspect the new image first."],
                  "limitations": ["Full rows and contact clearance remain unverified."],
                  "evidence_ids": ["view-probe-1"]}
        result.update(updates)
        return result

    def skill(self, **updates):
        result = {"id": "observe-view", "source_lines": list(SOURCE), "profile": {"purpose": "inspect"},
                  "applicability": ["Read-only current state inspection."],
                  "limitations": ["Does not establish keyboard contact geometry."],
                  "evidence_ids": ["view-probe-1"]}
        result.update(updates)
        return result

    def test_registered_evidence_is_frozen_and_idempotent(self):
        record_path = Path(self.evidence["path"])
        self.assertEqual(self.evidence["sha256"], hashlib.sha256(record_path.read_bytes()).hexdigest())
        self.assertEqual(self.store.register_evidence("view-probe-1", [self.receipt]), self.evidence)
        digest = hashlib.sha256(RECEIPT).hexdigest()
        self.assertEqual((self.store.namespace / "objects" / digest).read_bytes(), RECEIPT)
        self.receipt.write_text("changed original")
        candidate = self.store.propose("lesson", self.lesson())
        saved = json.loads((Path(candidate["path"]) / "candidate.json").read_bytes())
        self.assertEqual(saved["evidence"][0]["files"][0]["sha256"], digest)
        self.assertEqual((self.store.namespace / "objects" / digest).read_bytes(), RECEIPT)
        with self.assertRaises(KnowledgeError):
            self.store.register_evidence("view-probe-1", [self.receipt])

    def test_missing_and_forged_evidence_references_are_rejected(self):
        forged = self.store.namespace / "evidence" / "invented.json"
        forged.write_text(json.dumps({"scope": SCOPE, "evidence_id": "invented", "files": []}))
        for ids in ([], ["missing"], ["invented"], ["view-probe-1", "view-probe-1"], [True]):
            with self.subTest(ids=ids), self.assertRaises(KnowledgeError):
                self.store.propose("lesson", self.lesson(evidence_ids=ids))
        with self.assertRaises(KnowledgeError):
            self.store.propose("lesson", dict(self.lesson(), evidence_path=str(self.receipt)))

    def test_candidate_versions_append_without_overwriting(self):
        first = self.store.propose("lesson", self.lesson())
        frozen = {p.name: p.read_bytes() for p in Path(first["path"]).iterdir()}
        self.assertEqual(self.store.propose("lesson", self.lesson()), first)
        second = self.store.propose("lesson", self.lesson(claim="The expanded view remains incomplete."))
        self.assertEqual((first["version"], second["version"]), (1, 2))
        self.assertNotEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual({p.name: p.read_bytes() for p in Path(first["path"]).iterdir()}, frozen)
        saved = json.loads((Path(second["path"]) / "candidate.json").read_bytes())
        self.assertEqual(saved["origin"], "astra_generated")
        self.assertEqual(saved["status"], "candidate_not_promoted")
        self.assertEqual(saved["scope"], SCOPE)

    def test_skill_static_check_never_executes_and_has_safe_host_label(self):
        with patch("astra_robotics_agent.physical_keyboard.skill_language.run_skill",
                   side_effect=AssertionError("must not execute")):
            result = self.store.propose("skill", self.skill())
            skill = self.store.get_skill(result["candidate_id"])
        self.assertEqual(skill["source"], "\n".join(SOURCE) + "\n")
        self.assertEqual(skill["profile"], {"purpose": "inspect"})
        self.assertRegex(skill["label"], r"^[A-Za-z0-9_-]{1,64}$")
        lesson = self.store.propose("lesson", self.lesson())
        with self.assertRaises(KnowledgeError):
            self.store.get_skill(lesson["candidate_id"])

    def test_operator_revision_keeps_history_and_lessons_with_distinct_provenance(self):
        versions = [self.store.propose('skill', self.skill(profile={'angle': angle})) for angle in (3, -3, 2)]
        lesson = self.store.propose('lesson', self.lesson())
        receipt = self.store.record_verification(versions[-1]['candidate_id'], ['view-probe-1'], 'ok')
        prior_files = {str(path): path.read_bytes() for item in versions + [lesson]
                       for path in Path(item['path']).iterdir()}
        prior_files[receipt['path']] = Path(receipt['path']).read_bytes()
        original_index = (self.store.namespace / 'index.jsonl').read_bytes()
        updated = self.store.revise_skill(versions[-1]['candidate_id'], self.skill(profile={'offset_xyz_m': [0, 0, 0]}),
                                          reason='User requested removal of former exploration constraints.')
        self.assertEqual(updated['version'], 4)
        saved = self.store._candidate(updated['candidate_id'])
        self.assertEqual(saved['origin'], 'human_requested_revision')
        self.assertEqual(saved['revision']['parent_candidate_id'], versions[-1]['candidate_id'])
        self.assertEqual(saved['status'], 'candidate_not_promoted')
        self.assertTrue((self.store.namespace / 'index.jsonl').read_bytes().startswith(original_index))
        for name, data in prior_files.items():
            self.assertEqual(Path(name).read_bytes(), data)
        reopened = KnowledgeStore(self.base / 'store', dict(SCOPE), 'next-run')
        context = reopened.context_text()
        self.assertIn(lesson['candidate_id'], context)
        self.assertIn(updated['candidate_id'], context)
        self.assertIn('"origin":"human_requested_revision"', context)
        self.assertIn('"last_execution_verification":"none"', context)
        for old in versions:
            self.assertNotIn(old['candidate_id'], context)
            self.assertIn('return state', reopened.get_skill(old['candidate_id'])['source'])

    def test_bad_operator_revision_does_not_supersede_or_relabel_candidates(self):
        original = self.store.propose('skill', self.skill())
        newer = self.store.propose('skill', self.skill(profile={'revision': 2}))
        lesson = self.store.propose('lesson', self.lesson())
        index = (self.store.namespace / 'index.jsonl').read_bytes()
        directory_names = sorted(path.name for path in (self.store.namespace / 'candidates').iterdir())
        invalid = [
            (original['candidate_id'], self.skill(), 'stale parent'),
            (lesson['candidate_id'], self.skill(), 'wrong kind'),
            (newer['candidate_id'], self.skill(id='different'), 'wrong identity'),
            (newer['candidate_id'], self.skill(), ''),
            (newer['candidate_id'], self.skill(evidence_ids=['imaginary']), 'invalid evidence'),
            (newer['candidate_id'], self.skill(source_lines=['import os']), 'invalid AST'),
        ]
        for parent, arguments, reason in invalid:
            with self.subTest(reason=reason), self.assertRaises(KnowledgeError):
                self.store.revise_skill(parent, arguments, reason=reason)
            self.assertEqual((self.store.namespace / 'index.jsonl').read_bytes(), index)
            self.assertEqual(sorted(path.name for path in (self.store.namespace / 'candidates').iterdir()), directory_names)
        with self.assertRaises(KnowledgeError):
            self.store.propose('skill', dict(self.skill(), origin='human_requested_revision'))
        other = KnowledgeStore(self.base / 'store', dict(SCOPE, layout_id='different'), 'other')
        with self.assertRaises(KnowledgeError):
            other.revise_skill(newer['candidate_id'], self.skill(), reason='cross-scope')

    def test_ast_rejects_dangerous_dead_branch_and_invalid_entries(self):
        bad_sources = [
            ["def main(robot, profile):", "    if False:", "        import os", "    return None"],
            ["def main(robot, profile):", "    if False:", "        open('/tmp/forged', 'w')", "    return None"],
            ["def main(robot, profile):", "    return robot.__class__"],
            ["robot.stop()", "def main(robot, profile):", "    return None"],
            ["def main(robot):", "    return None"],
            ["def helper(robot, profile):", "    return None"],
        ]
        for source in bad_sources:
            with self.subTest(source=source), self.assertRaises(KnowledgeError):
                self.store.propose("skill", self.skill(source_lines=source))

    def test_profile_authority_and_nonfinite_values_are_rejected(self):
        for key in ("execute", "safety", "max_translation", "max_rotation_deg", "service_tier",
                    "workspace_min_xyz", "max_step_translation_m", "speed_percent"):
            with self.subTest(key=key), self.assertRaises(KnowledgeError):
                self.store.propose("skill", self.skill(profile={"nested": {key: True}}))
        for value in (float("nan"), float("inf"), object(), (1, 2)):
            with self.subTest(value=repr(value)), self.assertRaises(KnowledgeError):
                self.store.propose("skill", self.skill(profile={"value": value}))

    def test_candidate_source_profile_and_evidence_hashes_are_checked(self):
        result = self.store.propose("skill", self.skill())
        for name in ("source.py", "profile.json", "candidate.json", "manifest.json"):
            path = Path(result["path"]) / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaises(KnowledgeError):
                self.store.get_skill(result["candidate_id"])
            path.write_bytes(original)
        digest = hashlib.sha256(RECEIPT).hexdigest()
        (self.store.namespace / "objects" / digest).write_bytes(b"corrupt")
        with self.assertRaises(KnowledgeError):
            self.store.propose("lesson", self.lesson())

    def test_paths_and_scope_are_not_model_controlled(self):
        for identifier in ("../outside", "a/b", "", "unicode-\u4e2d", "a" * 65):
            with self.subTest(identifier=identifier), self.assertRaises(KnowledgeError):
                self.store.propose("lesson", self.lesson(id=identifier))
        for candidate in ("../outside", "/private/tmp/forged", "skill-unknown"):
            with self.subTest(candidate=candidate), self.assertRaises(KnowledgeError):
                self.store.get_skill(candidate)
        with self.assertRaises(KnowledgeError):
            self.store.propose("lesson", dict(self.lesson(), scope={"device_id": "other"}))

    def test_reopen_preserves_candidates_and_filters_other_layout(self):
        candidate = self.store.propose("skill", self.skill())
        reopened = KnowledgeStore(self.base / "store", dict(SCOPE), "EXP-016-a02")
        self.assertEqual(reopened.get_skill(candidate["candidate_id"]), self.store.get_skill(candidate["candidate_id"]))
        self.assertIn(candidate["candidate_id"], reopened.context_text())
        other = KnowledgeStore(self.base / "store", dict(SCOPE, layout_id="different"), "EXP-016-a03")
        self.assertNotIn(candidate["candidate_id"], other.context_text())
        with self.assertRaises(KnowledgeError):
            other.get_skill(candidate["candidate_id"])
        with self.assertRaises(KnowledgeError):
            other.propose("skill", self.skill())

    def test_verification_unknown_cannot_pass_or_promote(self):
        candidate = self.store.propose("skill", self.skill())
        expected = {"ok": "verified_execution_only", "blocked": "diagnostic_failed",
                    "error": "diagnostic_failed", "unknown": "verification_unknown"}
        records = []
        for status, verdict in expected.items():
            record = self.store.record_verification(candidate["candidate_id"], ["view-probe-1"], status)
            self.assertEqual(record["status"], verdict)
            self.assertEqual(record["text_success"], "unknown")
            self.assertIs(record["stable"], False)
            records.append(record["path"])
        self.assertEqual(len(set(records)), 4)
        self.assertIn('"last_execution_verification":"verification_unknown"', self.store.context_text())
        self.assertIn('"status":"candidate_not_promoted"', self.store.context_text())
        with self.assertRaises(KnowledgeError):
            self.store.record_verification(candidate["candidate_id"], ["view-probe-1"], "success")

    def test_evidence_limits_symlinks_and_short_context(self):
        for paths in ([], [self.receipt] * 17):
            with self.assertRaises(KnowledgeError):
                self.store.register_evidence("new", paths)
        alias = self.base / "alias"
        alias.symlink_to(self.receipt)
        with self.assertRaises(KnowledgeError):
            self.store.register_evidence("alias", [alias])
        large = self.base / "large"
        large.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
        with self.assertRaises(KnowledgeError):
            self.store.register_evidence("large", [large])
        for index in range(3):
            self.store.propose("lesson", self.lesson(id="view" + str(index), claim="word " * 500))
        context = self.store.context_text(limit=2)
        self.assertIn("KNOWLEDGE CANDIDATES — NOT STABLE", context)
        self.assertEqual(context.count('"candidate_id"'), 2)
        self.assertLess(len(context), 3000)

    def test_default_import_has_no_sdk_network_or_execution(self):
        code = '''
import builtins, sys
original = builtins.__import__
forbidden = {'socket','urllib','http','requests','easydp','startouch','torch'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def audit(event, args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        raise AssertionError('unexpected side effect: ' + event)
sys.addaudithook(audit)
from astra_robotics_agent.physical_keyboard.knowledge import KnowledgeStore
print('import-only: no SDK/network/process calls')
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no SDK/network/process calls", result.stdout)


if __name__ == "__main__":
    unittest.main()
