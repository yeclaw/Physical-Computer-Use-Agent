"""Verified per-operation memoization; no SDK, model, or real knowledge root."""
from copy import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeError, KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources


SCOPE = dict(task_id='read-scale-fixture', device_id='fake', layout_id='fake')


class KnowledgeReadScaleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.access = LearningAccess(SCOPE)
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'read-scale', learning_access=self.access)
        evidence = self.root / 'response.json'; evidence.write_text(json.dumps(dict(state={}, snapshot={})))
        self.store.register_evidence('obs', [evidence])
        self.library = SkillDevelopment(self.store)
        self.resources = LearningResources(self.store, access=self.access)
        self.changes = LearningChanges(self.library, self.resources, access=self.access)

    def package(self, name, dependencies=None):
        return self.library.save_package(dict(id=name,
            source_lines=['def main(robot, profile):', '    return profile["value"]'], profile={'value': 1},
            contract=dict(description='Inert CPU fixture', input_schema=dict(type='object', properties={},
                additionalProperties=False), preconditions=['Fixture only'], postconditions=['No robot effect']),
            applicability=['Test fixture'], limitations=['No physical evidence'], evidence_ids=['obs'],
            dependencies={} if dependencies is None else dependencies))

    def test_shared_dag_complete_closure_nodes_once_and_journal_once(self):
        base = self.package('base'); previous = [base['package_id'], base['package_id']]
        identifiers = [base['package_id']]
        for layer in range(4):
            current = [self.package('node%d%s' % (layer, side),
                {'left': previous[0], 'right': previous[1]})['package_id'] for side in ('a', 'b')]
            identifiers.extend(current); previous = current
        root = self.package('root', {'left': previous[0], 'right': previous[1]})['package_id']
        identifiers.append(root)
        with patch.object(self.library, '_load_record_access', wraps=self.library._load_record_access) as loaded:
            with self.store.read_snapshot() as snapshot:
                for _ in range(3):
                    self.assertEqual(self.library.inspect(root)['record_id'], root)
                # One canonical dependency map makes the complete shared DAG linear.
                self.assertEqual(loaded.call_count, len(identifiers))
                self.assertEqual(snapshot.stats['events_misses'], 1)
                self.assertEqual(snapshot.stats['development_index_misses'], 1)
                self.assertEqual(snapshot.stats['evidence_misses'], 1)
                self.assertEqual(snapshot.stats['closure_misses'], len(identifiers))
                self.assertGreater(snapshot.stats['closure_hits'], 0)
        self.assertFalse(snapshot.memo)

    def test_nested_copied_store_same_policy_reuses_one_snapshot(self):
        saved = self.package('nested')
        alias = copy(self.store)
        with self.store.read_snapshot() as outer:
            with alias.read_snapshot() as inner:
                self.assertIs(outer, inner)
                self.store.get_skill(saved['candidate_id'])
                alias.get_skill(saved['candidate_id'])
                self.assertEqual(inner.stats['candidate_misses'], 1)
                self.assertEqual(inner.stats['evidence_misses'], 1)

    def test_caller_profile_mutation_does_not_poison_verified_candidate(self):
        saved = self.package('profile')
        with self.store.read_snapshot():
            self.store.get_skill(saved['candidate_id'])['profile']['value'] = 'changed'
            self.assertEqual(self.store.get_skill(saved['candidate_id'])['profile']['value'], 1)

    def test_next_operation_detects_candidate_corruption_after_successful_read(self):
        saved = self.package('corrupt')
        with self.store.read_snapshot(): self.library.inspect(saved['package_id'])
        path = self.store.namespace / 'candidates' / saved['candidate_id'] / 'candidate.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(KnowledgeError): self.library.inspect(saved['package_id'])

    def test_file_mutation_during_snapshot_cannot_leave_the_boundary(self):
        saved = self.package('mutated')
        path = self.store.namespace / 'candidates' / saved['candidate_id'] / 'source.py'
        with self.assertRaisesRegex(KnowledgeError, 'changed'):
            with self.store.read_snapshot():
                self.store.get_skill(saved['candidate_id'])
                path.write_bytes(path.read_bytes() + b'\n')
                # Cached historical data can never be published after this change.
                self.store.get_skill(saved['candidate_id'])

    def test_policy_changes_and_store_policy_replacement_rejected(self):
        saved = self.package('policy')
        with self.assertRaisesRegex(KnowledgeError, 'policy changed'):
            with self.store.read_snapshot():
                self.store.get_skill(saved['candidate_id'])
                self.access.allowed_sources.add('new-authority')
                self.store.get_skill(saved['candidate_id'])
        self.access.allowed_sources.remove('new-authority')
        with self.store.read_snapshot():
            self.store.learning_access = self.access  # Same object, no mutation.
            with self.assertRaisesRegex(KnowledgeError, 'writes are forbidden'):
                self.store.learning_access = LearningAccess(SCOPE)

    def test_all_real_write_routes_reject_before_any_files_change(self):
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob('*') if path.is_file()}
        with self.store.read_snapshot():
            for call in (
                lambda: self.store.register_evidence('new', []),
                lambda: self.store.propose('lesson', {}),
                lambda: self.library.save_package({}),
                lambda: self.library.record_improvement({}),
                lambda: self.changes.commit({}),
                lambda: self.resources.put('new', {}, source_ids=['source']),
            ):
                with self.assertRaisesRegex(KnowledgeError, 'writes are forbidden'): call()
        after = {str(path.relative_to(self.root)): path.read_bytes()
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_journal_append_or_new_registration_invalidates_the_read(self):
        with self.assertRaisesRegex(KnowledgeError, 'changed'):
            with self.store.read_snapshot():
                self.store._events()
                path = self.store.namespace / 'index.jsonl'
                with path.open('ab') as stream: stream.write(b'\n')
        with self.assertRaisesRegex(KnowledgeError, 'changed'):
            with self.store.read_snapshot():
                path = self.store.namespace / 'learning/changes/commits/new.json'
                path.write_text('{}')

    def test_directory_membership_change_rejected_even_when_all_stat_fields_match(self):
        directory = self.store.namespace / 'learning/changes/commits'
        frozen = directory.lstat()
        original_lstat = Path.lstat
        before_names = sorted(path.name for path in directory.iterdir())
        def unchanged_directory_stat(path, *args, **kwargs):
            actual = original_lstat(path, *args, **kwargs)
            return frozen if path == directory else actual
        def signature(value):
            return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                    value.st_mtime_ns, value.st_ctime_ns)
        # Deterministic on Linux and Darwin: do not depend on timing granularity
        # or whether the filesystem changes a directory's byte-size field.
        with patch.object(Path, 'lstat', new=unchanged_directory_stat):
            with self.assertRaisesRegex(KnowledgeError, 'changed'):
                with self.store.read_snapshot():
                    self.store._events()
                    (directory / 'same-stat-new-member.json').write_text('{}')
                    self.assertEqual(signature(directory.lstat()), signature(frozen))
                    self.assertNotEqual(sorted(path.name for path in directory.iterdir()), before_names)


if __name__ == '__main__': unittest.main()
