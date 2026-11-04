"""Request-local verified reads; no persistent cache or physical observation."""
from contextlib import contextmanager
import hashlib
import unittest
from unittest.mock import patch

import test_keyboard_continual_actions as action_fixture
from astra_robotics_agent.physical_keyboard.records.journal import SegmentedJournal
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeError


class ReadSnapshotBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = action_fixture.ContinualActionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.development = self.fixture.development
        self.learning = self.fixture.learning
        self.observations = []
        self.depth = 0
        original = type(self.store).read_snapshot

        @contextmanager
        def tracked(current_store):
            with original(current_store) as snapshot:
                self.depth += 1
                self.observations.append(snapshot)
                try:
                    yield snapshot
                finally:
                    self.depth -= 1
        self.addCleanup(patch.stopall)
        patch.object(type(self.store), 'read_snapshot', tracked).start()

    def package(self, name='bounded-reading', **extra):
        args = dict(id=name, source_lines=['def main(robot, profile):', '    return profile["value"]'],
            profile={'value': 1}, contract=dict(description='Only a fixture',
                input_schema={'type':'object','properties':{},'additionalProperties':False},
                preconditions=['Fixture start'], postconditions=['Observe fixture result']),
            applicability=['Fixture'], limitations=['Not a physical verification'], evidence_ids=['obs'])
        args.update(extra)
        return args

    def saved(self, name='bounded-reading'):
        return self.learning.save_package(self.package(name))

    def handle(self, action, arguments):
        return self.learning.handle(action, arguments, facts=self.fixture.facts())

    def one_snapshot(self, callback):
        self.observations.clear()
        result = callback()
        self.assertTrue(self.observations, 'the read must enter an explicit request boundary')
        self.assertEqual(len({id(snapshot) for snapshot in self.observations}), 1)
        self.assertEqual(self.depth, 0)
        return result, self.observations[0]

    def test_directory_nested_selection_and_relation_read_only_once_per_request(self):
        package = self.saved()
        self.learning.changes.commit(dict(selections=[dict(logical_id='bounded-reading',
            identifier=package['package_id'], conditions=['Fixture neutral'], reason='Fixture condition')]))
        counts = {}
        original = SegmentedJournal.lines
        def counted(journal):
            counts[str(journal.path)] = counts.get(str(journal.path), 0) + 1
            yield from original(journal)
        with patch.object(SegmentedJournal, 'lines', counted):
            first, previous = self.one_snapshot(self.learning.view.directory)
        self.assertEqual(first['items'][0]['logical_id'], 'bounded-reading')
        self.assertTrue(first['items'][0]['selected'])
        self.assertEqual(counts[str(self.store.namespace / 'index.jsonl')], 1)
        self.assertEqual(counts[str(self.development.library.index)], 1)
        second, current = self.one_snapshot(self.learning.view.directory)
        self.assertIsNot(previous, current)
        self.assertEqual(first, second)

    def test_continual_context_shares_directory_attention_and_next_request_revalidates(self):
        package = self.saved()
        self.development.focus_skill_ids = [package['package_id']]
        result, first = self.one_snapshot(lambda:self.learning.context('obs'))
        self.assertIn('directory', result)
        self.assertIn('attention', result)
        again, second = self.one_snapshot(lambda:self.learning.context('obs'))
        self.assertIsNot(first, second)
        self.assertEqual(result, again)

    def test_legacy_context_recall_inspect_and_call_resolution_use_explicit_boundaries(self):
        package = self.saved()
        self.development.focus_skill_ids = [package['package_id']]
        self.one_snapshot(self.development.context)
        self.one_snapshot(lambda:self.development.before_use([package['package_id']], 'unrelated fixture source'))
        self.one_snapshot(lambda:self.development.knowledge_action('inspect_knowledge',
            {'identifier':package['package_id']}, 'obs'))
        resolved, _ = self.one_snapshot(lambda:self.development.resolve_call(
            {'package_id':package['package_id'], 'arguments':{}}))
        self.assertEqual(resolved['package_id'], package['package_id'])
        self.assertEqual(resolved['profile'], {'value':1})

    def test_reader_handle_discards_verified_bytes_between_requests(self):
        package = self.saved()
        self.one_snapshot(lambda:self.handle('read_knowledge', {'identifier':package['package_id']}))
        record = self.store._evidence('obs')
        path = self.store.namespace / 'objects' / record['files'][0]['sha256']
        path.chmod(0o600)
        path.write_bytes(b'corrupted immutable fixture bytes')
        with self.assertRaises(ValueError):
            self.handle('read_knowledge', {'identifier':package['package_id']})

    def test_learning_attention_evidence_cursor_reuses_only_request_local_objects(self):
        initial, first = self.one_snapshot(self.learning.attention.new_evidence)
        empty, second = self.one_snapshot(lambda:self.learning.attention.new_evidence(initial['next_cursor']))
        self.assertIsNot(first, second)
        self.assertEqual(empty['items'], [])

    def test_knowledge_writers_and_analysis_resource_put_are_outside_read_scope(self):
        put = self.learning.resources.put
        committed = self.learning.changes.commit
        observed = []
        def checked_put(*args, **kwargs):
            observed.append(('put', self.depth)); self.assertEqual(self.depth, 0)
            return put(*args, **kwargs)
        def checked_commit(*args, **kwargs):
            observed.append(('commit', self.depth)); self.assertEqual(self.depth, 0)
            return committed(*args, **kwargs)
        with patch.object(self.learning.resources, 'put', checked_put), \
                patch.object(self.learning.changes, 'commit', checked_commit):
            self.handle('analyze_learning_sources', dict(id='analysis',
                source_lines=['def analyze(data, parameters):','    return {"value": len(data["input"]["samples"])}'],
                inputs={'input':{'source_id':'numeric'}}, parameters={}))
            self.handle('commit_learning', {'packages':[self.package()]})
        self.assertIn(('put', 0), observed)
        self.assertIn(('commit', 0), observed)
        self.assertEqual(self.depth, 0)

    def test_old_save_and_outcome_routes_are_not_wrapped_in_read_scope(self):
        with patch.object(self.development.library, 'save_package', side_effect=lambda args:
                self.assertEqual(self.depth, 0) or {'package_id':'fixture'}) as save:
            self.development.knowledge_action('save_skill_package', self.package(), 'obs')
        save.assert_called_once()
        with patch.object(self.development.library, 'record_improvement', side_effect=lambda args:
                self.assertEqual(self.depth, 0) or {'record_id':'fixture'}) as writer:
            self.development.knowledge_action('record_improvement', {'skill_ids':[]}, 'obs')
        writer.assert_called_once()

    def test_explicit_read_scope_rejects_all_knowledge_writers_before_any_files(self):
        def files():
            return {str(path.relative_to(self.store.namespace)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.store.namespace.rglob('*') if path.is_file()}
        before = files()
        with self.store.read_snapshot():
            with self.assertRaises(KnowledgeError):
                self.learning.resources.put('not-published', {'value':1}, source_ids=['numeric'])
            with self.assertRaises(KnowledgeError):
                self.handle('commit_learning', {'packages':[self.package()]})
            with self.assertRaises(KnowledgeError):
                self.learning.save_package(self.package('no-write'))
        self.assertEqual(before, files())


if __name__ == '__main__':
    unittest.main()
