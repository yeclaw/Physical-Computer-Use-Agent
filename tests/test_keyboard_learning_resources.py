"""Temporary-only resources with exact hashes and source closure."""
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources


SCOPE = {'task_id': 'fixture', 'device_id': 'fixture_robot', 'layout_id': 'fixture_layout'}


class Fixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'fixture-run')
        self.evidence = self.root / 'evidence.json'
        self.evidence.write_text('{"status":"fixture only, not physical truth"}')
        self.store.register_evidence('obs', [self.evidence])
        self.library = SkillDevelopment(self.store)
        self.resources = LearningResources(self.store)

    def package(self, name='press', **updates):
        result = dict(id=name, source_lines=['def main(robot, profile):', '    return profile["text"]'],
            profile={'text': 'exact'}, contract=dict(description='Fixture declaration',
                input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
                preconditions=['Explicit neutral start'], postconditions=['Observe fresh result']),
            applicability=['Fixture scope'], limitations=['Not physically verified'], evidence_ids=['obs'])
        result.update(updates)
        return result

    def lesson(self, name='lesson', **updates):
        result = dict(id=name, claim='A model-authored hypothesis, not measured truth',
            applicability=['Fixture scope'], limitations=['Counterexamples remain'], evidence_ids=['obs'],
            source_ids=['source-a'])
        result.update(updates)
        return result

    def access(self, sources=('source-a',), knowledge=()):
        access = LearningAccess(SCOPE, dict(protocol=PROTOCOLS['access'], scope=SCOPE,
            allowed_source_ids=list(sources), allowed_knowledge_ids=list(knowledge),
            allowed_evidence_ids=['obs'], allow_own_experience=True))
        access.bind_evidence('obs', ['source-a'])
        return access


class ResourceTests(Fixture, unittest.TestCase):
    def test_profile_roundtrip_is_exact_immutable_and_hash_checked(self):
        value = {'text': 'exact', 'parameters': [0.12345678912345678, -0.0, 17]}
        result = self.resources.put('parameters', value, source_ids=['source-a'], kind='profile')
        self.assertNotIn('value', result)
        loaded = self.resources.get(result['resource_id'], expected_sha256=result['sha256'])
        self.assertEqual(loaded['value'], value)
        self.assertEqual(self.resources.put('parameters', value, source_ids=['source-a'], kind='profile'), result)
        self.assertEqual(len(self.resources.records()), 1)
        with self.assertRaises(ValueError):
            self.resources.get(result['resource_id'], expected_sha256='0' * 64)
        loaded['value']['text'] = 'changed'
        self.assertEqual(self.resources.get(result['resource_id'])['value'], value)

    def test_profile_keeps_original_authority_and_structure_checks(self):
        for value in ({'nested': {'speed_percent': .2}}, {'limits': {}}, {'max_tool_calls': None}, [], {'x': float('nan')}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resources.put('bad', value, source_ids=['source-a'], kind='profile')
        self.assertEqual(self.resources.records(), [])

    def test_analysis_is_inert_finite_bounded_and_not_an_execution_profile(self):
        result = self.resources.put('analysis', {'status': 'hypothesis', 'limits': {'descriptive': 3}},
            source_ids=['source-a'])
        self.assertEqual(result['kind'], 'analysis_result')
        self.assertEqual(result['authority'], 'inert_model_proposal_not_execution_permission')
        for value in (float('inf'), {'x': {1: 2}}, 'x' * (512 * 1024 + 1)):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                self.resources.put('bad', value, source_ids=['source-a'])
        with self.assertRaises(ValueError):
            self.resources.put('../escape', {}, source_ids=['source-a'])
        with self.assertRaises(ValueError):
            self.resources.put('unproven', {}, source_ids=[])

    def test_corrupted_object_and_registration_fail_closed(self):
        result = self.resources.put('one', {'x': 1}, source_ids=['source-a'])
        objects = list((self.resources.root / 'objects').glob('*.json'))
        objects[0].write_text('{}')
        with self.assertRaises(ValueError):
            self.resources.get(result['resource_id'])

    def test_source_closure_is_enforced_on_write_read_and_filtered_directory(self):
        visible = self.resources.put('visible', {'x': 1}, source_ids=['source-a'], evidence_ids=['obs'])
        hidden = self.resources.put('hidden', {'x': 2}, source_ids=['source-b'])
        restricted = LearningResources(self.store, access=self.access())
        self.assertEqual(restricted.get(visible['resource_id'])['value'], {'x': 1})
        with self.assertRaises(ValueError):
            restricted.get(hidden['resource_id'])
        self.assertEqual([row['resource_id'] for row in restricted.records()], [visible['resource_id']])
        with self.assertRaises(ValueError):
            restricted.put('launder', {}, source_ids=['source-b'], evidence_ids=['obs'])

    def test_cross_scope_and_symlink_namespace_are_rejected(self):
        result = self.resources.put('one', {}, source_ids=['source-a'])
        other = KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='other'), 'other')
        with self.assertRaises(ValueError):
            LearningResources(other).get(result['resource_id'])
        rogue = self.root / 'rogue'
        rogue.mkdir()
        third = KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='third'), 'third')
        third.namespace.joinpath('learning').symlink_to(rogue, target_is_directory=True)
        with self.assertRaises(ValueError):
            LearningResources(third)


if __name__ == '__main__':
    unittest.main()
