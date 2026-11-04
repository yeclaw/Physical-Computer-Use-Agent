"""Real temporary legacy writers, finite commits and interrupted visibility."""
from copy import deepcopy
import hashlib
import unittest

from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources
from test_keyboard_learning_resources import Fixture


class LearningChangeTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.changes = LearningChanges(self.library, self.resources)

    def batch(self):
        package = self.package(local_ref='press')
        del package['profile']
        package['profile_resource_id'] = {'local_ref': 'parameters'}
        return dict(resources=[dict(id='parameters', kind='profile', value={'text': '0.12345678912345678'},
            source_ids=['source-a'], local_ref='parameters')],
            lessons=[self.lesson(local_ref='lesson')], packages=[package],
            improvements=[dict(id='repeated', summary='Compensated repeated character',
                rationale='Compensation does not demonstrate a repaired primitive', status='open',
                skill_ids=[{'local_ref': 'press'}], evidence_ids=['obs'], local_ref='issue')],
            relations=[dict(kind='uses_lesson', from_id={'local_ref': 'press'}, to_id={'local_ref': 'lesson'}),
                dict(kind='addresses_issue', from_id={'local_ref': 'press'}, to_id={'local_ref': 'issue'})],
            selections=[dict(logical_id='press', identifier={'local_ref': 'press'},
                conditions=['neutral start'], reason='Explicit model selection')])

    def snapshot(self):
        return {path.relative_to(self.store.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.store.root.rglob('*') if path.is_file()}

    def test_preflight_is_write_free_and_commit_exact_matches_legacy_calls(self):
        args = self.batch()
        before = self.snapshot()
        prepared = self.changes.preflight(args)
        self.assertEqual(self.snapshot(), before)
        result = self.changes.commit(args)
        self.assertEqual(prepared['bindings'], result['bindings'])
        self.assertTrue(all(self.changes.is_committed(value) for value in result['record_ids']))
        package = self.library.inspect(result['bindings']['press'])
        self.assertEqual(package['declared_profile'], {'text': '0.12345678912345678'})
        executable = self.library.resolve_call({'package_id': package['record_id'], 'arguments': {}})
        self.assertEqual(run_skill([('skill.py', executable['source'])], tools={}, profile=executable['profile']),
                         '0.12345678912345678')
        self.assertEqual(self.library.inspect(result['bindings']['issue'])['status'], 'open')
        self.assertEqual(self.changes.source_ids(package['candidate_id']), ['source-a'])

    def test_invalid_late_declaration_does_not_write_earlier_valid_objects(self):
        for mutate in (
            lambda args: args['selections'][0].update(identifier={'local_ref': 'missing'}),
            lambda args: args['packages'][0].update(profile={}),
            lambda args: args['relations'][0].update(kind='execute'),
            lambda args: args['packages'][0].update(profile_resource_id={'json_path': 'x'}),
            lambda args: args['lessons'][0].update(claim=float('nan')),
        ):
            args = self.batch(); mutate(args)
            before = self.snapshot()
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.changes.commit(args)
            self.assertEqual(self.snapshot(), before)

    def test_failure_before_and_during_objects_never_activates_pending_ids(self):
        for stage in ('before_preflight', 'after_preflight', 'after_intent', 'after_object_write:0',
                      'after_object_write:2', 'before_final_commit'):
            with self.subTest(stage=stage):
                args = self.batch()
                args['resources'][0]['id'] += stage.replace(':', '-')
                args['lessons'][0]['id'] += stage.replace(':', '-')
                args['packages'][0]['id'] = 'press'
                plan = self.changes.preflight(args)
                previous = self.changes.commits()
                def fail(current):
                    if current == stage:
                        raise RuntimeError('Injected ' + stage)
                self.changes.failure_hook = fail
                with self.assertRaisesRegex(RuntimeError, 'Injected'):
                    self.changes.commit(args)
                self.changes.failure_hook = None
                self.assertEqual(self.changes.commits(), previous)
                if stage not in ('before_preflight', 'after_preflight'):
                    self.assertTrue(all(not self.changes.is_committed(value) for value in plan['new_ids']))
                restored = LearningChanges(self.library, self.resources)
                self.assertEqual(restored.commits(), previous)

    def test_after_final_failure_is_committed_and_restart_keeps_view(self):
        def fail(stage):
            if stage == 'after_final_commit':
                raise RuntimeError('Client delivery failed after commit')
        self.changes.failure_hook = fail
        with self.assertRaises(RuntimeError):
            self.changes.commit(self.batch())
        restored = LearningChanges(self.library, self.resources)
        self.assertEqual(len(restored.commits()), 1)
        result = restored.commits()[0]
        self.assertTrue(all(restored.is_committed(value) for value in result['record_ids']))

    def test_identical_retry_can_commit_previously_pending_immutable_resource_and_lesson(self):
        args = dict(resources=[dict(id='result', value={'x': 1}, source_ids=['source-a'], local_ref='r')],
                    lessons=[self.lesson(local_ref='l')])
        def fail(stage):
            if stage == 'before_final_commit':
                raise RuntimeError('Interrupted')
        self.changes.failure_hook = fail
        with self.assertRaises(RuntimeError):
            self.changes.commit(args)
        plan = self.changes.preflight(args)
        self.assertTrue(all(not self.changes.is_committed(value) for value in plan['record_ids']))
        self.changes.failure_hook = None
        result = self.changes.commit(args)
        self.assertTrue(all(self.changes.is_committed(value) for value in result['record_ids']))

    def test_failed_redeclaration_does_not_disable_legacy_exact_id(self):
        legacy = self.store.propose('lesson', {key: value for key, value in self.lesson().items() if key != 'source_ids'})
        args = dict(lessons=[self.lesson()])
        def fail(stage):
            if stage == 'before_final_commit':
                raise RuntimeError('Interrupted')
        self.changes.failure_hook = fail
        with self.assertRaises(RuntimeError):
            self.changes.commit(args)
        self.assertTrue(self.changes.is_committed(legacy['candidate_id']))
        self.assertEqual(self.library.inspect(legacy['candidate_id'])['claim'], self.lesson()['claim'])

    def test_profile_resource_requires_complete_object_not_array_or_path(self):
        resource = self.resources.put('analysis', [1, 2], source_ids=['source-a'])
        args = self.package(profile_resource_id=resource['resource_id'], source_ids=['source-a'])
        del args['profile']
        with self.assertRaises(ValueError):
            self.changes.commit({'packages': [args]})
        args['profile_resource_id'] = '/tmp/profile.json'
        with self.assertRaises(ValueError):
            self.changes.commit({'packages': [args]})

    def test_pending_dependency_is_rejected_and_old_pins_do_not_drift(self):
        first = self.changes.commit(self.batch())
        old = self.library.resolve_call({'package_id': first['bindings']['press'], 'arguments': {}})
        next_package = self.package(source_ids=['source-a'], profile={'text': 'new version'})
        next_package['parent_id'] = first['bindings']['press']
        second = self.changes.commit({'packages': [next_package]})
        self.assertEqual(old, self.library.resolve_call({'package_id': first['bindings']['press'], 'arguments': {}}))
        pending = self.package('pending', source_ids=['source-a'])
        def fail(stage):
            if stage == 'before_final_commit':
                raise RuntimeError('Interrupted')
        self.changes.failure_hook = fail
        plan = self.changes.preflight({'packages': [pending]})
        with self.assertRaises(RuntimeError):
            self.changes.commit({'packages': [pending]})
        self.changes.failure_hook = None
        child = self.package('child', source_ids=['source-a'], dependencies={'step': plan['operations'][0]['identifier']})
        with self.assertRaises(ValueError):
            self.changes.commit({'packages': [child]})

    def test_source_policy_closure_and_reconstruction_cannot_launder_legacy(self):
        access = self.access()
        resources = LearningResources(self.store, access=access)
        changes = LearningChanges(self.library, resources, access=access)
        result = changes.commit({'lessons': [self.lesson()]})
        self.assertEqual(changes.source_ids(result['record_ids'][0]), ['source-a'])
        forbidden = self.store.propose('lesson', {key: value for key, value in self.lesson('hidden').items() if key != 'source_ids'})
        with self.assertRaises(ValueError):
            changes.commit({'relations': [dict(kind='derived_from', from_id=result['record_ids'][0],
                to_id=forbidden['candidate_id'])]})
        hidden = self.changes.commit({'lessons': [self.lesson('hidden-b', source_ids=['source-b'])]})
        restored = LearningChanges(self.library, resources, access=self.access())
        with self.assertRaises(ValueError):
            restored.source_ids(hidden['record_ids'][0])

    def test_derived_summary_keeps_transitive_sources_when_read_in_filtered_session(self):
        origin = self.changes.commit({'lessons': [self.lesson('origin', source_ids=['source-b'])]})['record_ids'][0]
        result = self.changes.commit(dict(lessons=[self.lesson('derived', local_ref='summary')], relations=[
            dict(kind='derived_from', from_id={'local_ref': 'summary'}, to_id=origin)]))
        summary = result['bindings']['summary']
        self.assertEqual(self.changes.source_ids(summary), ['source-a', 'source-b'])
        access = self.access()
        restored = LearningChanges(self.library, LearningResources(self.store, access=access), access=access)
        with self.assertRaises(ValueError):
            restored.source_ids(summary)

    def test_forward_package_and_issue_references_are_finitely_topologically_ordered(self):
        base = self.package('base', source_ids=['source-a'], local_ref='base')
        child = self.package('child', source_ids=['source-a'], local_ref='child',
            dependencies={'step': {'local_ref': 'base'}},
            source_lines=['def main(robot, profile):', '    return step(robot, profile)'])
        result = self.changes.commit({'packages': [child, base]})
        child_record = self.library.inspect(result['bindings']['child'])
        self.assertEqual(child_record['dependencies']['step'], result['bindings']['base'])
        issue = dict(id='new-issue', summary='Explicit improvement proposal', rationale='A reusable concern',
            status='open', skill_ids=[result['bindings']['base']], evidence_ids=['obs'], local_ref='issue')
        revision = self.package('revision', source_ids=['source-a'], local_ref='revision',
                                improvement_ids=[{'local_ref': 'issue'}])
        next_result = self.changes.commit({'packages': [revision], 'improvements': [issue]})
        self.assertEqual(self.library.inspect(next_result['bindings']['revision'])['improvement_ids'],
                         [next_result['bindings']['issue']])

    def test_content_hash_cycle_is_rejected_and_policy_bindings_stay_unchanged_in_preflight(self):
        before = deepcopy(self.changes.access.__dict__)
        self.changes.preflight(self.batch())
        self.assertEqual(self.changes.access.__dict__, before)
        one = self.package('one', source_ids=['source-a'], local_ref='one',
                           improvement_ids=[{'local_ref': 'issue'}])
        issue = dict(id='issue', summary='Concern', rationale='Both contents would depend on each other',
            status='open', skill_ids=[{'local_ref': 'one'}], evidence_ids=['obs'], local_ref='issue')
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'cyclic finite local_ref'):
            self.changes.commit({'packages': [one], 'improvements': [issue]})
        self.assertEqual(snapshot, self.snapshot())
        self.assertEqual(self.changes.access.__dict__, before)

    def test_explicit_legacy_id_permission_does_not_supply_missing_derivation_provenance(self):
        policy = self.store.learning_access
        self.store.learning_access = None
        legacy = self.store.propose('lesson', {key: value for key, value in self.lesson('legacy').items() if key != 'source_ids'})
        self.store.learning_access = policy
        access = self.access(knowledge=[legacy['candidate_id']])
        changes = LearningChanges(self.library, LearningResources(self.store, access=access), access=access)
        self.assertEqual(changes.source_ids(legacy['candidate_id']), [])
        before = self.snapshot()
        with self.assertRaises(ValueError):
            changes.commit(dict(lessons=[self.lesson('derived', local_ref='d')], relations=[
                dict(kind='derived_from', from_id={'local_ref': 'd'}, to_id=legacy['candidate_id'])]))
        self.assertEqual(before, self.snapshot())


if __name__ == '__main__':
    unittest.main()
