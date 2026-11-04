"""T4 boundary and source-closure tests; no provider, robot or network calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.agent.work_context import WorkContext
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources


SCOPE = dict(task_id='fixture', device_id='robot', layout_id='board')


class Session:
    model, effort, service_tier = 'gpt-6-astra', 'xhigh', 'default'
    def __init__(self):
        self.thread_id = 'thread-1'
        self.total_usage = {'inputTokens': 23}
        self.total_usage_complete = False
        self.calls = []
        self.closed = False
        self.unsupported = False
    def compact_context(self):
        self.calls.append('compact')
        return dict(status='unsupported' if self.unsupported else 'completed')
    def rotate_thread(self, reason):
        self.calls.append(('rebuild', reason))
        self.thread_id = 'thread-2'
        return dict(thread_id=self.thread_id)
    def close(self):
        self.closed = True


class WorkContextTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'run')
        for evidence in ('obs', 'execution', 'new-obs'):
            path = self.root / (evidence + '.json')
            path.write_text(json.dumps(dict(fixture=evidence)))
            self.store.register_evidence(evidence, [path])
        self.legacy = self.store.propose('lesson', dict(id='legacy', claim='pre-access fixture hypothesis',
            applicability=['fixture only'], limitations=['not verified'], evidence_ids=['obs']))
        self.session = Session()
        self.context = WorkContext(self.session, self.store, output=self.root / 'context')
        self.args = dict(summary='Fixture observable work remains', knowledge_ids=[],
                         evidence_ids=['obs'], pending_issues=[], completed=['fixture-only check'],
                         remaining=['obtain a fresh observation before deciding'])
        self.facts = dict(task_contract_sha256='a' * 64, observation_evidence_id='obs', generation=3,
                          last_execution=dict(evidence_id='execution', status='ok',
                                              motion_outcome_uncertain=False, controller_terminal=False),
                          motion_available=True, inflight_rpc=False, inflight_action=False,
                          inflight_model=False, stop_requested=False, assistance=None, fault=None,
                          source_policy_sha256='b' * 64)

    def request(self, descriptor, method='rebuild', facts=None):
        return self.context.request(dict(checkpoint_id=descriptor['checkpoint_id'], method=method,
                                         reason='explicit fixture context request'), facts or self.facts)

    def access(self, *, sources=(), knowledge=(), evidence=('obs', 'execution', 'new-obs')):
        return LearningAccess(SCOPE, dict(protocol=PROTOCOLS['access'], scope=SCOPE,
            allowed_source_ids=list(sources), allowed_knowledge_ids=list(knowledge),
            allowed_evidence_ids=list(evidence), allow_own_experience=True))

    def test_save_read_rebuild_preserves_authority_usage_and_requires_new_observation(self):
        saved = self.context.save(self.args, self.facts)
        record = self.context.read(saved['checkpoint_id'])
        self.assertEqual(record['authoritative_facts'], self.facts)
        self.assertFalse(record['model_reasoning_saved'])
        result = self.request(saved)
        self.assertEqual(result['previous_thread_id'], 'thread-1')
        self.assertEqual(result['thread_id'], 'thread-2')
        self.assertEqual(result['total_usage'], {'inputTokens': 23})
        self.assertFalse(result['total_usage_complete'])
        self.assertEqual(result['generation'], 3)
        state = result['working_state']
        self.assertTrue(state['fresh_observation_required'])
        self.assertIsNone(state['current_observation_evidence_id'])
        self.assertEqual(state['historical_observation_evidence_id'], 'obs')
        self.assertEqual(state['authoritative_facts'], self.facts)
        self.assertEqual(self.session.calls, [('rebuild', 'explicit fixture context request')])
        self.assertNotIn('actions', state)

    def test_stop_and_each_inflight_flag_reject_before_provider(self):
        for flag in ('stop_requested', 'inflight_rpc', 'inflight_action', 'inflight_model'):
            with self.subTest(flag=flag):
                facts = copy.deepcopy(self.facts); facts[flag] = True
                saved = self.context.save(self.args, facts)
                with self.assertRaises(ValueError):
                    self.request(saved, facts=facts)
        self.assertEqual(self.session.calls, [])
        self.assertFalse(self.session.closed)

    def test_unknown_fault_and_terminal_require_read_only_and_preserve_all_restrictions(self):
        for change in ('unknown', 'motion_outcome_uncertain', 'controller_terminal', 'fault'):
            with self.subTest(change=change):
                facts = copy.deepcopy(self.facts)
                if change == 'unknown': facts['last_execution']['status'] = 'unknown'
                elif change == 'fault': facts['fault'] = dict(evidence_id='execution', kind='fixture_fault')
                else: facts['last_execution'][change] = True
                facts['assistance'] = dict(kind='operator review remains pending')
                saved = self.context.save(self.args, facts)
                with self.assertRaises(ValueError): self.request(saved, facts=facts)
                facts['motion_available'] = False
                saved = self.context.save(self.args, facts)
                result = self.request(saved, method='native_compact', facts=facts)
                self.assertFalse(result['motion_available'])
                self.assertEqual(result['generation'], facts['generation'])
                self.assertEqual(result['working_state']['authoritative_facts'], facts)
                self.assertEqual(result['working_state']['current_state'], 'unknown')
        self.assertEqual(self.session.calls, ['compact'] * 4)

    def test_initial_no_execution_and_missing_observation_only_in_read_only_mode(self):
        facts = dict(self.facts, last_execution=None)
        saved = self.context.save(self.args, facts)
        self.assertEqual(self.request(saved, 'native_compact', facts)['status'], 'completed')
        facts['observation_evidence_id'] = None
        with self.assertRaises(ValueError): self.context.save(self.args, facts)
        facts['motion_available'] = False
        saved = self.context.save(self.args, facts)
        self.assertEqual(self.request(saved, 'native_compact', facts)['working_state']['current_state'], 'unknown')

    def test_stale_observation_execution_contract_generation_or_source_policy_rejects(self):
        saved = self.context.save(self.args, self.facts)
        changes = [dict(observation_evidence_id='new-obs'), dict(generation=4),
                   dict(task_contract_sha256='c' * 64), dict(source_policy_sha256='d' * 64),
                   dict(last_execution=None)]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'stale'):
                self.request(saved, facts=dict(self.facts, **change))
        self.assertEqual(self.session.calls, [])

    def test_model_authority_fields_extra_facts_and_fake_references_are_rejected(self):
        for updates in (dict(generation=44), dict(authoritative_facts=self.facts),
                        dict(internal_reasoning='forbidden'), dict(evidence_ids=['missing']),
                        dict(knowledge_ids=['missing']), dict(pending_issues=['missing'])):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.context.save(dict(self.args, **updates), self.facts)
        with self.assertRaises(ValueError): self.context.save(self.args, dict(self.facts, goal_contract={}))
        with self.assertRaises(ValueError): self.context.save(self.args, dict(self.facts, generation=True))

    def test_bad_checkpoint_hash_or_evidence_object_fails_before_provider(self):
        saved = self.context.save(self.args, self.facts)
        path = self.context.checkpoints / (saved['checkpoint_id'] + '.json')
        original = path.read_bytes()
        value = json.loads(original); value['declaration']['summary'] = 'tampered'
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): self.request(saved)
        path.write_bytes(original)
        evidence = self.store._evidence('obs')
        (self.store.namespace / 'objects' / evidence['files'][0]['sha256']).write_text('corrupted')
        with self.assertRaises(ValueError): self.request(saved)
        self.assertFalse(self.session.calls)

    def test_resource_full_payload_hash_and_source_permissions_are_checked(self):
        resources = LearningResources(self.store)
        resource = resources.put('fixture', {'exact': [1.0, 0.2]}, source_ids=['demo-hidden'])
        self.args['knowledge_ids'] = [resource['resource_id']]
        restricted = WorkContext(self.session, self.store, output=self.root / 'restricted', access=self.access())
        with self.assertRaises(ValueError): restricted.save(self.args, self.facts)
        saved = self.context.save(self.args, self.facts)
        obj = next((resources.root / 'objects').glob('*.json')); obj.write_text('{}')
        with self.assertRaises(ValueError): self.request(saved)
        self.assertEqual(self.session.calls, [])

    def test_checkpoint_payload_cannot_wash_broader_previous_permissions(self):
        saved = self.context.save(self.args, self.facts)
        restricted = WorkContext(self.session, self.store, output=self.root / 'context', access=self.access())
        with self.assertRaisesRegex(ValueError, 'unrestricted checkpoint payload'):
            restricted.read(saved['checkpoint_id'])

    def test_default_access_inherits_existing_store_policy(self):
        resource = LearningResources(self.store).put('hidden', {'fixture': 1}, source_ids=['demo-hidden'])
        self.store.learning_access = self.access()
        inherited = WorkContext(self.session, self.store, output=self.root / 'inherited-access')
        self.assertIs(inherited.access, self.store.learning_access)
        with self.assertRaises(ValueError):
            inherited.save(dict(self.args, knowledge_ids=[resource['resource_id']]), self.facts)

    def test_explicitly_allowed_legacy_knowledge_and_evidence_need_real_registration(self):
        lesson = self.legacy
        access = self.access(knowledge=[lesson['candidate_id']])
        restricted = WorkContext(self.session, self.store, output=self.root / 'restricted', access=access)
        args = dict(self.args, knowledge_ids=[lesson['candidate_id']])
        saved = restricted.save(args, self.facts)
        self.assertEqual(restricted.read(saved['checkpoint_id'])['declaration']['knowledge_ids'], args['knowledge_ids'])
        with self.assertRaises(ValueError):
            restricted.save(dict(args, evidence_ids=['allowed-but-unregistered']), self.facts)

    def test_package_dependencies_and_stored_evidence_provenance_cannot_be_laundered(self):
        def package(name, **updates):
            args = dict(id=name, source_lines=['def main(robot, profile):', '    return 1'], profile={},
                contract=dict(description='Fixture only', input_schema={'type': 'object', 'properties': {},
                    'additionalProperties': False}, preconditions=['Fixture start'], postconditions=['Read fresh observation']),
                applicability=['Fixture scope'], limitations=['Not physically verified'], evidence_ids=['obs'])
            args.update(updates)
            return self.context.library.save_package(args)['package_id']
        base = package('base')
        outer = package('outer', dependencies={'base': base})
        restricted = WorkContext(self.session, self.store, output=self.root / 'restricted-dependency',
                                 access=self.access(knowledge=[outer]))
        with self.assertRaises(ValueError):
            restricted.save(dict(self.args, knowledge_ids=[outer]), self.facts)
        hidden = self.root / 'hidden.json'; hidden.write_text('{"fixture":"source hidden"}')
        self.store.register_evidence('hidden', [hidden], source_ids=['demo-hidden'])
        restricted = WorkContext(self.session, self.store, output=self.root / 'restricted-evidence',
            access=self.access(evidence=['obs', 'execution', 'new-obs', 'hidden']))
        with self.assertRaises(ValueError):
            restricted.save(dict(self.args, evidence_ids=['hidden']), self.facts)
        self.assertFalse(self.session.calls)

    def test_wrong_scope_and_wrong_pending_issue_kind_are_rejected(self):
        saved = self.context.save(self.args, self.facts)
        other_store = KnowledgeStore(self.root / 'other-knowledge', dict(SCOPE, device_id='other'), 'other-run')
        other = WorkContext(self.session, other_store, output=self.root / 'context')
        with self.assertRaises(ValueError): other.read(saved['checkpoint_id'])
        lesson = self.store.propose('lesson', dict(id='fixture', claim='Fixture claim',
            applicability=['fixture'], limitations=['unverified'], evidence_ids=['obs']))
        with self.assertRaisesRegex(ValueError, 'pending_issues'):
            self.context.save(dict(self.args, pending_issues=[lesson['candidate_id']]), self.facts)

    def test_unsupported_native_compaction_never_automatically_rebuilds(self):
        self.session.unsupported = True
        saved = self.context.save(self.args, self.facts)
        result = self.request(saved, 'native_compact')
        self.assertEqual(result['status'], 'unsupported')
        self.assertEqual(self.session.calls, ['compact'])
        self.assertEqual(result['thread_id'], result['previous_thread_id'])
        self.assertFalse(self.session.closed)

    def test_provider_configuration_change_invalid_receipt_or_failure_closes_and_records(self):
        for kind in ('configuration', 'receipt', 'failure'):
            with self.subTest(kind=kind):
                self.session = Session()
                self.context = WorkContext(self.session, self.store, output=self.root / kind)
                saved = self.context.save(self.args, self.facts)
                def compact():
                    if kind == 'configuration': self.session.effort = 'low'; return {'status': 'completed'}
                    if kind == 'receipt': return {'status': 'failed'}
                    raise RuntimeError('fixture provider failure')
                self.session.compact_context = compact
                with self.assertRaises((ValueError, RuntimeError)): self.request(saved, 'native_compact')
                self.assertTrue(self.session.closed)
                events = (self.context.output / 'context-events.jsonl').read_text()
                self.assertIn('context_operation_rejected', events)

    def test_old_checkpoint_after_rebuild_cannot_be_used_as_current_thread(self):
        saved = self.context.save(self.args, self.facts)
        self.request(saved)
        with self.assertRaisesRegex(ValueError, 'provider configuration or thread is stale'):
            self.request(saved, 'native_compact')
        self.assertEqual(len(self.session.calls), 1)


if __name__ == '__main__':
    unittest.main()
