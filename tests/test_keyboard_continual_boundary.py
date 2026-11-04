"""Authority/freshness counterexamples and exact feedback evidence projections.

Provider operations, observations and robot effects are explicitly fake. All
mutable stores live in resolved temporary directories; no API or SDK is used.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop
from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
from astra_robotics_agent.physical_keyboard.agent.execution_feedback import for_model
from astra_robotics_agent.physical_keyboard.agent.learning import ContinualLearning
from astra_robotics_agent.physical_keyboard.agent.learning_boundary import LearningBoundary
from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
from test_keyboard_continual_integration import Environment, Model, package, partial


class ContextModel(Model):
    def __init__(self, choose, *, operation=None, events=None):
        super().__init__(choose)
        self.operation = operation
        self.context_calls = []
        self.events = events if events is not None else []

    def run_turn(self, prompt, callback, images):
        context = json.loads(prompt.split('\n')[0])
        self.events.append(('turn', len(self.contexts), context['current_observation_evidence_id']))
        return super().run_turn(prompt, callback, images)

    def rotate_thread(self, reason):
        self.context_calls.append(('rebuild', reason))
        self.events.append(('provider', 'rebuild'))
        self.thread_id = 'fixture-rebuilt-thread'
        if self.operation is not None:
            self.operation()
        return dict(status='completed', thread_id=self.thread_id)

    def compact_context(self):
        self.context_calls.append(('native_compact', None))
        self.events.append(('provider', 'native_compact'))
        if self.operation is not None:
            self.operation()
        return dict(status='unsupported', requires_explicit_rebuild_choice=True)


class ContinualBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='continual-boundary-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.environment = Environment(self.root)

    def declaration(self, observation='observation-1'):
        return dict(summary='Auditable fixture task remains pending', knowledge_ids=[],
                    evidence_ids=[observation], pending_issues=[], remaining=['Get a fresh observation'])

    def context_model(self, method, *, operation=None, events=None, after=None):
        def choose(index, context):
            if index == 0:
                return dict(action='save_working_context', arguments=self.declaration(context['current_observation_evidence_id']))
            if index == 1:
                return dict(action='manage_working_context', arguments=dict(
                    checkpoint_id=context['feedback']['result']['checkpoint_id'],
                    method=method, reason='Explicit fixture context boundary'))
            if after is not None:
                after(context)
            return partial()
        return ContextModel(choose, operation=operation, events=events)

    def run_context(self, model, *, observe=None, stop_requested=None):
        return run_episode(session=model, observe=observe or self.environment.observe,
            execute=self.environment.execute, store=self.environment.store,
            output=self.environment.output, skill_development=True, continual_learning=True,
            learning_access=self.environment.access, max_decisions=5, stop_requested=stop_requested)

    def test_rebuild_success_requires_fresh_observation_before_next_model_turn(self):
        events = []
        def observe():
            result = self.environment.observe()
            events.append(('observe', result['evidence_id']))
            return result
        def after(context):
            feedback = context['feedback']
            self.assertFalse(feedback['observation_unchanged'])
            self.assertEqual(context['current_observation_evidence_id'], 'observation-2')
            self.assertEqual(feedback['current_observation_evidence_id'], 'observation-2')
            working = feedback['result']['working_state']
            self.assertEqual(working['historical_observation_evidence_id'], 'observation-1')
            self.assertIsNone(working['current_observation_evidence_id'])
            self.assertTrue(working['fresh_observation_required'])
            self.assertEqual(working['generation'] if 'generation' in working else feedback['result']['generation'], 0)
        model = self.context_model('rebuild', events=events, after=after)
        result = self.run_context(model, observe=observe)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(events, [('observe', 'observation-1'), ('turn', 0, 'observation-1'),
            ('turn', 1, 'observation-1'), ('provider', 'rebuild'),
            ('observe', 'observation-2'), ('turn', 2, 'observation-2')])
        self.assertEqual(self.environment.executions, [])

    def test_stop_arriving_inside_provider_prevents_observation_and_next_turn(self):
        stopped = {'value': False}
        def stop_during_provider():
            stopped['value'] = True
        model = self.context_model('rebuild', operation=stop_during_provider)
        with self.assertRaises(OperatorStop):
            self.run_context(model, stop_requested=lambda: stopped['value'])
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(self.environment.observations, 1)
        self.assertEqual(self.environment.executions, [])
        self.assertEqual(len(model.context_calls), 1)

    def test_native_unsupported_never_implicitly_rebuilds_or_changes_thread(self):
        def after(context):
            result = context['feedback']['result']
            self.assertEqual(result['status'], 'unsupported')
            self.assertEqual(result['previous_thread_id'], result['thread_id'])
            self.assertTrue(result['provider']['requires_explicit_rebuild_choice'])
        model = self.context_model('native_compact', after=after)
        result = self.run_context(model)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(model.context_calls, [('native_compact', None)])
        self.assertEqual(model.thread_id, 'fixture-thread')
        self.assertEqual(self.environment.executions, [])

    def test_new_timestamp_or_key_cannot_make_same_historical_evidence_current(self):
        old = self.environment.observe()
        calls = []
        def stale_observe():
            calls.append(len(calls))
            # Deliberately reuse identical registered evidence with a tempting
            # new timestamp/key. The identity is still the old frozen snapshot.
            return dict(old, timestamp_ns=100000000000 + len(calls), key='new-looking-key')
        model = self.context_model('rebuild')
        with self.assertRaisesRegex((RuntimeError, ValueError), 'observation|fresh|trusted'):
            self.run_context(model, observe=stale_observe)
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.environment.executions, [])

    def test_unknown_terminal_and_inflight_state_cannot_gain_motion_through_context(self):
        observation = self.environment.observe()
        prior = self.root / 'prior-response.json'
        prior.write_text(json.dumps({'status': 'unknown', 'fixture': 'unresolved execution'}))
        self.environment.store.register_evidence('prior-execution', [prior])
        model = ContextModel(lambda *_: partial())
        facade = ContinualLearning(DevelopmentActions(self.environment.store), model,
            output=self.root / 'direct-context', access=self.environment.access)
        boundary = LearningBoundary(self.environment.store)
        boundary.generation = 7
        conditions = ('unknown', 'terminal', 'inflight_rpc', 'inflight_action', 'inflight_model')
        for condition in conditions:
            with self.subTest(condition=condition):
                boundary.last_execution = None
                boundary.inflight_rpc = boundary.inflight_action = boundary.inflight_model = False
                if condition == 'unknown':
                    boundary.execution('prior-execution', {'status': 'unknown'}, unknown=True, terminal=False)
                elif condition == 'terminal':
                    boundary.execution('prior-execution', {'status': 'ok'}, unknown=False, terminal=True)
                else:
                    setattr(boundary, condition, True)
                facts = boundary.facts(observation['evidence_id'])
                checkpoint = facade.working.save(self.declaration(), facts)
                with self.assertRaises(ValueError):
                    facade.working.request(dict(checkpoint_id=checkpoint['checkpoint_id'],
                        method='rebuild', reason='Must not restore authority'), facts)
                self.assertEqual(model.context_calls, [])
                self.assertEqual(boundary.generation, 7)
        self.assertEqual(self.environment.executions, [])

    def test_read_only_unknown_context_preserves_fault_generation_and_old_observation(self):
        observation = self.environment.observe()
        prior = self.root / 'prior.json'; prior.write_text('{"status":"unknown"}')
        self.environment.store.register_evidence('prior-execution', [prior])
        model = ContextModel(lambda *_: partial())
        facade = ContinualLearning(DevelopmentActions(self.environment.store), model,
            output=self.root / 'read-only-context', access=self.environment.access)
        boundary = LearningBoundary(self.environment.store)
        boundary.generation = 11
        boundary.execution('prior-execution', {'status': 'unknown'}, unknown=True, terminal=True)
        facts = boundary.facts(observation['evidence_id'], motion_available=False,
            fault={'fault_id': 'still-unresolved'}, assistance={'operator_review': 'pending'})
        checkpoint = facade.working.save(self.declaration(), facts)
        result = facade.working.request(dict(checkpoint_id=checkpoint['checkpoint_id'],
            method='native_compact', reason='Read-only factual context maintenance'), facts)
        self.assertFalse(result['motion_available'])
        self.assertEqual(result['generation'], 11)
        self.assertEqual(result['working_state']['current_state'], 'unknown')
        self.assertEqual(result['working_state']['authoritative_facts'], facts)
        self.assertEqual(boundary.last_execution['status'], 'unknown')
        self.assertEqual(self.environment.executions, [])

    def test_boundary_rejects_generation_or_policy_change_during_provider_operation(self):
        observation = self.environment.observe()
        boundary = LearningBoundary(self.environment.store)
        before = boundary.facts(observation['evidence_id'])
        boundary.generation = 1
        with self.assertRaisesRegex(RuntimeError, 'authority changed'):
            boundary.require_unchanged(before, observation['evidence_id'])
        boundary.generation = 0
        self.environment.access.allowed_sources.add('new-source-after-checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'authority changed'):
            boundary.require_unchanged(before, observation['evidence_id'])

    def test_complete_learning_commit_candidates_are_owned_by_later_reflection(self):
        retained = {}
        def choose(index, context):
            if index == 0:
                evidence_id = context['current_observation_evidence_id']
                return dict(action='commit_learning', arguments=dict(
                    packages=[package(evidence_id, local_ref='skill')],
                    lessons=[dict(id='explicit-fixture-lesson', claim='An auditable fixture hypothesis',
                        applicability=['Only the temporary fixture'], limitations=['Not independently verified'],
                        evidence_ids=[evidence_id], local_ref='lesson')]))
            candidates = context['feedback']['result']['saved_candidates']
            self.assertEqual(len({row['candidate_id'] for row in candidates}), 2)
            retained.update({row['kind']: row['candidate_id'] for row in candidates})
            decision = partial()
            decision['result']['reflection']['skills_saved'] = [retained['skill']]
            decision['result']['reflection']['lessons_saved'] = [retained['lesson']]
            return decision
        model = ContextModel(choose)
        result = self.run_context(model)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual({row['candidate_id'] for row in result['saved_candidates']}, set(retained.values()))
        self.assertEqual(result['reply']['reflection']['skills_saved'], [retained['skill']])
        self.assertEqual(result['reply']['reflection']['lessons_saved'], [retained['lesson']])
        self.assertEqual(self.environment.executions, [])


class ExecutionFeedbackProjectionTests(unittest.TestCase):
    def response(self):
        return dict(status='ok', result=dict(motion_outcome_uncertain=False,
            controller_terminal=False, actual_state=dict(joint=[.1, .2, .3],
                observation_evidence_id='historical-observation', timestamp_ns=500, key='same-key'),
            value={'large_exact_sequence': [index / 1000 for index in range(4000)]},
            source_sha256='a' * 64, profile_sha256='b' * 64,
            receipts=[dict(index=index, method='play_trajectory', status='ok',
                replay={'samples': [index] * 2000, 'motion_outcome_uncertain': False})
                for index in range(20)]))

    def test_success_projection_preserves_actual_state_and_exact_evidence_hash_references(self):
        response = self.response()
        original = deepcopy(response)
        compact = for_model(response, 'registered-execution-1', compact=True)
        value = compact['result']
        self.assertEqual(response, original)
        self.assertEqual(value['actual_state'], original['result']['actual_state'])
        self.assertEqual(value['full_execution_evidence_id'], 'registered-execution-1')
        self.assertEqual(value['source_sha256'], 'a' * 64)
        self.assertEqual(value['profile_sha256'], 'b' * 64)
        self.assertEqual(value['full_receipt_count'], 20)
        self.assertEqual(value['earlier_boundaries_omitted'], 12)
        self.assertEqual([row['index'] for row in value['completed_boundaries']], list(range(12, 20)))
        raw = json.dumps(original['result']['value'], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(value['value']['full_value_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(value['value']['full_value_bytes'], len(raw))
        self.assertTrue(value['value']['available_in_execution_evidence'])
        self.assertLess(len(json.dumps(compact)), len(json.dumps(response)) // 10)
        self.assertEqual(value['actual_state']['observation_evidence_id'], 'historical-observation')
        self.assertNotIn('current_observation_evidence_id', value)
        value['actual_state']['joint'][0] = 999
        self.assertEqual(response, original)

    def test_unknown_error_terminal_or_failed_receipt_remain_byte_exact(self):
        mutations = (
            lambda response: response.update(status='unknown'),
            lambda response: response.update(status='error'),
            lambda response: response['result'].update(motion_outcome_uncertain=True),
            lambda response: response['result'].update(controller_terminal=True),
            lambda response: response['result']['receipts'][0].update(status='unknown'),
            lambda response: response['result']['receipts'][0].update(error='Exact original failure'),
            lambda response: response['result']['receipts'][0]['replay'].update(audit_error='Audit failed'),
            lambda response: response['result']['receipts'][0]['replay'].update(stop_error='Stop failed'),
            lambda response: response['result']['receipts'][0]['replay'].update(stop_called=True),
            lambda response: response['result']['receipts'][0]['replay'].update(motion_outcome_uncertain=True),
        )
        for mutate in mutations:
            response = self.response(); mutate(response)
            before = json.dumps(response, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            with self.subTest(mutate=mutate):
                projected = for_model(response, 'registered-error', compact=True)
                self.assertIs(projected, response)
                self.assertEqual(json.dumps(projected, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(), before)

    def test_disabled_projection_keeps_legacy_success_response_unchanged(self):
        response = self.response()
        self.assertIs(for_model(response, 'registered-execution', compact=False), response)


if __name__ == '__main__':
    unittest.main()
