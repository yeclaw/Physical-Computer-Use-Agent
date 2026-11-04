"""Continuous chat stays live across replies, RSI and ordinary operator help.

These tests exercise real scoped knowledge/evidence and model-loop bookkeeping.
All observations/executions/wakes are deterministic fixtures, never real devices.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from astra_robotics_agent.physical_keyboard.campaign import load_task_contract
from astra_robotics_agent.physical_keyboard.learning_loop import (
    learning_instructions, parse_decision, run_episode,
)
from test_keyboard_learning_loop import Fixture, Session, SCOPE, finish, skill

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = ROOT / 'configs/physical_keyboard/wechat_self_chat_contract.json'


def program(label='reply'):
    return dict(action='run_skill', arguments=dict(label=label,
        source_lines=['def main(robot, profile):', '    return robot.observe()'],
        profile={}), hypothesis='Fixture execution; inspect returned evidence, never infer real sending.')


def completed(skills=(), lessons=()):
    decision = finish(skills, lessons)
    decision['result']['status'] = 'completed'
    decision['result']['reflection']['summary'] = 'A fixture conversation turn has evidence; keep the session open.'
    return decision


def assistance(prompt, kind='operator_action'):
    return dict(action='request_assistance', arguments=dict(kind=kind,
        reason='Fixture input focus needs the operator, while historical send evidence remains valid.',
        required_human_action='Restore input focus or provide an operator clarification.',
        observation_evidence_id=prompt['current_observation_evidence_id']))


class ServiceFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='keyboard-chat-service-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.document = json.loads(CHAT_TEMPLATE.read_text())
        self.document.update(scope=SCOPE, session_mode='continuous', send_key='Return',
                             objective='Converse using physical keyboard skills and retain learning evidence.')
        self.path = self.root / 'contract.json'
        self.write()
        self.f = Fixture(self.root)
        self.f.statuses = ['ok'] * 16
        self.waits = []

    def write(self):
        self.path.write_text(json.dumps(self.document))

    def contract(self):
        return load_task_contract(self.path, SCOPE)

    def wake(self, arguments):
        self.waits.append(copy.deepcopy(arguments))
        return self.f.observe()

    def loop(self, choose, *, max_decisions=8, **kwargs):
        self.session = Session(choose)
        return run_episode(session=self.session, observe=kwargs.pop('observe', self.f.observe),
            execute=kwargs.pop('execute', self.f.execute), store=self.f.store,
            output=self.f.output, task_contract=kwargs.pop('task_contract', self.contract()),
            wait_for_update=kwargs.pop('wait_for_update', self.wake),
            max_decisions=max_decisions, **kwargs)


class ServiceContractTests(ServiceFixture):
    def test_continuous_is_explicit_and_absent_mode_keeps_single_exchange(self):
        continuous = self.contract()
        self.assertTrue(continuous.is_chat)
        self.assertTrue(continuous.is_continuous_chat)
        continuous.document['session_mode'] = 'single_exchange'
        self.assertTrue(continuous.is_continuous_chat)
        del self.document['session_mode']; self.write()
        old = self.contract()
        self.assertTrue(old.is_chat)
        self.assertFalse(old.is_continuous_chat)
        self.assertNotIn('target_text', old.document)

    def test_invalid_modes_and_typing_contract_mode_do_not_silently_enable_service(self):
        for value in (True, None, [], {}, '', 'any_forever_mode'):
            self.document['session_mode'] = value; self.write()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.contract()
        typing = json.loads((ROOT / 'configs/physical_keyboard/task_contract.json').read_text())
        typing.update(scope=SCOPE, session_mode='continuous')
        self.path.write_text(json.dumps(typing))
        with self.assertRaises(ValueError):
            self.contract()

    def test_persistent_prompt_keeps_rsi_and_uses_wait_instead_of_single_send_review(self):
        text = learning_instructions(continuous=True, chat=True, continuous_chat=True)
        self.assertIn('wait_for_update', text)
        for name in ('save_skill', 'save_lesson', 'verify_skill'):
            self.assertIn(name, text)
        self.assertNotIn('"action":"confirm_completion"', text)
        self.assertNotIn('A fresh completion-review image follows.', text)
        self.assertNotIn('robot.play_sequence', text)
        old = learning_instructions(continuous=True, chat=True)
        self.assertIn('"action":"confirm_completion"', old)

    def test_persistent_prompt_cannot_be_accidentally_enabled_for_a_typing_or_legacy_task(self):
        for options in (dict(continuous=False, chat=True), dict(continuous=True, chat=False)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                learning_instructions(continuous_chat=True, **options)


class ServiceLifecycleTests(ServiceFixture):
    def test_two_completed_replies_wait_and_keep_same_session_all_usage_without_extra_review(self):
        def choose(index, prompt):
            self.assertFalse(prompt['training_reset_available'])
            self.assertNotIn('demo_trajectories', prompt)
            self.assertIsNone(prompt['task_state']['pending_completion_observation_id'])
            for field in ('typed_text', 'expected_text', 'caret', 'cursor_position'):
                self.assertNotIn(field, prompt['task_state'])
            if index in (0, 3):
                return program('reply_one' if index == 0 else 'reply_two')
            if index in (1, 4):
                return completed()
            # A completed phase does not demand another view of an old bubble.
            self.assertEqual(self.f.observations, 2 if index == 2 else 4)
            return dict(action='wait_for_update', arguments={})
        status = Mock()
        result = self.loop(choose, max_decisions=6, status=status)
        self.assertEqual((self.f.executions, self.f.observations), (2, 5))
        self.assertEqual(self.waits, [{}, {}])
        self.assertEqual(len(result['turns']), 6)
        self.assertEqual(sum(t['usage']['input_tokens'] + t['usage']['output_tokens']
                             for t in result['turns']), 12)
        self.assertFalse(result['task_completed'])
        self.assertFalse(result['finished_by_model'])
        self.assertEqual(result['completion_reviews'], [])
        self.assertEqual(result['stopped_reason'], 'bounded_episode_decision_limit')
        status.assert_called()
        events = [json.loads(s) for s in (self.f.output/'learning-events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['event'] == 'chat_turn_recorded' for e in events), 2)

    def test_ordinary_assistance_waits_for_operator_then_continues_without_motion_replay(self):
        for kind in ('operator_action', 'missing_external_capability'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve()); f.statuses = ['ok']
                waits = []
                session = None
                def wake(arguments):
                    waits.append(arguments)
                    observation = f.observe()
                    observation['operator_feedback'] = [dict(source='user', text='keyboard focus restored')]
                    return observation
                def choose(index, prompt):
                    if index == 0: return assistance(prompt, kind)
                    if index == 1:
                        self.assertIn('keyboard focus restored', json.dumps(prompt))
                        self.assertEqual(prompt['current_observation_evidence_id'], 'obs-0002')
                        return program('continue_after_help')
                    return completed()
                session = Session(choose)
                result = run_episode(session=session, observe=f.observe, execute=f.execute, store=f.store,
                    output=f.output, task_contract=self.contract(), wait_for_update=wake, max_decisions=3)
                self.assertEqual(len(session.prompts), 3)
                self.assertEqual(len(waits), 1)
                self.assertEqual(waits[0], {})
                self.assertEqual((f.executions, f.observations), (1, 3))
                self.assertFalse(result['task_completed'])
                self.assertFalse(result['finished_by_model'])
                self.assertEqual(result['stopped_reason'], 'bounded_episode_decision_limit')

    def test_partial_checkpoint_and_new_skill_do_not_end_conversation_or_clear_evidence(self):
        saved = []
        def choose(index, prompt):
            if index == 0:
                return skill([prompt['current_observation_evidence_id']], 1)
            if index == 1:
                saved.append(prompt['feedback']['candidate']['candidate_id'])
                result = finish(saved); result['action'] = 'checkpoint'
                return result
            if index == 2:
                return dict(action='inspect_skill', arguments=dict(candidate_id=saved[0]))
            if index == 3:
                self.assertEqual(prompt['feedback']['skill']['profile']['revision'], 1)
                return completed(saved)
            return dict(action='wait_for_update', arguments={})
        result = self.loop(choose, max_decisions=5)
        self.assertEqual(len(result['turns']), 5)
        self.assertEqual(len(result['saved_candidates']), 1)
        self.assertEqual(self.f.store.get_skill(saved[0])['profile']['revision'], 1)
        self.assertEqual(self.f.executions, 0)
        self.assertFalse(result['stable_promoted'])
        self.assertFalse(result['task_completed'])

    def test_no_demo_existing_return_candidate_can_fail_revise_and_verify_in_same_loop(self):
        historical = self.f.evidence('historical-return', {'source': 'old-scoped-fixture'})
        prior_args = skill([historical], 1)['arguments']; prior_args['id'] = 'learned_return'
        prior = self.f.store.propose('skill', prior_args)['candidate_id']
        self.f.statuses = ['blocked', 'ok']
        newly_saved = []
        lesson_ids = []
        def choose(index, prompt):
            self.assertNotIn('demo_trajectories', prompt)
            if index == 0:
                return dict(action='inspect_skill', arguments=dict(candidate_id=prior))
            if index == 1:
                self.assertEqual(prompt['feedback']['skill']['profile']['revision'], 1)
                return dict(action='verify_skill', arguments=dict(candidate_id=prior),
                            hypothesis='Try the existing released-to-released Return candidate once.')
            if index == 2:
                value = skill([prompt['current_observation_evidence_id']], 2)
                value['arguments']['id'] = 'learned_return'
                return value
            if index == 3:
                newly_saved.append(prompt['feedback']['candidate']['candidate_id'])
                return dict(action='verify_skill', arguments=dict(candidate_id=newly_saved[0]),
                            hypothesis='Check the revised candidate after a known failed attempt.')
            if index == 4:
                return dict(action='save_lesson', arguments=dict(id='return_revision_observation',
                    claim='Fixture version one was blocked and version two executed; real sending remains unmeasured.',
                    applicability=['This fixture scope only'], limitations=['No real robot or text inference'],
                    evidence_ids=[prompt['current_observation_evidence_id']]))
            if index == 5:
                lesson_ids.append(prompt['feedback']['candidate']['candidate_id'])
                return completed(newly_saved, lesson_ids)
            if index == 6: return dict(action='wait_for_update', arguments={})
            return completed(newly_saved, lesson_ids)
        result = self.loop(choose, max_decisions=8)
        self.assertEqual(self.f.store.get_skill(prior)['profile']['revision'], 1)
        self.assertEqual(self.f.store.get_skill(newly_saved[0])['profile']['revision'], 2)
        self.assertEqual([v['execution_status'] for v in result['verifications']], ['blocked', 'ok'])
        self.assertEqual((self.f.executions, self.f.observations), (2, 4))
        self.assertEqual(len(result['saved_candidates']), 2)
        self.assertEqual(len(result['turns']), 8)
        self.assertFalse(result['stable_promoted'])
        self.assertFalse(result['task_completed'])

    def test_confirm_completion_is_recoverably_rejected_instead_of_ending_service(self):
        def choose(index, prompt):
            if index == 0: return completed()
            if index == 1: return dict(action='confirm_completion', arguments={})
            self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
            return program('still_alive_after_bad_final_confirmation')
        result = self.loop(choose, max_decisions=3)
        self.assertEqual(self.f.executions, 1)
        self.assertEqual(len(result['turns']), 3)
        self.assertFalse(result['task_completed'])
        self.assertEqual(result['completion_reviews'], [])

    def test_wait_blocks_model_dispatch_until_callback_returns_and_forwards_roi(self):
        roi = [.2, .1, .8, .6]
        polls = []
        def wake(arguments):
            self.assertEqual(arguments, {'roi': roi})
            # Represent idle checks, with no model or motion dispatch between them.
            for _ in range(7):
                polls.append(len(self.session.prompts))
                self.assertEqual(self.f.executions, 0)
            return self.f.observe()
        def choose(index, prompt):
            if index == 0: return dict(action='wait_for_update', arguments=dict(roi=roi))
            self.assertEqual(prompt['current_observation_evidence_id'], 'obs-0002')
            return completed()
        result = self.loop(choose, max_decisions=2, wait_for_update=wake)
        self.assertEqual(polls, [1] * 7)
        self.assertEqual(len(result['turns']), 2)
        self.assertEqual(self.f.observations, 2)
        self.assertEqual(self.f.executions, 0)

    def test_operator_feedback_arrives_before_normal_model_turns_without_wait(self):
        historical = self.f.evidence('retained-return-evidence', {'fixture': 'prior learning'})
        prior = self.f.store.propose('skill', skill([historical], 3)['arguments'])['candidate_id']
        original_skill = self.f.store.get_skill(prior)
        records = [
            dict(protocol='physical-chat-operator-feedback-v1', id='operator-one', source='user',
                 utc_ns=1001, text='Return 已学会，继续当前群。\n保留原技能。'),
            dict(protocol='physical-chat-operator-feedback-v1', id='operator-two', source='user',
                 utc_ns=1002, text='刚才的 here 已发送；不要重复发送。'),
        ]
        supplied = [[copy.deepcopy(records[0])], [copy.deepcopy(records[1])], []]
        callback_order = []
        def updates():
            index = len(callback_order)
            # The callback must be checked before, not after, each model turn.
            self.assertEqual(len(self.session.prompts), index)
            callback_order.append(index)
            return supplied[index]
        def choose(index, prompt):
            if index < 2:
                self.assertEqual(prompt['operator_feedback'], [records[index]])
                return program('normal_one' if index == 0 else 'normal_two')
            self.assertNotIn('operator_feedback', prompt)  # No stale feedback replay.
            return completed()
        wait = Mock(side_effect=AssertionError('Active-turn feedback must not require wait'))
        result = self.loop(choose, max_decisions=3, wait_for_update=wait, operator_updates=updates)
        wait.assert_not_called()
        self.assertEqual(callback_order, [0, 1, 2])
        self.assertEqual((self.f.executions, self.f.observations), (2, 3))
        self.assertEqual(self.f.store.get_skill(prior), original_skill)
        self.assertEqual(result['saved_candidates'], [])
        self.assertEqual(len(result['turns']), 3)
        self.assertEqual(sum(t['usage']['input_tokens'] + t['usage']['output_tokens']
                             for t in result['turns']), 6)
        self.assertFalse(result['task_completed'])
        self.assertFalse(result['finished_by_model'])
        events = [json.loads(s) for s in (self.f.output/'learning-events.jsonl').read_text().splitlines()]
        self.assertEqual([e['data'] for e in events if e['event'] == 'operator_feedback'],
                         [[records[0]], [records[1]]])
        for index in range(2):
            request = json.loads((self.f.output/('turn-%02d' % index)/'request.txt').read_text().split('\n')[0])
            self.assertEqual(request['operator_feedback'], [records[index]])

    def test_missing_wait_capability_rejected_before_initial_model_observe_or_execute(self):
        observe, execute, session = Mock(), Mock(), Mock()
        with self.assertRaises(ValueError):
            run_episode(session=session, observe=observe, execute=execute, store=self.f.store,
                        output=self.f.output, task_contract=self.contract(), wait_for_update=None)
        observe.assert_not_called(); execute.assert_not_called(); session.run_turn.assert_not_called()

    def test_wait_is_unavailable_in_single_exchange_mode(self):
        del self.document['session_mode']; self.write()
        wait = Mock(side_effect=AssertionError('No wait in legacy mode'))
        def choose(index, prompt):
            if index == 0: return dict(action='wait_for_update', arguments={})
            self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
            return assistance(prompt, 'hardware_fault')
        result = self.loop(choose, max_decisions=2, wait_for_update=wait)
        wait.assert_not_called()
        self.assertEqual(len(result['turns']), 2)
        self.assertEqual(self.f.executions, 0)

    def test_invalid_wait_fields_or_roi_reject_before_callback(self):
        for arguments in ({'path': '/arbitrary'}, {'roi': [0, 0, 2, 1]},
                          {'roi': [.9, .1, .2, .8]}, {'roi': [True, 0, 1, 1]},
                          {'roi': [0, 0, 1]}, {'roi': 'whole image'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                parse_decision(json.dumps(dict(action='wait_for_update', arguments=arguments)))

    def test_physical_or_hardware_assistance_still_ends_without_wait_or_motion(self):
        for kind in ('physical_safety', 'hardware_fault'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve()); wait = Mock()
                session = Session(lambda index, prompt: assistance(prompt, kind))
                result = run_episode(session=session, observe=f.observe, execute=f.execute, store=f.store,
                    output=f.output, task_contract=self.contract(), wait_for_update=wait)
                wait.assert_not_called()
                self.assertEqual((len(session.prompts), f.observations, f.executions), (1, 1, 0))
                self.assertFalse(result['task_completed'])
                self.assertEqual(result['reply']['assistance']['kind'], kind)

    def test_unknown_motion_and_terminal_controller_remain_terminal_without_replay(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve()); calls = []; wait = Mock()
                def execute(arguments):
                    calls.append(arguments)
                    result = dict(receipts=[], controller_terminal=True) if terminal else {}
                    return dict(evidence_id=f.evidence('exec-stop', {}), response=dict(
                        status='ok' if terminal else 'unknown', result=result))
                session = Session(lambda *_: program('single_dispatch'))
                result = run_episode(session=session, observe=f.observe, execute=execute, store=f.store,
                    output=f.output, task_contract=self.contract(), wait_for_update=wait)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(session.prompts), 1)
                self.assertEqual(f.observations, 1)
                self.assertFalse(result['task_completed'])
                wait.assert_not_called()
                self.assertEqual(result['stopped_reason'],
                    'native_controller_terminal_no_retry' if terminal else 'unknown_motion_no_retry')

    def test_wait_callback_fault_and_repeated_observation_do_not_become_model_retry(self):
        for failure in ('connection', 'untrusted_frame', 'duplicate', 'operator_interrupt'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve()); initial = f.observe()
                def wake(arguments):
                    if failure == 'connection': raise OSError('Fixture connection lost')
                    if failure == 'untrusted_frame': raise ValueError('Fixture frame verification failed')
                    if failure == 'operator_interrupt': raise KeyboardInterrupt()
                    return initial
                session = Session(lambda *_: dict(action='wait_for_update', arguments={}))
                expected = (OSError if failure == 'connection' else ValueError if failure == 'untrusted_frame'
                            else KeyboardInterrupt if failure == 'operator_interrupt' else RuntimeError)
                with self.assertRaises(expected):
                    run_episode(session=session, observe=lambda: initial, execute=f.execute, store=f.store,
                        output=f.output, task_contract=self.contract(), wait_for_update=wake, max_decisions=4)
                self.assertEqual(len(session.prompts), 1)
                self.assertEqual(f.executions, 0)
                self.assertFalse(list(f.output.glob('turn-*/protocol-rejection.json')))


if __name__ == '__main__':
    unittest.main()
