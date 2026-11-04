"""Persistent final-goal lifecycle; real temporary knowledge, no robot/model."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

from astra_robotics_agent.physical_keyboard.campaign import load_task_contract
from astra_robotics_agent.physical_keyboard.learning_loop import run_episode
from test_keyboard_learning_loop import Fixture, Session, SCOPE, finish, skill
from test_keyboard_training_reset import arguments as reset_arguments


ROOT = Path(__file__).resolve().parents[1]


class ContinuousLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.f = Fixture(self.root)
        document = json.loads((ROOT / 'configs/physical_keyboard/task_contract.json').read_text())
        document['scope'] = SCOPE
        path = self.root / 'task.json'
        path.write_text(json.dumps(document))
        self.contract = load_task_contract(path, SCOPE)
        evidence = self.f.evidence('prior-evidence', {'historical_fixture': True})
        self.prior = self.f.store.propose('skill', skill([evidence], 1)['arguments'])['candidate_id']

    def run_loop(self, choose, **kwargs):
        self.session = Session(choose)
        return run_episode(session=self.session, observe=kwargs.pop('observe', self.f.observe),
            execute=self.f.execute, store=self.f.store, output=self.f.output,
            task_contract=self.contract, **kwargs)

    def completion(self):
        decision = finish()
        decision['result'].update(status='completed', observed_text='hello')
        return decision

    def confirm(self, prompt, **updates):
        args = dict(review_observation_evidence_id=prompt['current_observation_evidence_id'],
                    observed_text='hello', released_and_input_stable=True,
                    skill_candidate_ids=[self.prior])
        args.update(updates)
        return dict(action='confirm_completion', arguments=args)

    def test_many_phase_summaries_keep_goal_thread_knowledge_and_all_usage(self):
        newly_saved = []
        def choose(index, prompt):
            self.assertEqual(prompt['task_state']['contract'], self.contract.document)
            self.assertIsNone(prompt['remaining_decisions'])
            if index == 0:
                return skill([prompt['current_observation_evidence_id']], 2)
            if index == 1:
                newly_saved.append(prompt['feedback']['candidate']['candidate_id'])
            if index <= 24:
                self.assertEqual(prompt['task_state']['checkpoints_completed'], max(0, index - 1))
                decision = finish(newly_saved)
                decision['action'] = 'checkpoint' if index % 2 else 'finish'
                decision['result']['status'] = ('partial', 'blocked', 'view_ready', 'calibrated_partial')[index % 4]
                return decision
            if index == 25:
                return self.completion()
            self.assertEqual(prompt['feedback']['phase'], 'completion_review')
            self.assertTrue(self.session.images[index])
            self.assertEqual(prompt['feedback']['result']['observed_text'], 'hello')
            return self.confirm(prompt)
        result = self.run_loop(choose)
        self.assertTrue(result['task_completed'])
        self.assertEqual(result['task_status'], 'completed_by_visual_claim')
        self.assertEqual(len(result['phase_checkpoints']), 24)
        self.assertEqual(len(result['turns']), 27)
        self.assertEqual(sum(t['usage']['input_tokens'] + t['usage']['output_tokens']
                             for t in result['turns']), 54)
        self.assertEqual(self.f.store.get_skill(newly_saved[0])['profile']['revision'], 2)
        self.assertEqual((self.f.executions, self.f.observations), (0, 26))
        self.assertFalse(result['stable_promoted'])
        self.assertFalse(result['independently_verified'])
        self.assertEqual(result['reply']['completion_review']['skill_candidate_ids'], [self.prior])

    def test_completed_proposal_alone_is_not_completion(self):
        result = self.run_loop(lambda *_: self.completion(), max_decisions=1)
        self.assertFalse(result['task_completed'])
        self.assertFalse(result['completed'])
        self.assertFalse(result['finished_by_model'])
        self.assertEqual(len(result['completion_reviews']), 1)
        self.assertEqual(self.f.observations, 2)
        self.assertEqual(result['stopped_reason'], 'bounded_episode_decision_limit')

    def test_wrong_text_release_forged_skill_and_stale_id_are_rejected_then_recover(self):
        bad_values = [dict(observed_text='Goal: hello'), dict(observed_text='helloo'),
                      dict(released_and_input_stable=False), dict(skill_candidate_ids=['imaginary']),
                      dict(review_observation_evidence_id='obs-0001')]
        def choose(index, prompt):
            if index == 0:
                return self.completion()
            if index > 1:
                self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
            return self.confirm(prompt, **(bad_values[index-1] if index <= len(bad_values) else {}))
        result = self.run_loop(choose)
        self.assertTrue(result['task_completed'])
        self.assertEqual(len(result['turns']), 7)
        self.assertEqual(self.f.observations, 2)
        self.assertEqual(self.f.executions, 0)

    def test_execution_reset_observe_and_checkpoint_invalidate_review(self):
        for action in ('run_skill', 'verify_skill', 'reset', 'observe', 'checkpoint'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve())
                reset_calls = []
                def reset(args, recovery):
                    reset_calls.append((args, recovery))
                    return dict(evidence_id=f.evidence('reset-proof', {}), response=dict(
                        status='ok', result=dict(receipts=[], motion_outcome_uncertain=False,
                                                reset_outcome_uncertain=False)))
                def choose(index, prompt):
                    if index == 0:
                        return self.completion()
                    if index == 1:
                        if action == 'run_skill':
                            return dict(action=action, arguments=dict(label='probe',
                                source_lines=['def main(robot, profile):', '    return 1'], profile={}),
                                hypothesis='Fixture execution only')
                        if action == 'verify_skill':
                            prior = f.store.propose('skill', skill([prompt['current_observation_evidence_id']], 1)['arguments'])
                            return dict(action=action, arguments=dict(candidate_id=prior['candidate_id']),
                                        hypothesis='Fixture verification only')
                        if action == 'reset':
                            return dict(action=action, arguments=reset_arguments())
                        if action == 'observe':
                            return dict(action=action, arguments={})
                        decision = finish(); decision['action'] = 'checkpoint'
                        return decision
                    if index == 2:
                        return self.confirm(prompt)
                    self.assertIn('no pending', prompt['feedback']['error'])
                    return finish()
                result = run_episode(session=Session(choose), observe=f.observe, execute=f.execute,
                    store=f.store, output=f.output, task_contract=self.contract,
                    reset=reset, max_decisions=4)
                self.assertFalse(result['task_completed'])
                self.assertEqual(f.executions, int(action in ('run_skill', 'verify_skill')))
                self.assertEqual(len(reset_calls), int(action == 'reset'))

    def test_knowledge_save_and_inspect_preserve_current_completion_review(self):
        def choose(index, prompt):
            if index == 0: return self.completion()
            if index == 1: return skill([prompt['current_observation_evidence_id']], 2)
            if index == 2:
                return dict(action='inspect_skill', arguments=dict(candidate_id=self.prior))
            self.assertIsNotNone(prompt['task_state']['pending_completion_observation_id'])
            return self.confirm(prompt)
        result = self.run_loop(choose)
        self.assertTrue(result['task_completed'])
        self.assertEqual((self.f.observations, self.f.executions), (2, 0))

    def test_repeated_observation_and_callback_fault_stop_without_schema_retry(self):
        for mode in ('duplicate', 'old', 'fault'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve())
                saved = []
                def observe():
                    if not saved:
                        saved.append(f.observe())
                        return saved[0]
                    if mode == 'fault': raise ValueError('untrusted camera evidence')
                    if mode == 'old' and len(saved) == 1:
                        saved.append(f.observe()); return saved[-1]
                    return saved[0]
                session = Session(lambda *_: self.completion())
                report = {}
                error = ValueError if mode == 'fault' else RuntimeError
                with self.assertRaises(error):
                    run_episode(session=session, observe=observe, execute=f.execute, store=f.store,
                        output=f.output, task_contract=self.contract, report=report)
                self.assertFalse(report['task_completed'])
                self.assertEqual(len(session.prompts), 2 if mode == 'old' else 1)
                self.assertFalse(list(f.output.glob('turn-*/protocol-rejection.json')))
                self.assertEqual(f.executions, 0)

    def test_unknown_execution_and_reset_remain_terminal_incomplete_without_replay(self):
        for action in ('run_skill', 'reset'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                f = Fixture(Path(folder).resolve()); f.statuses = ['unknown']
                calls = []
                def reset(*args):
                    calls.append(args)
                    return dict(evidence_id='reset-unknown', response=dict(status='unknown', result={}))
                decision = dict(action=action, arguments=reset_arguments()) if action == 'reset' else dict(
                    action=action, arguments=dict(label='probe', source_lines=[
                        'def main(robot, profile):', '    return 1'], profile={}), hypothesis='Fixture only')
                session = Session(lambda *_: decision)
                result = run_episode(session=session, observe=f.observe, execute=f.execute,
                    store=f.store, output=f.output, task_contract=self.contract, reset=reset)
                self.assertFalse(result['task_completed'])
                self.assertEqual(result['task_status'], 'stopped_incomplete')
                self.assertEqual((len(session.prompts), f.observations), (1, 1))
                self.assertEqual(f.executions + len(calls), 1)

    def test_assistance_is_explicit_unfinished_pause_with_current_evidence(self):
        result = self.run_loop(lambda i, p: dict(action='request_assistance', arguments=dict(
            kind='hardware_fault', reason='Fixture camera mount disconnected.',
            required_human_action='Restore the camera mount.',
            observation_evidence_id=p['current_observation_evidence_id'])))
        self.assertFalse(result['completed'])
        self.assertFalse(result['task_completed'])
        self.assertEqual(result['task_status'], 'awaiting_assistance')
        self.assertEqual(result['stopped_reason'], 'external_assistance_required')
        self.assertEqual(self.f.executions, 0)




if __name__ == '__main__':
    unittest.main()
