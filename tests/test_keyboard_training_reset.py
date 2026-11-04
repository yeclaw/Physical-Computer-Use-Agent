"""Training reset contract tests using no model, SDK, camera or desktop APIs."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.learning_loop import parse_decision, run_episode
from astra_robotics_agent.physical_keyboard.training_reset import (
    HOME_POSE, perform_reset, resolve_recovery, validate_reset_arguments,
)


SCOPE = dict(task_id='typing', device_id='fake-startouch', layout_id='fake-qwerty')
RECOVERY_LINES = ['def main(robot, profile):', '    return robot.observe()']


def arguments(return_pose='home', recovery=None):
    return dict(reason='Restart this training input after ambiguous repeated characters.',
                return_pose=copy.deepcopy(return_pose),
                recovery=copy.deepcopy(recovery) if recovery is not None else dict(
                    source_lines=list(RECOVERY_LINES), profile={'reference': 'synthetic'},
                    label='return_to_observation'))


def finished(skills=(), lessons=()):
    return dict(action='finish', result=dict(status='partial', reflection=dict(
        summary='Preserved diagnostic learning; reset is not a physical typing result.',
        skills_saved=list(skills), lessons_saved=list(lessons),
        limitations=['Synthetic reset only; no physical key-release verification.'])))


def known_motion(status='ok', receipts=None):
    return dict(status=status, result=dict(
        receipts=[{'status': 'ok'}] if receipts is None else receipts,
        motion_outcome_uncertain=False))


def editor_success():
    return dict(status='ok', result=dict(cleared=True, focus_requested=True,
                                        ime_reset_requested=True, buffer_read=False))


class Session:
    def __init__(self, choose):
        self.choose = choose
        self.prompts = []
        self.images = []
        self.usage = []

    def run_turn(self, prompt, callback, images):
        index = len(self.prompts)
        value = json.loads(prompt.split('\n')[0])
        self.prompts.append(value)
        self.images.append(images)
        usage = dict(input_tokens=10 + index, output_tokens=1 + index)
        self.usage.append(usage)
        decision = self.choose(index, value)
        return dict(turn_id='fake-turn-%d' % index, usage=usage,
                    final_text=json.dumps(decision, allow_nan=False))


class Fixture:
    def __init__(self, root):
        self.root = root = root.resolve()
        self.store = KnowledgeStore(root / 'knowledge', SCOPE, 'fake-reset-run')
        self.output = root / 'output'
        self.output.mkdir()
        self.events = []
        self.observations = 0
        self.reset_calls = 0
        self.executions = 0
        self.states = 0
        self.clears = 0
        self.motion_response = known_motion()
        self.actual_pose = copy.deepcopy(HOME_POSE)
        self.editor_reply = editor_success()
        self.execute_error = self.state_error = self.editor_error = None

    def evidence(self, name, value):
        path = self.root / (name + '.json')
        path.write_text(json.dumps(value, allow_nan=False))
        self.store.register_evidence(name, [path])
        return name

    def observe(self):
        self.events.append('observe')
        self.observations += 1
        evidence_id = self.evidence('observation-%d' % self.observations,
                                    {'synthetic_frame': self.observations})
        return dict(evidence_id=evidence_id, contentItems=[
            {'type': 'inputText', 'text': 'Synthetic camera evidence ' + evidence_id},
            {'type': 'inputImage', 'imageUrl': 'data:image/jpeg;base64,fixture'}])

    def execute(self, recovery):
        self.events.append('execute')
        self.executions += 1
        if self.execute_error is not None:
            raise self.execute_error
        response = copy.deepcopy(self.motion_response)
        return dict(response=response, evidence_id=self.evidence(
            'recovery-%d' % self.executions, dict(recovery=recovery, response=response)))

    def state(self):
        self.events.append('state')
        self.states += 1
        if self.state_error is not None:
            raise self.state_error
        return {'arms': {'left': copy.deepcopy(self.actual_pose)}}

    def clear(self):
        self.events.append('clear')
        self.clears += 1
        if self.editor_error is not None:
            raise self.editor_error
        return copy.deepcopy(self.editor_reply)

    def reset(self, args, recovery):
        self.reset_calls += 1
        response = perform_reset(args, recovery, execute=self.execute,
                                 state=self.state, clear_editor=self.clear)
        return dict(response=response, evidence_id=self.evidence(
            'reset-%d' % self.reset_calls, dict(arguments=args, recovery=recovery,
                                             response=response)))

    def perform(self, args=None):
        args = arguments() if args is None else args
        return perform_reset(args, resolve_recovery(args, self.store),
                             execute=self.execute, state=self.state, clear_editor=self.clear)

    def run(self, session, **kwargs):
        kwargs.setdefault('max_decisions', 10)
        kwargs.setdefault('reset', self.reset)
        return run_episode(session=session, observe=self.observe, execute=self.execute,
                           store=self.store, output=self.output, **kwargs)

    def seed(self, kind, identifier, store=None):
        store = self.store if store is None else store
        path = self.root / (identifier + '-evidence.json')
        path.write_text('{"fixture": true}')
        store.register_evidence(identifier + '-evidence', [path])
        values = dict(id=identifier, applicability=['Synthetic fixture only'],
                      limitations=['Not a validated physical skill'],
                      evidence_ids=[identifier + '-evidence'])
        if kind == 'skill':
            values.update(source_lines=list(RECOVERY_LINES), profile={'seed': identifier})
        else:
            values['claim'] = 'Synthetic observation retained across training reset.'
        return store.propose(kind, values)


class ResetArgumentTests(unittest.TestCase):
    def test_home_explicit_pose_and_both_recovery_forms_preserve_values(self):
        for pose in ('home', HOME_POSE):
            for recovery in (None, {'candidate_id': 'skill-return-' + 'a' * 64}):
                with self.subTest(pose=pose, recovery=recovery):
                    args = arguments(pose, recovery)
                    before = copy.deepcopy(args)
                    self.assertIs(validate_reset_arguments(args), args)
                    decision = dict(action='reset', arguments=args)
                    self.assertEqual(parse_decision(json.dumps(decision)), decision)
                    self.assertEqual(args, before)

    def test_malformed_reset_arguments_are_rejected_without_callbacks(self):
        cases = [None, [], {}, dict(arguments(), extra=True)]
        for field in ('reason', 'return_pose', 'recovery'):
            value = arguments()
            del value[field]
            cases.append(value)
        for value in ('', ' ', True, 5, 'x' * 2049):
            cases.append(dict(arguments(), reason=value))
        for value in ('HOME', 'learned_pose', [],
                      {'xyz_m': [0, 0, 0]},
                      dict(HOME_POSE, extra=0),
                      dict(HOME_POSE, xyz_m=(0, 0, 0)),
                      dict(HOME_POSE, xyz_m=[0, True, 0]),
                      dict(HOME_POSE, xyz_m=[0, math.nan, 0]),
                      dict(HOME_POSE, xyz_m=[0, math.inf, 0]),
                      dict(HOME_POSE, quaternion_xyzw=[0, 0, 0, 0]),
                      dict(HOME_POSE, quaternion_xyzw=[0, 0, 0, 2]),
                      dict(HOME_POSE, quaternion_xyzw=[0, 0, 1])):
            cases.append(dict(arguments(), return_pose=value))
        bad_recoveries = [None, [], {}, {'candidate_id': ''}, {'candidate_id': True},
                          {'candidate_id': 'valid-looking', 'profile': {}},
                          {'candidate_id': 'x', 'source_lines': RECOVERY_LINES}]
        for field, values in (
            ('source_lines', [[], '', [None], ['def main(robot, profile):\n    return 1'], ['\r']]),
            ('profile', [None, [], 'profile']),
            ('label', ['', 'has spaces', '../path', 'x' * 65, 0]),
        ):
            for value in values:
                bad_recoveries.append(dict(arguments()['recovery'], **{field: value}))
        for recovery in bad_recoveries:
            cases.append(dict(arguments(), recovery=recovery))
        for args in cases:
            with self.subTest(args=args):
                calls = []
                def unexpected(*_):
                    calls.append('called')
                    self.fail('Invalid reset must be rejected before any callback')
                with self.assertRaises(ValueError):
                    perform_reset(args, {}, execute=unexpected, state=unexpected,
                                  clear_editor=unexpected)
                self.assertEqual(calls, [])

    def test_reset_has_strict_top_level_action_fields(self):
        for decision in (dict(action='reset', arguments=arguments(), hypothesis='not allowed'),
                         dict(action='reset', arguments=arguments(), extra=True),
                         dict(action='reset', arguments={})):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                parse_decision(json.dumps(decision))

    def test_inline_recovery_and_scoped_frozen_skill_resolve_without_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            inline = arguments()
            resolved = resolve_recovery(inline, fixture.store)
            self.assertEqual(resolved, dict(source='\n'.join(RECOVERY_LINES) + '\n',
                                            profile={'reference': 'synthetic'},
                                            label='return_to_observation'))
            saved = fixture.seed('skill', 'return_pose')
            args = arguments(recovery={'candidate_id': saved['candidate_id']})
            resolved = resolve_recovery(args, fixture.store)
            skill = fixture.store.get_skill(saved['candidate_id'])
            self.assertEqual(resolved, {key: skill[key] for key in ('source', 'profile', 'label')})
            self.assertEqual((fixture.reset_calls, fixture.executions, fixture.clears), (0, 0, 0))


class ResetExecutionTests(unittest.TestCase):
    def test_complete_reset_executes_once_then_measures_then_clears(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            args = arguments()
            before = copy.deepcopy(args)
            result = fixture.perform(args)
            self.assertEqual(fixture.events, ['execute', 'state', 'clear'])
            self.assertEqual((fixture.executions, fixture.states, fixture.clears), (1, 1, 1))
            self.assertEqual(args, before)
            self.assertEqual(result['status'], 'ok')
            details = result['result']
            self.assertEqual(details['stage'], 'reset_completed')
            self.assertTrue(details['reset_applied'])
            self.assertFalse(details['motion_outcome_uncertain'])
            self.assertFalse(details['reset_outcome_uncertain'])
            self.assertEqual(details['return_pose'], HOME_POSE)
            self.assertEqual(details['physical_release'], 'unverified_external_camera_required')
            self.assertEqual(details['text_success'], 'not_a_physical_typing_or_correction_result')
            self.assertEqual(details['editor_receipt'], editor_success())

    def test_measured_learned_pose_and_equivalent_quaternion_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            pose = dict(xyz_m=[.1, -.02, .2], quaternion_xyzw=[0, math.sin(.1), 0, math.cos(.1)])
            fixture.actual_pose = dict(xyz_m=[.101, -.02, .2],
                                       quaternion_xyzw=[-x for x in pose['quaternion_xyzw']])
            result = fixture.perform(arguments(pose))
            self.assertEqual(result['status'], 'ok')
            self.assertAlmostEqual(result['result']['position_error_m'], .001)
            self.assertAlmostEqual(result['result']['rotation_error_rad'], 0)
            self.assertEqual(fixture.clears, 1)

    def test_measured_position_or_orientation_mismatch_never_clears_editor(self):
        for error in ('position', 'rotation'):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                if error == 'position':
                    fixture.actual_pose['xyz_m'][0] += .0051
                else:
                    fixture.actual_pose['quaternion_xyzw'] = [0, 0, math.sin(.051/2), math.cos(.051/2)]
                result = fixture.perform()
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['result']['stage'], 'return_pose_not_reached')
                self.assertFalse(result['result']['reset_applied'])
                self.assertEqual((fixture.executions, fixture.states, fixture.clears), (1, 1, 0))

    def test_known_blocked_or_failed_recovery_never_measures_or_clears(self):
        for response in (known_motion('blocked'), known_motion('error'),
                         known_motion(receipts=[{'status': 'rejected'}]),
                         known_motion(receipts=[{'status': 'error'}])):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                fixture.motion_response = response
                result = fixture.perform()
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['result']['stage'], 'recovery_not_completed')
                self.assertFalse(result['result']['motion_outcome_uncertain'])
                self.assertEqual((fixture.executions, fixture.states, fixture.clears), (1, 0, 0))

    def test_unknown_or_untrusted_motion_never_retries_or_clears(self):
        responses = [dict(status='unknown', result={'receipts': []}),
                     dict(status='ok', result=dict(receipts=[], motion_outcome_uncertain=True)),
                     known_motion(receipts=[{'status': 'unknown'}]),
                     dict(status='ok', result={'receipts': []}),
                     dict(status='error', error={'type': 'TransportLost'})]
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                fixture.motion_response = response
                result = fixture.perform()
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['result']['stage'], 'recovery_unknown_no_retry')
                self.assertTrue(result['result']['motion_outcome_uncertain'])
                self.assertEqual((fixture.executions, fixture.states, fixture.clears), (1, 0, 0))

    def test_execute_state_or_editor_exception_escapes_without_retry(self):
        for stage, counts in (('execute', (1, 0, 0)), ('state', (1, 1, 0)),
                              ('editor', (1, 1, 1))):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                setattr(fixture, stage + '_error', ValueError(stage + ' dispatched failure'))
                with self.assertRaisesRegex(ValueError, 'dispatched failure'):
                    fixture.perform()
                self.assertEqual((fixture.executions, fixture.states, fixture.clears), counts)

    def test_editor_failure_classifies_known_vs_unknown_without_retry(self):
        for uncertain in (False, True, None):
            with self.subTest(uncertain=uncertain), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                fixture.editor_reply = dict(status='error', error={'uncertain': uncertain})
                result = fixture.perform()
                self.assertEqual(result['status'], 'blocked' if uncertain is False else 'unknown')
                self.assertEqual(result['result']['stage'], 'editor_reset_failed')
                self.assertEqual(result['result']['reset_outcome_uncertain'], uncertain is not False)
                self.assertFalse(result['result']['reset_applied'])
                self.assertEqual((fixture.executions, fixture.states, fixture.clears), (1, 1, 1))

    def test_untrusted_editor_success_does_not_claim_completion_or_retry(self):
        for reply in (None, {}, dict(status='ok', result={}),
                      dict(status='ok', result=dict(editor_success()['result'], buffer_read=True))):
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                fixture.editor_reply = reply
                if reply == {}:
                    result = fixture.perform()
                    self.assertEqual(result['status'], 'unknown')
                else:
                    with self.assertRaisesRegex(RuntimeError, 'receipt'):
                        fixture.perform()
                self.assertEqual(fixture.clears, 1)


class ResetLoopTests(unittest.TestCase):
    def test_success_refreshes_visual_evidence_and_records_reset_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            def choose(index, prompt):
                self.assertTrue(prompt['training_reset_available'])
                if index == 0:
                    return dict(action='reset', arguments=arguments())
                self.assertEqual(prompt['feedback']['action'], 'reset')
                self.assertEqual(prompt['feedback']['status'], 'ok')
                self.assertEqual(prompt['feedback']['verification'], 'training_reset_not_physical_correction')
                self.assertEqual(prompt['current_observation_evidence_id'], 'observation-2')
                self.assertFalse(prompt['observation_unchanged'])
                return finished()
            session = Session(choose)
            result = fixture.run(session)
            self.assertTrue(result['finished_by_model'])
            self.assertEqual(fixture.events, ['observe', 'execute', 'state', 'clear', 'observe'])
            self.assertEqual([bool(items) for items in session.images], [True, True])
            self.assertEqual(len(result['reset_attempts']), 1)
            self.assertEqual(result['reset_attempts'][0]['evidence_ids'],
                             ['observation-1', 'reset-1', 'observation-2'])
            self.assertEqual(result['verifications'], [])
            self.assertFalse(result['stable_promoted'])

    def test_disabled_and_legacy_reset_reject_before_any_reset_callback(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                def choose(index, prompt):
                    self.assertFalse(prompt['training_reset_available'])
                    if index == 0:
                        return dict(action='reset', arguments=arguments())
                    self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
                    self.assertIn('unavailable', prompt['feedback']['error'])
                    return finished()
                result = fixture.run(Session(choose), legacy_view_setup=legacy,
                                     reset=fixture.reset if legacy else None)
                self.assertTrue(result['finished_by_model'])
                self.assertEqual(result['reset_attempts'], [])
                self.assertEqual((fixture.reset_calls, fixture.executions, fixture.states, fixture.clears), (0, 0, 0, 0))

    def test_missing_cross_scope_non_skill_and_path_candidate_reject_before_callback(self):
        for kind in ('missing', 'cross_scope', 'lesson', 'path'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                if kind == 'cross_scope':
                    other = KnowledgeStore(fixture.root / 'knowledge', dict(SCOPE, layout_id='other'), 'other')
                    candidate_id = fixture.seed('skill', 'other_recovery', store=other)['candidate_id']
                elif kind == 'lesson':
                    candidate_id = fixture.seed('lesson', 'not_executable')['candidate_id']
                elif kind == 'path':
                    candidate_id = str(fixture.root / 'untrusted.py')
                else:
                    candidate_id = 'skill-missing-' + 'f' * 64
                def choose(index, prompt):
                    if index == 0:
                        return dict(action='reset', arguments=arguments(recovery={'candidate_id': candidate_id}))
                    self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
                    return finished()
                result = fixture.run(Session(choose))
                self.assertTrue(result['finished_by_model'])
                self.assertEqual(result['reset_attempts'], [])
                self.assertEqual((fixture.reset_calls, fixture.executions, fixture.clears), (0, 0, 0))

    def test_unknown_motion_or_editor_reset_stops_without_new_model_turn_or_observation(self):
        for unknown in ('motion', 'editor'):
            with self.subTest(unknown=unknown), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                if unknown == 'motion':
                    fixture.motion_response = dict(status='unknown', result={'receipts': []})
                else:
                    fixture.editor_reply = dict(status='error', error={'uncertain': True})
                session = Session(lambda *_: dict(action='reset', arguments=arguments()))
                result = fixture.run(session)
                self.assertEqual(result['stopped_reason'], 'unknown_training_reset_no_retry')
                self.assertFalse(result['completed'])
                self.assertEqual((fixture.reset_calls, fixture.executions, fixture.observations, len(session.prompts)), (1, 1, 1, 1))
                self.assertEqual(fixture.clears, 0 if unknown == 'motion' else 1)

    def test_post_dispatch_callback_exceptions_are_fatal_not_model_rejections(self):
        for stage in ('execute', 'state', 'editor'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder))
                setattr(fixture, stage + '_error', ValueError('dispatched ' + stage + ' failure'))
                session = Session(lambda *_: dict(action='reset', arguments=arguments()))
                report = {}
                with self.assertRaisesRegex(ValueError, 'dispatched'):
                    fixture.run(session, report=report)
                self.assertTrue(report['reset_exception_no_retry'])
                self.assertEqual((fixture.reset_calls, len(session.prompts), fixture.observations), (1, 1, 1))
                self.assertFalse((fixture.output / 'turn-00' / 'protocol-rejection.json').exists())

    def test_reset_preserves_saved_knowledge_and_all_turn_usage_then_learning_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            saved = {}
            snapshot = {}
            def candidate_action(kind, identifier, evidence):
                args = dict(id=identifier, applicability=['Synthetic test scope'],
                            limitations=['Not physical success'], evidence_ids=[evidence])
                if kind == 'skill':
                    args.update(source_lines=list(RECOVERY_LINES), profile={'reference': 'synthetic'})
                else:
                    args['claim'] = 'Observed synthetic evidence remains available after reset.'
                return dict(action='save_' + kind, arguments=args)
            def choose(index, prompt):
                feedback = prompt['feedback']
                if index == 0:
                    return candidate_action('skill', 'learned_return', prompt['current_observation_evidence_id'])
                if index == 1:
                    saved['skill'] = feedback['candidate']['candidate_id']
                    return candidate_action('lesson', 'before_reset', prompt['current_observation_evidence_id'])
                if index == 2:
                    saved['lesson'] = feedback['candidate']['candidate_id']
                    snapshot.update({str(path): path.read_bytes() for path in
                                     (fixture.store.namespace / 'candidates').rglob('*') if path.is_file()})
                    return dict(action='reset', arguments=arguments(recovery={'candidate_id': saved['skill']}))
                if index == 3:
                    self.assertEqual(feedback['status'], 'ok')
                    self.assertEqual(feedback['evidence_ids'][-1], 'observation-2')
                    for path, original in snapshot.items():
                        self.assertEqual(Path(path).read_bytes(), original)
                    return dict(action='inspect_skill', arguments={'candidate_id': saved['skill']})
                if index == 4:
                    self.assertEqual(feedback['skill']['profile'], {'reference': 'synthetic'})
                    return candidate_action('lesson', 'after_reset', prompt['current_observation_evidence_id'])
                saved['after'] = feedback['candidate']['candidate_id']
                return finished(skills=[saved['skill']], lessons=[saved['lesson'], saved['after']])
            session = Session(choose)
            result = fixture.run(session)
            self.assertTrue(result['finished_by_model'])
            self.assertEqual(len(result['saved_candidates']), 3)
            self.assertEqual([turn['usage'] for turn in result['turns']], session.usage)
            self.assertEqual([turn['turn_id'] for turn in result['turns']],
                             ['fake-turn-%d' % index for index in range(6)])
            self.assertEqual(sum(turn['usage']['input_tokens'] + turn['usage']['output_tokens']
                                 for turn in result['turns']), 96)
            self.assertEqual((fixture.reset_calls, fixture.executions, fixture.clears), (1, 1, 1))
            self.assertEqual(len(result['reset_attempts']), 1)
            reopened = KnowledgeStore(fixture.root / 'knowledge', SCOPE, 'next-synthetic-run')
            self.assertEqual(reopened.get_skill(saved['skill'])['profile'], {'reference': 'synthetic'})
            self.assertIn(saved['lesson'], reopened.context_text())
            self.assertIn(saved['after'], reopened.context_text())
            for path, original in snapshot.items():
                self.assertEqual(Path(path).read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
