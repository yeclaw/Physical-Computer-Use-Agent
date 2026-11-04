"""Synthetic trajectory evaluation; no physical keyboard, camera, or model."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.contracts import EvaluationPlan, Scope, Usage, canonical, digest
from astra_robotics_agent.physical_keyboard.ledger import Ledger
from astra_robotics_agent.physical_keyboard.rpc import RpcClient, default_safety
from astra_robotics_agent.physical_keyboard.synthetic_keyboard import SyntheticKeyboard, IncompleteProgram
from astra_robotics_agent.physical_keyboard.trial_runner import TrialRunner


def config():
    return dict(protocol='synthetic-keyboard-config-v1', keys={'a': [0, 0], 'b': [.025, 0]},
                initial_xyz_m=[0, 0, .22], contact_z_m=.19, registration_z_m=.182,
                release_z_m=.21, minimum_safe_z_m=.175, key_radius_m=.008)


def report_path(points, *, initial=None):
    receipts = []
    if initial is not None:
        receipts.append(dict(method='observe', arguments={}, result=dict(status='ok', observation=dict(
            evidence_mode='synthetic', pose={'xyz_m': initial, 'quaternion_xyzw': [0, 0, 0, 1]}))))
    for index, point in enumerate(points):
        receipts.append(dict(method='move_tcp', arguments={'xyz_m': list(point)}, result=dict(
            status='ok', executed=True, request_id='move-' + str(index), method='move_tcp',
            evidence_mode='synthetic', pose={'xyz_m': list(point), 'quaternion_xyzw': [0, 0, 0, 1]})))
    return dict(evidence_mode='synthetic', receipts=receipts)


class SyntheticKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.keyboard = SyntheticKeyboard(config())

    def test_complete_program_requires_release_and_home_beyond_literal_text(self):
        text_path = [[0, 0, .18], [0, 0, .22], [.025, 0, .22], [.025, 0, .18], [.025, 0, .22]]
        self.assertEqual(self.keyboard.evaluate(report_path(text_path))['observed_text'], 'ab')
        with self.assertRaises(IncompleteProgram) as caught:
            self.keyboard.evaluate_completed_trial(report_path(text_path))
        evidence = json.loads(caught.exception.evidence)
        self.assertEqual(evidence['observed_text'], 'ab')
        self.assertIn('final_pose_not_home', evidence['completion_reasons'])
        complete = self.keyboard.evaluate_completed_trial(report_path(text_path + [[0, 0, .22]]))
        self.assertEqual(complete['observed_text'], 'ab')
        self.assertTrue(json.loads(complete['evidence'])['completion_valid'])

    def test_correct_text_then_unreleased_press_cannot_validate_complete_program(self):
        points = [[0, 0, .18], [0, 0, .22], [.025, 0, .22], [.025, 0, .18],
                  [.025, 0, .22], [0, 0, .22], [0, 0, .18]]
        self.assertEqual(self.keyboard.evaluate(report_path(points))['observed_text'], 'ab')
        with self.assertRaises(IncompleteProgram) as caught:
            self.keyboard.evaluate_completed_trial(report_path(points))
        evidence = json.loads(caught.exception.evidence)
        self.assertIn('unreleased_stroke', evidence['completion_reasons'])
        self.assertFalse(evidence['completion_valid'])

    def evaluate(self, points, **kwargs):
        result = self.keyboard.evaluate(report_path(points, **kwargs))
        self.assertEqual(set(result), {'observed_text', 'evidence'})
        self.assertIsInstance(result['evidence'], bytes)
        evidence = json.loads(result['evidence'])
        self.assertEqual(result['evidence'], canonical(evidence))
        self.assertEqual(evidence['evidence_mode'], 'synthetic')
        self.assertFalse(evidence['physical_or_ocr'])
        self.assertEqual(evidence['evaluator_config_sha256'], self.keyboard.config_sha256)
        return result['observed_text'], evidence

    def test_nominal_contact_is_not_a_registered_key(self):
        text, evidence = self.evaluate([[0, 0, .19], [0, 0, .22]])
        self.assertEqual(text, '')
        self.assertEqual(evidence['actual_minimum_z_m'], .19)
        self.assertFalse(evidence['strokes'][0]['registered'])
        self.assertTrue(evidence['strokes'][0]['released'])

    def test_deeper_complete_stroke_registers_after_release(self):
        text, evidence = self.evaluate([[0, 0, .18], [0, 0, .22]])
        self.assertEqual(text, 'a')
        self.assertEqual(evidence['registered_strokes'], 1)
        self.assertEqual(evidence['released_strokes'], 1)
        self.assertTrue(evidence['strokes'][0]['counted'])

    def test_contact_then_deeper_motion_can_register_same_stroke(self):
        text, evidence = self.evaluate([[0, 0, .19], [0, 0, .185], [0, 0, .18], [0, 0, .22]])
        self.assertEqual(text, 'a')
        self.assertEqual(len(evidence['strokes']), 1)

    def test_press_without_release_cannot_count(self):
        text, evidence = self.evaluate([[0, 0, .18]])
        self.assertEqual(text, '')
        self.assertTrue(evidence['strokes'][0]['registered'])
        self.assertFalse(evidence['strokes'][0]['released'])

    def test_partial_lift_below_release_does_not_count(self):
        text, evidence = self.evaluate([[0, 0, .18], [0, 0, .205]])
        self.assertEqual(text, '')
        self.assertEqual(evidence['released_strokes'], 0)

    def test_long_hold_and_depth_jitter_register_only_once(self):
        text, evidence = self.evaluate([[0, 0, .18], [0, 0, .179], [0, 0, .183],
                                        [0, 0, .18], [0, 0, .22]])
        self.assertEqual(text, 'a')
        self.assertEqual(evidence['registered_strokes'], 1)

    def test_full_release_allows_same_key_again(self):
        text, _ = self.evaluate([[0, 0, .18], [0, 0, .22], [0, 0, .18], [0, 0, .22]])
        self.assertEqual(text, 'aa')

    def test_multiple_keys_need_lateral_move_at_clearance(self):
        text, _ = self.evaluate([[0, 0, .18], [0, 0, .22], [.025, 0, .22],
                                 [.025, 0, .18], [.025, 0, .22]])
        self.assertEqual(text, 'ab')

    def test_space_is_a_valid_key(self):
        value = config()
        value['keys'] = {' ': [0, 0]}
        result = SyntheticKeyboard(value).evaluate(report_path([[0, 0, .18], [0, 0, .22]]))
        self.assertEqual(result['observed_text'], ' ')

    def test_wrong_xy_cannot_register(self):
        text, evidence = self.evaluate([[.06, 0, .18], [.06, 0, .22]])
        self.assertEqual(text, '')
        self.assertEqual(evidence['registered_strokes'], 0)

    def test_diagonal_crossing_through_key_is_not_a_press(self):
        text, evidence = self.evaluate([[.04, 0, .18], [.04, 0, .22]], initial=[-.04, 0, .22])
        self.assertEqual(text, '')
        self.assertEqual(evidence['registered_strokes'], 0)
        self.assertTrue(evidence['skipped_contacts'])

    def test_horizontal_sweep_below_contact_never_arms_a_key(self):
        text, evidence = self.evaluate([[-.04, 0, .18], [.04, 0, .18], [.04, 0, .22]])
        self.assertEqual(text, '')
        self.assertEqual(evidence['registered_strokes'], 0)

    def test_escaping_key_while_held_invalidates_stroke_even_if_returned(self):
        text, evidence = self.evaluate([[0, 0, .18], [.025, 0, .18], [0, 0, .18], [0, 0, .22]])
        self.assertEqual(text, '')
        self.assertIn('lateral_escape', evidence['strokes'][0]['invalid_reasons'])

    def test_excessive_depth_cannot_succeed_after_release(self):
        text, evidence = self.evaluate([[0, 0, .174], [0, 0, .22]])
        self.assertEqual(text, '')
        self.assertTrue(evidence['safety_violation'])
        self.assertIn('unsafe_depth', evidence['strokes'][0]['invalid_reasons'])

    def test_unsafe_earlier_motion_invalidates_later_valid_stroke(self):
        text, evidence = self.evaluate([[.06, 0, .17], [.06, 0, .22], [0, 0, .22],
                                        [0, 0, .18], [0, 0, .22]])
        self.assertEqual(text, '')
        self.assertEqual(evidence['valid_stroke_text'], 'a')
        self.assertTrue(evidence['safety_violation'])

    def test_registration_release_and_safe_boundaries_are_inclusive(self):
        text, _ = self.evaluate([[0, 0, .182], [0, 0, .21], [0, 0, .175], [0, 0, .21]])
        self.assertEqual(text, 'aa')

    def test_skill_result_and_candidate_text_claims_are_ignored(self):
        report = report_path([[0, 0, .19], [0, 0, .22]])
        report.update(skill_result={'observed_text': 'ab', 'success': True}, observed_text='ab', text_verified=True)
        result = self.keyboard.evaluate(report)
        self.assertEqual(result['observed_text'], '')

    def test_executed_actual_pose_wins_over_commanded_target(self):
        report = report_path([[0, 0, .19], [0, 0, .22]])
        report['receipts'][0]['arguments']['xyz_m'] = [0, 0, .18]
        self.assertEqual(self.keyboard.evaluate(report)['observed_text'], '')

    def test_unexecuted_rejected_or_truthy_integer_acknowledgements_do_not_count(self):
        for result_patch in [{'executed': False}, {'status': 'rejected'}, {'executed': 1}, {'executed': None}]:
            with self.subTest(patch=result_patch):
                report = report_path([[0, 0, .18], [0, 0, .22]])
                report['receipts'][0]['result'].update(result_patch)
                self.assertEqual(self.keyboard.evaluate(report)['observed_text'], '')

    def test_unknown_move_breaks_trace_and_cannot_be_overridden_by_later_stroke(self):
        report = report_path([[0, 0, .18], [0, 0, .22]])
        report['receipts'].insert(0, dict(method='move_tcp', result={'status': 'unknown', 'executed': None}))
        result = self.keyboard.evaluate(report)
        self.assertEqual(result['observed_text'], '')
        self.assertFalse(json.loads(result['evidence'])['trajectory_complete'])

    def test_duplicate_historical_receipt_cannot_fake_another_keypress(self):
        report = report_path([[0, 0, .18], [0, 0, .22]])
        report['receipts'] += list(report['receipts'])
        result = self.keyboard.evaluate(report)
        evidence = json.loads(result['evidence'])
        self.assertEqual(result['observed_text'], 'a')
        self.assertEqual(evidence['executed_move_count'], 2)
        self.assertEqual(evidence['duplicate_receipt_count'], 2)

    def test_conflicting_same_request_id_receipts_are_rejected(self):
        report = report_path([[0, 0, .18], [0, 0, .22]])
        report['receipts'][1]['result']['request_id'] = 'move-0'
        with self.assertRaises(ValueError):
            self.keyboard.evaluate(report)

    def test_source_initial_pose_prevents_invented_downstroke(self):
        text, evidence = self.evaluate([[0, 0, .18], [0, 0, .22]], initial=[0, 0, .18])
        self.assertEqual(text, '')
        self.assertEqual(evidence['initial_pose_source'], 'source_observation')

    def test_config_is_deeply_copied_and_public_description_hides_registration(self):
        value = config()
        keyboard = SyntheticKeyboard(value)
        fingerprint = keyboard.config_sha256
        value['registration_z_m'] = .195
        value['keys']['a'][0] = .1
        public = keyboard.public_description()
        self.assertNotIn('registration_z_m', public)
        self.assertNotIn('0.182', json.dumps(public))
        public['keys']['a'][0] = .1
        result = keyboard.evaluate(report_path([[0, 0, .18], [0, 0, .22]]))
        self.assertEqual(result['observed_text'], 'a')
        self.assertEqual(keyboard.config_sha256, fingerprint)

    def test_invalid_config_nonfinite_overlap_and_non_english_keys_rejected(self):
        variants = []
        for field, value in [('key_radius_m', 0), ('contact_z_m', True), ('release_z_m', float('nan')),
                             ('registration_z_m', .2), ('initial_xyz_m', [0, 0, .20]),
                             ('keys', {'A': [0, 0]}), ('keys', {'ab': [0, 0]}),
                             ('keys', {'a': [0, 0], 'b': [.016, 0]}),
                             ('keys', {'a': [float('inf'), 0]})]:
            c = config()
            c[field] = value
            variants.append(c)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                SyntheticKeyboard(value)

    def test_physical_report_and_nonfinite_executed_pose_are_rejected(self):
        report = report_path([[0, 0, .18]])
        report['evidence_mode'] = 'physical'
        with self.assertRaises(ValueError):
            self.keyboard.evaluate(report)
        report = report_path([[0, 0, .18]])
        report['receipts'][0]['result']['pose']['xyz_m'][2] = float('nan')
        with self.assertRaises(ValueError):
            self.keyboard.evaluate(report)

    def test_actual_rpc_and_trial_runner_use_motion_dependent_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safety = default_safety()
            scope = Scope('synthetic', 'keyboard-test', 'keyboard-layout',
                          digest(b'fixed-geometry'), digest(b'synthetic-protocol'), self.keyboard.config_sha256,
                          digest(b'target-a'), digest(canonical(safety)))
            with Ledger(root / 'ledger') as ledger:
                def bundle(label, depth, parent=None):
                    actions = [dict(method='move_tcp', arguments={'xyz_m': [0, 0, depth], 'quaternion_xyzw': [0, 0, 0, 1]}),
                               dict(method='observe', arguments={}),
                               dict(method='move_tcp', arguments={'xyz_m': [0, 0, .22], 'quaternion_xyzw': [0, 0, 0, 1]})]
                    return ledger.freeze_bundle(label=label, kind='skill_program' if parent else 'seed',
                        scope=scope, parent=parent, files={'profile.json': b'{}', 'actions.json': canonical(dict(
                            protocol='physical-keyboard-action-plan-v1', actions=actions))},
                        provenance={'origin': 'original', 'evidence_mode': 'synthetic'})
                seed = bundle('synthetic-shallow', .19)
                ledger.bootstrap(seed, reason='Handwritten software fixture, not model learning')
                candidate = bundle('synthetic-deeper', .18, seed)
                ledger.open_round('synthetic-round', candidate=candidate,
                    plan=EvaluationPlan({'a': 'a'}, 1, 1, 0), generation_usage=Usage.zero(),
                    metadata={'evidence_mode': 'synthetic'})
                rpc_config = dict(protocol='physical-keyboard-rpc-config-v1', backend='synthetic-fake',
                    scope=asdict(scope), safety=safety, allowed_profiles=[digest(b'{}')],
                    ledger_root=str(ledger.root.resolve()), initial_pose={'xyz_m': [0, 0, .22], 'quaternion_xyzw': [0, 0, 0, 1]})
                path = root / 'rpc-config.json'
                path.write_bytes(canonical(rpc_config))
                rpc = RpcClient(path, root / 'journal', root / 'server')
                try:
                    runner = TrialRunner(ledger, rpc, root / 'trials')
                    parent = runner.run('synthetic-round', 'parent-a', 'parent', 'a', 0, 'a-0', evaluator=self.keyboard.evaluate)
                    improved = runner.run('synthetic-round', 'candidate-a', 'candidate', 'a', 0, 'a-0', evaluator=self.keyboard.evaluate)
                    self.assertEqual(parent['observed_text'], '')
                    self.assertEqual(improved['observed_text'], 'a')
                    self.assertEqual(ledger.decide('synthetic-round')['correct'], {'parent': 0, 'candidate': 1})
                finally:
                    exit_report = rpc.close()
                    self.assertTrue(exit_report['waited'])


if __name__ == '__main__':
    unittest.main()
