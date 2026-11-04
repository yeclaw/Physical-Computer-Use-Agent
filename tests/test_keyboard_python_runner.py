"""EXP-010 integration contracts for frozen Python skills, never real robots.

Every execution below uses the actual stdio RPC service and a fresh worker.
Only the synthetic-fake adapter runs; candidate output never supplies an
independent observation of a screen or a successful keypress.
"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.contracts import (
    EvaluationPlan, Scope, Usage, canonical, digest,
)
from astra_robotics_agent.physical_keyboard.ledger import Ledger, LedgerError
from astra_robotics_agent.physical_keyboard.rpc import RpcClient, default_safety
from astra_robotics_agent.physical_keyboard.trial_runner import TrialRunner
from astra_robotics_agent.physical_keyboard.transport import JsonProcess


DESCRIPTOR = {
    'protocol': 'physical-keyboard-executable-v1',
    'kind': 'bounded-python-v1',
    'entry': 'skill.py',
    'includes': ['helpers.py'],
}
PROFILE = {
    'evidence_mode': 'synthetic',
    'approach': [.01, 0, .2],
    'press': [.01, 0, .185],
    'release': [.01, 0, .21],
    'q': [0, 0, 0, 1],
    'target_x': .03,
    'step': .01,
}
HELPER = '''
def approach(robot, profile):
    return robot.move_tcp(profile['approach'], profile['q'])
'''
PRESS_RELEASE = '''
def main(robot, profile):
    approach(robot, profile)
    robot.move_tcp(profile['press'], profile['q'])
    robot.wait_bounded(0)
    robot.move_tcp(profile['release'], profile['q'])
    return {'status': 'success', 'observed_text': 'cat'}
'''


def source(value):
    return textwrap.dedent(value).lstrip().encode()


class AuditedRpc(RpcClient):
    """Real service transport with host-owned observation of grant ordering."""

    def __init__(self, config_path, journal_dir, artifacts_dir, ledger,
                 grant_overrides=None):
        self.ledger = ledger
        self.calls = []
        self.grant_overrides = grant_overrides or {}
        super().__init__(config_path, journal_dir, artifacts_dir)

    def call(self, method, arguments, timeout_s=5., *, rpc_id=None):
        arguments = dict(arguments)
        if method == 'authorize_trial':
            state = self.ledger.trial_state(arguments['round_id'], arguments['trial_id'])
            if (state is None or state['result'] is not None
                    or state['start_event_sha256'] != arguments['start_event_sha256']):
                raise AssertionError('Python authority preceded a durable unfinished start')
            arguments.update(self.grant_overrides)
        self.calls.append((method, arguments))
        return super().call(method, arguments, timeout_s, rpc_id=rpc_id)


class PythonTrialRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = Ledger(self.root / 'ledger')
        self.addCleanup(self.ledger.close)
        self.safety = default_safety()
        # Process startup timing is irrelevant to these synthetic logic tests.
        self.safety['observation_max_age_s'] = 5.
        self.scope = Scope(
            'synthetic', 'python-fixture-device', 'python-layout',
            *[digest(s.encode()) for s in ('config', 'protocol', 'evaluator', 'goal')],
            digest(canonical(self.safety)),
        )
        self.seed = self.bundle('seed', kind='seed')
        self.ledger.bootstrap(self.seed, reason='Handwritten synthetic seed, not a learned skill')
        self.candidate = self.bundle('candidate', parent=self.seed)
        self.open_round()
        self.rpc = None
        self.runner = None

    def bundle(self, label, *, parent=None, kind='skill_program', main=PRESS_RELEASE,
               helper=HELPER, profile=None, descriptor=None, omit=(), scope=None):
        files = {
            'execution.json': canonical(DESCRIPTOR if descriptor is None else descriptor),
            'skill.py': source(main),
            'helpers.py': source(helper),
            'profile.json': canonical(PROFILE if profile is None else profile),
        }
        for name in omit:
            files.pop(name)
        return self.ledger.freeze_bundle(
            label=label, kind=kind, parent=parent, scope=scope or self.scope,
            files=files, provenance={'origin': 'original', 'synthetic_fixture': True},
        )

    def open_round(self, name='python-round', candidate=None, evidence_mode='synthetic'):
        self.ledger.open_round(
            name, candidate=candidate or self.candidate,
            plan=EvaluationPlan({'a': 'cat', 'b': 'dog'}, 1, 1, 0),
            generation_usage=Usage.zero(),
            metadata={'evidence_mode': evidence_mode, 'experiment_id': 'SOFTWARE-ONLY-010'},
        )

    def start_service(self, *extra_bundles, allowed_profiles=None, scope=None,
                      grant_overrides=None):
        if allowed_profiles is None:
            allowed_profiles = [
                json.loads(self.ledger.read_object(bundle))['files']['profile.json']['sha256']
                for bundle in (self.seed, self.candidate, *extra_bundles)
            ]
        config = {
            'protocol': 'physical-keyboard-rpc-config-v1',
            'backend': 'synthetic-fake',
            'scope': asdict(scope or self.scope),
            'safety': self.safety,
            'allowed_profiles': allowed_profiles,
            'ledger_root': str(self.ledger.root.resolve()),
            'initial_pose': {'xyz_m': [0, 0, .2], 'quaternion_xyzw': [0, 0, 0, 1]},
        }
        config_path = self.root / 'server-config.json'
        config_path.write_bytes(canonical(config))
        self.rpc = AuditedRpc(
            config_path.resolve(), self.root / 'action-journal', self.root / 'server',
            self.ledger, grant_overrides,
        )
        self.addCleanup(self.rpc.close)
        self.runner = TrialRunner(self.ledger, self.rpc, self.root / 'runs')
        return self.rpc

    def run_trial(self, trial='trial', *, round_id='python-round', variant='candidate',
                  case='a', **kwargs):
        return self.runner.run(round_id, trial, variant, case, 0, case + '-0', **kwargs)

    def report(self, result):
        for ref in result['after_evidence']:
            try:
                value = json.loads(self.ledger.read_object(ref))
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and value.get('protocol') == 'physical-keyboard-trial-receipt-v1':
                return value
        self.fail('No original trial report retained as ledger evidence')

    @staticmethod
    def moves(report):
        return [r for r in report['receipts'] if r['method'] == 'move_tcp']

    def assert_reaped(self, report):
        worker = report['worker']
        self.assertGreater(worker['pid'], 0)
        self.assertNotEqual(worker['pid'], self.rpc.pid)
        self.assertTrue(worker['exit']['waited'])
        self.assertIsNotNone(worker['exit']['returncode'])
        self.assertEqual(worker['exit']['cleanup_errors'], [])
        with self.assertRaises(ProcessLookupError):
            os.kill(worker['pid'], 0)

    def assert_no_dispatch(self):
        observation = self.rpc.call('observe', {})['observation']
        self.assertEqual(observation['move_calls'], 0)
        self.assertEqual(
            [method for method, _ in self.rpc.calls if method in {'move_tcp', 'wait_bounded'}], [],
        )
        with sqlite3.connect(self.root / 'action-journal' / 'actions.sqlite3') as db:
            count = db.execute(
                "SELECT count(*) FROM events WHERE kind='action_dispatching'"
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_real_press_release_uses_frozen_profile_and_fresh_observations(self):
        self.start_service()
        result = self.run_trial()
        report = self.report(result)
        self.assertEqual(result['status'], 'uncertain')
        self.assertIsNone(result['observed_text'])
        self.assertTrue(report['motion_completed'])
        self.assertFalse(report['text_verified'])
        self.assertEqual(report['execution_kind'], 'bounded-python-v1')
        self.assertEqual(report['program_file'], 'execution.json')
        self.assertEqual(report['worker']['done']['status'], 'completed')
        self.assertEqual(report['skill_result'], {'status': 'success', 'observed_text': 'cat'})
        moves = self.moves(report)
        self.assertEqual([r['result']['status'] for r in moves], ['ok'] * 3)
        self.assertEqual([r['arguments']['xyz_m'] for r in moves],
                         [PROFILE['approach'], PROFILE['press'], PROFILE['release']])
        self.assertEqual(len({r['arguments']['observation_id'] for r in moves}), 3)
        observation = self.rpc.call('observe', {})['observation']
        self.assertEqual(observation['move_calls'], 3)
        self.assertEqual(observation['pose']['xyz_m'], PROFILE['release'])
        self.assert_reaped(report)
        manifest = json.loads(self.ledger.read_object(self.candidate))
        grant = next(args for method, args in self.rpc.calls if method == 'authorize_trial')
        self.assertEqual(grant['program_sha256'], manifest['files']['execution.json']['sha256'])
        self.assertEqual(grant['profile_id'], manifest['files']['profile.json']['sha256'])
        self.assertEqual(report['start_event_sha256'], grant['start_event_sha256'])
        for item in manifest['files'].values():
            self.assertIn(item['sha256'], result['after_evidence'])
        self.assertIn(self.candidate, result['after_evidence'])
        self.assertEqual(result['usage'], asdict(Usage.zero()))

    def test_only_independent_host_evaluator_can_record_observed_text(self):
        self.start_service()
        evaluated = []

        def evaluate(report):
            self.assertTrue(report['motion_completed'])
            evaluated.append(report['worker']['pid'])
            return {'observed_text': 'dog', 'evidence': b'Independent synthetic literal: dog'}

        result = self.run_trial(evaluator=evaluate)
        report = self.report(result)
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['observed_text'], 'dog')
        self.assertEqual(report['skill_result']['observed_text'], 'cat')
        self.assertEqual(evaluated, [report['worker']['pid']])
        self.assertIn(digest(b'Independent synthetic literal: dog'), result['after_evidence'])

    def test_feedback_while_and_branch_read_actual_server_pose(self):
        main = '''
        def main(robot, profile):
            receipt = robot.observe()
            x = receipt['observation']['pose']['xyz_m'][0]
            count = 0
            while x < profile['target_x'] - 0.0001:
                robot.move_tcp([x + profile['step'], 0, 0.2], profile['q'])
                receipt = robot.observe()
                x = receipt['observation']['pose']['xyz_m'][0]
                count = count + 1
            if x >= profile['target_x'] - 0.0001:
                robot.move_tcp([x, 0, 0.21], profile['q'])
                branch = 'release'
            else:
                branch = 'not_reached'
            return {'x': x, 'iterations': count, 'branch': branch}
        '''
        candidate = self.bundle('feedback', parent=self.seed, main=main)
        self.open_round('feedback-round', candidate)
        self.start_service(candidate)
        result = self.run_trial(round_id='feedback-round')
        report = self.report(result)
        self.assertTrue(report['motion_completed'])
        self.assertEqual(report['skill_result'], {'x': .03, 'iterations': 3, 'branch': 'release'})
        xs = [r['result']['pose']['xyz_m'][0] for r in self.moves(report)]
        self.assertEqual(xs, [.01, .02, .03, .03])
        self.assertEqual(self.rpc.call('observe', {})['observation']['pose']['xyz_m'], [.03, 0, .21])

    def test_rejected_press_retains_receipt_and_finally_releases_without_evaluation(self):
        main = '''
        def main(robot, profile):
            approach(robot, profile)
            caught = False
            try:
                robot.move_tcp([0.4, 0, 0.185], profile['q'])
            except ToolError:
                caught = True
            finally:
                robot.move_tcp(profile['release'], profile['q'])
            return {'caught': caught, 'status': 'success'}
        '''
        candidate = self.bundle('finally-release', parent=self.seed, main=main)
        self.open_round('finally-round', candidate)
        self.start_service(candidate)

        def forbidden(_):
            self.fail('A rejected press must never reach the text evaluator')

        result = self.run_trial(round_id='finally-round', evaluator=forbidden)
        report = self.report(result)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(report['motion_completed'])
        self.assertFalse(report['text_verified'])
        self.assertEqual(report['skill_result'], {'caught': True, 'status': 'success'})
        moves = self.moves(report)
        self.assertEqual([r['result']['status'] for r in moves], ['ok', 'rejected', 'ok'])
        self.assertEqual(moves[1]['result']['reason'], 'workspace_bounds')
        self.assertFalse(moves[1]['result']['executed'])
        self.assertEqual(moves[-1]['result']['pose']['xyz_m'], PROFILE['release'])
        self.assertEqual(report['worker']['done']['status'], 'blocked')
        self.assertEqual(report['worker']['done']['tool_failure'], 'rejected')
        self.assert_reaped(report)

    def test_compute_budget_preserves_earlier_action_then_stops_and_reaps(self):
        candidate = self.bundle('loop-budget', parent=self.seed, main='''
        def main(robot, profile):
            approach(robot, profile)
            i = 0
            while True:
                i = i + 1
        ''')
        self.open_round('loop-round', candidate)
        self.start_service(candidate)
        started = time.monotonic()
        result = self.run_trial(round_id='loop-round')
        self.assertLess(time.monotonic() - started, 5.)
        report = self.report(result)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(report['worker']['done']['status'], 'error')
        self.assertEqual(len(self.moves(report)), 1)
        self.assertEqual(self.moves(report)[0]['result']['status'], 'ok')
        self.assertIn('stop', [r['method'] for r in report['receipts']])
        self.assertTrue(self.rpc.call('observe', {})['observation']['stopped'])
        self.assert_reaped(report)

    def test_tool_budget_retains_all_64_calls_and_host_stop(self):
        candidate = self.bundle('tool-budget', parent=self.seed, main='''
        def main(robot, profile):
            while True:
                robot.wait_bounded(0)
        ''')
        self.open_round('tool-budget-round', candidate)
        self.start_service(candidate)
        result = self.run_trial(round_id='tool-budget-round')
        report = self.report(result)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(report['worker']['done']['status'], 'error')
        waits = [r for r in report['receipts'] if r['method'] == 'wait_bounded']
        self.assertEqual(len(waits), 64)
        self.assertTrue(all(r['result']['status'] == 'ok' for r in waits))
        self.assertEqual(len({r['arguments']['request_id'] for r in waits}), 64)
        self.assertIn('stop', [r['method'] for r in report['receipts']])
        self.assertTrue(self.rpc.call('observe', {})['observation']['stopped'])
        self.assert_reaped(report)

    def test_unreachable_bad_helper_is_validated_before_any_tool_call(self):
        candidate = self.bundle('bad-helper', parent=self.seed, helper='''
        def approach(robot, profile):
            return robot.move_tcp(profile['approach'], profile['q'])
        def unused():
            import os
        ''')
        self.open_round('bad-helper-round', candidate)
        self.start_service(candidate)
        result = self.run_trial(round_id='bad-helper-round')
        report = self.report(result)
        self.assertEqual(result['status'], 'uncertain')
        self.assertFalse(report['motion_completed'])
        self.assertEqual(report['worker']['done']['status'], 'error')
        self.assertFalse(any(
            m['message'].get('type') == 'tool_call'
            for m in report['messages'] if m['direction'] == 'worker_to_host'
        ))
        self.assertEqual(self.moves(report), [])
        self.assertEqual(self.rpc.call('observe', {})['observation']['move_calls'], 0)
        self.assert_reaped(report)

    def test_missing_dependency_is_rejected_before_authority_or_worker(self):
        candidate = self.bundle('missing-helper', parent=self.seed, omit=['helpers.py'])
        self.open_round('missing-round', candidate)
        self.start_service(candidate)
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess',
                   side_effect=AssertionError('Missing frozen helper spawned a worker')):
            with self.assertRaises((ValueError, LedgerError)):
                self.run_trial(round_id='missing-round')
        self.assertEqual(self.rpc.calls, [])
        self.assert_no_dispatch()

    def test_tampered_dependency_is_rejected_before_any_rpc_or_worker(self):
        self.start_service()
        manifest = json.loads(self.ledger.read_object(self.candidate))
        helper = manifest['files']['helpers.py']['sha256']
        (self.ledger.objects / helper).write_bytes(b'altered frozen helper')
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess',
                   side_effect=AssertionError('Corrupted helper spawned a worker')):
            with self.assertRaises(LedgerError):
                self.run_trial()
        self.assertEqual(self.rpc.calls, [])
        self.assert_no_dispatch()

    def test_worker_rechecks_dependency_after_host_materialization(self):
        self.start_service()

        def tampering_worker(command, stderr_path):
            manifest = Path(command[command.index('--manifest') + 1])
            (manifest.parent / 'helpers.py').write_bytes(b'def approach():\n    return 0\n')
            return JsonProcess(command, stderr_path)

        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess', tampering_worker):
            result = self.run_trial()
        report = self.report(result)
        self.assertEqual(result['status'], 'uncertain')
        self.assertFalse(report['motion_completed'])
        self.assertEqual(report['worker']['done']['status'], 'error')
        self.assertIn('integrity mismatch', report['worker']['done']['error'].lower())
        self.assertFalse(any(
            m['message'].get('type') == 'tool_call'
            for m in report['messages'] if m['direction'] == 'worker_to_host'
        ))
        self.assertEqual(self.moves(report), [])
        self.assertEqual(self.rpc.call('observe', {})['observation']['move_calls'], 0)
        self.assert_reaped(report)

    def test_non_dict_profile_cannot_execute_python(self):
        candidate = self.bundle('bad-profile', parent=self.seed, profile=['not', 'a', 'dict'])
        self.open_round('bad-profile-round', candidate)
        self.start_service(candidate)
        with self.assertRaises((ValueError, LedgerError)):
            self.run_trial(round_id='bad-profile-round')
        self.assertEqual(self.rpc.calls, [])
        self.assert_no_dispatch()

    def test_descriptor_cannot_add_authority_or_runtime_fields(self):
        descriptor = {**DESCRIPTOR, 'profile_id': 'attacker-selected-profile'}
        candidate = self.bundle('extra-descriptor-field', parent=self.seed, descriptor=descriptor)
        self.open_round('extra-descriptor-round', candidate)
        self.start_service(candidate)
        with self.assertRaises((ValueError, LedgerError)):
            self.run_trial(round_id='extra-descriptor-round')
        self.assertEqual(self.rpc.calls, [])
        self.assert_no_dispatch()

    def test_parent_and_child_use_their_own_frozen_helpers_and_fresh_workers(self):
        candidate = self.bundle('helper-change', parent=self.seed, helper='''
        def approach(robot, profile):
            return robot.move_tcp([0.02, 0, 0.2], profile['q'])
        ''')
        self.open_round('helper-round', candidate)
        self.start_service(candidate)
        parent_result = self.run_trial('parent', round_id='helper-round', variant='parent')
        child_result = self.run_trial('child', round_id='helper-round', variant='candidate')
        parent, child = self.report(parent_result), self.report(child_result)
        self.assertTrue(parent['motion_completed'])
        self.assertTrue(child['motion_completed'])
        self.assertEqual(self.moves(parent)[0]['arguments']['xyz_m'], [.01, 0, .2])
        self.assertEqual(self.moves(child)[0]['arguments']['xyz_m'], [.02, 0, .2])
        self.assertNotEqual(parent['worker']['pid'], child['worker']['pid'])
        self.assertEqual(parent['bundle_id'], self.seed)
        self.assertEqual(child['bundle_id'], candidate)
        # The descriptor may stay identical while its frozen helper changes.
        self.assertEqual(parent['program_sha256'], child['program_sha256'])
        self.assert_reaped(parent)
        self.assert_reaped(child)
        calls = list(self.rpc.calls)
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess',
                   side_effect=AssertionError('Completed trial was executed again')):
            self.assertEqual(
                self.run_trial('child', round_id='helper-round', variant='candidate'), child_result,
            )
        self.assertEqual(self.rpc.calls, calls)

    def test_ledger_start_failure_never_grants_python_authority(self):
        self.start_service()
        with patch.object(self.ledger, 'start_trial', side_effect=OSError('Synthetic disk failure')):
            with self.assertRaises(OSError):
                self.run_trial()
        self.assertEqual([method for method, _ in self.rpc.calls], ['observe'])
        self.assertIsNone(self.ledger.trial_state('python-round', 'trial'))
        self.assert_no_dispatch()

    def test_server_requires_a_real_start_for_python_descriptor(self):
        self.start_service()
        manifest = json.loads(self.ledger.read_object(self.candidate))
        # Bypass only the auditing wrapper to present an adversarial wire claim.
        result = RpcClient.call(self.rpc, 'authorize_trial', {
            'trial_id': 'invented', 'round_id': 'python-round', 'variant': 'candidate',
            'bundle_id': self.candidate, 'profile_id': manifest['files']['profile.json']['sha256'],
            'scope_key': self.scope.key, 'start_event_sha256': '0' * 64,
            'program_sha256': manifest['files']['execution.json']['sha256'],
        })
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], 'unfinished_durable_start_required')
        self.assert_no_dispatch()

    def test_server_binds_python_authorization_to_descriptor_sha_not_entry_sha(self):
        self.start_service()
        before = self.ledger.put_evidence(b'Synthetic direct grant fingerprint check')
        self.ledger.start_trial(
            'python-round', 'trial', variant='candidate', case_id='a', repetition=0,
            condition_id='a-0', before_evidence=[before],
        )
        state = self.ledger.trial_state('python-round', 'trial')
        manifest = json.loads(self.ledger.read_object(self.candidate))
        args = {
            'trial_id': 'trial', 'round_id': 'python-round', 'variant': 'candidate',
            'bundle_id': self.candidate, 'profile_id': manifest['files']['profile.json']['sha256'],
            'scope_key': self.scope.key, 'start_event_sha256': state['start_event_sha256'],
            'program_sha256': manifest['files']['skill.py']['sha256'],
        }
        rejected = self.rpc.call('authorize_trial', args)
        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['reason'], 'frozen_bundle_or_scope_mismatch')
        granted = self.rpc.call('authorize_trial', {
            **args, 'program_sha256': manifest['files']['execution.json']['sha256'],
        })
        self.assertEqual(granted['status'], 'ok')
        self.assertEqual(self.rpc.call('close_trial', {'permit_id': granted['permit_id']})['status'], 'ok')
        self.assert_no_dispatch()

    def test_server_still_rejects_wrong_scope_for_python(self):
        self.start_service(grant_overrides={'scope_key': '0' * 64})
        result = self.run_trial()
        report = self.report(result)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(report['worker'])
        self.assert_no_dispatch()

    def test_server_still_requires_an_allowed_profile_for_python(self):
        self.start_service(allowed_profiles=['0' * 64])
        result = self.run_trial()
        report = self.report(result)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(report['worker'])
        self.assert_no_dispatch()

    def test_python_bundle_cannot_switch_synthetic_runner_to_physical_mode(self):
        physical = replace(self.scope, evidence_mode='physical')
        seed = self.bundle('physical-seed', kind='seed', scope=physical)
        self.ledger.bootstrap(seed, reason='Nonexecuting physical-mode rejection fixture')
        child = self.bundle('physical-child', parent=seed, scope=physical)
        self.open_round('physical-round', child, evidence_mode='physical')
        self.start_service()
        with self.assertRaises(LedgerError):
            self.run_trial(round_id='physical-round')
        self.assertEqual(self.rpc.calls, [])
        self.assert_no_dispatch()


if __name__ == '__main__':
    unittest.main()
