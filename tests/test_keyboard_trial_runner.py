"""Synthetic host-coordination tests, with actual fresh declarative workers.

The RecordingRpc fixture intentionally is not a device/simulator. Full server
integration is separately exercised by the EXP-009 smoke and RPC tests.
"""

import fcntl
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.contracts import EvaluationPlan, Scope, Usage, canonical, digest
from astra_robotics_agent.physical_keyboard.ledger import Ledger, LedgerError
from astra_robotics_agent.physical_keyboard.rpc import RpcClient, default_safety
from astra_robotics_agent.physical_keyboard.trial_runner import CoordinatorBusy, RecoveryRequired, TrialRunner
from astra_robotics_agent.physical_keyboard.transport import JsonProcess


def program(actions=None):
    if actions is None:
        actions = [dict(method='move_tcp', arguments=dict(xyz_m=[0.01, 0, .2], quaternion_xyzw=[0, 0, 0, 1]))]
    return canonical(dict(protocol='physical-keyboard-action-plan-v1', actions=actions))


class RecordingRpc:
    """Host test double; records authority order, never accesses real hardware."""

    pid = None

    def __init__(self, ledger):
        self.ledger = ledger
        self.calls = []
        self.move_receipt = {'status': 'ok', 'evidence_mode': 'synthetic'}
        self.move_error = None
        self.observe_error = None
        self.authorize_error = None
        self.closed_permits = []

    def call(self, method, arguments, timeout_s=5):
        self.calls.append((method, dict(arguments)))
        if method == 'observe':
            if self.observe_error:
                raise self.observe_error
            return {'status': 'ok', 'observation': {'observation_id': 'observation-' + str(len(self.calls)),
                                                   'evidence_mode': 'synthetic'}}
        if method == 'authorize_trial':
            if self.authorize_error:
                raise self.authorize_error
            state = self.ledger.trial_state(arguments['round_id'], arguments['trial_id'])
            if state is None or state['result'] is not None or state['start_event_sha256'] != arguments['start_event_sha256']:
                raise AssertionError('Authorization preceded a durable unfinished start')
            return {'status': 'ok', 'permit_id': 'permit-1'}
        if method == 'move_tcp':
            if self.move_error:
                raise self.move_error
            return dict(self.move_receipt)
        if method == 'close_trial':
            self.closed_permits.append(arguments['permit_id'])
            return {'status': 'ok'}
        if method in {'wait_bounded', 'stop'}:
            return {'status': 'ok'}
        raise AssertionError('Unexpected RPC method: ' + method)


class TrialRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / 'ledger')
        self.addCleanup(self.ledger.close)
        self.scope = Scope('synthetic', 'test-device', 'test-layout',
                           *[digest(value.encode()) for value in ['config', 'protocol', 'evaluator', 'goal']],
                           digest(canonical(default_safety())))
        self.seed = self.bundle('seed', kind='seed')
        self.ledger.bootstrap(self.seed, reason='Synthetic test seed, not a learned keyboard skill')
        self.candidate = self.bundle('candidate', parent=self.seed)
        self.open_round()
        self.rpc = RecordingRpc(self.ledger)
        self.runner = TrialRunner(self.ledger, self.rpc, self.root / 'runs')

    def bundle(self, label, *, parent=None, kind='skill_program', content=None, scope=None):
        return self.ledger.freeze_bundle(label=label, kind=kind, scope=scope or self.scope,
                    files={'actions.json': program() if content is None else content,
                           'profile.json': canonical({'evidence_mode': 'synthetic', 'label': label})},
                    provenance={'origin': 'original', 'fixture': True}, parent=parent)

    def open_round(self, name='software-rpc-round', candidate=None):
        self.ledger.open_round(name, candidate=candidate or self.candidate,
                plan=EvaluationPlan({'a': 'cat', 'b': 'dog'}, 1, 1, 0), generation_usage=Usage.zero(),
                metadata={'evidence_mode': 'synthetic', 'experiment_id': 'software-only'})

    def run_trial(self, trial='trial-one', variant='candidate', case='a', **kwargs):
        return self.runner.run('software-rpc-round', trial, variant, case, 0, case + '-0', **kwargs)

    def report(self, result):
        for ref in result['after_evidence']:
            try:
                value = json.loads(self.ledger.read_object(ref))
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and value.get('protocol') == 'physical-keyboard-trial-receipt-v1':
                return value
        self.fail('No raw coordinator report stored in ledger evidence')

    def test_motion_completed_is_uncertain_without_text_evaluator(self):
        result = self.run_trial()
        self.assertEqual(result['status'], 'uncertain')
        self.assertIsNone(result['observed_text'])
        report = self.report(result)
        self.assertTrue(report['motion_completed'])
        self.assertFalse(report['text_verified'])
        self.assertGreater(report['worker']['pid'], 0)
        self.assertEqual(report['worker']['exit']['returncode'], 0)
        self.assertTrue(report['worker']['exit']['waited'])
        self.assertEqual([m for m, _ in self.rpc.calls], ['observe', 'authorize_trial', 'move_tcp', 'close_trial'])
        self.assertEqual(result['usage'], {'input_tokens': 0, 'cached_input_tokens': 0,
                                          'output_tokens': 0, 'reasoning_output_tokens': 0})
        directory = Path(report['artifact_directory'])
        audit = [json.loads(line) for line in (directory / 'audit.jsonl').read_bytes().splitlines()]
        self.assertEqual([e['sequence'] for e in audit], list(range(1, len(audit) + 1)))
        self.assertEqual(report['audit_events'], len(audit))
        kinds = [e['kind'] for e in audit]
        self.assertLess(kinds.index('trial_started'), kinds.index('worker_started'))
        self.assertEqual(json.loads((directory / 'worker-start.json').read_bytes())['pid'], report['worker']['pid'])
        self.assertEqual(json.loads((directory / 'trial-start.json').read_bytes())['start_event_sha256'], report['start_event_sha256'])
        self.assertIn(digest((directory / 'audit.jsonl').read_bytes()), result['after_evidence'])

    def test_separate_evaluator_supplies_literal_text_and_raw_evidence(self):
        seen = []
        def evaluate(report):
            seen.append(report['motion_completed'])
            return {'observed_text': 'cat', 'evidence': b'Synthetic evaluator observation: cat'}
        result = self.run_trial(evaluator=evaluate)
        self.assertEqual(seen, [True])
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['observed_text'], 'cat')
        self.assertIn(digest(b'Synthetic evaluator observation: cat'), result['after_evidence'])

    def test_bounds_rejection_stops_worker_without_evaluator(self):
        self.rpc.move_receipt = {'status': 'rejected', 'reason': 'outside_workspace'}
        def forbidden(_):
            self.fail('Evaluator must not be called for blocked movement')
        result = self.run_trial(evaluator=forbidden)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(self.report(result)['motion_completed'])
        self.assertEqual(self.report(result)['worker']['done']['status'], 'blocked')
        self.assertEqual(self.rpc.closed_permits, ['permit-1'])

    def test_duplicate_completed_trial_does_not_spawn_or_call_rpc(self):
        first = self.run_trial()
        calls = list(self.rpc.calls)
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess', side_effect=AssertionError('rerun')):
            self.assertEqual(self.run_trial(), first)
        self.assertEqual(self.rpc.calls, calls)
        with self.assertRaises(LedgerError):
            self.run_trial(variant='parent')
        self.assertEqual(self.rpc.calls, calls)

    def test_bool_repetition_cannot_match_a_completed_zero_slot(self):
        self.run_trial()
        calls = list(self.rpc.calls)
        with self.assertRaises(ValueError):
            self.runner.run('software-rpc-round', 'trial-one', 'candidate', 'a', False, 'a-0')
        self.assertEqual(self.rpc.calls, calls)

    def start_pending(self, trial='pending'):
        ref = self.ledger.put_evidence(b'Synthetic interrupted start')
        self.ledger.start_trial('software-rpc-round', trial, variant='candidate', case_id='a',
                               repetition=0, condition_id='a-0', before_evidence=[ref])

    def test_inflight_same_trial_never_reexecutes_after_coordinator_restart(self):
        self.start_pending()
        with self.assertRaises(RecoveryRequired):
            self.run_trial(trial='pending')
        self.assertEqual(self.rpc.calls, [])

    def test_inflight_other_id_cannot_bypass_recovery(self):
        self.start_pending()
        with self.assertRaises(RecoveryRequired):
            self.run_trial(trial='different-id', case='b')
        self.assertEqual(self.rpc.calls, [])

    def test_explicit_abort_is_nonexecuting_and_preserves_unknown_cost(self):
        self.start_pending()
        result = self.runner.abort_pending('software-rpc-round', 'pending', reason='Manual recovery fixture',
                                           evidence=b'Unknown execution history retained')
        self.assertEqual(result['status'], 'aborted')
        self.assertTrue(all(v is None for v in result['usage'].values()))
        self.assertEqual(self.rpc.calls, [])
        with self.assertRaises(RecoveryRequired):
            self.runner.abort_pending('software-rpc-round', 'pending', reason='Duplicate abort', evidence=b'proof')

    def test_evaluator_exception_and_malformed_output_are_uncertain(self):
        def broken(_):
            raise RuntimeError('Injected evaluator unavailable')
        result = self.run_trial(evaluator=broken)
        self.assertEqual(result['status'], 'uncertain')
        self.assertIn('evaluator', [e['stage'] for e in self.report(result)['errors']])
        malformed = self.run_trial(trial='malformed-evaluator', case='b',
                                   evaluator=lambda _: {'observed_text': True, 'evidence': b'bad'})
        self.assertEqual(malformed['status'], 'uncertain')
        self.assertIsNone(malformed['observed_text'])

    def test_broken_frozen_program_fails_before_any_worker_action(self):
        candidate = self.bundle('invalid-program', parent=self.seed, content=b'{bad json')
        self.open_round('invalid-round', candidate)
        result = self.runner.run('invalid-round', 'invalid', 'candidate', 'a', 0, 'a-0')
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual([m for m, _ in self.rpc.calls], ['observe', 'authorize_trial', 'close_trial'])
        self.assertEqual(self.report(result)['worker']['exit']['returncode'], 2)

    def test_object_corruption_cannot_dispatch(self):
        bundle = json.loads(self.ledger.read_object(self.candidate))
        ref = bundle['files']['actions.json']['sha256']
        (self.ledger.objects / ref).write_bytes(b'changed after freeze')
        with self.assertRaises(LedgerError):
            self.run_trial()
        self.assertEqual(self.rpc.calls, [])

    def test_ledger_start_failure_grants_no_permission_and_saves_error(self):
        with patch.object(self.ledger, 'start_trial', side_effect=RuntimeError('Injected disk failure')):
            with self.assertRaises(RuntimeError):
                self.run_trial()
        self.assertEqual([m for m, _ in self.rpc.calls], ['observe'])
        self.assertIsNone(self.ledger.trial_state('software-rpc-round', 'trial-one'))
        errors = list((self.root / 'runs').glob('*/start-error.json'))
        self.assertEqual(len(errors), 1)
        self.assertEqual(json.loads(errors[0].read_bytes())['errors'][-1]['stage'], 'ledger_start')

    def test_ledger_finish_failure_retains_inflight_and_raw_report(self):
        with patch.object(self.ledger, 'finish_trial', side_effect=RuntimeError('Injected finish disk failure')):
            with self.assertRaises(RuntimeError):
                self.run_trial()
        self.assertIsNone(self.ledger.trial_state('software-rpc-round', 'trial-one')['result'])
        reports = list((self.root / 'runs').glob('*/run-report.json'))
        self.assertEqual(len(reports), 1)
        self.assertTrue(json.loads(reports[0].read_bytes())['motion_completed'])
        count = len(self.rpc.calls)
        with self.assertRaises(RecoveryRequired):
            self.run_trial()
        self.assertEqual(len(self.rpc.calls), count)

    def test_rpc_exception_preserves_receipt_and_attempts_stop_then_revokes(self):
        self.rpc.move_error = TimeoutError('Injected transport timeout after dispatch')
        result = self.run_trial()
        self.assertEqual(result['status'], 'uncertain')
        methods = [m for m, _ in self.rpc.calls]
        self.assertEqual(methods[-2:], ['stop', 'close_trial'])
        report = self.report(result)
        move = next(r for r in report['receipts'] if r['method'] == 'move_tcp')
        self.assertEqual(move['error']['type'], 'TimeoutError')
        self.assertTrue(report['worker']['exit']['waited'])

    def test_unknown_receipt_remains_uncertain_and_stop_is_available(self):
        self.rpc.move_receipt = {'status': 'unknown', 'reason': 'dispatch_without_terminal_receipt'}
        result = self.run_trial()
        self.assertEqual(result['status'], 'uncertain')
        self.assertIn('stop', [m for m, _ in self.rpc.calls])

    def test_worker_launch_exception_is_recorded_and_permission_revoked(self):
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess', side_effect=OSError('Injected fork failure')):
            result = self.run_trial()
        self.assertEqual(result['status'], 'uncertain')
        self.assertIsNone(self.report(result)['worker'])
        self.assertEqual([m for m, _ in self.rpc.calls][-2:], ['stop', 'close_trial'])

    def test_interrupted_host_retains_durable_start_worker_pid_and_audit(self):
        class InterruptedWorker(JsonProcess):
            def receive(self, timeout_s):
                value = super().receive(timeout_s)
                if value.get('type') == 'tool_call':
                    raise KeyboardInterrupt('Injected coordinator interruption before forwarding')
                return value
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess', InterruptedWorker):
            with self.assertRaises(KeyboardInterrupt):
                self.run_trial()
        state = self.ledger.trial_state('software-rpc-round', 'trial-one')
        self.assertIsNone(state['result'])
        directory = next((self.root / 'runs').iterdir())
        self.assertTrue((directory / 'trial-start.json').is_file())
        self.assertGreater(json.loads((directory / 'worker-start.json').read_bytes())['pid'], 0)
        audit = [json.loads(line) for line in (directory / 'audit.jsonl').read_bytes().splitlines()]
        self.assertIn('worker_started', [e['kind'] for e in audit])
        self.assertIn('worker_exit', [e['kind'] for e in audit])
        self.assertFalse((directory / 'run-report.json').exists())
        count = len(self.rpc.calls)
        with self.assertRaises(RecoveryRequired):
            self.run_trial()
        self.assertEqual(len(self.rpc.calls), count)

    def test_silent_worker_timeout_is_bounded_and_process_reaped(self):
        def silent_worker(command, stderr_path):
            return JsonProcess([sys.executable, '-c', 'import time;time.sleep(10)'], stderr_path)
        start = time.monotonic()
        with patch('astra_robotics_agent.physical_keyboard.trial_runner.JsonProcess', silent_worker):
            result = self.run_trial(timeout_s=.2)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(result['status'], 'uncertain')
        report = self.report(result)
        self.assertTrue(report['worker']['exit']['waited'])
        self.assertTrue(report['worker']['exit']['terminate_sent'])
        self.assertIn('TimeoutError', [e['type'] for e in report['errors']])
        self.assertEqual([m for m, _ in self.rpc.calls][-2:], ['stop', 'close_trial'])

    def test_empty_or_malformed_deadline_rejected_before_rpc(self):
        for value in [0, -1, float('nan'), True, 61, '5']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_trial(timeout_s=value)
        self.assertEqual(self.rpc.calls, [])

    def test_other_artifact_root_cannot_bypass_execution_lock(self):
        other = TrialRunner(self.ledger, self.rpc, self.root / 'another-runs')
        with (self.ledger.root / 'execution.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(CoordinatorBusy):
                other.run('software-rpc-round', 'trial-two', 'candidate', 'a', 0, 'a-0')
        self.assertEqual(self.rpc.calls, [])

    def test_frozen_bundle_selection_is_bound_to_requested_variant(self):
        self.run_trial(variant='parent')
        grant = next(args for method, args in self.rpc.calls if method == 'authorize_trial')
        parent_manifest = json.loads(self.ledger.read_object(self.seed))
        self.assertEqual(grant['bundle_id'], self.seed)
        self.assertEqual(grant['profile_id'], parent_manifest['files']['profile.json']['sha256'])
        self.assertNotEqual(grant['bundle_id'], self.candidate)

    def real_rpc(self, *bundle_ids):
        profiles = [json.loads(self.ledger.read_object(bundle))['files']['profile.json']['sha256']
                    for bundle in [self.seed, self.candidate, *bundle_ids]]
        config = dict(protocol='physical-keyboard-rpc-config-v1', backend='synthetic-fake',
                      scope=asdict(self.scope), safety=default_safety(), allowed_profiles=profiles,
                      ledger_root=str(self.ledger.root.resolve()),
                      initial_pose={'xyz_m': [0, 0, .2], 'quaternion_xyzw': [0, 0, 0, 1]})
        config_path = self.root / 'server-config.json'
        config_path.write_bytes(canonical(config))
        rpc = RpcClient(config_path.resolve(), self.root / 'action-journal', self.root / 'server')
        self.addCleanup(rpc.close)
        self.runner = TrialRunner(self.ledger, rpc, self.root / 'real-process-runs')
        return rpc

    def test_real_server_worker_and_ledger_complete_transport_without_claiming_keypress(self):
        rpc = self.real_rpc()
        result = self.run_trial()
        report = self.report(result)
        self.assertEqual(result['status'], 'uncertain')
        self.assertTrue(report['motion_completed'])
        self.assertNotEqual(report['worker']['pid'], rpc.pid)
        self.assertGreater(rpc.pid, 0)
        receipt = next(r['result'] for r in report['receipts'] if r['method'] == 'move_tcp')
        self.assertEqual(receipt['status'], 'ok')
        self.assertEqual(receipt['start_event_sha256'], report['start_event_sha256'])
        self.assertEqual(receipt['bundle_id'], self.candidate)
        self.assertFalse(receipt['text_verified'])
        self.assertEqual(receipt['move_calls'], 1)
        # Duplicate whole trial returns its existing ledger result, without even
        # observing again or incrementing the backend's synthetic move count.
        self.assertEqual(self.run_trial(), result)
        self.assertEqual(rpc.call('observe', {})['observation']['move_calls'], 1)

    def test_real_server_bounds_receipt_is_preserved_with_no_dispatch(self):
        content = program([dict(method='move_tcp', arguments=dict(xyz_m=[1, 0, .2], quaternion_xyzw=[0, 0, 0, 1]))])
        candidate = self.bundle('outside', parent=self.seed, content=content)
        self.open_round('outside-round', candidate)
        rpc = self.real_rpc(candidate)
        result = self.runner.run('outside-round', 'outside-trial', 'candidate', 'a', 0, 'a-0')
        self.assertEqual(result['status'], 'blocked')
        report = self.report(result)
        receipt = next(r['result'] for r in report['receipts'] if r['method'] == 'move_tcp')
        self.assertEqual(receipt['status'], 'rejected')
        self.assertFalse(receipt['executed'])
        self.assertEqual(rpc.call('observe', {})['observation']['move_calls'], 0)
        with sqlite3.connect(self.root / 'action-journal' / 'actions.sqlite3') as connection:
            count = connection.execute("SELECT COUNT(*) FROM events WHERE kind='action_dispatching'").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
