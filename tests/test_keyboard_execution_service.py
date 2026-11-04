"""Persistent fault waiting/recovery with injected operator events, no hardware."""
import json
import os
from pathlib import Path
import time
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.agent.execution_service import (
    PersistentExecutionSupervisor, REVIEW_PROTOCOL, make_resume_event, make_review_event,
)
from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop


class ExecutionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.calls, self.statuses, self.sleeps = [], [], 0
        self.stop = False
        self.fault = dict(stopped_reason='unknown_motion_no_retry',
                          last_observation=dict(evidence_id='old'), physical_outcome='unknown')

    def write(self, value, mode=0o600, *, path=None):
        raw = json.dumps(value).encode()
        path = self.supervisor.resume_file if path is None else path
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)
        path.chmod(mode)

    def write_review(self, value, mode=0o600):
        self.write(value, mode, path=self.supervisor.review_file)

    def current_fault(self):
        return next(s['recovery']['fault_id'] for s in reversed(self.statuses) if 'recovery' in s)

    def create(self, *, close=None, review=None, restore=None, observe=None, sleep=None,
               observation_home=False):
        def callback(name, result):
            def called(*args):
                self.calls.append((name, args))
                return result
            return called
        self.supervisor = PersistentExecutionSupervisor(output=self.root / 'service',
            close_host=close or callback('close', dict(closed=True)),
            review_fault=review or callback('review', dict(finished_by_model=True,
                saved_candidates=[{'candidate_id': 'model-authored-unverified'}])),
            restore_host=restore or callback('restore', dict(generation=1)),
            observe=observe or callback('observe', dict(evidence_id='fresh', contentItems=[])),
            status=lambda value: self.statuses.append(value),
            stop_requested=lambda: self.stop, sleep_fn=sleep or (lambda seconds: None),
            observation_home=observation_home)
        return self.supervisor

    def test_observation_home_requires_separate_explicit_fault_bound_confirmation(self):
        service=self.create(observation_home=True)
        fid='fault-current';started=time.time_ns()-1
        self.write(make_resume_event(fid))
        self.assertIsNone(service.read_resume(fid,started))
        self.write(dict(make_resume_event(fid,observation_home=True),allow_observation_home=1))
        self.assertIsNone(service.read_resume(fid,started))
        record=make_resume_event(fid,observation_home=True)
        self.write(record)
        self.assertEqual(service.read_resume(fid,started),record)
        self.assertIsNone(service.read_resume(fid,started))
        self.assertEqual(self.calls,[])

    def test_default_service_rejects_extra_observation_home_permission(self):
        service=self.create()
        self.write(make_resume_event('fault-current',observation_home=True))
        self.assertIsNone(service.read_resume('fault-current',0))
        self.assertEqual(self.calls,[])

    def test_observation_home_contract_is_published_and_still_waits_before_restore(self):
        def idle(seconds):
            self.sleeps+=1
            request=self.statuses[-1]['recovery']
            self.assertIn('allow_observation_home',request['required_confirmations'])
            self.assertTrue(request['observation_arm_home_requested'])
            self.assertIn('Reference pose restoration',request['initialization_side_effect'])
            self.assertEqual([c[0] for c in self.calls],['close','review'])
            if self.sleeps==1:
                self.write(make_resume_event(self.current_fault()))
            else:
                self.write(make_resume_event(self.current_fault(),observation_home=True))
        service=self.create(sleep=idle,observation_home=True)
        result=service.handle_fault(self.fault)
        self.assertEqual(self.sleeps,2)
        self.assertEqual([c[0] for c in self.calls],['close','review','restore','observe'])
        self.assertFalse(result['recovery']['original_request_replayed'])


    def test_idle_waits_without_model_or_hardware_then_restores_once_with_fresh_observation(self):
        def sleep(seconds):
            self.sleeps += 1
            self.assertEqual([x[0] for x in self.calls], ['close', 'review'])
            if self.sleeps == 3:
                self.write(make_resume_event(self.current_fault()))
        service = self.create(sleep=sleep)
        result = service.handle_fault(self.fault)
        self.assertEqual(self.sleeps, 3)
        self.assertEqual([x[0] for x in self.calls], ['close', 'review', 'restore', 'observe'])
        self.assertEqual(result['observation']['evidence_id'], 'fresh')
        self.assertFalse(result['recovery']['original_request_replayed'])
        self.assertFalse(result['recovery']['implicit_home'])
        self.assertFalse(result['recovery']['implicit_editor_reset'])
        self.assertEqual(result['recovery']['saved_candidates'][0]['candidate_id'], 'model-authored-unverified')
        self.assertEqual(self.fault['last_observation']['evidence_id'], 'old')

    def test_failed_restoration_requires_a_different_explicit_event_and_does_not_repeat_review(self):
        attempts = []
        def restore(event, fault):
            self.calls.append(('restore', (event,)))
            attempts.append(event['id'])
            if len(attempts) == 1:
                raise RuntimeError('fixture constructor failed')
            return dict(generation=2)
        def sleep(seconds):
            self.sleeps += 1
            if self.sleeps in (1, 4):
                self.write(make_resume_event(self.current_fault(), event_id='event-' + str(self.sleeps)))
        service = self.create(restore=restore, sleep=sleep)
        result = service.handle_fault(self.fault)
        self.assertEqual(attempts, ['event-1', 'event-4'])
        self.assertEqual([x[0] for x in self.calls].count('review'), 1)
        self.assertEqual([x[0] for x in self.calls].count('close'), 2)
        self.assertEqual(result['recovery']['restore_receipt'], dict(generation=2))

    def test_operator_stop_ends_wait_without_restore(self):
        def sleep(seconds):
            self.stop = True
        service = self.create(sleep=sleep)
        with self.assertRaises(OperatorStop):
            service.handle_fault(self.fault)
        self.assertEqual([x[0] for x in self.calls], ['close', 'review'])

    def test_review_or_close_failure_remains_recorded_and_can_wait_for_restoration(self):
        def close():
            self.calls.append(('close', ()))
            raise RuntimeError('fixture cleanup failure')
        def review(fault, output):
            self.calls.append(('review', ()))
            self.assertIn('fixture cleanup failure', fault['control_cleanup']['error'])
            raise RuntimeError('fixture model failure')
        def sleep(seconds):
            self.write(make_resume_event(self.current_fault()))
        service = self.create(close=close, review=review, sleep=sleep)
        self.assertEqual(service.handle_fault(self.fault)['observation']['evidence_id'], 'fresh')
        log=(service.output / 'learning-events.jsonl').read_text()
        self.assertIn('fault_learning_failed', log)
        self.assertIn('fixture model failure', log)

    def test_stale_or_malformed_events_cannot_restore_or_reuse_prior_fault_authorization(self):
        service = self.create()
        fid = 'fault-current'
        import time
        start = time.time_ns()
        invalid = [make_resume_event('fault-old'), make_resume_event(fid, utc_ns=start-1),
                   dict(make_resume_event(fid), onsite_takeover_ready=False),
                   dict(make_resume_event(fid), extra=True)]
        for event in invalid:
            self.write(event)
            self.assertIsNone(service.read_resume(fid, start))
        valid = make_resume_event(fid)
        self.write(valid)
        self.assertEqual(service.read_resume(fid, start), valid)
        self.assertIsNone(service.read_resume(fid, start))
        self.write(make_resume_event(fid, event_id=valid['id']))
        self.assertIsNone(service.read_resume(fid, start))
        self.assertEqual(self.calls, [])

    def test_untrusted_file_types_and_permissions_are_rejected(self):
        service = self.create()
        event = make_resume_event('fault-current')
        self.write(event, 0o644)
        self.assertIsNone(service.read_resume('fault-current', 1))
        service.resume_file.unlink()
        target = self.root / 'other.json'
        target.write_text(json.dumps(event))
        service.resume_file.symlink_to(target)
        self.assertIsNone(service.read_resume('fault-current', 1))
        service.resume_file.unlink()
        service.resume_file.mkdir()
        self.assertIsNone(service.read_resume('fault-current', 1))
        self.assertEqual(self.calls, [])

    def test_stale_recovery_observation_is_not_accepted_or_retried(self):
        def sleep(seconds):
            self.sleeps += 1
            if self.sleeps == 1:
                self.write(make_resume_event(self.current_fault()))
            else:
                self.stop = True
        service = self.create(observe=lambda: dict(evidence_id='old', contentItems=[]), sleep=sleep)
        with self.assertRaises(OperatorStop):
            service.handle_fault(self.fault)
        self.assertEqual([x[0] for x in self.calls].count('restore'), 1)
        self.assertIn('execution_restore_failed', (service.output / 'learning-events.jsonl').read_text())

    def test_explicit_review_once_returns_to_suspended_without_restore_or_observe(self):
        def sleep(seconds):
            self.sleeps += 1
            expected_reviews = 1 if self.sleeps == 1 else 2
            self.assertEqual([item[0] for item in self.calls], ['close'] + ['review'] * expected_reviews)
            if self.sleeps == 1:
                self.write_review(make_review_event(self.current_fault(), reason='New registered execution evidence'))
            if self.sleeps == 4:
                self.stop = True
        service = self.create(sleep=sleep)
        with self.assertRaises(OperatorStop):
            service.handle_fault(self.fault)
        reviews = [item[1] for item in self.calls if item[0] == 'review']
        self.assertEqual(len(reviews), 2)
        self.assertEqual(reviews[0][1].name, 'review')
        self.assertEqual(reviews[1][1].name, 'review-0001')
        self.assertTrue(reviews[1][1].is_dir())
        self.assertEqual(reviews[0][1].parent, reviews[1][1].parent)
        self.assertTrue(reviews[1][0]['further_dispatch_disabled'])
        self.assertEqual(reviews[1][0]['last_observation']['evidence_id'], 'old')
        self.assertEqual(reviews[1][0]['read_only_review_event']['reason'], 'New registered execution evidence')
        self.assertNotIn('read_only_review_event', reviews[0][0])
        self.assertNotIn('read_only_review_event', self.fault)
        self.assertEqual(self.statuses[-1]['phase'], 'execution_suspended')
        events = [json.loads(line) for line in (service.output / 'learning-events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(item['event'] == 'fault_review_requested' for item in events), 1)
        self.assertEqual(sum(item['event'] == 'fault_review_completed' for item in events), 1)

    def test_review_can_precede_independent_resume_and_preserves_both_candidate_reports(self):
        review_calls = []
        def review(fault, output):
            review_calls.append(output)
            self.calls.append(('review', (fault, output)))
            return dict(finished_by_model=True, saved_candidates=[{'candidate_id': 'candidate-' + str(len(review_calls))}])
        def sleep(seconds):
            self.sleeps += 1
            if self.sleeps == 1:
                self.write_review(make_review_event(self.current_fault(), reason='New diagnostics'))
            if self.sleeps == 3:
                self.write(make_resume_event(self.current_fault()))
        service = self.create(review=review, sleep=sleep)
        result = service.handle_fault(self.fault)
        self.assertEqual([item[0] for item in self.calls], ['close', 'review', 'review', 'restore', 'observe'])
        self.assertEqual([item['candidate_id'] for item in result['recovery']['saved_candidates']],
                         ['candidate-1', 'candidate-2'])
        self.assertEqual(result['observation']['evidence_id'], 'fresh')

    def test_failed_requested_review_is_audited_without_automatic_model_loop(self):
        def review(fault, output):
            self.calls.append(('review', (fault, output)))
            if 'read_only_review_event' in fault:
                raise RuntimeError('fixture requested review failure')
            return dict(finished_by_model=True, saved_candidates=[])
        def sleep(seconds):
            self.sleeps += 1
            if self.sleeps == 1:
                self.write_review(make_review_event(self.current_fault(), reason='New fault evidence'))
            if self.sleeps == 4:
                self.stop = True
        service = self.create(review=review, sleep=sleep)
        with self.assertRaises(OperatorStop):
            service.handle_fault(self.fault)
        self.assertEqual([item[0] for item in self.calls], ['close', 'review', 'review'])
        events = [json.loads(line) for line in (service.output / 'learning-events.jsonl').read_text().splitlines()]
        failed = [item for item in events if item['event'] == 'fault_review_failed']
        self.assertEqual(len(failed), 1)
        self.assertFalse(failed[0]['data']['automatic_retry'])
        self.assertEqual(self.statuses[-1]['phase'], 'execution_suspended')

    def test_multiple_fresh_review_events_have_separate_inert_output_directories(self):
        def sleep(seconds):
            self.sleeps += 1
            if self.sleeps in (1, 3):
                self.write_review(make_review_event(self.current_fault(), reason='Additional source ' + str(self.sleeps),
                                                   event_id='review-' + str(self.sleeps)))
            if self.sleeps == 5:
                self.stop = True
        service = self.create(sleep=sleep)
        with self.assertRaises(OperatorStop):
            service.handle_fault(self.fault)
        self.assertEqual([item[0] for item in self.calls], ['close', 'review', 'review', 'review'])
        self.assertEqual([item[1][1].name for item in self.calls if item[0] == 'review'],
                         ['review', 'review-0001', 'review-0002'])

    def test_failed_restore_can_be_followed_by_readonly_review_without_second_restore(self):
        def restore(event, fault):
            self.calls.append(('restore', (event, fault)))
            raise RuntimeError('fixture restore failure')
        def sleep(seconds):
            self.sleeps += 1
            if self.sleeps == 1:
                self.write(make_resume_event(self.current_fault()))
            if self.sleeps == 2:
                self.assertEqual(self.statuses[-1]['phase'], 'execution_suspended')
                self.assertIn('review_request', self.statuses[-1])
                self.write_review(make_review_event(self.current_fault(), reason='Review after failed restoration'))
            if self.sleeps == 4:
                self.stop = True
        service = self.create(restore=restore, sleep=sleep)
        with self.assertRaises(OperatorStop):
            service.handle_fault(self.fault)
        self.assertEqual([item[0] for item in self.calls], ['close', 'review', 'restore', 'close', 'review'])
        self.assertEqual(self.statuses[-1]['phase'], 'execution_suspended')

    def test_review_requires_fresh_fault_bound_exact_schema_and_reason(self):
        service = self.create()
        start, fid = time.time_ns(), 'fault-current'
        event = make_review_event(fid, reason='New evidence')
        invalid = [make_review_event('fault-old', reason='Other fault'),
            dict(event, utc_ns=start - 1), dict(event, utc_ns=time.time_ns() + 10_000_000_000),
            dict(event, utc_ns=True), dict(event, id=[]), dict(event, id='../unsafe'),
            dict(event, reason=''), dict(event, reason=' '), dict(event, reason='x' * 2049),
            dict(event, reason=[]), dict(event, reason='\ud800'), dict(event, source='model'),
            dict(event, allow_sdk_reinitialize=True), make_resume_event(fid)]
        for value in invalid:
            with self.subTest(value=value):
                self.write_review(value)
                self.assertIsNone(service.read_review(fid, start))
        self.assertEqual(self.calls, [])

    def test_review_payload_and_event_id_are_read_once_and_errors_are_not_relogged(self):
        service = self.create()
        start, fid = time.time_ns(), 'fault-current'
        value = make_review_event(fid, reason='A source became available')
        self.write_review(value)
        self.assertEqual(service.read_review(fid, start), value)
        self.assertIsNone(service.read_review(fid, start))
        self.write_review(dict(value, reason='Changed payload with reused nonce'))
        self.assertIsNone(service.read_review(fid, start))
        before = (service.output / 'learning-events.jsonl').read_bytes()
        for unused in range(5):
            self.assertIsNone(service.read_review(fid, start))
        self.assertEqual(before, (service.output / 'learning-events.jsonl').read_bytes())
        self.write_review(make_review_event(fid, reason='Independent new event'))
        self.assertIsNotNone(service.read_review(fid, start))

    def test_review_rejects_permissions_links_nonregular_owner_and_oversized_payload(self):
        service = self.create()
        value = make_review_event('fault-current', reason='New evidence')
        self.write_review(value, 0o644)
        self.assertIsNone(service.read_review('fault-current', 1))
        service.review_file.unlink()
        target = self.root / 'other-review.json'
        target.write_text(json.dumps(value))
        service.review_file.symlink_to(target)
        self.assertIsNone(service.read_review('fault-current', 1))
        service.review_file.unlink()
        service.review_file.mkdir()
        self.assertIsNone(service.read_review('fault-current', 1))
        service.review_file.rmdir()
        os.mkfifo(service.review_file, 0o600)
        self.assertIsNone(service.read_review('fault-current', 1))
        service.review_file.unlink()
        self.write_review(make_review_event('fault-current', reason='Owner mismatch'))
        with patch('astra_robotics_agent.physical_keyboard.agent.execution_service.os.getuid', return_value=os.getuid() + 1):
            self.assertIsNone(service.read_review('fault-current', 1))
        service.review_file.write_bytes(b'x' * 16385)
        self.assertIsNone(service.read_review('fault-current', 1))
        self.assertEqual(self.calls, [])

    def test_review_rejects_duplicate_json_invalid_encoding_and_file_changes(self):
        service = self.create()
        value = make_review_event('fault-current', reason='New evidence')
        self.write_review(value)
        for raw in (b'{"id":"one","id":"two"}', b'\xffnot-json', b'[' * 2000):
            service.review_file.write_bytes(raw)
            self.assertIsNone(service.read_review('fault-current', 1))
        self.write_review(value)
        original_read = os.read
        def changed(fd, maximum):
            raw = original_read(fd, maximum)
            with service.review_file.open('ab') as stream:
                stream.write(b' ')
            return raw
        with patch('astra_robotics_agent.physical_keyboard.agent.execution_service.os.read', side_effect=changed):
            self.assertIsNone(service.read_review('fault-current', 1))
        self.assertIn('unchanged bounded owner-only', (service.output / 'learning-events.jsonl').read_text())

    def test_review_and_resume_event_contracts_cannot_authorize_each_other(self):
        service = self.create()
        fid, started = 'fault-current', time.time_ns()
        review = make_review_event(fid, reason='Read only')
        self.write(review)
        self.assertIsNone(service.read_resume(fid, started))
        self.write_review(make_resume_event(fid))
        self.assertIsNone(service.read_review(fid, started))
        self.write_review(review)
        self.assertEqual(service.read_review(fid, started), review)
        self.assertEqual(self.calls, [])



if __name__ == '__main__':
    unittest.main()
