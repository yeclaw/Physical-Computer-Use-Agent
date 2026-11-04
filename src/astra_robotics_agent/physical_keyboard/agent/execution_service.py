"""Persistent fault lifecycle. No SDK, action source repair or request replay.

The application owns the supplied close/review/restore/observe callbacks.
Unknown physical execution can resume only after a fresh, fault-bound operator
event. Waiting invokes neither the model nor the robot.

The v1 resume event retains its historical ``allow_sdk_reinitialize`` field
name for protocol compatibility. It authorizes only the explicitly supplied
restore callback; this module provides no SDK, device initialization or pose.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop
from astra_robotics_agent.physical_keyboard.records.run_recording import RunRecorder


RESUME_PROTOCOL = 'physical-keyboard-execution-resume-v1'
RESUME_FIELDS = {'protocol', 'id', 'fault_id', 'source', 'utc_ns',
                 'onsite_takeover_ready', 'physical_state_checked', 'allow_sdk_reinitialize'}
REVIEW_PROTOCOL = 'physical-keyboard-execution-review-v1'
REVIEW_FIELDS = {'protocol', 'id', 'fault_id', 'source', 'utc_ns', 'reason'}


def _unique_pairs(pairs, label='resume'):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate ' + label + ' event field')
        result[key] = value
    return result


def make_resume_event(fault_id, *, event_id=None, utc_ns=None, observation_home=False):
    """Called by a trusted operator integration after actual confirmation."""
    if (type(fault_id) is not str or not fault_id.startswith('fault-')
            or re.fullmatch(r'[A-Za-z0-9_-]{1,96}', fault_id) is None):
        raise ValueError('invalid fault ID')
    if type(observation_home) is not bool:
        raise ValueError('Observation HOME confirmation must be boolean')
    record = dict(protocol=RESUME_PROTOCOL, id=event_id or uuid.uuid4().hex,
        fault_id=fault_id, source='operator', utc_ns=utc_ns or time.time_ns(),
        onsite_takeover_ready=True, physical_state_checked=True, allow_sdk_reinitialize=True)
    if observation_home:
        record['allow_observation_home'] = True
    return record


def _review_reason(value):
    if type(value) is not str or not value.strip() or len(value) > 2048:
        raise ValueError('review reason must be a nonempty string of at most 2048 characters')
    value.encode('utf-8')
    return value


def make_review_event(fault_id, *, reason, event_id=None, utc_ns=None):
    """Request another inert review; this conveys no restoration permission."""
    if (type(fault_id) is not str or not fault_id.startswith('fault-')
            or re.fullmatch(r'[A-Za-z0-9_-]{1,96}', fault_id) is None):
        raise ValueError('invalid fault ID')
    identifier = uuid.uuid4().hex if event_id is None else event_id
    timestamp = time.time_ns() if utc_ns is None else utc_ns
    if (type(identifier) is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identifier) is None
            or type(timestamp) is not int or timestamp < 0):
        raise ValueError('bounded review event ID and nonnegative UTC nanoseconds required')
    return dict(protocol=REVIEW_PROTOCOL, id=identifier, fault_id=fault_id,
                source='operator', utc_ns=timestamp, reason=_review_reason(reason))


class PersistentExecutionSupervisor:
    def __init__(self, *, output, close_host, review_fault, restore_host, observe,
                 status=None, stop_requested=None, sleep_fn=time.sleep, poll_seconds=1.0,
                 observation_home=False, recovery_instructions=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.close_host, self.review_fault = close_host, review_fault
        self.restore_host, self.observe = restore_host, observe
        if any(not callable(fn) for fn in (close_host, review_fault, restore_host, observe)):
            raise ValueError('explicit lifecycle callbacks required')
        self.status = status
        if type(observation_home) is not bool:
            raise ValueError('Observation HOME restoration must be explicit boolean')
        self.observation_home = observation_home
        if recovery_instructions is None:
            recovery_instructions = (
                'Confirm actual device state and execution ownership, then use the configured recovery. '
                'Device initialization effects and reference poses are supplied by the adapter.')
        if (type(recovery_instructions) is not str or not recovery_instructions.strip()
                or len(recovery_instructions) > 4096):
            raise ValueError('Recovery instructions must be a nonempty string of at most 4096 characters')
        recovery_instructions.encode('utf-8')
        self.recovery_instructions = recovery_instructions
        self.stop_requested = stop_requested or (lambda: False)
        self.sleep_fn, self.poll_seconds = sleep_fn, poll_seconds
        if type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= 5:
            raise ValueError('bounded local event polling interval required')
        self.event = RunRecorder(self.output).event
        self.resume_file = self.output / 'resume.json'
        self.review_file = self.output / 'review.json'
        self.seen_events = set()
        self.seen_payloads = set()
        self.seen_review_events = set()
        self.seen_review_payloads = set()
        self.fault_count = 0

    def check_stop(self):
        if self.stop_requested():
            raise OperatorStop('explicit operator stop while execution is suspended')

    def publish(self, phase, **fields):
        if self.status is not None:
            self.status(dict(phase=phase, **fields))

    def close(self):
        try:
            return dict(receipt=self.close_host(), physical_release='unknown')
        except OperatorStop:
            raise
        except Exception as exc:
            return dict(error=repr(exc), physical_release='unknown')

    def read_resume(self, fault_id, started_ns):
        try:
            fd = os.open(self.resume_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        except OSError as exc:
            payload = ('open-error:' + str(exc)).encode()
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in self.seen_payloads:
                self.seen_payloads.add(digest)
                self.event('resume_rejected', dict(error=str(exc), payload_sha256=digest))
            return None
        digest = 'unreadable-resume-file'
        try:
            info = os.fstat(fd)
            raw = os.read(fd, 16385)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in self.seen_payloads:
                return None
            self.seen_payloads.add(digest)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or len(raw) > 16384):
                raise ValueError('resume event must be a bounded owner-only regular file')
            record = json.loads(raw.decode(), object_pairs_hook=_unique_pairs)
            expected_fields = RESUME_FIELDS | ({'allow_observation_home'} if self.observation_home else set())
            if (type(record) is not dict or set(record) != expected_fields
                    or record['protocol'] != RESUME_PROTOCOL or record['source'] != 'operator'
                    or record['fault_id'] != fault_id
                    or type(record['id']) is not str or not 1 <= len(record['id']) <= 128
                    or record['id'] in self.seen_events
                    or type(record['utc_ns']) is not int
                    or not started_ns <= record['utc_ns'] <= time.time_ns() + 5_000_000_000
                    or any(record[key] is not True for key in
                           ('onsite_takeover_ready', 'physical_state_checked', 'allow_sdk_reinitialize'))
                    or (self.observation_home and record['allow_observation_home'] is not True)):
                raise ValueError('fresh fault-bound operator restoration confirmation required')
            self.seen_events.add(record['id'])
            return record
        except (ValueError, TypeError, UnicodeError, OSError) as exc:
            if digest == 'unreadable-resume-file':
                digest = hashlib.sha256(repr(exc).encode()).hexdigest()
                if digest in self.seen_payloads:
                    return None
                self.seen_payloads.add(digest)
            self.event('resume_rejected', dict(error=str(exc), payload_sha256=digest))
            return None
        finally:
            os.close(fd)

    def read_review(self, fault_id, started_ns):
        """Consume at most one fresh operator event; unchanged errors log once."""
        try:
            fd = os.open(self.review_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        except OSError as exc:
            digest = hashlib.sha256(('open-error:' + str(exc)).encode()).hexdigest()
            if digest not in self.seen_review_payloads:
                self.seen_review_payloads.add(digest)
                self.event('fault_review_rejected', dict(error=str(exc), payload_sha256=digest))
            return None
        digest = None
        try:
            before = os.fstat(fd)
            raw = os.read(fd, 16385)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in self.seen_review_payloads:
                return None
            self.seen_review_payloads.add(digest)
            after = os.fstat(fd)
            fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_uid')
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_mode & 0o077 or len(raw) > 16384 or len(raw) != before.st_size
                    or any(getattr(before, key) != getattr(after, key) for key in fields)):
                raise ValueError('review event must be an unchanged bounded owner-only regular file')
            record = json.loads(raw.decode('utf-8'),
                object_pairs_hook=lambda pairs: _unique_pairs(pairs, 'review'))
            if (type(record) is not dict or set(record) != REVIEW_FIELDS
                    or record['protocol'] != REVIEW_PROTOCOL or record['source'] != 'operator'
                    or record['fault_id'] != fault_id
                    or type(record['id']) is not str
                    or re.fullmatch(r'[A-Za-z0-9_-]{1,128}', record['id']) is None
                    or record['id'] in self.seen_review_events
                    or type(record['utc_ns']) is not int
                    or not started_ns <= record['utc_ns'] <= time.time_ns() + 5_000_000_000):
                raise ValueError('fresh fault-bound operator read-only review event required')
            _review_reason(record['reason'])
            self.seen_review_events.add(record['id'])
            return record
        except (ValueError, TypeError, UnicodeError, OSError, RecursionError) as exc:
            if digest is None:
                digest = hashlib.sha256(repr(exc).encode()).hexdigest()
                if digest in self.seen_review_payloads:
                    return None
                self.seen_review_payloads.add(digest)
            self.event('fault_review_rejected', dict(error=str(exc), payload_sha256=digest))
            return None
        finally:
            os.close(fd)

    def handle_fault(self, fault):
        self.check_stop()
        self.fault_count += 1
        fault_id = 'fault-' + uuid.uuid4().hex
        started_ns = time.time_ns()
        folder = self.output / fault_id
        folder.mkdir()
        snapshot = json.loads(json.dumps(fault, allow_nan=False))
        snapshot.update(fault_id=fault_id, suspended_utc_ns=started_ns,
                        further_dispatch_disabled=True)
        (folder / 'original-fault.json').write_text(json.dumps(snapshot, indent=2) + '\n')
        self.publish('execution_stopping', fault_id=fault_id)
        cleanup = self.close()
        snapshot['control_cleanup'] = cleanup
        self.event('execution_suspended', dict(fault_id=fault_id, stopped_reason=fault.get('stopped_reason'),
                                             control_cleanup=cleanup))
        self.check_stop()
        review = {}
        try:
            review = self.review_fault(snapshot, folder / 'review')
            self.event('fault_learning_completed', dict(fault_id=fault_id,
                finished_by_model=review.get('finished_by_model'),
                saved_candidate_count=len(review.get('saved_candidates', []))))
        except OperatorStop:
            raise
        except Exception as exc:
            self.event('fault_learning_failed', dict(fault_id=fault_id, error=repr(exc)))
        request = dict(protocol=RESUME_PROTOCOL, fault_id=fault_id, resume_file=str(self.resume_file),
            required_confirmations=['onsite_takeover_ready', 'physical_state_checked', 'allow_sdk_reinitialize'],
            initialization_side_effect=self.recovery_instructions,
            motion_request_replayed=False, current_state='unknown')
        if self.observation_home:
            request['required_confirmations'].append('allow_observation_home')
            request['observation_arm_home_requested'] = True
            request['initialization_side_effect'] = (self.recovery_instructions +
                ' Reference pose restoration was explicitly requested; use only the configured recovery callback.')
        (folder / 'recovery-required.json').write_text(json.dumps(request, indent=2) + '\n')
        review_request = dict(protocol=REVIEW_PROTOCOL, fault_id=fault_id, review_file=str(self.review_file),
            required_fields=sorted(REVIEW_FIELDS), read_only=True, motion_available=False,
            after_review='execution_suspended', restoration_permission=False)
        (folder / 'review-required.json').write_text(json.dumps(review_request, indent=2) + '\n')
        self.publish('execution_suspended', recovery=request, review_request=review_request)
        review_count = 0
        while True:
            self.check_stop()
            event = self.read_resume(fault_id, started_ns)
            if event is None:
                review_event = self.read_review(fault_id, started_ns)
                if review_event is not None:
                    review_count += 1
                    review_folder = folder / ('review-%04d' % review_count)
                    self.event('fault_review_requested', dict(review_event, output=str(review_folder),
                                                              further_dispatch_disabled=True))
                    self.check_stop()
                    review_snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
                    review_snapshot['read_only_review_event'] = review_event
                    try:
                        review_folder.mkdir()
                        reviewed = self.review_fault(review_snapshot, review_folder)
                        saved = list(review.get('saved_candidates', [])) + list(reviewed.get('saved_candidates', []))
                        review = dict(reviewed, saved_candidates=saved)
                        self.event('fault_review_completed', dict(fault_id=fault_id,
                            review_event_id=review_event['id'], finished_by_model=reviewed.get('finished_by_model'),
                            saved_candidate_count=len(reviewed.get('saved_candidates', [])),
                            further_dispatch_disabled=True))
                    except OperatorStop:
                        raise
                    except Exception as exc:
                        self.event('fault_review_failed', dict(fault_id=fault_id,
                            review_event_id=review_event['id'], error=repr(exc), automatic_retry=False,
                            further_dispatch_disabled=True))
                    self.publish('execution_suspended', recovery=request, review_request=review_request,
                                 processed_review_event_id=review_event['id'])
                    continue
                self.sleep_fn(self.poll_seconds)
                continue
            self.event('execution_restore_requested', event)
            self.publish('execution_restoring', fault_id=fault_id, recovery_event_id=event['id'])
            self.check_stop()
            try:
                restored = self.restore_host(event, snapshot)
                self.check_stop()
                observation = self.observe()
                old = snapshot.get('last_observation') or {}
                if (type(observation) is not dict or not observation.get('evidence_id')
                        or observation['evidence_id'] == old.get('evidence_id')):
                    raise ValueError('restoration requires a fresh registered observation')
            except OperatorStop:
                self.close()
                raise
            except Exception as exc:
                cleanup = self.close()
                failure = dict(fault_id=fault_id, recovery_event_id=event['id'], error=repr(exc),
                               control_cleanup=cleanup, automatic_retry=False)
                self.event('execution_restore_failed', failure)
                self.publish('execution_suspended', recovery=request, review_request=review_request,
                             restore_failure=failure)
                continue
            recovery = dict(fault_id=fault_id, recovery_event_id=event['id'], restore_receipt=restored,
                saved_candidates=review.get('saved_candidates', []),
                old_stopped_reason=snapshot.get('stopped_reason'), original_request_replayed=False,
                implicit_home=False, implicit_editor_reset=False,
                observation_evidence_id=observation['evidence_id'],
                instruction='Use this new observation to decide the remaining task yourself. '
                            'Prior motion was not replayed; partial effects and unresolved issues remain.')
            self.event('execution_restored', recovery)
            self.publish('execution_active', recovery=recovery, stopped_reason=None)
            return dict(observation=observation, recovery=recovery)
