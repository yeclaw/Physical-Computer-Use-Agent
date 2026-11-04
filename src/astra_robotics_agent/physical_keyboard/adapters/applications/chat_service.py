"""Persistent read-only waiting and explicit operator input; no robot/model imports."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

from astra_robotics_agent.physical_keyboard.runtime.visual_update import validate_roi

FEEDBACK_PROTOCOL = 'physical-chat-operator-feedback-v1'


class OperatorStop(Exception):
    """An explicit local operator stop, never an inferred chat message."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate operator record member')
        result[key] = value
    return result


def append_operator_feedback(path, text, *, message_id=None):
    if type(text) is not str or not text.strip() or len(text) > 8192:
        raise ValueError('operator feedback must be nonempty text of at most8192 characters')
    identifier = message_id or uuid.uuid4().hex
    if type(identifier) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identifier):
        raise ValueError('invalid operator message id')
    record = dict(protocol=FEEDBACK_PROTOCOL, id=identifier, source='user', text=text,
                  utc_ns=time.time_ns())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('operator feedback must be a regular file')
        raw = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
        while raw:
            count = os.write(fd, raw); raw = raw[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


class OperatorFeedbackTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b''
        self.seen = set()
        self.rejections = []

    def read_new(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except FileNotFoundError:
            return []
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('operator feedback must be a regular nonsymlink file')
            if info.st_size < self.offset:
                self.offset = 0; self.pending = b''
            os.lseek(fd, self.offset, os.SEEK_SET)
            raw = os.read(fd, 64 * 1024)
            self.offset += len(raw)
        finally:
            os.close(fd)
        parts = (self.pending + raw).split(b'\n')
        self.pending = parts.pop()
        if len(self.pending) > 64 * 1024:
            self.rejections.append(dict(reason='oversize partial operator record'))
            self.pending = b''
        result = []
        for line in parts:
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode('utf-8'), object_pairs_hook=_unique_pairs)
                if (type(record) is not dict
                        or set(record) != {'protocol', 'id', 'source', 'text', 'utc_ns'}
                        or record['protocol'] != FEEDBACK_PROTOCOL or record['source'] != 'user'
                        or type(record['id']) is not str
                        or re.fullmatch(r'[A-Za-z0-9_-]{1,128}', record['id']) is None
                        or type(record['text']) is not str or not record['text'].strip()
                        or len(record['text']) > 8192
                        or type(record['utc_ns']) is not int or record['utc_ns'] <= 0):
                    raise ValueError('invalid explicit operator record')
                if record['id'] not in self.seen:
                    self.seen.add(record['id']); result.append(record)
            except (ValueError, UnicodeError, TypeError) as exc:
                self.rejections.append(dict(reason=str(exc), sha256=hashlib.sha256(line).hexdigest()))
        return result


class ChatWaiter:
    """Read-only visual scheduling shared by idle waits and safe checkpoints.

    ``checkpoint`` never sleeps or invokes a model. It can acquire a registered
    observation only after a visual/operator event or an explicitly configured
    recheck deadline. The owner calls ``note_observation`` whenever it accepts
    another outer observation; checking a frame never changes that baseline.
    """

    def __init__(self, *, check_update, observe, feedback_path, stop_path,
                 status=None, register_feedback=None, sleep_fn=time.sleep,
                 poll_seconds=1.0, persistence_frames=2, responsive_visual=False,
                 recheck_after_seconds=None, clock_fn=time.monotonic):
        if not callable(check_update) or not callable(observe) or not callable(clock_fn):
            raise ValueError('read-only checker, observer and clock callbacks required')
        if (type(persistence_frames) is not int or persistence_frames < 1
                or type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= 5):
            raise ValueError('bounded poll interval and positive persistence required')
        if type(responsive_visual) is not bool:
            raise ValueError('responsive_visual must be an explicit boolean')
        if recheck_after_seconds is not None:
            if type(recheck_after_seconds) not in (int, float):
                raise ValueError('recheck_after_seconds must be a positive finite number or None')
            try:
                valid_recheck = math.isfinite(recheck_after_seconds) and recheck_after_seconds > 0
            except OverflowError:
                valid_recheck = False
            if not valid_recheck:
                raise ValueError('recheck_after_seconds must be a positive finite number or None')
        self.check_update, self.observe = check_update, observe
        self.feedback = OperatorFeedbackTail(feedback_path)
        self.stop_path = Path(stop_path)
        self.status = status or (lambda fields: None)
        self.register_feedback = register_feedback or (lambda records: records)
        self.sleep_fn, self.poll_seconds = sleep_fn, poll_seconds
        self.persistence_frames = persistence_frames
        self.responsive_visual = responsive_visual
        self.recheck_after_seconds, self.clock_fn = recheck_after_seconds, clock_fn
        self.roi = validate_roi(None)
        self._last_observation_id = None
        self._last_observed_monotonic = self.clock_fn()
        self._consecutive = 0
        self._last_frame = None
        self._baseline_index = None
        self._pending_operator = []
        self._polls = 0
        self._waiting_since_utc_ns = time.time_ns()

    def descriptor(self):
        return dict(protocol='physical-chat-observation-scheduling-v1',
                    responsive_visual=self.responsive_visual,
                    recheck_after_seconds=self.recheck_after_seconds,
                    poll_seconds=self.poll_seconds, persistence_frames=self.persistence_frames,
                    checkpoint='one read-only poll at an owner-declared safe action boundary',
                    meaning='visual changes and explicit rechecks, not message recognition')

    def note_observation(self, observation):
        if (type(observation) is not dict or type(observation.get('evidence_id')) is not str
                or not observation['evidence_id']):
            raise ValueError('registered outer observation evidence ID required')
        if observation['evidence_id'] != self._last_observation_id:
            self._last_observation_id = observation['evidence_id']
            self._last_observed_monotonic = self.clock_fn()
            self._consecutive = 0
            self._last_frame = None
            self._baseline_index = None
            self._waiting_since_utc_ns = time.time_ns()

    def _arguments(self, arguments):
        if type(arguments) is not dict or set(arguments) not in (set(), {'roi'}):
            raise ValueError('wait_for_update takes only an optional normalized ROI')
        if 'roi' in arguments:
            roi = validate_roi(arguments['roi'])
            if roi != self.roi:
                self._consecutive = 0
                self._last_frame = None
            self.roi = roi

    def _stopped(self):
        if self.stop_path.exists():
            raise OperatorStop('explicit operator stop file')

    def _deliver(self, reason, metric):
        self._stopped()
        waiting_since = self._waiting_since_utc_ns
        observation = self.observe()
        if (type(observation) is not dict or type(observation.get('evidence_id')) is not str
                or not observation['evidence_id'] or observation['evidence_id'] == self._last_observation_id):
            raise ValueError('wakeup requires a new registered outer observation evidence ID')
        if self._pending_operator:
            observation['operator_feedback'] = self.register_feedback(self._pending_operator)
            count = len(self._pending_operator)
            self._pending_operator = []
        else:
            count = 0
        if reason == 'visual_update':
            observation['visual_update'] = metric
        elif reason == 'recheck_deadline':
            observation['observation_recheck'] = dict(
                reason='explicit_recheck_deadline',
                recheck_after_seconds=self.recheck_after_seconds,
                elapsed_seconds=max(0.0, self.clock_fn() - self._last_observed_monotonic),
                baseline_observation_evidence_id=self._last_observation_id,
                visual_metric=metric, semantic_message_detected=False)
        self.note_observation(observation)
        self.status(dict(phase=reason + '_ready', waiting_since_utc_ns=waiting_since,
                         idle_polls=self._polls, operator_records=count))
        return observation

    def checkpoint(self, arguments=None):
        """Check once at a safe boundary; return a new observation or ``None``.

        Callers must never invoke this during an in-flight robot program. No
        sleeps, retries, SDK queries or model calls are hidden in this method.
        Capturing an outer observation remains the supplied trusted callback.
        """
        self._arguments({} if arguments is None else arguments)
        self._stopped()
        self._pending_operator.extend(self.feedback.read_new())
        query = {'roi': self.roi}
        if self.responsive_visual:
            query.update(local_change=True, full_frame_guard=True)
        metric = self.check_update(query)
        if (type(metric) is not dict or type(metric.get('changed')) is not bool
                or type(metric.get('frame_id')) is not str or not metric['frame_id']):
            raise ValueError('visual checker must return changed and source frame identity')
        self._polls += 1
        self._stopped()
        if metric.get('observation_pending') is True:
            self._consecutive = 0
            self._last_frame = None
            self.status(dict(phase='waiting_for_camera', waiting_since_utc_ns=self._waiting_since_utc_ns,
                             idle_polls=self._polls, visual_metric=metric,
                             operator_records_pending=len(self._pending_operator)))
            return None
        if self._pending_operator:
            return self._deliver('operator_update', metric)
        # The trusted Runtime reports its frozen reference snapshot. A new
        # baseline invalidates an old persistence count even if an embedding
        # caller forgot to announce its separate outer observation.
        baseline = metric.get('baseline_snapshot_index')
        if baseline != self._baseline_index:
            self._consecutive = 0
            self._last_frame = None
            self._baseline_index = baseline
        if metric['frame_id'] != self._last_frame:
            self._consecutive = self._consecutive + 1 if metric['changed'] else 0
            self._last_frame = metric['frame_id']
        self.status(dict(phase='waiting_for_update', waiting_since_utc_ns=self._waiting_since_utc_ns,
                         idle_polls=self._polls, changed_frames=self._consecutive, visual_metric=metric,
                         operator_input_rejections=self.feedback.rejections[-8:]))
        if self._consecutive >= self.persistence_frames:
            return self._deliver('visual_update', metric)
        if (self.recheck_after_seconds is not None
                and self.clock_fn() - self._last_observed_monotonic >= self.recheck_after_seconds):
            return self._deliver('recheck_deadline', metric)
        return None

    def __call__(self, arguments):
        self._arguments(arguments)
        while True:
            observation = self.checkpoint()
            if observation is not None:
                return observation
            self.sleep_fn(self.poll_seconds)
