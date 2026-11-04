"""Continuous physical typing task state, without GUI/model/robot side effects.

Text is evaluation data. Only the separate strict identity document may be
read by the actor for recording; it never contains the editor buffer or goal.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import stat
import time

from astra_robotics_agent.physical_keyboard.adapters.environments.typing_cases import validate_manifest, manifest_sha256
from astra_robotics_agent.physical_keyboard.records.typing_metrics import validate_case_identity


EVENT_PROTOCOL = 'physical-keyboard-typing-lab-events-v1'
IDENTITY_PROTOCOL = 'physical-keyboard-typing-case-identity-v1'


class TrainingSession:
    """One continuous editor session; Return is a task commit, not a retry."""
    def __init__(self, manifest, *, run_id, emit, publish_identity,
                 clock_ns=time.monotonic_ns, utc_ns=time.time_ns, settle_seconds=.5,
                 key_state_provider=None):
        if (type(run_id) is not str or not run_id or len(run_id) > 160
                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-' for c in run_id)):
            raise ValueError('bounded run identity required')
        if (type(settle_seconds) not in (int, float) or not math.isfinite(settle_seconds)
                or not 0 < settle_seconds <= 10):
            raise ValueError('finite positive UI release-settle interval up to10 seconds required')
        if not callable(emit) or not callable(publish_identity):
            raise ValueError('explicit recording callbacks required')
        self.run_id, self._emit_callback, self._publish = run_id, emit, publish_identity
        self._clock, self._utc, self._settle_ns = clock_ns, utc_ns, int(settle_seconds * 1000000000)
        self._seq, self._last_clock = 0, -1
        self._pressed, self._focused, self._last_activity = {}, True, 0
        self._return_released, self._return_code = False, None
        if key_state_provider is not None and not callable(key_state_provider):
            raise ValueError('read-only key state provider must be callable')
        self._key_state_provider = key_state_provider
        self._platform_pressed = None
        self._event_protocol = (EVENT_PROTOCOL if key_state_provider is None else
                                'physical-keyboard-typing-lab-events-v2')
        self._seen_batches, self._index = set(), -1
        self.phase, self.current_text, self.last_result = 'batch_complete', '', None
        self._current = None
        self.load_next_manifest(manifest)

    def _now(self):
        value = self._clock()
        if type(value) is not int or value < self._last_clock or value < 0:
            raise ValueError('environment monotonic clock must not move backwards')
        self._last_clock = value
        return value

    def _event(self, kind, data):
        mono, utc = self._now(), self._utc()
        if type(utc) is not int or utc < 0:
            raise ValueError('nonnegative UTC nanoseconds required')
        self._seq += 1
        self._emit_callback(dict(protocol=self._event_protocol, run_id=self.run_id, seq=self._seq,
                                event=kind, utc_ns=utc, monotonic_ns=mono, data=copy.deepcopy(data)))

    def identity(self):
        value = dict(protocol=IDENTITY_PROTOCOL, run_id=self.run_id,
            batch_id=self.manifest['batch_id'], manifest_sha256=self.manifest_hash,
            case_id=self._current['case_id'], occurrence_id=self._current['occurrence_id'],
            repeat_index=self._current['repeat_index'], role=self._current['role'], phase=self.phase)
        return validate_case_identity(value, expected_run_id=self.run_id)

    def _published(self):
        self._publish(self.identity())

    def _begin_case(self):
        occurrence = self.manifest['occurrences'][self._index]
        base = self._base[occurrence['case_id']]
        self._current = dict(base, **occurrence)
        self.current_text, self.phase = '', 'typing'
        self._pressed, self._return_released, self._return_code = {}, False, None
        self._platform_pressed = None
        self._last_activity = self._now()
        self._event('case_begin', dict(self._current, batch_id=self.manifest['batch_id'],
                                      manifest_sha256=self.manifest_hash))
        self._published()

    def load_next_manifest(self, manifest):
        if self.phase != 'batch_complete':
            raise ValueError('a new batch cannot replace an active occurrence')
        validated = validate_manifest(manifest)
        digest = manifest_sha256(validated)
        if digest in self._seen_batches or validated['batch_id'] in self._seen_batches:
            raise ValueError('a batch identity cannot be reused in this session')
        self._seen_batches.update((digest, validated['batch_id']))
        self.manifest, self.manifest_hash = validated, digest
        self._base = {item['case_id']: item for item in validated['base_cases']}
        self._event('batch_begin', dict(manifest_sha256=digest, batch_id=validated['batch_id'],
                                       manifest=validated))
        self._index = 0
        self._begin_case()

    def snapshot(self):
        return copy.deepcopy(dict(phase=self.phase, target_text=self._current['target_text'],
            current_text=self.current_text, case_id=self._current['case_id'],
            occurrence_id=self._current['occurrence_id'], repeat_index=self._current['repeat_index'],
            role=self._current['role'], batch_id=self.manifest['batch_id'], last_result=self.last_result))

    def _key(self, key, code):
        if (type(key) is not str or not 1 <= len(key) <= 128
                or type(code) is not int or not 0 <= code <= 65535):
            raise ValueError('bounded key name and hardware code required')

    def key_down(self, key, code):
        self._key(key, code)
        if self.phase in {'batch_complete', 'closed'}:
            return dict(consume=True, submitted=False)
        repeated = code in self._pressed
        self._pressed[code] = key
        self._last_activity = self._now()
        self._event('key_down', dict(key=key, code=code, repeated=repeated))
        if key in {'Return', 'KP_Enter'} and self.phase == 'typing':
            self._return_released, self._return_code = False, code
            self.phase = 'awaiting_release'
            self.last_result = dict(exact_correct=self.current_text == self._current['target_text'])
            self._event('submit', dict(text=self.current_text))
            self._published()
            return dict(consume=True, submitted=True)
        return dict(consume=self.phase != 'typing', submitted=False)

    def key_up(self, key, code):
        self._key(key, code)
        if self.phase in {'batch_complete', 'closed'}:
            return
        self._pressed.pop(code, None)
        self._last_activity = self._now()
        self._event('key_up', dict(key=key, code=code))
        if code == self._return_code and key in {'Return', 'KP_Enter'}:
            self._return_released = True

    def text_changed(self, text):
        if type(text) is not str or len(text) > 4096:
            raise ValueError('bounded actual editor text required')
        if self.phase != 'typing':
            raise ValueError('text cannot enter a new task before the commit is released')
        self.current_text = text
        self._last_activity = self._now()
        self._event('text_changed', dict(text=text, source='physical_input'))

    def focus_changed(self, focused):
        if type(focused) is not bool:
            raise ValueError('explicit focus boolean required')
        if self.phase == 'closed':
            return
        self._focused = focused
        self._last_activity = self._now()
        self._event('focus', dict(focused=focused))

    def reset_input(self):
        if self.phase in {'batch_complete', 'closed'}:
            raise ValueError('no active training input to reset')
        # This clears only the editor state; robot recovery remains model-owned.
        self.current_text = ''
        self._last_activity = self._now()
        self._event('reset', dict(reason='explicit_training_reset'))
        self._event('text_changed', dict(text='', source='training_reset'))

    def tick(self):
        if self.phase != 'awaiting_release':
            return False
        if self._key_state_provider is not None:
            # A query exception propagates to the environment error handler;
            # never reinterpret an unavailable OS map as released keys.
            self.reconcile_key_state(self._key_state_provider())
        now = self._now()
        elapsed = now - self._last_activity
        if (not self._return_released or self._pressed or not self._focused
                or elapsed < self._settle_ns or (self._key_state_provider is not None
                    and (self._platform_pressed is None or self._platform_pressed))):
            return False
        self._event('case_end', dict(status='submitted', keyboard_released=True,
            return_released=True, pressed_codes=[], focused=True, settled_ns=elapsed, reason='return_released'))
        self._index += 1
        if self._index >= len(self.manifest['occurrences']):
            self.phase = 'batch_complete'
            self._published()
        else:
            self._begin_case()
        return True

    def reconcile_key_state(self, codes):
        """Record genuine OS state separately from GUI key-up events."""
        if (type(codes) not in (list, tuple) or any(type(c) is not int or not 0 <= c <= 255 for c in codes)
                or list(codes) != sorted(set(codes))):
            raise ValueError('sorted unique X11 keycodes required')
        if self.phase != 'awaiting_release':
            raise ValueError('key state reconciliation requires a committed task')
        current = set(codes)
        stale = set(self._pressed) - current
        if self._platform_pressed != current or stale:
            self._pressed = {code: key for code, key in self._pressed.items() if code in current}
            self._platform_pressed = current
            if self._return_code is not None and self._return_code not in current:
                self._return_released = True
            self._last_activity = self._now()
            self._event('key_state_snapshot', dict(source='x11_query_keymap', pressed_codes=list(codes)))

    def close(self, reason='operator_stop'):
        if self.phase == 'closed':
            return
        if type(reason) is not str or not reason or len(reason) > 256:
            raise ValueError('bounded stop reason required')
        if self.phase != 'batch_complete':
            self._event('case_end', dict(status='aborted', keyboard_released=None,
                return_released=self._return_released, pressed_codes=sorted(self._pressed),
                focused=self._focused, settled_ns=0, reason=reason))
        self.phase = 'closed'
        self._published()
        self._event('ended', dict(reason=reason))


def read_case_identity(path, *, expected_run_id=None):
    """Read a bounded regular identity file; no buffer or evaluator API."""
    path = Path(path)
    if not path.is_absolute() or any(part.is_symlink() for part in (path,) + tuple(path.parents)):
        raise ValueError('canonical absolute case identity path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 4096:
            raise ValueError('bounded owner case identity file required')
        raw = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    if len(raw) > 4096:
        raise ValueError('case identity is too large')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate case identity field')
            result[key] = value
        return result
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs)
    return validate_case_identity(value, expected_run_id=expected_run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read recording-only training case identity; no GUI or text API.')
    parser.add_argument('--read-identity', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        value = read_case_identity(args.read_identity, expected_run_id=args.run_id)
    except (ValueError, OSError):
        print(json.dumps({'error': 'typing_case_identity_unavailable'}))
        return 2
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
