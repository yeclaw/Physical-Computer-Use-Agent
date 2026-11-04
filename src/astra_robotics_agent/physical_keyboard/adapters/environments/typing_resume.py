"""Verified pure-state reconstruction of an active typing event prefix.

This replays only TrainingSession methods into an exact event verifier. It
never injects keyboard events, opens a display, or touches robot interfaces.
"""
import hashlib
import json
import time

from .typing_lab import TrainingSession, EVENT_PROTOCOL


def decode_prefix(raw):
    if not raw or not raw.endswith(b'\n') or len(raw) > 64 * 1024 * 1024:
        raise ValueError('bounded complete event prefix required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate event field')
            result[key] = value
        return result
    events = [json.loads(line.decode('utf-8'), object_pairs_hook=pairs) for line in raw.splitlines()]
    if any(e.get('event') in ('ended',) or
           (e.get('event') == 'case_end' and e.get('data', {}).get('status') != 'submitted') for e in events):
        raise ValueError('only an active event prefix can resume')
    if sum(e.get('event') == 'batch_begin' for e in events) != 1:
        raise ValueError('resume currently requires one exact batch')
    return events


def restore_session(manifest, raw, *, run_id, emit, publish_identity,
                    clock_ns=time.monotonic_ns, utc_ns=time.time_ns,
                    key_state_provider=None, settle_seconds=.5):
    events = decode_prefix(raw)
    position = [0]

    def expected():
        if position[0] >= len(events):
            raise ValueError('event prefix ended during a state transition')
        return events[position[0]]

    def verify(event):
        if event != expected():
            raise ValueError('event prefix disagrees with deterministic task state at seq %s' % event['seq'])
        position[0] += 1

    first = events[0]
    if (first.get('protocol') not in (EVENT_PROTOCOL, 'physical-keyboard-typing-lab-events-v2')
            or first.get('run_id') != run_id):
        raise ValueError('event prefix run/protocol mismatch')
    # No identity is published and no output is appended during reconstruction.
    session = TrainingSession(manifest, run_id=run_id, emit=verify, publish_identity=lambda unused: None,
        clock_ns=lambda: expected()['monotonic_ns'], utc_ns=lambda: expected()['utc_ns'],
        settle_seconds=settle_seconds,
        key_state_provider=(lambda: []) if first['protocol'].endswith('v2') else None)
    session._key_state_provider = None  # Replay recorded snapshots, never query OS.
    while position[0] < len(events):
        event = expected()
        if (event.get('run_id') != run_id or event.get('seq') != position[0] + 1
                or event.get('protocol') not in (EVENT_PROTOCOL, 'physical-keyboard-typing-lab-events-v2')):
            raise ValueError('event identity or sequence mismatch')
        if session._event_protocol.endswith('v2') and event['protocol'] == EVENT_PROTOCOL:
            raise ValueError('event protocol cannot downgrade')
        session._event_protocol = event['protocol']
        kind, data = event['event'], event['data']
        if kind == 'key_down':
            session.key_down(data['key'], data['code'])
        elif kind == 'key_up':
            session.key_up(data['key'], data['code'])
        elif kind == 'text_changed' and data.get('source') == 'physical_input':
            session.text_changed(data['text'])
        elif kind == 'focus':
            session.focus_changed(data['focused'])
        elif kind == 'reset':
            session.reset_input()
        elif kind == 'key_state_snapshot':
            if data.get('source') != 'x11_query_keymap':
                raise ValueError('unknown key state source')
            session.reconcile_key_state(data['pressed_codes'])
        elif kind == 'case_end' and data.get('status') == 'submitted':
            if type(data.get('settled_ns')) is not int or data['settled_ns'] < session._settle_ns:
                raise ValueError('invalid original release settling evidence')
            session._last_activity = event['monotonic_ns'] - data['settled_ns']
            if not session.tick():
                raise ValueError('recorded case completion has no release evidence')
        else:
            raise ValueError('unsupported or incomplete state transition: ' + kind)
    if session.phase not in ('typing', 'awaiting_release'):
        raise ValueError('event prefix has no active task')
    if clock_ns() < session._last_clock:
        raise ValueError('resume requires the original monotonic-clock boot')
    session._clock, session._utc = clock_ns, utc_ns
    session._emit_callback, session._publish = emit, publish_identity
    session._key_state_provider = key_state_provider
    session._platform_pressed = None  # Require a fresh actual map after restart.
    session._event_protocol = ('physical-keyboard-typing-lab-events-v2'
                               if key_state_provider is not None else session._event_protocol)
    session._published()
    return session


def prefix_metadata(raw, *, source_path):
    events = decode_prefix(raw)
    return dict(protocol='physical-keyboard-typing-resume-prefix-v1', source_path=str(source_path),
                prefix_sha256=hashlib.sha256(raw).hexdigest(), prefix_bytes=len(raw),
                last_seq=events[-1]['seq'], last_monotonic_ns=events[-1]['monotonic_ns'],
                run_id=events[0]['run_id'], raw_source_unchanged=True,
                policy_feedback='external_camera_only')
