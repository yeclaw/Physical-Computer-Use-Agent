"""Synthetic-only stdio RPC owner and durable action journal.

No robot SDK, network, camera or model is imported. This is neither a hardware
exactly-once protocol nor a sandbox. Unknown dispatch requires reconciliation;
there is deliberately no API here to clear it.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from astra_robotics_agent.physical_keyboard.records.contracts import Scope, canonical, digest, identifier, integer, sha256
from astra_robotics_agent.physical_keyboard.records.ledger import Ledger
from astra_robotics_agent.physical_keyboard.runtime.executable import resolve_execution

PROTOCOL = 'physical-keyboard-rpc-v1'
CONFIG_PROTOCOL = 'physical-keyboard-rpc-config-v1'


def default_safety():
    """Software fixture bounds only; these are NOT approved physical limits."""
    return dict(xyz_min_m=[-.3, -.3, 0.], xyz_max_m=[.3, .3, .5],
                max_translation_m=.05, max_rotation_rad=.5, max_wait_s=.2,
                observation_max_age_s=.5, lease_s=5., max_actions=64)


def number(value, name, minimum=None, maximum=None):
    if type(value) not in (int, float):
        raise ValueError(name + ': finite number required')
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(name + ': finite float range required') from exc
    if not finite:
        raise ValueError(name + ': finite number required')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(name + ': outside bounds')
    return float(value)


def vector(value, count, name):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(name + ': invalid dimension')
    return [number(x, name) for x in value]


def pose(value):
    if not isinstance(value, dict) or set(value) != {'xyz_m', 'quaternion_xyzw'}:
        raise ValueError('Invalid pose fields')
    xyz = vector(value['xyz_m'], 3, 'xyz_m')
    q = vector(value['quaternion_xyzw'], 4, 'quaternion_xyzw')
    if abs(sum(x*x for x in q) - 1.) > 1e-6:
        raise ValueError('Unit quaternion required')
    return dict(xyz_m=xyz, quaternion_xyzw=q)


def check_safety(value):
    if not isinstance(value, dict) or set(value) != set(default_safety()):
        raise ValueError('Exact safety fields required')
    lo, hi = [vector(value[k], 3, k) for k in ('xyz_min_m', 'xyz_max_m')]
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError('Empty workspace')
    for k in ('max_translation_m', 'max_rotation_rad', 'max_wait_s',
              'observation_max_age_s', 'lease_s'):
        number(value[k], k, 1e-6, 60.)
    integer(value['max_actions'], 'max_actions', 1)
    if value['max_actions'] > 64:
        raise ValueError('Too many actions')
    return value


class Journal:
    """Single-owner hash-chained journal. FULL SQLite commit precedes dispatch."""
    def __init__(self, root, fingerprint):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.root / 'device.lock', 'a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise ValueError('Journal already owned')
        self.db = sqlite3.connect(self.root / 'actions.sqlite3', isolation_level=None)
        try:
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, kind TEXT NOT NULL,
                    key TEXT NOT NULL, payload TEXT NOT NULL, timestamp_utc TEXT NOT NULL, previous_sha256 TEXT NOT NULL,
                    sha256 TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
                CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
            ''')
            self.calls, self.actions, self.events = {}, {}, []
            self.head = '0'*64
            for row in self.db.execute('SELECT * FROM events ORDER BY seq'):
                event = {k: row[k] for k in ('seq', 'kind', 'key', 'timestamp_utc', 'previous_sha256')}
                event['payload'] = json.loads(row['payload'])
                if (row['seq'] != len(self.events)+1 or row['previous_sha256'] != self.head
                        or row['payload'].encode() != canonical(event['payload'])
                        or digest(canonical(event)) != row['sha256']):
                    raise ValueError('Action journal integrity failure')
                self._apply(event)
                self.head = row['sha256']
            if not self.events:
                self.append('config', 'config', {'fingerprint': fingerprint})
            if self.events[0]['payload'] != {'fingerprint': fingerprint}:
                raise ValueError('Journal configuration changed')
        except BaseException:
            self.close()
            raise

    def _apply(self, event):
        kind, key, payload = event['kind'], event['key'], event['payload']
        if kind == 'rpc_received':
            if key in self.calls:
                raise ValueError('Duplicate RPC reservation')
            self.calls[key] = {'request': payload, 'response': None}
        elif kind == 'rpc_response':
            if self.calls[key]['response'] is not None:
                raise ValueError('Duplicate RPC result')
            self.calls[key]['response'] = payload
        elif kind == 'action_reserved':
            if key in self.actions:
                raise ValueError('Duplicate action reservation')
            self.actions[key] = {'request': payload, 'dispatched': False, 'receipt': None}
        elif kind == 'action_dispatching':
            if self.actions[key]['dispatched'] or self.actions[key]['receipt'] is not None:
                raise ValueError('Invalid action dispatch transition')
            self.actions[key]['dispatched'] = True
        elif kind == 'action_receipt':
            if self.actions[key]['receipt'] is not None:
                raise ValueError('Duplicate action receipt')
            self.actions[key]['receipt'] = payload
        elif kind != 'config' or self.events:
            raise ValueError('Unknown journal event')
        self.events.append(event)

    def append(self, kind, key, payload):
        event = dict(seq=len(self.events)+1, kind=kind, key=key, payload=payload,
                     timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     previous_sha256=self.head)
        encoded = canonical(event)
        head = digest(encoded)
        self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',
                        (event['seq'], kind, key, canonical(payload).decode(), event['timestamp_utc'], self.head, head))
        self._apply(event)
        self.head = head

    @property
    def pending(self):
        return [key for key, value in self.actions.items() if value['receipt'] is None]

    def close(self):
        try:
            self.db.close()
        finally:
            self.lock.close()


class SyntheticServer:
    def __init__(self, config, journal_dir):
        required = {'protocol', 'backend', 'scope', 'safety', 'allowed_profiles',
                    'ledger_root', 'initial_pose'}
        if not isinstance(config, dict) or not required <= set(config) or set(config)-required-{'fault'}:
            raise ValueError('Invalid RPC config fields')
        if config['protocol'] != CONFIG_PROTOCOL or config['backend'] != 'synthetic-fake':
            raise ValueError('Only explicit synthetic-fake backend is implemented')
        self.scope = Scope(**config['scope'])
        if self.scope.evidence_mode != 'synthetic':
            raise ValueError('Physical mode is not implemented or authorized')
        self.safety = check_safety(config['safety'])
        if self.scope.safety_sha256 != digest(canonical(self.safety)):
            raise ValueError('Safety fingerprint mismatch')
        if not isinstance(config['allowed_profiles'], list) or not config['allowed_profiles']:
            raise ValueError('Frozen profiles required')
        for profile in config['allowed_profiles']:
            sha256(profile)
        self.config = config
        self.pose = pose(config['initial_pose'])
        self._workspace(self.pose)
        self.ledger_root = Path(config['ledger_root'])
        if not (self.ledger_root / 'events.sqlite3').is_file():
            raise ValueError('Existing durable ledger required')
        self.fault = config.get('fault', {})
        if not isinstance(self.fault, dict) or set(self.fault)-{'crash_at', 'request_id', 'adapter_delay_s'}:
            raise ValueError('Invalid synthetic fault fields')
        if self.fault.get('crash_at') not in (None, 'after_dispatch', 'after_adapter'):
            raise ValueError('Invalid synthetic crash point')
        if 'request_id' in self.fault:
            identifier(self.fault['request_id'])
        number(self.fault.get('adapter_delay_s', 0), 'adapter_delay_s', 0, 2)
        self.journal = Journal(journal_dir, digest(canonical(config)))
        self.session = uuid.uuid4().hex
        self.revision, self.move_calls = 0, 0
        self.observation, self.permit = None, None
        self.stopped = any(a['request']['method'] == 'stop'
                           and (a['receipt'] or {}).get('status') == 'ok'
                           for a in self.journal.actions.values())

    def _workspace(self, target):
        if any(x < lo or x > hi for x, lo, hi in
               zip(target['xyz_m'], self.safety['xyz_min_m'], self.safety['xyz_max_m'])):
            raise ValueError('workspace_bounds')

    def _context(self, args):
        with Ledger(self.ledger_root) as ledger:
            trial = ledger.trial_state(args['round_id'], args['trial_id'])
            context = ledger.execution_context(args['round_id'], args['variant'])
            executable = resolve_execution(context['bundle'], ledger.read_object)
        if trial is None or trial['result'] is not None or context['round_decided']:
            raise ValueError('unfinished_durable_start_required')
        if trial['variant'] != args['variant'] or trial['start_event_sha256'] != args['start_event_sha256']:
            raise ValueError('trial_start_mismatch')
        bundle = context['bundle']
        if (args['bundle_id'] != context['bundle_id'] or args['scope_key'] != self.scope.key
                or context['scope_key'] != self.scope.key or bundle['scope'] != asdict(self.scope)
                or bundle['files']['profile.json']['sha256'] != args['profile_id']
                or executable['program_sha256'] != args['program_sha256']
                or args['profile_id'] not in self.config['allowed_profiles']):
            raise ValueError('frozen_bundle_or_scope_mismatch')
        return trial

    def _permit(self, args):
        if self.stopped:
            raise ValueError('stopped')
        p = self.permit
        if p is None or args['permit_id'] != p['permit_id']:
            raise ValueError('invalid_permit')
        if args['scope_key'] != self.scope.key or args['profile_id'] != p['profile_id']:
            raise ValueError('scope_or_profile_mismatch')
        if time.monotonic() >= p['expires_at']:
            raise ValueError('lease_expired')
        if p['actions'] >= self.safety['max_actions']:
            raise ValueError('action_budget')
        self._context(p)

    def _fresh(self, observation_id):
        o = self.observation
        if (o is None or observation_id != o['observation_id'] or o['session'] != self.session
                or o['revision'] != self.revision):
            raise ValueError('stale_observation')
        if time.monotonic() - o['source_monotonic_s'] > self.safety['observation_max_age_s']:
            raise ValueError('expired_observation')

    def _bound_move(self, args):
        target = pose({key: args[key] for key in ('xyz_m', 'quaternion_xyzw')})
        self._workspace(target)
        distance = math.dist(target['xyz_m'], self.pose['xyz_m'])
        qdot = abs(sum(x*y for x, y in zip(target['quaternion_xyzw'], self.pose['quaternion_xyzw'])))
        angle = 2*math.acos(min(1., qdot))  # q and -q are the same rotation
        if distance > self.safety['max_translation_m'] + 1e-12:
            raise ValueError('translation_limit')
        if angle > self.safety['max_rotation_rad'] + 1e-12:
            raise ValueError('rotation_limit')
        return target

    def _preflight(self, method, args):
        self._permit(args)
        if self.journal.pending:
            # The request being preflighted is not reserved yet; a later recheck
            # excludes it explicitly below. Unknown actions survive restart/stop.
            raise ValueError('unreconciled_action')
        if method == 'move_tcp':
            self._fresh(args['observation_id'])
            self._bound_move(args)
        else:
            duration = number(args['seconds'], 'seconds', 0, self.safety['max_wait_s'])
            if time.monotonic() + duration >= self.permit['expires_at']:
                raise ValueError('wait_exceeds_lease')

    def _action(self, method, args):
        fields = {'request_id'}
        if method != 'stop':
            fields |= {'permit_id', 'scope_key', 'profile_id'}
            fields |= ({'observation_id', 'xyz_m', 'quaternion_xyzw'} if method == 'move_tcp' else {'seconds'})
        if not isinstance(args, dict) or set(args) != fields:
            raise ValueError('action_schema')
        key = identifier(args['request_id'])
        request = {'method': method, 'arguments': args}
        existing = self.journal.actions.get(key)
        if existing is not None:
            if existing['request'] != request:
                return dict(status='rejected', reason='conflicting_request', request_id=key, executed=False)
            return existing['receipt'] or dict(status='unknown', reason='uncompleted_dispatch',
                                                request_id=key, executed=None)
        # Preflight is performed BEFORE reservation. Rejections also get receipts.
        reason = None
        if method != 'stop':
            try:
                self._preflight(method, args)
            except (ValueError, KeyError, TypeError) as exc:
                reason = str(exc)
        self.journal.append('action_reserved', key, request)
        receipt = dict(protocol=PROTOCOL, evidence_mode='synthetic', request_id=key,
                       method=method, session=self.session, source_monotonic_s=time.monotonic(),
                       executed=False, text_verified=False)
        if self.permit is not None:
            receipt.update({k: self.permit[k] for k in
                            ('trial_id', 'bundle_id', 'start_event_sha256', 'scope_key', 'profile_id')})
        if method != 'stop':
            receipt.update({k: args[k] for k in ('permit_id', 'scope_key', 'profile_id')})
        if reason is None:
            try:
                if method != 'stop':
                    self._permit(args)  # commit may have consumed the lease/freshness
                    if method == 'move_tcp':
                        self._fresh(args['observation_id'])
                        target = self._bound_move(args)
                    elif time.monotonic() + args['seconds'] >= self.permit['expires_at']:
                        raise ValueError('wait_exceeds_lease')
                receipt['dispatched_at_monotonic_s'] = time.monotonic()
                self.journal.append('action_dispatching', key,
                                    {'session': self.session, 'source_monotonic_s': receipt['dispatched_at_monotonic_s']})
                if self.fault.get('request_id') == key and self.fault.get('crash_at') == 'after_dispatch':
                    os._exit(81)
                delay = self.fault.get('adapter_delay_s', 0)
                if delay:
                    time.sleep(delay)  # synthetic fault only, never real hardware
                if method == 'move_tcp':
                    self.pose = target
                    self.move_calls += 1
                    self.revision += 1
                    self.observation = None
                elif method == 'wait_bounded':
                    time.sleep(args['seconds'])
                else:
                    self.stopped = True
                    self.permit = None
                if method != 'stop':
                    self.permit['actions'] += 1
                if self.fault.get('request_id') == key and self.fault.get('crash_at') == 'after_adapter':
                    os._exit(82)
                receipt.update(status='ok', executed=True, pose=self.pose,
                               completed_at_monotonic_s=time.monotonic(),
                               revision=self.revision, move_calls=self.move_calls)
                if self.permit is not None:
                    receipt.update(trial_id=self.permit['trial_id'],
                                   start_event_sha256=self.permit['start_event_sha256'],
                                   bundle_id=self.permit['bundle_id'])
            except (ValueError, KeyError, TypeError) as exc:
                if self.journal.actions[key]['dispatched']:
                    # After dispatch no false failure acknowledgement is allowed.
                    # Leave the action pending; its effect cannot be safely inferred.
                    return {**receipt, 'status': 'unknown', 'reason': str(exc), 'executed': None}
                reason = str(exc)
        if reason is not None:
            receipt.update(status='rejected', reason=reason)
        self.journal.append('action_receipt', key, receipt)
        return receipt

    def handle(self, method, args):
        if method in {'move_tcp', 'wait_bounded', 'stop'}:
            return self._action(method, args)
        if method == 'observe':
            if args != {}:
                raise ValueError('observe_schema')
            self.observation = dict(observation_id=uuid.uuid4().hex, session=self.session,
                                    source_monotonic_s=time.monotonic(), revision=self.revision,
                                    scope_key=self.scope.key, pose=self.pose,
                                    evidence_mode='synthetic', text_verified=False,
                                    move_calls=self.move_calls, stopped=self.stopped,
                                    pending_requests=self.journal.pending)
            return dict(status='ok', observation=self.observation)
        if method == 'authorize_trial':
            expected = {'trial_id', 'round_id', 'variant', 'bundle_id', 'profile_id',
                        'scope_key', 'start_event_sha256', 'program_sha256'}
            if not isinstance(args, dict) or set(args) != expected:
                raise ValueError('authorize_schema')
            if self.stopped or self.journal.pending or self.permit is not None:
                raise ValueError('stopped_pending_or_active_permit')
            self._context(args)
            self.permit = {**args, 'permit_id': uuid.uuid4().hex,
                           'expires_at': time.monotonic() + self.safety['lease_s'], 'actions': 0}
            return dict(status='ok', permit_id=self.permit['permit_id'],
                        start_event_sha256=args['start_event_sha256'], session=self.session,
                        expires_at_monotonic_s=self.permit['expires_at'])
        if method == 'close_trial':
            if not isinstance(args, dict) or set(args) != {'permit_id'}:
                raise ValueError('close_schema')
            if self.permit is not None and args['permit_id'] != self.permit['permit_id']:
                raise ValueError('invalid_permit')
            self.permit = None
            return {'status': 'ok'}
        raise ValueError('unsupported_method')

    def rpc(self, message):
        if not isinstance(message, dict) or set(message) != {'rpc_id', 'method', 'arguments'}:
            raise ValueError('rpc_schema')
        rpc_id = identifier(message['rpc_id'])
        if not isinstance(message['method'], str) or not isinstance(message['arguments'], dict):
            raise ValueError('rpc_schema')
        old = self.journal.calls.get(rpc_id)
        if old is not None:
            if old['request'] != message:
                result = {'status': 'rejected', 'reason': 'conflicting_rpc_id'}
            else:
                result = old['response'] or {'status': 'unknown', 'reason': 'uncompleted_rpc'}
            return {'rpc_id': rpc_id, 'result': result}
        self.journal.append('rpc_received', rpc_id, message)
        try:
            result = self.handle(message['method'], message['arguments'])
        except (ValueError, KeyError, TypeError) as exc:
            result = {'status': 'rejected', 'reason': str(exc)}
        self.journal.append('rpc_response', rpc_id, result)
        return {'rpc_id': rpc_id, 'result': result}

    def close(self):
        self.journal.close()


class RpcClient:
    """Owns one synthetic service subprocess; never retries an action implicitly."""
    def __init__(self, config_path, journal_dir, artifacts_dir):
        from astra_robotics_agent.physical_keyboard.runtime.transport import JsonProcess
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        self.transport = JsonProcess([sys.executable, '-m', __name__, '--config', str(config_path),
                                      '--journal', str(journal_dir)],
                                     Path(artifacts_dir) / 'rpc-stderr.log')
        self.pid = self.transport.pid
        self.broken = False

    def call(self, method, arguments, timeout_s=5., *, rpc_id=None):
        if self.broken:
            raise RuntimeError('RPC connection is broken; no implicit retry')
        rpc_id = rpc_id or uuid.uuid4().hex
        timeout_s = number(timeout_s, 'RPC timeout_s', 1e-6, 60.)
        deadline = time.monotonic() + timeout_s
        try:
            self.transport.send(dict(rpc_id=rpc_id, method=method, arguments=arguments), timeout_s=timeout_s)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('RPC deadline expired during send')
            response = self.transport.receive(remaining)
            if set(response) != {'rpc_id', 'result'} or response['rpc_id'] != rpc_id or not isinstance(response['result'], dict):
                raise ValueError('Invalid RPC correlation/response')
            return response['result']
        except BaseException:
            self.broken = True
            raise

    def close(self):
        return self.transport.close()


def main():
    from astra_robotics_agent.physical_keyboard.runtime.transport import strict_loads, encode_line
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--journal', required=True)
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    if len(config_bytes) > 65536:
        raise ValueError('Config too large')
    server = SyntheticServer(strict_loads(config_bytes), args.journal)
    try:
        while line := sys.stdin.buffer.readline(65537):
            if len(line) > 65536 or not line.endswith(b'\n'):
                raise ValueError('Oversize or unterminated RPC frame')
            response = server.rpc(strict_loads(line))
            sys.stdout.buffer.write(encode_line(response))
            sys.stdout.buffer.flush()
    finally:
        server.close()


if __name__ == '__main__':
    main()
