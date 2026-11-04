"""Execute bounded frozen JSON declarations, never Python source or hardware.

stdout: ready, zero or more tool_call messages, then done. stdin replies must be
exactly {type: tool_result, index: <matching integer>, result: <receipt object>}.
Authority stays with the coordinator; only move arguments add observation_id.
An invalid program/job fails before ready or any tool call. Protocol failures
emit done/error and exit 2; a non-ok receipt emits done/blocked and exits 0.
This ordinary process is not a sandbox for future Python skills.
"""

import argparse
import hashlib
import math
from pathlib import Path
import sys

from astra_robotics_agent.physical_keyboard.records.contracts import identifier, sha256
from astra_robotics_agent.physical_keyboard.runtime.transport import MAX_LINE_BYTES, ProtocolError, encode_line, strict_loads


PROGRAM_PROTOCOL = 'physical-keyboard-action-plan-v1'
JOB_PROTOCOL = 'physical-keyboard-worker-job-v1'


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ProtocolError('Invalid ' + name + ' fields')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ProtocolError('Action values must be finite numbers, not bool')


def _vector(value, size):
    if not isinstance(value, list) or len(value) != size:
        raise ProtocolError('Invalid action vector length')
    for number in value:
        _number(number)


def validate_program(program):
    _exact(program, {'protocol', 'actions'}, 'action plan')
    if program['protocol'] != PROGRAM_PROTOCOL:
        raise ProtocolError('Unsupported action plan protocol')
    actions = program['actions']
    if not isinstance(actions, list) or len(actions) > 64:
        raise ProtocolError('Action plan must contain at most 64 declarations')
    for action in actions:
        _exact(action, {'method', 'arguments'}, 'action')
        method, arguments = action['method'], action['arguments']
        if method in ('observe', 'stop'):
            _exact(arguments, set(), method)
        elif method == 'move_tcp':
            _exact(arguments, {'xyz_m', 'quaternion_xyzw'}, method)
            _vector(arguments['xyz_m'], 3)
            _vector(arguments['quaternion_xyzw'], 4)
            if not math.isclose(math.hypot(*arguments['quaternion_xyzw']), 1.0,
                                rel_tol=0.0, abs_tol=1e-6):
                raise ProtocolError('Quaternion must be unit length')
        elif method == 'wait_bounded':
            _exact(arguments, {'seconds'}, method)
            _number(arguments['seconds'])
            if arguments['seconds'] < 0:
                raise ProtocolError('Wait duration must be nonnegative')
        else:
            raise ProtocolError('Unsupported action method')
    return actions


def validate_job(job):
    _exact(job, {'protocol', 'scope_key', 'profile_id', 'permit_id', 'trial_id',
                 'observation_id'}, 'worker job')
    if job['protocol'] != JOB_PROTOCOL:
        raise ProtocolError('Unsupported worker job protocol')
    sha256(job['scope_key'])
    for field in ('profile_id', 'permit_id', 'trial_id'):
        identifier(job[field])
    if job['observation_id'] is not None:
        identifier(job['observation_id'])
    return job


def _emit(message):
    sys.stdout.buffer.write(encode_line(message))
    sys.stdout.buffer.flush()


def _reply(index):
    line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
    if not line or not line.endswith(b'\n'):
        raise ProtocolError('Missing complete coordinator response')
    value = strict_loads(line)
    _exact(value, {'type', 'index', 'result'}, 'coordinator response')
    if value['type'] != 'tool_result' or type(value['index']) is not int or value['index'] != index:
        raise ProtocolError('Unexpected coordinator response type/index')
    result = value['result']
    if not isinstance(result, dict) or not isinstance(result.get('status'), str) or not result['status']:
        raise ProtocolError('Tool result needs a receipt status')
    return result


def run(program_path, expected_sha256, job_path):
    sha256(expected_sha256)
    with Path(program_path).open('rb') as stream:
        content = stream.read(MAX_LINE_BYTES + 1)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ProtocolError('Frozen program SHA-256 mismatch')
    actions = validate_program(strict_loads(content))
    with Path(job_path).open('rb') as stream:
        job = validate_job(strict_loads(stream.read(MAX_LINE_BYTES + 1)))
    observation_id = job['observation_id']
    _emit({'type': 'ready', 'program_sha256': expected_sha256})
    for index, action in enumerate(actions):
        method = action['method']
        arguments = dict(action['arguments'])
        if method == 'move_tcp':
            arguments['observation_id'] = observation_id
        _emit({'type': 'tool_call', 'index': index, 'method': method, 'arguments': arguments})
        result = _reply(index)
        if result['status'] != 'ok':
            _emit({'type': 'done', 'status': 'blocked'})
            return 0
        if method == 'observe':
            observation = result.get('observation')
            if not isinstance(observation, dict):
                raise ProtocolError('Observe receipt needs an observation')
            observation_id = identifier(observation.get('observation_id'))
        elif method == 'move_tcp':
            observation_id = None
    _emit({'type': 'done', 'status': 'completed'})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.program, args.sha256, args.job)
    except (ValueError, OSError, ProtocolError, TypeError, OverflowError) as exc:
        _emit({'type': 'done', 'status': 'error', 'error': str(exc)[:1024]})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
