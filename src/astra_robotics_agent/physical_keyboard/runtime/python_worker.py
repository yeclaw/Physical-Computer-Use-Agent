"""Run one frozen bounded Python skill through host-owned JSON tool receipts.

This worker contains no SDK, model, network client or exec. The AST interpreter
is a deliberately limited language implementation, not an OS security sandbox.
"""

import argparse
from pathlib import Path
import sys

from astra_robotics_agent.physical_keyboard.records.contracts import PROTOCOL as LEDGER_PROTOCOL, Scope, digest, identifier, sha256
from astra_robotics_agent.physical_keyboard.runtime.executable import PYTHON_KIND, python_source_names, resolve_execution, safe_name
from astra_robotics_agent.physical_keyboard.runtime.skill_language import BudgetExceeded, SkillError, ToolError, run_skill
from astra_robotics_agent.physical_keyboard.runtime.transport import MAX_LINE_BYTES, ProtocolError, encode_line, strict_loads


JOB_PROTOCOL = 'physical-keyboard-python-worker-job-v1'
MAX_TOOL_CALLS = 64


def _emit(message):
    sys.stdout.buffer.write(encode_line(message))
    sys.stdout.buffer.flush()


def _reply(index):
    line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
    if not line or not line.endswith(b'\n'):
        raise SkillError('Missing complete coordinator response; tool execution is unknown')
    try:
        value = strict_loads(line)
        if (set(value) != {'type', 'index', 'result'} or value['type'] != 'tool_result'
                or type(value['index']) is not int or value['index'] != index):
            raise ProtocolError('Unexpected coordinator response type/index')
        result = value['result']
        if not isinstance(result, dict) or result.get('status') not in {'ok', 'rejected', 'unknown'}:
            raise ProtocolError('Invalid coordinator tool receipt')
        return result
    except (ValueError, ProtocolError, TypeError) as exc:
        raise SkillError('Invalid coordinator response; no tool retry is permitted') from exc


class Bridge:
    def __init__(self, job):
        self.observation_id = job['observation_id']
        self.calls = 0
        self.failure = None

    def call(self, method, arguments):
        if self.calls >= MAX_TOOL_CALLS:
            raise BudgetExceeded('Actual tool-call budget exhausted')
        index = self.calls
        self.calls += 1
        try:
            _emit(dict(type='tool_call', index=index, method=method, arguments=arguments))
            receipt = _reply(index)
        except (OSError, ProtocolError) as exc:
            raise SkillError('Tool transport failed; no tool retry is permitted') from exc
        if receipt['status'] != 'ok':
            if receipt['status'] == 'unknown' or self.failure is None:
                self.failure = receipt['status']
            if receipt['status'] == 'unknown':
                self.observation_id = None
            raise ToolError('Tool ' + method + ': ' + str(receipt.get('reason', receipt['status']))[:512])
        return receipt

    def observe(self):
        receipt = self.call('observe', {})
        observation = receipt.get('observation')
        if not isinstance(observation, dict):
            raise SkillError('Observe receipt is missing source observation')
        try:
            self.observation_id = identifier(observation.get('observation_id'))
        except ValueError as exc:
            raise SkillError('Invalid source observation ID') from exc
        return receipt

    def move_tcp(self, xyz_m, quaternion_xyzw):
        if self.failure == 'unknown':
            raise ToolError('Unreconciled unknown action prohibits further movement')
        if self.observation_id is None:
            self.observe()
        receipt = self.call('move_tcp', dict(xyz_m=xyz_m, quaternion_xyzw=quaternion_xyzw,
                                            observation_id=self.observation_id))
        self.observation_id = None
        return receipt

    def wait_bounded(self, seconds):
        if self.failure == 'unknown':
            raise ToolError('Unreconciled unknown action prohibits further waiting')
        return self.call('wait_bounded', {'seconds': seconds})

    def stop(self):
        return self.call('stop', {})


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Runtime material must be a regular nonsymlink file')
    with path.open('rb') as stream:
        content = stream.read(MAX_LINE_BYTES + 1)
    if len(content) > MAX_LINE_BYTES:
        raise ValueError('Runtime material exceeds 64 KiB')
    return content


def _read_named(root, bundle, name):
    safe_name(name)
    path = root
    for part in Path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Runtime material cannot contain symlink components')
    content = _read(path)
    try:
        entry = bundle['files'][name]
        if (not isinstance(entry, dict) or set(entry) != {'bytes', 'sha256'}
                or type(entry['bytes']) is not int or entry['bytes'] < 0
                or len(content) != entry['bytes'] or digest(content) != sha256(entry['sha256'])):
            raise ValueError('Materialized frozen source integrity mismatch')
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing frozen runtime file: ' + name) from exc
    return content


def _load(manifest_path, expected_sha256, job_path):
    sha256(expected_sha256)
    manifest_path, job_path = Path(manifest_path), Path(job_path)
    content = _read(manifest_path)
    if digest(content) != expected_sha256:
        raise ValueError('Frozen bundle manifest SHA-256 mismatch')
    bundle = strict_loads(content)
    if bundle.get('protocol') != LEDGER_PROTOCOL or not isinstance(bundle.get('files'), dict):
        raise ValueError('Unsupported frozen bundle manifest')
    for name in bundle['files']:
        safe_name(name)
    scope = Scope(**bundle['scope'])
    if scope.evidence_mode != 'synthetic':
        raise ValueError('Python worker currently accepts synthetic fixtures only')
    root = manifest_path.parent.resolve()
    execution = _read_named(root, bundle, 'execution.json')
    descriptor = strict_loads(execution)
    names = python_source_names(descriptor)
    material = {'execution.json': execution, 'profile.json': _read_named(root, bundle, 'profile.json')}
    for name in names:
        material[name] = _read_named(root, bundle, name)
    objects = {digest(value): value for value in material.values()}
    resolved = resolve_execution(bundle, objects.__getitem__)
    if resolved['kind'] != PYTHON_KIND:
        raise ValueError('Python worker requires bounded-python-v1')
    job = strict_loads(_read(job_path))
    if set(job) != {'protocol', 'scope_key', 'profile_id', 'permit_id', 'trial_id', 'observation_id', 'bundle_id'}:
        raise ValueError('Exact Python worker job fields required')
    if job['protocol'] != JOB_PROTOCOL:
        raise ValueError('Unsupported Python worker job protocol')
    sha256(job['scope_key'])
    sha256(job['profile_id'])
    sha256(job['bundle_id'])
    identifier(job['permit_id'])
    identifier(job['trial_id'])
    if job['observation_id'] is not None:
        identifier(job['observation_id'])
    if (job['bundle_id'] != expected_sha256 or job['scope_key'] != scope.key
            or job['profile_id'] != bundle['files']['profile.json']['sha256']):
        raise ValueError('Python worker job does not match frozen scope/profile/bundle')
    return resolved, job, [(name, material[name].decode('utf-8')) for name in names]


def run(manifest_path, expected_sha256, job_path):
    bridge = None
    try:
        resolved, job, sources = _load(manifest_path, expected_sha256, job_path)
        bridge = Bridge(job)
        _emit(dict(type='ready', program_sha256=resolved['program_sha256'], bundle_id=expected_sha256))
        result = run_skill(sources, tools={'observe': bridge.observe, 'move_tcp': bridge.move_tcp,
                    'wait_bounded': bridge.wait_bounded, 'stop': bridge.stop}, profile=resolved['profile'])
        _emit(dict(type='done', status='blocked' if bridge.failure else 'completed',
                   tool_failure=bridge.failure, skill_result=result))
        return 0
    except ToolError as exc:
        _emit(dict(type='done', status='blocked', tool_failure=bridge.failure if bridge else 'rejected',
                   error=str(exc)[:1024]))
        return 0
    except (SkillError, ValueError, OSError, ProtocolError, TypeError, KeyError, UnicodeError) as exc:
        _emit(dict(type='done', status='error', tool_failure=bridge.failure if bridge else None,
                   error=str(exc)[:1024]))
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    return run(args.manifest, args.sha256, args.job)


if __name__ == '__main__':
    raise SystemExit(main())
