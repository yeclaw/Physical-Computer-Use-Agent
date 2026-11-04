"""Bounded, read-only facts from frozen native replay files.

This records module uses standard-library and shared Records schemas. It neither imports a robot
backend nor proves a physical cause, present pose, release, or safe recovery.
All filenames come from a saved response or the fixed native plan directory;
callers must not expose this filesystem reader as an arbitrary model tool.
"""
from __future__ import annotations

from collections import deque
import codecs
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from astra_robotics_agent.physical_keyboard.records.admission import validate_rejection

PROTOCOL = 'physical-keyboard-saved-diagnostics-v1'
ADMISSION_DIAGNOSTIC_PROTOCOL = 'physical-keyboard-saved-admission-diagnostics-v1'
ADMISSION_EVIDENCE_PROTOCOL = 'physical-keyboard-admission-evidence-v1'
PLAN_PROTOCOL = 'physical-keyboard-native-timed-joint-plan-v1'
MAX_OUTPUT_BYTES = 64 * 1024
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_PLAN_BYTES = 128 * 1024 * 1024
MAX_AUDIT_BYTES = 512 * 1024 * 1024
MAX_AUDIT_LINE_BYTES = 256 * 1024  # Complete target-event decode budget.
_AUDIT_SCAN_CHUNK_BYTES = 64 * 1024
_AUDIT_SCAN_MAX_DEPTH = 64
_AUDIT_SCAN_MAX_KEYS = 4096
_AUDIT_SCAN_MAX_KEY_CHARS = 128 * 1024
_AUDIT_SCAN_MAX_TOKEN_CHARS = 4096
_NATIVE_AUDIT_EVENTS = frozenset(('native_timed_prepared', 'native_timed_dispatch_check',
    'native_timed_dispatching', 'native_timed_dispatched', 'native_timed_dispatch_failed'))
_JSON_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')
_JSON_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')
MAX_PLANS = 512
MAX_POINTS = 100_000
_SHA = re.compile(r'[a-f0-9]{64}\Z')


class DiagnosticUnavailable(ValueError):
    """Saved facts cannot be established without guessing or unsafe reads."""


def _canonical(value):
    # Identical to native_timed_backend._encoded/timed_trajectory._canonical.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticUnavailable('duplicate_json_key')
        result[key] = value
    return result


def _constant(value):
    raise DiagnosticUnavailable('nonfinite_json_number')


def decode_json(raw):
    """Decode strict JSON for the SDK-free CLI as well as saved documents."""
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        _canonical(value)  # Also rejects float overflow such as 1e999.
        return value
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise DiagnosticUnavailable('invalid_json') from exc


def _object(value, label):
    if type(value) is not dict:
        raise DiagnosticUnavailable(label + '_must_be_object')
    return value


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise DiagnosticUnavailable(label + '_must_be_finite')
    return value


def _index(value, label):
    if type(value) is not int or value < 0:
        raise DiagnosticUnavailable(label + '_must_be_nonnegative_integer')
    return value


def _text(value, label, limit=4096):
    if type(value) is not str or len(value.encode('utf-8')) > limit:
        raise DiagnosticUnavailable(label + '_invalid_or_oversized')
    return value


def _digest(value, label):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise DiagnosticUnavailable(label + '_must_be_sha256')
    return value


def _vector(value, size, label):
    if type(value) is not list or len(value) != size:
        raise DiagnosticUnavailable(label + '_invalid_vector')
    return [_number(part, label) for part in value]


def _sample(value):
    value = _object(value, 'sample')
    result = dict(t_s=_number(value.get('t_s'), 'sample_time'),
                  joint_rad=_vector(value.get('joint_rad'), 6, 'sample_joint'))
    for key, size in (('xyz_m', 3), ('quaternion_xyzw', 4)):
        if key in value:
            result[key] = _vector(value[key], size, key)
    if 'opening_m' in value:
        result['opening_m'] = _number(value['opening_m'], 'opening_m')
    return result


def _anchor(value):
    if value is None:
        return None
    value = _object(value, 'anchor')
    return {key: _index(value[key], key) for key in ('monotonic_ns', 'utc_ns', 'sampling_span_ns') if key in value}


class _RunFiles:
    """Anchor every open at a directory descriptor, refusing child symlinks."""
    def __init__(self, root):
        path = Path(root)
        if '..' in path.parts:
            raise DiagnosticUnavailable('run_root_traversal')
        self.original = Path(os.path.abspath(path))
        try:
            if self.original.is_symlink():
                raise DiagnosticUnavailable('run_root_symlink')
            self.root = self.original.resolve(strict=True)
            self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise DiagnosticUnavailable('run_root_unavailable') from exc

    def close(self):
        os.close(self.fd)

    def relative(self, value):
        _text(value, 'saved_path')
        path = Path(value)
        if '..' in path.parts or not path.parts:
            raise DiagnosticUnavailable('saved_path_traversal')
        if path.is_absolute():
            for root in (self.original, self.root):
                try:
                    path = path.relative_to(root)
                    break
                except ValueError:
                    pass
            else:
                raise DiagnosticUnavailable('saved_path_outside_run_root')
        if not path.parts or any(part in ('', '.', '..') for part in path.parts):
            raise DiagnosticUnavailable('invalid_saved_path')
        return path.as_posix()

    def open(self, value, *, directory=False):
        relative = self.relative(value)
        parent = os.dup(self.fd)
        try:
            parts = Path(relative).parts
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if directory:
                flags |= os.O_DIRECTORY
            result = os.open(parts[-1], flags, dir_fd=parent)
            info = os.fstat(result)
            if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
                os.close(result)
                raise DiagnosticUnavailable('saved_path_not_regular')
            return result, relative, info
        except OSError as exc:
            raise DiagnosticUnavailable('saved_file_missing_unsafe_or_unreadable') from exc
        finally:
            os.close(parent)

    def read_json(self, value, expected_sha=None):
        raw, reference = self.read_bytes(value, expected_sha=expected_sha, limit=MAX_JSON_BYTES)
        return decode_json(raw), reference

    def read_bytes(self, value, expected_sha=None, *, limit=MAX_INPUT_BYTES):
        descriptor, relative, before = self.open(value)
        try:
            if before.st_size > limit:
                raise DiagnosticUnavailable('saved_json_size_limit')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                raw = stream.read(limit + 1)
            after = os.fstat(descriptor)
            if len(raw) > limit:
                raise DiagnosticUnavailable('saved_json_size_limit')
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise DiagnosticUnavailable('saved_file_changed_during_read')
        finally:
            os.close(descriptor)
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None and digest != _digest(expected_sha, 'expected_file_sha256'):
            raise DiagnosticUnavailable('saved_file_sha256_mismatch')
        return raw, dict(relative_path=relative, file_sha256=digest, bytes=len(raw))

    def plan_paths(self):
        descriptor, _, _ = self.open('native-joint-plans', directory=True)
        paths = []
        try:
            with os.scandir(descriptor) as entries:
                seen = 0
                for entry in entries:
                    seen += 1
                    if seen > MAX_PLANS:
                        raise DiagnosticUnavailable('native_plan_directory_count_limit')
                    if entry.name.startswith('plan-'):
                        if not entry.is_dir(follow_symlinks=False):
                            raise DiagnosticUnavailable('native_plan_directory_unsafe')
                        paths.append('native-joint-plans/' + entry.name + '/joint-plan.json')
        finally:
            os.close(descriptor)
        return sorted(paths)


def _unwrap(execution):
    value = _object(execution, 'execution')
    for _ in range(4):
        if 'receipts' in value:
            return value
        for key in ('response', 'result', 'execution'):
            if type(value.get(key)) is dict:
                value = value[key]
                break
        else:
            raise DiagnosticUnavailable('execution_receipts_missing')
    raise DiagnosticUnavailable('execution_wrapper_depth_limit')


def _receipt_summary(receipt, replay):
    result = {key: receipt[key] for key in ('tool_index', 'method', 'status', 'motion_dispatch_attempted') if key in receipt}
    for key in ('error',):
        if key in receipt:
            result[key] = _text(receipt[key], 'receipt_' + key)
    fields = ('trajectory_id', 'source_sha256', 'effective_sha256', 'status', 'point_count',
              'dispatch_attempted_count', 'dispatched_count', 'window_start_index', 'window_end_index',
              'speed_scale', 'patch_count', 'planned_duration_s', 'elapsed_s', 'max_lateness_s',
              'motion_outcome_uncertain', 'stop_called', 'stop_error', 'error', 'text_success')
    result['replay'] = {key: replay[key] for key in fields if key in replay}
    if len(_canonical(result)) > 12 * 1024:
        raise DiagnosticUnavailable('receipt_summary_size_limit')
    return result


def _admission_facts(value):
    try:
        value = validate_rejection(value)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise DiagnosticUnavailable('admission_rejection_invalid') from exc
    constraint, details = value['constraint'], value['details']
    for name in ('joint_index',):
        if name in constraint and _index(constraint[name], name) >= 6:
            raise DiagnosticUnavailable('admission_joint_index_outside_arm')
    quantity = constraint.get('quantity')
    special = set()
    if quantity == 'collision_free':
        if type(constraint.get('observed')) is not bool or type(constraint.get('bound')) is not bool:
            raise DiagnosticUnavailable('admission_collision_constraint_invalid')
        special = {'observed', 'bound'}
    elif quantity == 'eef_position':
        for name in ('observed', 'lower_bound', 'upper_bound'):
            _vector(constraint.get(name), 3, 'admission_' + name)
        special = {'observed', 'lower_bound', 'upper_bound'}
    for name in ('observed', 'signed_value', 'bound', 'margin', 'value', 'limit', 'lower_bound', 'upper_bound'):
        if name in special:
            continue
        if name in constraint:
            _number(constraint[name], 'admission_' + name)
    if not special and {'observed', 'bound', 'margin'} <= constraint.keys():
        if constraint['margin'] != constraint['bound'] - constraint['observed']:
            raise DiagnosticUnavailable('admission_constraint_margin_mismatch')
    if {'observed', 'signed_value'} <= constraint.keys() and constraint['observed'] != abs(constraint['signed_value']):
        raise DiagnosticUnavailable('admission_constraint_observed_mismatch')
    for name in ('estimate_index', 'bridge_before_key_index', 'index', 'sample_index', 'source_index', 'violation_count'):
        if name in details:
            _index(details[name], 'admission_' + name)
    if 'derivative_order' in details and _index(details['derivative_order'], 'derivative_order') not in (1, 2, 3):
        raise DiagnosticUnavailable('admission_derivative_order_invalid')
    for name in ('dt_s', 'duration_s', 'peak_u', 'peak_t_s', 'bridge_duration_s'):
        if name in details:
            _number(details[name], 'admission_' + name)
    for name in ('joint_rad', 'actual_joint_rad', 'reference_joint_rad'):
        if name in details:
            _vector(details[name], 6, 'admission_' + name)
    for name in ('requested_xyz_m', 'actual_fk_xyz_m'):
        if name in details:
            _vector(details[name], 3, 'admission_' + name)
    if 'source_points' in details:
        points = details['source_points']
        if type(points) is not list or not 1 <= len(points) <= 8:
            raise DiagnosticUnavailable('admission_source_window_invalid')
        for point in points:
            _sample(point)
    if 'input_t_s' in details:
        times = _vector(details['input_t_s'], 2, 'admission_input_times')
        if 'dt_s' in details and details['dt_s'] != times[1] - times[0]:
            raise DiagnosticUnavailable('admission_dt_mismatch')
    if 'input_values' in details:
        values = details['input_values']
        if type(values) is not list or len(values) != 2:
            raise DiagnosticUnavailable('admission_input_values_invalid')
        for vector in values:
            _vector(vector, 6, 'admission_input_values')
    for name in ('joint_coefficients_normalized_time', 'derivative_coefficients_normalized_time'):
        if name in details:
            coefficients = details[name]
            if type(coefficients) is not list or not 1 <= len(coefficients) <= 6:
                raise DiagnosticUnavailable('admission_coefficients_invalid')
            for number in coefficients:
                _number(number, 'admission_coefficient')
    return value


def _request(receipt):
    method = _text(receipt.get('method'), 'admission_method', 128)
    if not method:
        raise DiagnosticUnavailable('admission_method_empty')
    arguments = _object(receipt.get('arguments'), 'admission_arguments')
    value = dict(method=method, arguments=arguments)
    raw = _canonical(value)
    if len(raw) > 16 * 1024:
        raise DiagnosticUnavailable('admission_request_size_limit')
    return decode_json(raw)


def _no_dispatch_claim(receipt):
    if receipt.get('status') != 'rejected' or receipt.get('motion_dispatch_attempted') is not False:
        raise DiagnosticUnavailable('admission_zero_dispatch_unverified')
    replay = receipt.get('replay', {})
    if type(replay) is not dict:
        raise DiagnosticUnavailable('admission_replay_invalid')
    if replay and replay.get('status') not in ('rejected', 'not_started'):
        raise DiagnosticUnavailable('admission_replay_outcome_not_not_started')
    for source in (receipt, replay):
        for name in ('dispatch_attempted_count', 'dispatched_count'):
            if name in source and (type(source[name]) is not int or source[name] != 0):
                raise DiagnosticUnavailable('admission_dispatch_counter_not_zero')
        if 'motion_outcome_uncertain' in source and source['motion_outcome_uncertain'] is not False:
            raise DiagnosticUnavailable('admission_dispatch_outcome_uncertain')


def _tool_boundary(receipt, receipts, position):
    tool_index = _index(receipt.get('tool_index'), 'admission_tool_index')
    previous = []
    for row in receipts[max(0, position - 4):position]:
        if type(row) is not dict:
            raise DiagnosticUnavailable('admission_previous_receipt_invalid')
        previous.append({name: row[name] for name in ('tool_index', 'method', 'status') if name in row})
    return dict(failed_tool_index=tool_index, receipt_position=position, preceding_receipt_count=position,
                preceding_receipts=previous,
                meaning='failed request only; earlier tools may have dispatched motion')


def _declared_identity(result, receipt, request):
    contract = _object(receipt.get('execution_contract', {}), 'execution_contract')
    identity = dict(request_sha256=_sha(request), run_id=None)
    for name, source in (('source_sha256', result), ('profile_sha256', result),
                         ('policy_sha256', contract), ('catalog_sha256', contract)):
        identity[name] = None if source.get(name) is None else _digest(source[name], name)
    return identity


def _diagnose_admission(files, receipt, result, receipts, position):
    _no_dispatch_claim(receipt)
    admission = _admission_facts(receipt.get('admission_rejection'))
    skill_dir = files.relative(result.get('skill_dir'))
    tool_index = _index(receipt.get('tool_index'), 'admission_tool_index')
    artifact_path = files.relative(receipt.get('admission_artifact_path'))
    if artifact_path != skill_dir + '/admission-%03d.json' % tool_index:
        raise DiagnosticUnavailable('admission_artifact_tool_path_mismatch')
    document, reference = files.read_json(receipt.get('admission_artifact_path'),
        _digest(receipt.get('admission_artifact_sha256'), 'admission_artifact_sha256'))
    required = {'protocol', 'run_id', 'source_sha256', 'profile_sha256', 'policy_sha256',
                'catalog_sha256', 'tool_index', 'method', 'arguments', 'request_sha256', 'admission'}
    if type(document) is not dict or set(document) != required or document['protocol'] != ADMISSION_EVIDENCE_PROTOCOL:
        raise DiagnosticUnavailable('admission_artifact_schema_invalid')
    request = _request(receipt)
    identity = _declared_identity(result, receipt, request)
    for name in ('source_sha256', 'profile_sha256'):
        _digest(identity[name], name)
    if _canonical({key: document[key] for key in ('method', 'arguments')}) != _canonical(request):
        raise DiagnosticUnavailable('admission_request_identity_mismatch')
    if _digest(document['request_sha256'], 'request_sha256') != identity['request_sha256']:
        raise DiagnosticUnavailable('admission_request_sha256_mismatch')
    if _canonical(_admission_facts(document['admission'])) != _canonical(admission):
        raise DiagnosticUnavailable('admission_artifact_rejection_mismatch')
    if _index(document['tool_index'], 'admission_tool_index') != _index(receipt.get('tool_index'), 'admission_tool_index'):
        raise DiagnosticUnavailable('admission_tool_index_mismatch')
    for name in ('source_sha256', 'profile_sha256', 'policy_sha256', 'catalog_sha256'):
        if document[name] != identity[name]:
            raise DiagnosticUnavailable('admission_' + name + '_mismatch')
    execution_contract = result.get('trajectory_execution')
    if execution_contract is not None:
        execution_contract = _object(execution_contract, 'trajectory_execution')
        for name in ('policy_sha256', 'catalog_sha256'):
            if execution_contract.get(name) != identity[name]:
                raise DiagnosticUnavailable('admission_execution_' + name + '_mismatch')
    if receipt.get('execution_contract', {}).get('command') == 'original_native_joint_raw':
        for name in ('policy_sha256', 'catalog_sha256'):
            _digest(identity[name], name)
    start, start_ref = files.read_json('start.json')
    run_id = _text(document['run_id'], 'admission_run_id', 512)
    if not run_id or _object(start, 'run_start').get('run_id') != run_id:
        raise DiagnosticUnavailable('admission_run_id_mismatch')
    identity['run_id'] = run_id
    _, source_ref = files.read_bytes(skill_dir + '/skill.py', expected_sha=identity['source_sha256'])
    profile, profile_ref = files.read_json(skill_dir + '/profile.json', expected_sha=identity['profile_sha256'])
    _object(profile, 'admission_profile')
    return dict(protocol=ADMISSION_DIAGNOSTIC_PROTOCOL, status='diagnostic_available',
        evidence_kind='typed_admission', identity=identity, admission=admission, request=request,
        receipt_summary=_receipt_summary(receipt, receipt.get('replay', {})),
        failure=dict(stage=admission['failure_stage'], reason_code=admission['reason_code'],
            message=admission['message'], dispatch_state='not_started', scope='failed_tool_request_only',
            physical_cause='unknown', present_pose_and_release='unknown'),
        tool_boundary=_tool_boundary(receipt, receipts, position),
        file_refs=dict(admission=reference, skill_source=source_ref, profile=profile_ref, run_start=start_ref),
        current_state='unknown', motion_available=False, recovery_authority=False, issues=[])


def _diagnose_untyped(receipt, result, receipts, position):
    request = _request(receipt)
    message = _text(receipt.get('error'), 'receipt_error')
    return dict(protocol=ADMISSION_DIAGNOSTIC_PROTOCOL, status='diagnostic_available',
        evidence_kind='untyped_historical_reference', identity=_declared_identity(result, receipt, request),
        admission=None, request=request, receipt_summary=_receipt_summary(receipt, receipt.get('replay', {})),
        failure=dict(stage='untyped', reason_code=None, message=message, dispatch_state='unverified',
            scope='failed_tool_request_only', physical_cause='unknown', present_pose_and_release='unknown'),
        tool_boundary=_tool_boundary(receipt, receipts, position), file_refs={},
        current_state='unknown', motion_available=False, recovery_authority=False,
        issues=['typed_admission_facts_unavailable', 'dispatch_state_unverified', 'precise_plan_unavailable'])


def validate_admission_diagnostic(value):
    """Validate an SDK-free reader's admission summary; never grant authority."""
    value = _object(value, 'admission_diagnostic')
    required = {'protocol', 'status', 'evidence_kind', 'identity', 'admission', 'request',
                'receipt_summary', 'failure', 'tool_boundary', 'file_refs', 'current_state',
                'motion_available', 'recovery_authority', 'issues'}
    if (set(value) != required or value.get('protocol') != ADMISSION_DIAGNOSTIC_PROTOCOL or value.get('status') != 'diagnostic_available'
            or value.get('evidence_kind') not in ('typed_admission', 'untyped_historical_reference')
            or value.get('current_state') != 'unknown' or value.get('motion_available') is not False
            or value.get('recovery_authority') is not False):
        raise DiagnosticUnavailable('admission_diagnostic_schema_invalid')
    identity = _object(value.get('identity'), 'admission_identity')
    if set(identity) != {'request_sha256', 'run_id', 'source_sha256', 'profile_sha256', 'policy_sha256', 'catalog_sha256'}:
        raise DiagnosticUnavailable('admission_diagnostic_identity_invalid')
    for name in ('source_sha256', 'profile_sha256', 'policy_sha256', 'catalog_sha256'):
        if identity[name] is not None:
            _digest(identity[name], name)
    if type(value.get('request')) is not dict or set(value['request']) != {'method', 'arguments'}:
        raise DiagnosticUnavailable('admission_diagnostic_request_invalid')
    request = _request(value.get('request'))
    if _digest(identity.get('request_sha256'), 'request_sha256') != _sha(request):
        raise DiagnosticUnavailable('admission_diagnostic_request_mismatch')
    failure = _object(value.get('failure'), 'admission_failure')
    _text(failure.get('message'), 'admission_failure_message')
    if (failure.get('scope') != 'failed_tool_request_only' or failure.get('physical_cause') != 'unknown'
            or failure.get('present_pose_and_release') != 'unknown'):
        raise DiagnosticUnavailable('admission_diagnostic_authority_invalid')
    boundary = _object(value.get('tool_boundary'), 'admission_tool_boundary')
    _index(boundary.get('failed_tool_index'), 'failed_tool_index')
    for name in ('receipt_position', 'preceding_receipt_count'):
        _index(boundary.get(name), name)
    if boundary['receipt_position'] != boundary['preceding_receipt_count']:
        raise DiagnosticUnavailable('admission_diagnostic_receipt_position_mismatch')
    previous = boundary.get('preceding_receipts')
    if type(previous) is not list or len(previous) > min(4, boundary['preceding_receipt_count']):
        raise DiagnosticUnavailable('admission_diagnostic_previous_receipts_invalid')
    if type(value.get('issues')) is not list or any(type(issue) is not str for issue in value['issues']):
        raise DiagnosticUnavailable('admission_diagnostic_issues_invalid')
    summary = _object(value.get('receipt_summary'), 'admission_receipt_summary')
    if summary.get('method') != request['method'] or summary.get('tool_index') != boundary['failed_tool_index']:
        raise DiagnosticUnavailable('admission_diagnostic_boundary_mismatch')
    if value['evidence_kind'] == 'typed_admission':
        _no_dispatch_claim(summary)
        admission = _admission_facts(value.get('admission'))
        if any(failure.get(name) != expected for name, expected in (
                ('stage', admission['failure_stage']), ('reason_code', admission['reason_code']),
                ('message', admission['message']), ('dispatch_state', 'not_started'))):
            raise DiagnosticUnavailable('admission_diagnostic_facts_mismatch')
        refs = _object(value.get('file_refs'), 'admission_file_refs')
        if set(refs) != {'admission', 'skill_source', 'profile', 'run_start'}:
            raise DiagnosticUnavailable('admission_diagnostic_file_refs_invalid')
        if not _text(identity.get('run_id'), 'admission_run_id', 512):
            raise DiagnosticUnavailable('admission_run_id_empty')
        for name in ('admission', 'skill_source', 'profile', 'run_start'):
            ref = _object(refs.get(name), 'admission_file_ref')
            _digest(ref.get('file_sha256'), 'admission_file_sha256')
            path = Path(_text(ref.get('relative_path'), 'admission_relative_path'))
            if not path.parts or path.is_absolute() or '..' in path.parts:
                raise DiagnosticUnavailable('admission_diagnostic_reference_path_invalid')
            _index(ref.get('bytes'), 'admission_file_bytes')
        for name, ref_name in (('source_sha256', 'skill_source'), ('profile_sha256', 'profile')):
            if _digest(identity.get(name), name) != refs[ref_name]['file_sha256']:
                raise DiagnosticUnavailable('admission_diagnostic_source_mismatch')
    elif (value.get('admission') is not None or failure.get('dispatch_state') != 'unverified'
            or failure.get('stage') != 'untyped' or failure.get('reason_code') is not None
            or failure.get('message') != summary.get('error') or value.get('file_refs') != {}):
        raise DiagnosticUnavailable('untyped_diagnostic_cannot_assert_dispatch')
    if len(_canonical(value)) > MAX_OUTPUT_BYTES:
        raise DiagnosticUnavailable('admission_diagnostic_size_limit')
    return value


def _find_plan(files, asset, replay, receipt):
    matches, total = [], 0
    for path in files.plan_paths():
        plan, reference = files.read_json(path)
        total += reference['bytes']
        if total > MAX_PLAN_BYTES:
            raise DiagnosticUnavailable('native_plan_total_bytes_limit')
        plan = _object(plan, 'joint_plan')
        if plan.get('trajectory_id') != asset['trajectory_id'] or plan.get('effective_sha256') != replay['effective_sha256']:
            continue
        if plan.get('protocol') != PLAN_PROTOCOL:
            raise DiagnosticUnavailable('native_plan_protocol_mismatch')
        for key in ('source_sha256',):
            if key in replay and plan.get(key) != replay[key]:
                raise DiagnosticUnavailable('native_plan_' + key + '_mismatch')
        contract = receipt.get('execution_contract', {})
        for key in ('catalog_sha256', 'policy_sha256'):
            expected = contract.get(key) if type(contract) is dict else None
            if expected is not None and plan.get(key) != expected:
                raise DiagnosticUnavailable('native_plan_' + key + '_mismatch')
        reference['canonical_sha256'] = _sha(plan)
        matches.append((plan, reference))
    if len(matches) != 1:
        raise DiagnosticUnavailable('native_plan_ambiguous' if matches else 'native_plan_identity_not_found')
    plan, reference = matches[0]
    points = plan.get('samples')
    if type(points) is not list or len(points) != len(asset['samples']):
        raise DiagnosticUnavailable('native_plan_sample_count_mismatch')
    return plan, reference


def _event(event, line):
    result = dict(line=line, type=event['type'])
    if 'index' in event:
        result['index'] = _index(event['index'], 'audit_index')
    for key in ('anchor',):
        if key in event:
            result[key] = _anchor(event[key])
    if 'error' in event:
        result['error'] = _text(event['error'], 'audit_error')
    if 'source_t_s' in event:
        result['source_t_s'] = _number(event['source_t_s'], 'audit_source_time')
    if 'target_joint_rad' in event:
        result['target_joint_rad'] = _vector(event['target_joint_rad'], 6, 'target_joint')
    if 'actual_before' in event:
        actual = _object(event['actual_before'], 'actual_before')
        arms = _object(actual.get('arms'), 'actual_arms')
        result['actual_before'] = dict(arms={})
        for arm in ('left', 'right'):
            if arm in arms:
                result['actual_before']['arms'][arm] = _sample(dict(t_s=0, **arms[arm]))
                del result['actual_before']['arms'][arm]['t_s']
        if 'sample_anchor' in actual:
            result['actual_before']['sample_anchor'] = _anchor(actual['sample_anchor'])
        if 'feedback_meaning' in actual:
            result['actual_before']['feedback_meaning'] = _text(actual['feedback_meaning'], 'feedback_meaning', 1024)
    return result


class _AuditIdentityScanner:
    """Validate one oversized JSON object while retaining only top-level IDs.

    The large payload strings (notably RPC JPEG base64) are never accumulated.
    UTF-8, JSON grammar/escapes, duplicate object keys, and trailing content are
    checked across arbitrary chunk boundaries. Depth, keys, retained key text,
    and scalar/identity token lengths have independent fixed budgets.
    """
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.stack, self.identity = [], {}
        self.started = self.finished = False
        self.mode, self.role, self.capture = None, None, False
        self.parts, self.token_size, self.unicode_left = [], 0, 0
        self.key_count = self.key_chars = 0

    def _invalid(self):
        raise DiagnosticUnavailable('invalid_json')

    def _budget(self):
        raise DiagnosticUnavailable('audit_long_record_parse_budget')

    def _append(self, text):
        if self.capture:
            self.token_size += len(text)
            if self.token_size > _AUDIT_SCAN_MAX_TOKEN_CHARS:
                self._budget()
            self.parts.append(text)

    def _done(self, value=None):
        if not self.stack:
            self.finished = True
            return
        parent = self.stack[-1]
        if parent['kind'] == 'object':
            if parent['state'] != 'value':
                self._invalid()
            if len(self.stack) == 1 and parent['key'] in ('type', 'plan_sha256'):
                self.identity[parent['key']] = value
        parent['state'] = 'comma_or_end'

    def _string_done(self):
        value = json.loads('"' + ''.join(self.parts) + '"') if self.capture else None
        if self.role == 'key':
            parent = self.stack[-1]
            self.key_count += 1
            self.key_chars += len(value)
            if self.key_count > _AUDIT_SCAN_MAX_KEYS or self.key_chars > _AUDIT_SCAN_MAX_KEY_CHARS:
                self._budget()
            if value in parent['keys']:
                self._invalid()
            parent['keys'].add(value)
            parent.update(key=value, state='colon')
        else:
            self._done(value)
        self.mode, self.parts = None, []

    def _scalar_done(self):
        token = ''.join(self.parts)
        if token in ('true', 'false', 'null'):
            value = json.loads(token)
        elif _JSON_NUMBER.fullmatch(token):
            value = json.loads(token)
            if type(value) is float and not math.isfinite(value):
                self._invalid()
        else:
            self._invalid()
        self.mode, self.parts = None, []
        self._done(value)

    def _value(self, char):
        if not self.stack:
            if self.started or char != '{':
                self._invalid()
            self.started = True
        if char in '{[':
            if len(self.stack) >= _AUDIT_SCAN_MAX_DEPTH:
                self._budget()
            self.stack.append(dict(kind='object' if char == '{' else 'array',
                state='key_or_end' if char == '{' else 'value_or_end', keys=set(), key=None))
        elif char == '"':
            self.mode, self.role = 'string', 'value'
            parent = self.stack[-1]
            self.capture = len(self.stack) == 1 and parent['kind'] == 'object' and parent['key'] in ('type', 'plan_sha256')
            self.parts, self.token_size = [], 0
        elif char in '-0123456789tfn':
            self.mode, self.capture, self.parts, self.token_size = 'scalar', True, [], 0
            self._append(char)
        else:
            self._invalid()

    def _text(self, text):
        index = 0
        while index < len(text):
            char = text[index]
            if self.mode == 'string':
                match = _JSON_STRING_SPECIAL.search(text, index)
                end = len(text) if match is None else match.start()
                self._append(text[index:end])
                index = end
                if match is None:
                    return
                char = text[index]
                if char == '"':
                    self._string_done()
                elif char == '\\':
                    self._append(char)
                    self.mode = 'escape'
                else:
                    self._invalid()
            elif self.mode == 'escape':
                self._append(char)
                if char == 'u':
                    self.mode, self.unicode_left = 'unicode', 4
                elif char in '"\\/bfnrt':
                    self.mode = 'string'
                else:
                    self._invalid()
            elif self.mode == 'unicode':
                if char not in '0123456789abcdefABCDEF':
                    self._invalid()
                self._append(char)
                self.unicode_left -= 1
                if not self.unicode_left:
                    self.mode = 'string'
            elif self.mode == 'scalar':
                if char in ' \t\r\n,]}':
                    self._scalar_done()
                    continue  # The delimiter still belongs to the parent.
                self._append(char)
            elif char in ' \t\r\n':
                pass
            elif self.finished:
                self._invalid()
            elif not self.stack:
                self._value(char)
            else:
                parent = self.stack[-1]
                state, kind = parent['state'], parent['kind']
                if state in ('key_or_end', 'key'):
                    if char == '}' and state == 'key_or_end':
                        self.stack.pop(); self._done()
                    elif char == '"':
                        self.mode, self.role, self.capture = 'string', 'key', True
                        self.parts, self.token_size = [], 0
                    else:
                        self._invalid()
                elif state == 'colon':
                    if char != ':':
                        self._invalid()
                    parent['state'] = 'value'
                elif state == 'value_or_end' and char == ']':
                    self.stack.pop(); self._done()
                elif state in ('value', 'value_or_end'):
                    self._value(char)
                elif state == 'comma_or_end':
                    if char == ',':
                        parent['state'] = 'key' if kind == 'object' else 'value'
                    elif char == ('}' if kind == 'object' else ']'):
                        self.stack.pop(); self._done()
                    else:
                        self._invalid()
                else:
                    self._invalid()
            index += 1

    def feed(self, raw):
        try:
            self._text(self.decoder.decode(raw, final=False))
        except (UnicodeError, json.JSONDecodeError, OverflowError, RecursionError) as exc:
            raise DiagnosticUnavailable('invalid_json') from exc

    def finish(self):
        try:
            self._text(self.decoder.decode(b'', final=True))
        except UnicodeError as exc:
            raise DiagnosticUnavailable('invalid_json') from exc
        if self.mode == 'scalar':
            self._scalar_done()
        if not self.finished or self.stack or self.mode is not None:
            self._invalid()
        return self.identity


def _read_audit(files, plan_ref):
    descriptor, relative, before = files.open('audit.jsonl')
    plan_sha = plan_ref['canonical_sha256']
    digest, total, lines = hashlib.sha256(), 0, 0
    prepared, failures, previous = [], [], deque(maxlen=12)
    last_dispatched = last_dispatching = last_check = None
    counts, sent_count, contiguous = {}, 0, True
    events_after_failure = False
    oversized_skipped, maximum_record_bytes = 0, 0

    def consume(raw):
        nonlocal total
        total += len(raw)
        if total > MAX_AUDIT_BYTES:
            raise DiagnosticUnavailable('audit_size_limit')
        digest.update(raw)
    try:
        if before.st_size > MAX_AUDIT_BYTES:
            raise DiagnosticUnavailable('audit_size_limit')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            while True:
                raw = stream.readline(MAX_AUDIT_LINE_BYTES + 1)
                if not raw:
                    break
                lines += 1
                consume(raw)
                record_bytes = len(raw)
                if record_bytes > MAX_AUDIT_LINE_BYTES:
                    scanner = _AuditIdentityScanner()
                    scanner.feed(raw)
                    while not raw.endswith(b'\n'):
                        raw = stream.readline(_AUDIT_SCAN_CHUNK_BYTES)
                        if not raw:
                            break
                        consume(raw)
                        record_bytes += len(raw)
                        scanner.feed(raw)
                    identity = scanner.finish()
                    maximum_record_bytes = max(maximum_record_bytes, record_bytes)
                    kind, row_plan = identity.get('type'), identity.get('plan_sha256')
                    if type(kind) is not str:
                        raise DiagnosticUnavailable('audit_oversized_record_identity_unavailable')
                    if kind in _NATIVE_AUDIT_EVENTS:
                        if type(row_plan) is not str or _SHA.fullmatch(row_plan) is None:
                            raise DiagnosticUnavailable('audit_oversized_record_identity_unavailable')
                        if row_plan == plan_sha:
                            raise DiagnosticUnavailable('audit_target_event_size_limit')
                    oversized_skipped += 1
                    continue
                maximum_record_bytes = max(maximum_record_bytes, record_bytes)
                row = _object(decode_json(raw), 'audit_row')
                if row.get('plan_sha256') != plan_sha:
                    continue
                kind = row.get('type')
                if kind == 'native_timed_prepared':
                    prepared.append(files.relative(row.get('directory')))
                    if len(prepared) > 1:
                        raise DiagnosticUnavailable('audit_prepared_plan_ambiguous')
                    continue
                if kind not in ('native_timed_dispatch_check', 'native_timed_dispatching',
                                'native_timed_dispatched', 'native_timed_dispatch_failed'):
                    continue
                item = _event(row, lines)
                counts[kind] = counts.get(kind, 0) + 1
                if failures:
                    events_after_failure = True
                if kind == 'native_timed_dispatch_failed':
                    failures.append(dict(event=item, preceding_events=list(previous),
                        check=last_check, last_dispatched=last_dispatched,
                        dispatching=last_dispatching, sent_count=sent_count, contiguous=contiguous))
                    if len(failures) > 1:
                        raise DiagnosticUnavailable('audit_failure_ambiguous')
                elif kind == 'native_timed_dispatch_check':
                    last_check = item
                elif kind == 'native_timed_dispatching':
                    last_dispatching = item
                elif kind == 'native_timed_dispatched':
                    contiguous = contiguous and item['index'] == sent_count
                    sent_count += 1
                    last_dispatched = item
                previous.append(item)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or total != after.st_size:
            raise DiagnosticUnavailable('audit_changed_during_read')
    finally:
        os.close(descriptor)
    expected_directory = str(Path(plan_ref['relative_path']).parent)
    if prepared != [expected_directory]:
        raise DiagnosticUnavailable('audit_prepared_plan_identity_unverified')
    if len(failures) != 1 or events_after_failure:
        raise DiagnosticUnavailable('audit_failure_missing_or_nonterminal')
    reference = dict(relative_path=relative, file_sha256=digest.hexdigest(), bytes=total, lines=lines,
                     matched_plan_sha256=plan_sha, matched_event_counts=counts,
                     oversized_unrelated_records_skipped=oversized_skipped,
                     maximum_record_bytes=maximum_record_bytes)
    return failures[0], reference


def _source_location(asset, index, replay):
    original_index = index + _index(replay.get('window_start_index', 0), 'window_start_index')
    sequence = asset.get('sequence')
    if type(sequence) is not dict:
        crop = asset.get('crop', {})
        return dict(kind='source_fragment', key=asset.get('key'), trajectory_id=asset['trajectory_id'],
                    source_index=original_index, crop=crop if len(_canonical(crop)) <= 4096 else None)
    mapping = sequence.get('source_point_map')
    clips = asset.get('crop', {}).get('clips')
    bridges = sequence.get('bridges')
    if type(mapping) is not list or not 0 <= original_index < len(mapping) or type(clips) is not list or type(bridges) is not list:
        raise DiagnosticUnavailable('sequence_source_mapping_unavailable')
    mapped = mapping[original_index]
    if mapped is not None:
        mapped = _object(mapped, 'source_point_map')
        clip_index = _index(mapped.get('clip_index'), 'clip_index')
        if clip_index >= len(clips):
            raise DiagnosticUnavailable('source_clip_index_invalid')
        clip = _object(clips[clip_index], 'source_clip')
        start = _index(clip.get('program_start_index'), 'program_start_index')
        end = _index(clip.get('program_end_index'), 'program_end_index')
        source_index = _index(mapped.get('source_index'), 'source_index')
        if (not start <= original_index <= end or mapped.get('component_key') != clip.get('key')
                or source_index != clip.get('source_start_index', -1) + original_index - start
                or source_index > clip.get('source_end_index', -1)):
            raise DiagnosticUnavailable('source_point_map_clip_mismatch')
        return dict(kind='source_fragment', program_index=original_index, clip_index=clip_index,
                    key=mapped['component_key'], source_index=source_index,
                    trajectory_id=clip.get('trajectory_id'), clip=clip)
    matches = [bridge for bridge in bridges if type(bridge) is dict
               and type(bridge.get('program_start_index')) is int
               and type(bridge.get('program_end_index')) is int
               and bridge['program_start_index'] < original_index < bridge['program_end_index']]
    if len(matches) != 1:
        raise DiagnosticUnavailable('generated_bridge_mapping_ambiguous')
    bridge = matches[0]
    fields = ('from_key', 'to_key', 'program_start_index', 'program_end_index', 'start_t_s', 'duration_s',
              'generated_interior_count', 'requested_duration_s', 'kind')
    return dict(kind='generated_bridge', program_index=original_index, source_index=None,
                bridge={key: bridge[key] for key in fields if key in bridge})


def _diagnose(files, receipt):
    replay = _object(receipt.get('replay'), 'replay')
    summary = _receipt_summary(receipt, replay)
    asset, asset_ref = files.read_json(receipt.get('effective_asset_path'),
        _digest(receipt.get('effective_asset_file_sha256'), 'effective_asset_file_sha256'))
    asset = _object(asset, 'effective_asset')
    trajectory_id = _text(asset.get('trajectory_id'), 'trajectory_id', 512)
    effective_sha = _digest(replay.get('effective_sha256'), 'effective_sha256')
    if replay.get('trajectory_id') != trajectory_id or _sha(asset) != effective_sha:
        raise DiagnosticUnavailable('effective_asset_identity_mismatch')
    points = asset.get('samples')
    if type(points) is not list or not 0 < len(points) <= MAX_POINTS or replay.get('point_count') != len(points):
        raise DiagnosticUnavailable('effective_asset_sample_count_invalid')
    refs = dict(effective_asset=asset_ref)
    sequence = receipt.get('sequence')
    if sequence is not None:
        sequence = _object(sequence, 'receipt_sequence')
        metadata, metadata_ref = files.read_json(sequence.get('metadata_path'),
            _digest(sequence.get('metadata_sha256'), 'metadata_sha256'))
        metadata = _object(metadata, 'sequence_metadata')
        for key in ('trajectory_id', 'program_sha256', 'sample_count'):
            if metadata.get(key) != sequence.get(key):
                raise DiagnosticUnavailable('sequence_metadata_identity_mismatch')
        if metadata.get('trajectory_id') != trajectory_id or metadata.get('program_sha256') != replay.get('source_sha256'):
            raise DiagnosticUnavailable('sequence_metadata_source_mismatch')
        refs['sequence_metadata'] = metadata_ref
    elif type(asset.get('sequence')) is dict:
        raise DiagnosticUnavailable('sequence_metadata_reference_missing')
    plan, plan_ref = _find_plan(files, asset, replay, receipt)
    refs['joint_plan'] = plan_ref
    audit, audit_ref = _read_audit(files, plan_ref)
    refs['audit'] = audit_ref
    failed = audit['event']
    index = failed['index']
    if index >= len(points):
        raise DiagnosticUnavailable('failure_index_outside_asset')
    check = audit['check']
    if check is None or check['index'] != index:
        raise DiagnosticUnavailable('failure_check_missing_or_mismatched')
    sample, native_sample = _sample(points[index]), _sample(plan['samples'][index])
    if sample['t_s'] != native_sample['t_s'] or check.get('source_t_s') != native_sample['t_s']:
        raise DiagnosticUnavailable('failure_sample_time_mismatch')
    dispatching = audit['dispatching']
    attempted_send = dispatching is not None and dispatching['index'] == index
    last = audit['last_dispatched']
    last_index = None if last is None else last['index']
    if (last_index is not None and last_index >= index) or not audit['contiguous']:
        raise DiagnosticUnavailable('audit_dispatched_order_inconsistent')
    if replay.get('dispatched_count') != audit['sent_count']:
        raise DiagnosticUnavailable('receipt_audit_dispatch_count_mismatch')
    if replay.get('dispatch_attempted_count') != audit['sent_count'] + 1:
        raise DiagnosticUnavailable('receipt_audit_attempt_count_mismatch')
    if not attempted_send and last_index != (index - 1 if index else None):
        raise DiagnosticUnavailable('failure_previous_dispatch_unverified')
    if dispatching is not None:
        target_index = index if attempted_send else last_index
        if target_index is None or dispatching['index'] != target_index:
            raise DiagnosticUnavailable('last_dispatching_index_mismatch')
        target = _sample(plan['samples'][target_index])
        if (dispatching.get('target_joint_rad') != target['joint_rad']
                or dispatching.get('source_t_s') != target['t_s']):
            raise DiagnosticUnavailable('audit_target_plan_mismatch')
    elif last_index is not None:
        raise DiagnosticUnavailable('last_dispatching_target_missing')
    reference_index = max(0, index - 1)
    reference = _sample(plan['samples'][reference_index])
    actual = check.get('actual_before', {})
    actual_joint = actual.get('arms', {}).get('left', {}).get('joint_rad')
    if actual_joint is None:
        raise DiagnosticUnavailable('failure_actual_joint_missing')
    delta = [a - b for a, b in zip(actual_joint, reference['joint_rad'])]
    timing = dict(failure_anchor=failed.get('anchor'), check_anchor=check.get('anchor'),
                  actual_sample_anchor=actual.get('sample_anchor'),
                  last_dispatched_anchor=None if last is None else last.get('anchor'),
                  source_t_s=native_sample['t_s'])
    pairs = (('actual_sample_to_check_s', actual.get('sample_anchor'), check.get('anchor')),
             ('last_dispatch_return_to_check_s', None if last is None else last.get('anchor'), check.get('anchor')))
    for label, before, after in pairs:
        if before is not None and after is not None and 'monotonic_ns' in before and 'monotonic_ns' in after:
            timing[label] = (after['monotonic_ns'] - before['monotonic_ns']) / 1e9
    tolerance_match = re.search(r'\btolerance_rad=([0-9.eE+\-]+)', failed.get('error', ''))
    tolerance = None if tolerance_match is None else _number(float(tolerance_match[1]), 'tolerance_rad')
    window = [dict(index=i, effective_sample=_sample(points[i]), native_target=_sample(plan['samples'][i]),
                   source_location=_source_location(asset, i, replay))
              for i in range(max(0, index - 2), min(len(points), index + 3))]
    return dict(status='diagnostic_available', receipt_summary=summary,
        identity=dict(trajectory_id=trajectory_id, effective_sha256=effective_sha,
                      plan_sha256=plan_ref['canonical_sha256']), file_refs=refs,
        failure=dict(index=index, source_t_s=native_sample['t_s'], event=failed,
                     point_rejected_before_send=not attempted_send,
                     raw_send_attempt_observed=attempted_send,
                     physical_cause='unknown', present_pose_and_release='unknown'),
        source_location=_source_location(asset, index, replay), sample_window=window,
        last_dispatched=dict(index=last_index, event=last,
            native_target=None if last_index is None else _sample(plan['samples'][last_index])),
        reference=dict(index=reference_index, native_target=reference,
                       meaning='previous target used by native dispatch precheck; target calls are not arrival evidence'),
        actual_at_check=actual, actual_delta=dict(actual_minus_reference_joint_rad=delta,
            abs_joint_error_rad=[abs(value) for value in delta], max_abs_joint_error_rad=max(abs(value) for value in delta),
            tolerance_rad_from_error=tolerance), timing=timing,
        audit_context=audit['preceding_events'], issues=[])


def unavailable(reason):
    """Small JSON envelope used when even input/root validation cannot finish."""
    return dict(protocol=PROTOCOL, status='diagnostic_unavailable', issues=[reason], receipts=[],
                meaning='saved historical facts only; physical cause, current pose and release remain unknown')


def encode_diagnostics(value):
    """Return complete bounded JSON; never silently crop identity/failure fields."""
    raw = _canonical(value) + b'\n'
    if len(raw) > MAX_OUTPUT_BYTES:
        return _canonical(unavailable('diagnostics_output_size_limit')) + b'\n'
    return raw


def collect_saved_diagnostics(run_root, execution):
    """Read a saved failure response and files under ``run_root`` without writes.

    A successful return may explicitly be ``diagnostic_unavailable``. Ambiguous
    plans/events, malformed or missing files, and size limits never select a
    plausible last event or infer a root cause.
    """
    files = None
    try:
        result = _unwrap(execution)
        receipts = result.get('receipts')
        if type(receipts) is not list or len(receipts) > 256:
            raise DiagnosticUnavailable('execution_receipt_count_invalid')
        admission_fields = {'admission_rejection', 'admission_artifact_path', 'admission_artifact_sha256'}
        candidates = [(position, r) for position, r in enumerate(receipts) if type(r) is dict and (
            admission_fields & r.keys() or
            'effective_asset_path' in r and (r.get('status') in ('unknown', 'rejected', 'failed')
                or type(r.get('replay')) is dict and r['replay'].get('status') != 'ok') or
            r.get('status') == 'rejected' and {'method', 'arguments', 'error'} <= r.keys())]
        if not candidates or len(candidates) > 8:
            raise DiagnosticUnavailable('failed_trajectory_receipt_missing_or_count_limit')
        files = _RunFiles(run_root)
        items = []
        for position, receipt in candidates:
            try:
                if admission_fields & receipt.keys():
                    item = _diagnose_admission(files, receipt, result, receipts, position)
                    validate_admission_diagnostic(item)
                elif 'effective_asset_path' in receipt:
                    item = _diagnose(files, receipt)
                else:
                    item = _diagnose_untyped(receipt, result, receipts, position)
                    validate_admission_diagnostic(item)
                items.append(item)
            except (DiagnosticUnavailable, TypeError, KeyError, ValueError, OverflowError, RecursionError) as exc:
                reason = str(exc) if isinstance(exc, DiagnosticUnavailable) else 'malformed_saved_diagnostic_fields'
                try:
                    summary = _receipt_summary(receipt, _object(receipt.get('replay', {}), 'replay'))
                except (DiagnosticUnavailable, TypeError, KeyError, ValueError, RecursionError):
                    summary = None
                items.append(dict(status='diagnostic_unavailable', tool_index=receipt.get('tool_index'),
                                  receipt_summary=summary, failure=None, issues=[reason]))
        status = 'diagnostic_available' if all(item['status'] == 'diagnostic_available' for item in items) else 'diagnostic_unavailable'
        value = dict(protocol=PROTOCOL, status=status, issues=[], receipts=items,
                     meaning='saved historical facts only; physical cause, current pose and release remain unknown')
        return decode_json(encode_diagnostics(value))
    except (DiagnosticUnavailable, OSError, TypeError, ValueError, RecursionError) as exc:
        return unavailable(str(exc) if isinstance(exc, DiagnosticUnavailable) else 'diagnostic_input_or_files_unavailable')
    finally:
        if files is not None:
            files.close()
