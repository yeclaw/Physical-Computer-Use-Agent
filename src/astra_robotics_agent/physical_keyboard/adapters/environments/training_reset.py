"""Explicit training reset orchestration; no model, SDK, editor or OS imports.

The caller supplies a frozen recovery program and the existing single-owner
execution callbacks. This module never invents a robot recovery trajectory.
"""
from __future__ import annotations

import math
import re


HOME_POSE = {'xyz_m': [.125, 0., .225], 'quaternion_xyzw': [0., 0., 0., 1.]}
POSITION_TOLERANCE_M = .005
ROTATION_TOLERANCE_RAD = .05


def _vector(value, size, name):
    try:
        valid = (type(value) is list and len(value) == size
                 and all(type(x) in (int, float) and math.isfinite(x) for x in value))
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValueError(name + ' must be a finite vector')
    return list(value)


def _pose(value):
    if type(value) is str and value == 'home':
        return {k: list(v) for k, v in HOME_POSE.items()}
    if type(value) is not dict or set(value) != {'xyz_m', 'quaternion_xyzw'}:
        raise ValueError('return_pose must be home or xyz_m/quaternion_xyzw')
    xyz = _vector(value['xyz_m'], 3, 'return xyz')
    quat = _vector(value['quaternion_xyzw'], 4, 'return quaternion')
    if abs(math.hypot(*quat) - 1) > 1e-3:
        raise ValueError('return quaternion must be unit length')
    return dict(xyz_m=xyz, quaternion_xyzw=quat)


def validate_reset_arguments(args):
    """Validate before any callback; retained values are not repaired/defaulted."""
    if type(args) is not dict or set(args) != {'reason', 'return_pose', 'recovery'}:
        raise ValueError('reset requires reason, return_pose and recovery')
    if type(args['reason']) is not str or not args['reason'].strip() or len(args['reason']) > 2048:
        raise ValueError('reset reason must be a nonempty string, at most 2048 characters')
    _pose(args['return_pose'])
    recovery = args['recovery']
    if type(recovery) is not dict:
        raise ValueError('reset recovery must be an object')
    if set(recovery) == {'candidate_id'}:
        if type(recovery['candidate_id']) is not str or not recovery['candidate_id'].strip():
            raise ValueError('reset recovery candidate_id required')
        return args
    if (set(recovery) != {'source_lines', 'profile', 'label'}
            or type(recovery['source_lines']) is not list or not recovery['source_lines']
            or any(type(x) is not str or '\n' in x or '\r' in x for x in recovery['source_lines'])
            or type(recovery['profile']) is not dict or type(recovery['label']) is not str
            or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', recovery['label']) is None):
        raise ValueError('reset recovery must be candidate_id or source_lines/profile/label')
    return args


def resolve_recovery(args, store):
    """Use the current scoped store; candidate identity is never a filesystem path."""
    validate_reset_arguments(args)
    recovery = args['recovery']
    if 'candidate_id' in recovery:
        saved = store.get_skill(recovery['candidate_id'])
        return {k: saved[k] for k in ('source', 'profile', 'label')}
    return dict(source='\n'.join(recovery['source_lines']) + '\n',
                profile=recovery['profile'], label=recovery['label'])


def _known_motion(response):
    if type(response) is not dict or response.get('status') not in ('ok', 'blocked', 'error'):
        return False
    result = response.get('result')
    return (type(result) is dict and result.get('motion_outcome_uncertain') is False
            and type(result.get('receipts')) is list
            and all(type(r) is dict and r.get('status') in ('ok', 'rejected', 'error')
                    for r in result['receipts']))


def perform_reset(args, recovery, *, execute, state, clear_editor):
    """Recover via the existing executor, verify the declared pose, clear once.

    execute returns the normal {response,evidence_id}. state returns the direct
    SDK state. clear_editor returns a validated reset-only editor receipt.
    Callback exceptions escape: the actor stops, never retries a dispatched call.
    Pose verification is not a physical key-release or visual-success detector.
    """
    validate_reset_arguments(args)
    target = _pose(args['return_pose'])
    executed = execute(recovery)
    response = executed['response']
    payload = response.get('result') if type(response) is dict else None
    result = dict(reason=args['reason'], return_pose=target, recovery=response,
                  recovery_evidence_id=executed['evidence_id'],
                  receipts=payload.get('receipts', []) if type(payload) is dict else [],
                  motion_outcome_uncertain=not _known_motion(response),
                  reset_outcome_uncertain=False, reset_applied=False,
                  physical_release='unverified_external_camera_required',
                  text_success='not_a_physical_typing_or_correction_result')
    if result['motion_outcome_uncertain']:
        result['stage'] = 'recovery_unknown_no_retry'
        return dict(status='unknown', result=result)
    if (response['status'] != 'ok'
            or any(r.get('status') != 'ok' for r in result['receipts'])):
        result['stage'] = 'recovery_not_completed'
        return dict(status='blocked', result=result)
    actual = state()
    left = _pose({k: actual['arms']['left'][k] for k in ('xyz_m', 'quaternion_xyzw')})
    distance = math.dist(left['xyz_m'], target['xyz_m'])
    a, b = left['quaternion_xyzw'], target['quaternion_xyzw']
    # Normalize only for angular comparison; original feedback remains recorded.
    dot = abs(sum(x*y for x, y in zip(a, b))) / (math.hypot(*a)*math.hypot(*b))
    rotation = 2 * math.acos(min(1., dot))
    result.update(actual_return_state=actual, position_error_m=distance, rotation_error_rad=rotation,
                  pose_tolerances=dict(position_m=POSITION_TOLERANCE_M, rotation_rad=ROTATION_TOLERANCE_RAD))
    if distance > POSITION_TOLERANCE_M or rotation > ROTATION_TOLERANCE_RAD:
        result['stage'] = 'return_pose_not_reached'
        return dict(status='blocked', result=result)
    editor = clear_editor()
    result['editor_receipt'] = editor
    if type(editor) is not dict:
        raise RuntimeError('Untrusted editor reset receipt; no retry')
    if editor.get('status') != 'ok':
        result['stage'] = 'editor_reset_failed'
        error = editor.get('error')
        result['reset_outcome_uncertain'] = (type(error) is not dict
                                             or error.get('uncertain') is not False)
        return dict(status='unknown' if result['reset_outcome_uncertain'] else 'blocked', result=result)
    payload = editor.get('result')
    if (type(payload) is not dict or any(payload.get(key) is not True for key in
            ('cleared', 'focus_requested', 'ime_reset_requested')) or payload.get('buffer_read') is not False):
        raise RuntimeError('Editor reset completion receipt missing; no retry')
    result.update(stage='reset_completed', reset_applied=True)
    return dict(status='ok', result=result)
