"""Present actionable admission facts; preserve full execution as evidence."""
import copy
import hashlib
import json

from astra_robotics_agent.physical_keyboard.records.admission import validate_rejection


def for_model(response, execution_evidence_id, *, compact=False):
    result = response.get('result')
    if (compact and response.get('status') == 'ok' and type(result) is dict
            and result.get('motion_outcome_uncertain') is False
            and result.get('controller_terminal') is not True):
        receipts = result.get('receipts', [])
        if type(receipts) is list and all(type(row) is dict and row.get('status') == 'ok'
                and not any(row.get(key) for key in ('error', 'audit_error', 'stop_error'))
                and not (type(row.get('replay')) is dict and any(row['replay'].get(key)
                    for key in ('motion_outcome_uncertain', 'audit_error', 'stop_error', 'stop_called')))
                for row in receipts):
            def raw(value):
                return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            def view(value):
                data = raw(value)
                if len(data) <= 4096:
                    return copy.deepcopy(value)
                return dict(full_value_sha256=hashlib.sha256(data).hexdigest(),
                            full_value_bytes=len(data), available_in_execution_evidence=True)
            projected = {key: view(result[key]) for key in (
                'motion_outcome_uncertain', 'controller_terminal', 'execution_disposition',
                'actual_state', 'value', 'text_success', 'completed_tools', 'source_sha256', 'profile_sha256')
                if key in result}
            projected.update(full_execution_evidence_id=execution_evidence_id,
                full_receipt_count=len(receipts), receipt_view='successful tool boundaries; exact values in evidence',
                completed_boundaries=[{key: row[key] for key in ('index', 'method', 'status') if key in row}
                                      for row in receipts[-8:]],
                earlier_boundaries_omitted=max(0, len(receipts) - 8))
            return dict(response, result=projected)
    if (type(result) is not dict or result.get('recoverable_admission_error') is not True
            or result.get('motion_outcome_uncertain') is not False
            or result.get('controller_terminal') is True):
        return response
    receipts = result.get('receipts', [])
    if type(receipts) is not list:
        return response
    for receipt in receipts:
        if type(receipt) is not dict or receipt.get('status') == 'unknown':
            return response
        replay = receipt.get('replay')
        if type(replay) is dict and (replay.get('motion_outcome_uncertain') is True
                or replay.get('audit_error') or replay.get('stop_error')
                or replay.get('stop_called')):
            return response
    continuation = result.get('continuation')
    if (type(continuation) is not dict
            or continuation.get('protocol') != 'physical-keyboard-continuation-v1'
            or continuation.get('controller_health') != 'healthy'
            or continuation.get('dispatch_state') != 'not_started'
            or continuation.get('automatic_replay') is not False):
        return response
    try:
        validate_rejection(continuation['admission'])
    except (ValueError, KeyError):
        return response
    keep = ('error', 'motion_outcome_uncertain', 'recoverable_admission_error',
            'execution_disposition', 'continuation', 'actual_state',
            'skill_dir', 'source_sha256', 'profile_sha256', 'text_success',
            'trajectory_execution', 'completed_tools')
    projected = {key: copy.deepcopy(result[key]) for key in keep if key in result}
    projected.update(full_execution_evidence_id=execution_evidence_id,
                     full_receipt_count=len(receipts),
                     receipt_view='completed tool boundaries and failed admission; full exact receipts in evidence')
    return dict(response, result=projected)
