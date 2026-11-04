"""Small serializable admission facts, independent of execution authority."""
import copy
import json
import math


PROTOCOL = 'physical-keyboard-admission-rejection-v1'
STAGES = {'compile', 'preflight', 'catalog'}


def rejection(stage, code, message, *, constraint=None, details=None):
    value = dict(protocol=PROTOCOL, failure_stage=stage, reason_code=code,
                 message=message, dispatch_state='not_started',
                 constraint=constraint or {}, details=details or {})
    return validate_rejection(value)


def validate_rejection(value):
    fields = {'protocol', 'failure_stage', 'reason_code', 'message', 'dispatch_state', 'constraint', 'details'}
    if (type(value) is not dict or set(value) != fields or value['protocol'] != PROTOCOL
            or value['failure_stage'] not in STAGES or value['dispatch_state'] != 'not_started'
            or type(value['reason_code']) is not str or not 1 <= len(value['reason_code']) <= 128
            or type(value['message']) is not str or not 1 <= len(value['message']) <= 2048
            or type(value['constraint']) is not dict or type(value['details']) is not dict):
        raise ValueError('invalid admission rejection facts')
    def visit(item, depth=0):
        if depth > 12:
            raise ValueError('admission facts nesting exceeds limit')
        if type(item) in (str, int, bool) or item is None:
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child, depth + 1)
            return
        raise ValueError('admission facts must be finite JSON')
    visit(value)
    if len(json.dumps(value, allow_nan=False, ensure_ascii=False).encode('utf-8')) > 32768:
        raise ValueError('admission facts exceed size limit')
    return copy.deepcopy(value)
