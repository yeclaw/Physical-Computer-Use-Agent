"""Offline, evaluation-only typing evidence and descriptive comparisons.

Only explicit case IDs associate model/execution records. Keyboard release is
an editor event guarantee, never evidence of robot contact release or HOME.
This module performs no I/O and has no dependency on policy or environments.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re

from .usage import normalize_usage


EVENT_PROTOCOL = 'physical-keyboard-typing-lab-events-v1'
EVENT_PROTOCOL_V2 = 'physical-keyboard-typing-lab-events-v2'
IDENTITY_PROTOCOL = 'physical-keyboard-typing-case-identity-v1'
REPORT_PROTOCOL = 'physical-keyboard-typing-report-v1'
COMPARISON_PROTOCOL = 'physical-keyboard-typing-comparison-v1'
_CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789 '
_BINS = {'tiny': (1, 3), 'short': (4, 10), 'medium': (11, 25), 'long': (26, 60)}
_ROLES = {'initial': 1, 'immediate': 2, 'delayed': 3}
_IDENTITY_FIELDS = {'protocol', 'run_id', 'batch_id', 'manifest_sha256', 'case_id',
                    'occurrence_id', 'repeat_index', 'role', 'phase'}
_USAGE_FIELDS = ('input_tokens', 'cached_input_tokens', 'output_tokens',
                 'reasoning_output_tokens')


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}', value):
        raise ValueError('Expected bounded identifier')
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Expected lowercase SHA-256')
    return value


def _int(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Expected integer >= ' + str(minimum))
    return value


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Expected finite nonnegative elapsed_seconds')
    return float(value)


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError('Unexpected or missing fields')


def _pressed_codes(value):
    """Validate an X11 keymap without coercing booleans or hiding duplicates."""
    if not isinstance(value, list):
        raise ValueError('pressed_codes must be a sorted unique list of X11 keycodes')
    for code in value:
        if _int(code) > 255:
            raise ValueError('X11 keycode exceeds 255')
    if value != sorted(set(value)):
        raise ValueError('pressed_codes must be sorted and unique')
    return set(value)


def validate_case_identity(value, expected_run_id=None):
    """Return a fresh strict identity; reject all text/score/extra fields."""
    _fields(value, _IDENTITY_FIELDS)
    if value['protocol'] != IDENTITY_PROTOCOL:
        raise ValueError('Unsupported case identity protocol')
    for name in ('run_id', 'batch_id', 'case_id', 'occurrence_id'):
        _id(value[name])
    _sha(value['manifest_sha256'])
    if type(value['role']) is not str or value['role'] not in _ROLES or type(value['repeat_index']) is not int or value['repeat_index'] != _ROLES[value['role']]:
        raise ValueError('Repeat role and index disagree')
    if type(value['phase']) is not str or value['phase'] not in {'typing', 'awaiting_release', 'batch_complete', 'closed'}:
        raise ValueError('Unsupported case phase')
    if expected_run_id is not None and value['run_id'] != _id(expected_run_id):
        raise ValueError('Case identity belongs to another run')
    return dict(value)


def _target(value, length_bin):
    if not isinstance(value, str) or not value or value.strip(' ') != value or any(c not in _CHARS for c in value):
        raise ValueError('Invalid typing target')
    if type(length_bin) is not str or length_bin not in _BINS or not _BINS[length_bin][0] <= len(value) <= _BINS[length_bin][1]:
        raise ValueError('Target length disagrees with bin')
    return value


def _base(value):
    _fields(value, {'case_id', 'target_text', 'length_bin', 'content_type'})
    _id(value['case_id'])
    _target(value['target_text'], value['length_bin'])
    _id(value['content_type'])


def _occurrence(value):
    _fields(value, {'case_id', 'occurrence_id', 'repeat_index', 'role'})
    _id(value['case_id'])
    _id(value['occurrence_id'])
    if type(value['role']) is not str or value['role'] not in _ROLES or type(value['repeat_index']) is not int or value['repeat_index'] != _ROLES[value['role']]:
        raise ValueError('Repeat role and index disagree')


def _manifest(data):
    _fields(data, {'batch_id', 'manifest_sha256', 'manifest'})
    _id(data['batch_id'])
    _sha(data['manifest_sha256'])
    manifest = data['manifest']
    _fields(manifest, {'protocol', 'config', 'batch_id', 'base_cases', 'occurrences', 'coverage'})
    if (type(manifest['protocol']) is not str
            or manifest['protocol'] not in {'physical-keyboard-typing-cases-v1',
                                   'physical-keyboard-typing-cases-subset-v1'}
            or manifest['batch_id'] != data['batch_id']):
        raise ValueError('Invalid batch manifest identity')
    if not isinstance(manifest['config'], dict) or not isinstance(manifest['coverage'], dict):
        raise ValueError('Invalid manifest metadata')
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    if hashlib.sha256(raw).hexdigest() != data['manifest_sha256']:
        raise ValueError('Manifest SHA-256 mismatch')
    if not isinstance(manifest['base_cases'], list) or not isinstance(manifest['occurrences'], list):
        raise ValueError('Manifest cases must be lists')
    bases, occurrences = {}, {}
    for base in manifest['base_cases']:
        _base(base)
        if base['case_id'] in bases:
            raise ValueError('Duplicate base case ID')
        bases[base['case_id']] = base
    for occurrence in manifest['occurrences']:
        _occurrence(occurrence)
        if occurrence['case_id'] not in bases or occurrence['occurrence_id'] in occurrences:
            raise ValueError('Unregistered or duplicate occurrence')
        occurrences[occurrence['occurrence_id']] = {**bases[occurrence['case_id']], **occurrence,
            'batch_id': data['batch_id'], 'manifest_sha256': data['manifest_sha256']}
    if manifest['protocol'] == 'physical-keyboard-typing-cases-subset-v1':
        _subset_manifest(manifest)
    return occurrences


def _subset_manifest(manifest):
    """Independently verify parent bytes/membership; never import an adapter.

    The environment separately checks the parent's deterministic generation.
    Records verifies the self-contained frozen parent, exact selected demands,
    rescoping and repeat schedule rather than trusting a claimed parent hash.
    Original v1 report inputs retain their prior accepted shape and semantics.
    """
    config = manifest['config']
    _fields(config, {'protocol', 'parent_manifest', 'parent_manifest_sha256', 'selected_case_ids'})
    if config['protocol'] != 'physical-keyboard-typing-subset-config-v1':
        raise ValueError('Invalid subset config protocol')
    parent = config['parent_manifest']
    if not isinstance(parent, dict) or parent.get('protocol') != 'physical-keyboard-typing-cases-v1':
        raise ValueError('Subset parent must be an original v1 manifest')
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
            ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    pin = _sha(config['parent_manifest_sha256'])
    if digest(parent) != pin:
        raise ValueError('Subset parent manifest SHA-256 mismatch')
    _manifest(dict(batch_id=parent['batch_id'], manifest_sha256=pin, manifest=parent))
    selected = config['selected_case_ids']
    if (not isinstance(selected, list) or len(selected) != 3
            or any(type(value) is not str for value in selected) or len(set(selected)) != 3):
        raise ValueError('Subset must select three unique original case IDs')
    bases = {value['case_id']: value for value in parent['base_cases']}
    if any(value not in bases for value in selected):
        raise ValueError('Subset selected case is not registered in parent')
    if manifest['base_cases'] != [bases[value] for value in selected]:
        raise ValueError('Subset demands differ from pinned parent cases')
    config_pin = digest(config)
    if manifest['batch_id'] != 'typing-subset-' + config_pin[:24]:
        raise ValueError('Subset batch ID differs from config')
    expected, repeats = [], [0, 0, 0]
    for index in (0, 0, 1, 1, 2, 2, 0, 1, 2):
        repeats[index] += 1
        repeat, case_id = repeats[index], selected[index]
        expected.append(dict(case_id=case_id,
            occurrence_id=case_id + '-subset-' + config_pin[:12] + '-r%d' % repeat,
            repeat_index=repeat, role=('initial', 'immediate', 'delayed')[repeat - 1]))
    if manifest['occurrences'] != expected:
        raise ValueError('Subset occurrences differ from selected first/immediate/delayed schedule')
    def coverage(cases):
        joined = ''.join(case['target_text'] for case in cases)
        counts = {char: joined.count(char) for char in _CHARS[:-1]}
        return dict(requested_counts=counts, space_count=joined.count(' '),
                    missing_chars=[char for char in _CHARS[:-1] if not counts[char]])
    expected_coverage = coverage(manifest['base_cases'])
    expected_coverage.update(alphabet=_CHARS[:-1], counts_basis='base_cases',
        by_bin={name: coverage([case for case in manifest['base_cases'] if case['length_bin'] == name])
                for name in _BINS})
    if manifest['coverage'] != expected_coverage:
        raise ValueError('Subset coverage differs from its reduced demand')
    return dict(protocol=manifest['protocol'], batch_id=manifest['batch_id'],
        parent_manifest_sha256=pin, selected_case_ids=list(selected),
        base_case_count=3, occurrence_count=9, coverage_scope='explicit_subset_not_full_alphabet',
        missing_alphanumeric=''.join(expected_coverage['missing_chars']),
        repeat_interpretation='scheduled revisits; independence and knowledge freezing are not established by the manifest')


def _distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _usage(value):
    normalized = normalize_usage(value)
    # Keep the v1 report's four measured fields. Its derived total remains only
    # input + output; validating native total never fills a missing measurement.
    return {field: normalized[field] for field in _USAGE_FIELDS}


def _sum_usage(entries, attribution_complete=True, *, _categories=True):
    result = {'calls': len(entries), 'call_count_scope': 'observed_records_only',
              'evidence': 'model_turn_records' if entries else 'no_model_turn_records',
              'attribution_complete': bool(entries) and attribution_complete,
              'observed': {}, 'unknown_calls': {}}
    for field in _USAGE_FIELDS:
        values = [entry['usage'][field] for entry in entries]
        result['observed'][field] = sum(v for v in values if v is not None)
        unknown = [entry['usage'][field] is None or entry.get('usage_complete', True) is False
                   for entry in entries]
        result['unknown_calls'][field] = sum(unknown)
        result[field] = None if not values or not attribution_complete or any(unknown) else sum(values)
    result['complete'] = all(result[name] is not None for name in _USAGE_FIELDS)
    result['total_input_plus_output'] = (result['input_tokens'] + result['output_tokens']
        if result['input_tokens'] is not None and result['output_tokens'] is not None else None)
    result['elapsed_seconds'] = sum(entry['elapsed_seconds'] for entry in entries)
    result['trace_indices'] = [entry['trace_index'] for entry in entries]
    operations = [entry for entry in entries if entry.get('category') == 'context_operation']
    if _categories and operations:
        decisions = [entry for entry in entries if entry.get('category') != 'context_operation']
        context_usage = _sum_usage(operations, attribution_complete, _categories=False)
        context_usage.update(call_count_scope='observed_context_operation_attempt_records',
                             evidence='context_operation_records',
                             observed_total_input_plus_output=context_usage['observed']['input_tokens'] + context_usage['observed']['output_tokens'])
        result.update(call_count_scope='observed_decision_and_context_operation_attempt_records',
            evidence='model_turn_and_context_operation_records',
            observed_total_input_plus_output=result['observed']['input_tokens'] + result['observed']['output_tokens'],
            observed_total_scope='recorded input plus output lower bound; cached/reasoning subsets are not added',
            decision_calls=len(decisions), context_operations=len(operations),
            context_operation_scope='requested maintenance attempts; not necessarily provider RPCs or inference calls',
            by_category=dict(decision=_sum_usage(decisions, attribution_complete, _categories=False),
                             context_operation=context_usage),
            context_operation_methods=dict(Counter(entry['method'] or 'unknown' for entry in operations)),
            context_operation_statuses=dict(Counter(entry['status'] for entry in operations)),
            incomplete_context_operations=sum(entry['usage_complete'] is False
                or any(value is None for value in entry['usage'].values()) for entry in operations),
            decision_elapsed_seconds=sum(entry['elapsed_seconds'] for entry in decisions),
            context_operation_elapsed_seconds=sum(entry['elapsed_seconds'] for entry in operations),
            context_operation_records=[{key: entry[key] for key in
                ('trace_index', 'turn_index', 'method', 'status', 'usage_complete')} for entry in operations])
    return result


def _sum_executions(entries, attribution_complete=True):
    observed_wait, known_pairs, reasons = 0.0, 0, Counter()
    for before, after in zip(entries, entries[1:]):
        if after['execution_index'] != before['execution_index'] + 1:
            reasons['nonadjacent_segments'] += 1
        elif before['_case_key'] is None or before['_case_key'] != after['_case_key']:
            reasons['cross_case_or_unknown_identity'] += 1
        elif any(e[name] is None for e in (before, after)
                 for name in ('started_monotonic_ns', 'ended_monotonic_ns')):
            reasons['missing_interval'] += 1
        elif after['started_monotonic_ns'] < before['ended_monotonic_ns']:
            reasons['overlapping_intervals'] += 1
        else:
            observed_wait += (after['started_monotonic_ns'] - before['ended_monotonic_ns']) / 1e9
            known_pairs += 1
    return {'segments': len(entries), 'elapsed_seconds': sum(e['elapsed_seconds'] for e in entries),
            'count_scope': 'observed_records_only', 'attribution_complete': bool(entries) and attribution_complete,
            'by_action': dict(Counter(e['action'] for e in entries)),
            'by_status': dict(Counter(e['execution_status'] for e in entries)),
            'trace_indices': [e['trace_index'] for e in entries],
            'intervals': [{name: entry[name] for name in ('trace_index', 'started_monotonic_ns', 'ended_monotonic_ns')}
                          for entry in entries],
            'inter_segment_wait_seconds': (observed_wait if known_pairs and not reasons and attribution_complete else None),
            'observed_inter_segment_wait_seconds': observed_wait,
            'inter_segment_wait_pairs': known_pairs,
            'unknown_inter_segment_wait_pairs': sum(reasons.values()),
            'inter_segment_wait_unknown_reasons': dict(reasons),
            'wait_scope': 'Adjacent execution records with the same explicit case identity and actor clock; no environment clock subtraction.'}


def _normalized(rows, *, elapsed_seconds, model_usage, elapsed_basis):
    targets = sum(len(row['target_text']) for row in rows)
    correct = sum(len(row['target_text']) for row in rows if row['exact_correct'] is True)
    attributed = model_usage.get('case_attribution_complete', model_usage['attribution_complete'])
    def ratio(numerator, denominator):
        return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None
    result = {'target_characters': targets, 'correct_submitted_characters': correct,
            'elapsed_seconds': elapsed_seconds, 'elapsed_basis': elapsed_basis,
            'correct_characters_per_minute': ratio(correct * 60, elapsed_seconds),
            'backspaces_per_100_target_characters': ratio(sum(r['backspaces'] for r in rows) * 100, targets),
            'resets_per_100_target_characters': ratio(sum(r['reset_count'] for r in rows) * 100, targets),
            'tokens_per_correct_character': ratio(model_usage['total_input_plus_output'] if attributed else None, correct),
            'calls_per_correct_character': ratio(model_usage['calls'] if attributed else None, correct),
            'cost_scope': 'Observed model records; all failures/adaptation included; unassigned or missing usage is unknown.'}
    if 'context_operations' in model_usage:
        result.update(decision_calls_per_correct_character=ratio(model_usage['decision_calls'] if attributed else None, correct),
            context_operations_per_correct_character=ratio(model_usage['context_operations'] if attributed else None, correct),
            calls_per_correct_character_scope='observed decision plus context-operation attempt records',
            cost_scope='Decision and context-operation records; partial usage is an observed lower bound, not complete billing.')
    return result


def _row(data, event, focused, platform_codes=None):
    _fields(data, {'batch_id', 'manifest_sha256', 'case_id', 'occurrence_id', 'repeat_index', 'role',
                   'target_text', 'length_bin', 'content_type'})
    _base({key: data[key] for key in ('case_id', 'target_text', 'length_bin', 'content_type')})
    _occurrence({key: data[key] for key in ('case_id', 'occurrence_id', 'repeat_index', 'role')})
    _id(data['batch_id'])
    _sha(data['manifest_sha256'])
    return {**dict(data), 'begin_seq': event['seq'], 'end_seq': None, 'status': 'unfinished',
            'submitted_text': None, 'actual_text': '', 'exact_correct': None,
            'first_pass_correct': None, 'backspaces': 0, 'reset_count': 0,
            'correction_text_changes': 0, 'delete_keys': 0,
            'bad_prefix_snapshots': 0, 'focus_loss_count': 0, 'key_down_count': 0,
            'attempted': False, 'diagnostics': [], 'complete': False,
            '_begin_ns': event['monotonic_ns'], '_submit_ns': None, '_last_ns': event['monotonic_ns'],
            '_last_activity_ns': event['monotonic_ns'], '_keys': {}, '_return_down': False,
            '_return_up': False, '_focused': focused, '_training_reset_texts': 0,
            '_platform_keys': None if platform_codes is None else set(platform_codes),
            '_return_codes': set(), '_submit_text_matches_observed': False,
            '_snapshot_released_keys': {},
            'key_state_snapshots': 0, 'return_release_evidence': None,
            '_model': [], '_executions': []}


def _finish(row, event=None):
    diagnostics = row['diagnostics']
    data = event['data'] if event else {}
    if event:
        _fields(data, {'status', 'keyboard_released', 'reason'},
                {'return_released', 'focused', 'pressed_codes', 'settled_ns'})
        if type(data['status']) is not str or data['status'] not in {'submitted', 'aborted'} or data['keyboard_released'] not in (True, False, None):
            raise ValueError('Invalid case end')
        if data['keyboard_released'] is not None and type(data['keyboard_released']) is not bool:
            raise ValueError('keyboard_released must be boolean or null')
        if not isinstance(data['reason'], str):
            raise ValueError('Invalid case end reason')
        for name in ('return_released', 'focused'):
            if name in data and type(data[name]) is not bool:
                raise ValueError(name + ' must be boolean')
        if 'pressed_codes' in data:
            if not isinstance(data['pressed_codes'], list):
                raise ValueError('Invalid pressed_codes')
            for code in data['pressed_codes']:
                _int(code)
        if 'settled_ns' in data:
            _int(data['settled_ns'])
        row['end_seq'] = event['seq']
        row['_last_ns'] = event['monotonic_ns']
        row['status'] = data['status']
        row['end_reason'] = data['reason']
        row['keyboard_released'] = data['keyboard_released']
        if data['status'] != 'submitted':
            diagnostics.append('aborted')
        if data['keyboard_released'] is not True or data['reason'] != 'return_released':
            diagnostics.append('unconfirmed_keyboard_release')
        if data.get('return_released') is False:
            diagnostics.append('unconfirmed_return_release')
        if data.get('pressed_codes'):
            diagnostics.append('keys_still_down')
        if row['_platform_keys'] is not None:
            if row['_platform_keys']:
                diagnostics.append('keys_still_down')
            if ('pressed_codes' in data and _pressed_codes(data['pressed_codes']) != row['_platform_keys']
                    or data['keyboard_released'] is not None
                    and data['keyboard_released'] != (not row['_platform_keys'])):
                diagnostics.append('key_state_snapshot_mismatch')
        if data.get('focused', row['_focused']) is not True or row['_focused'] is False:
            diagnostics.append('focus_unconfirmed')
        if event['monotonic_ns'] <= row['_last_activity_ns'] or data.get('settled_ns', 1) <= 0:
            diagnostics.append('unconfirmed_settle')
    else:
        diagnostics.append('missing_case_end')
        row['keyboard_released'] = None
        row['end_reason'] = None
    if row['_submit_ns'] is None:
        diagnostics.append('missing_submit')
    if not row['_return_down']:
        diagnostics.append('missing_return_down')
    if not row['_return_up']:
        diagnostics.append('missing_return_release')
    if row['_keys']:
        diagnostics.append('keys_still_down')
    if row['_training_reset_texts'] > row['reset_count']:
        diagnostics.append('reset_text_without_reset_event')
    if row['_training_reset_texts'] < row['reset_count']:
        diagnostics.append('reset_without_reset_text_event')
    if row['submitted_text'] is not None and row['submitted_text'] != row['actual_text']:
        diagnostics.append('submit_text_mismatch')
    row['diagnostics'] = list(dict.fromkeys(diagnostics))
    row['final_text_exact'] = row['submitted_text'] == row['target_text'] if row['submitted_text'] is not None else None
    # A training reset can erase an already observed wrong submission. Keep its
    # failure and diagnostics without ever promoting a changed successful draft.
    reset_after_wrong_submit = (
        row['final_text_exact'] is False and row['_submit_text_matches_observed']
        and 'reset_after_submit' in diagnostics
        and not set(diagnostics) - {'reset_after_submit', 'text_changed_after_submit', 'submit_text_mismatch'})
    row['complete'] = (not row['diagnostics'] or reset_after_wrong_submit) and row['status'] == 'submitted'
    if row['complete']:
        row['exact_correct'] = row['final_text_exact']
        row['first_pass_correct'] = bool(row['exact_correct'] and not (
            row['backspaces'] or row['delete_keys'] or row['correction_text_changes']
            or row['reset_count'] or row['bad_prefix_snapshots']))
    score_text = row['submitted_text'] if row['complete'] and reset_after_wrong_submit else row['actual_text']
    row['edit_distance'] = _distance(score_text, row['target_text'])
    row['elapsed_seconds'] = (row['_last_ns'] - row['_begin_ns']) / 1e9
    row['segments'] = {
        'begin_to_submit_seconds': ((row['_submit_ns'] - row['_begin_ns']) / 1e9 if row['_submit_ns'] is not None else None),
        'submit_to_end_seconds': ((row['_last_ns'] - row['_submit_ns']) / 1e9 if row['_submit_ns'] is not None else None),
    }


def _coverage(planned, rows):
    def empty():
        return {name: {char: 0 for char in _CHARS} for name in ('requested', 'presented', 'attempted', 'correct')}
    result = {'unit': 'character occurrences; repeated targets count once per scheduled occurrence',
              'attempted_basis': 'Target demand of cases with key/text/submit activity; not per-key execution proof.',
              'correct_basis': 'Target demand of independently correct cases; not per-key physical reliability.',
              'overall': empty(), 'by_length_bin': {name: empty() for name in _BINS}, 'by_content_type': {}}
    def add(row, category):
        content = result['by_content_type'].setdefault(row['content_type'], empty())
        for group in (result['overall'], result['by_length_bin'][row['length_bin']], content):
            for char in row['target_text']:
                group[category][char] += 1
    for row in planned:
        add(row, 'requested')
    for row in rows:
        add(row, 'presented')
        if row['attempted']:
            add(row, 'attempted')
        if row['exact_correct'] is True:
            add(row, 'correct')
    for group in [result['overall'], *result['by_length_bin'].values(), *result['by_content_type'].values()]:
        group['not_requested_alphanumeric'] = ''.join(c for c in _CHARS[:-1] if not group['requested'][c])
        group['requested_unattempted'] = ''.join(c for c in _CHARS if group['requested'][c] and not group['attempted'][c])
        group['requested_uncorrect'] = ''.join(c for c in _CHARS if group['requested'][c] and not group['correct'][c])
    return result


def _quality(rows, *, model_usage=None, execution=None, elapsed_seconds=None, elapsed_basis='sum_case_windows'):
    n = len(rows)
    complete = sum(row['complete'] for row in rows)
    correct = sum(row['exact_correct'] is True for row in rows)
    first = sum(row['first_pass_correct'] is True for row in rows)
    if model_usage is None:
        model_usage = _sum_usage([entry for row in rows for entry in row['_model']],
            attribution_complete=all(row['model_usage']['attribution_complete'] for row in rows))
    if execution is None:
        execution = _sum_executions(sorted([entry for row in rows for entry in row['_executions']],
            key=lambda entry: entry['execution_index']),
            attribution_complete=all(row['execution']['attribution_complete'] for row in rows))
    case_seconds = sum(row['elapsed_seconds'] for row in rows)
    if elapsed_basis == 'sum_case_windows':
        elapsed_seconds = case_seconds
    return {'presented': n, 'attempted': sum(row['attempted'] for row in rows), 'complete': complete,
            'incomplete': n - complete, 'exact_correct': correct, 'first_pass_correct': first,
            'exact_correct_per_presented': correct / n if n else None,
            'first_pass_per_presented': first / n if n else None,
            'backspaces': sum(row['backspaces'] for row in rows),
            'delete_keys': sum(row['delete_keys'] for row in rows),
            'correction_text_changes': sum(row['correction_text_changes'] for row in rows),
            'reset_count': sum(row['reset_count'] for row in rows),
            'edit_distance_sum': sum(row['edit_distance'] for row in rows),
            'elapsed_seconds': case_seconds, 'model_usage': model_usage, 'execution': execution,
            'normalized': _normalized(rows, elapsed_seconds=elapsed_seconds, model_usage=model_usage,
                                      elapsed_basis=elapsed_basis)}


def _measure(row):
    return {'exact_correct': row['exact_correct'], 'first_pass_correct': row['first_pass_correct'],
            'backspaces': row['backspaces'], 'reset_count': row['reset_count'],
            'correction_text_changes': row['correction_text_changes'],
            'edit_distance': row['edit_distance'], 'elapsed_seconds': row['elapsed_seconds'],
            'model_calls': row['model_usage'].get('decision_calls', row['model_usage']['calls'])
                           if row['model_usage']['attribution_complete'] else None,
            'model_tokens': row['model_usage']['total_input_plus_output'],
            'execution_seconds': row['execution']['elapsed_seconds'] if row['execution']['attribution_complete'] else None,
            'inter_segment_wait_seconds': row['execution']['inter_segment_wait_seconds'],
            **{name: row['normalized'][name] for name in ('correct_characters_per_minute',
                'backspaces_per_100_target_characters', 'resets_per_100_target_characters',
                'tokens_per_correct_character', 'calls_per_correct_character')}}


def _delta(before, after):
    a, b = _measure(before), _measure(after)
    return {name: (b[name] - a[name] if a[name] is not None and b[name] is not None else None) for name in a}


def _repeats(rows):
    groups = {}
    for row in rows:
        key = (row['batch_id'], row['manifest_sha256'], row['case_id'], row['target_text'])
        group = groups.setdefault(key, {'batch_id': key[0], 'manifest_sha256': key[1], 'case_id': key[2],
            'target_text': key[3], 'roles': {}, 'unknowns': [], 'deltas': {}})
        role = row['role']
        if role in group['roles']:
            group['unknowns'].append('duplicate_role:' + role)
        else:
            group['roles'][role] = {'occurrence_id': row['occurrence_id'], 'metrics': _measure(row),
                                    'versions': deepcopy(row['versions'])}
    for key, group in groups.items():
        relevant = [row for row in rows if (row['batch_id'], row['manifest_sha256'], row['case_id'], row['target_text']) == key]
        by_role = {row['role']: row for row in relevant}
        group['missing_roles'] = [role for role in _ROLES if role not in group['roles']]
        group['complete'] = not group['missing_roles'] and not group['unknowns'] and all(row['complete'] for row in relevant)
        if not group['unknowns']:
            for role in ('immediate', 'delayed'):
                if 'initial' in by_role and role in by_role:
                    group['deltas'][role + '_minus_initial'] = _delta(by_role['initial'], by_role[role])
        group['interpretation'] = 'Descriptive repeated exposure; no causal improvement claim.'
    return list(groups.values())


def _identity_key(value):
    return tuple(value[name] for name in ('batch_id', 'manifest_sha256', 'case_id', 'occurrence_id', 'repeat_index', 'role'))


def _attach_traces(trace_events, report, unknowns):
    rows = {_identity_key(row): row for row in report['cases']}
    models, executions, unassigned, straddled = [], [], [], []
    unassigned_model = False
    unassigned_execution = False
    crossed_keys = set()
    def identity(value, trace_index, field):
        try:
            return validate_case_identity(value, report['run_id'])
        except (ValueError, TypeError):
            unknowns.append({'kind': 'invalid_or_missing_trace_identity', 'trace_index': trace_index, 'field': field})
            return None
    for index, event in enumerate(trace_events):
        if not isinstance(event, dict) or not isinstance(event.get('event'), str):
            raise ValueError('Invalid trace event envelope')
        if event['event'] not in {'typing_lab_model_turn', 'typing_lab_execution', 'typing_lab_context_operation'}:
            continue
        _int(event.get('utc_ns'))
        data = event.get('data')
        if not isinstance(data, dict):
            raise ValueError('Invalid trace event data')
        context = identity(data.get('case'), index, 'case')
        start_key = _identity_key(context) if context else None
        row = rows.get(start_key) if context and context['phase'] in {'typing', 'awaiting_release'} else None
        elapsed = _seconds(data.get('elapsed_seconds'))
        if event['event'] in {'typing_lab_model_turn', 'typing_lab_context_operation'}:
            _int(data.get('turn_index'))
            entry = {'trace_index': index, 'turn_index': data['turn_index'], 'usage': _usage(data.get('usage')),
                     'elapsed_seconds': elapsed}
            operation = event['event'] == 'typing_lab_context_operation'
            if operation:
                _fields(data, {'case', 'turn_index', 'method', 'status', 'usage', 'elapsed_seconds', 'usage_complete'})
                if type(data['status']) is not str or data['status'] not in {'completed', 'unsupported', 'failed'}:
                    raise ValueError('Invalid context operation status')
                if (data['method'] is None and data['status'] != 'failed'
                        or data['method'] is not None and (type(data['method']) is not str
                            or data['method'] not in {'native_compact', 'rebuild'})):
                    raise ValueError('Invalid context operation method')
                if type(data['usage_complete']) is not bool:
                    raise ValueError('Context usage_complete must be boolean')
                entry.update(category='context_operation', method=data['method'],
                             status=data['status'], usage_complete=data['usage_complete'])
            models.append(entry)
            if entry.get('usage_complete', True) is False or not all(value is not None for value in entry['usage'].values()):
                unknowns.append({'kind': 'incomplete_context_operation_usage' if operation else 'incomplete_model_usage',
                                 'trace_index': index})
            if row is not None:
                row['_model'].append(entry)
            else:
                unassigned_model = True
                unassigned.append(index)
                unknowns.append({'kind': 'unassigned_context_operation' if operation else 'unassigned_model_call',
                                 'trace_index': index})
        else:
            after = identity(data.get('case_after'), index, 'case_after')
            if type(data.get('action')) is not str or data['action'] not in {'run_skill', 'verify_skill', 'call_skill', 'reset'}:
                raise ValueError('Invalid typing execution action')
            if not isinstance(data.get('execution_status'), str):
                raise ValueError('Invalid execution_status')
            version = {}
            for name in ('program_sha256', 'profile_sha256'):
                value = data.get(name)
                version[name] = _sha(value) if value is not None else None
            for name in ('candidate_id', 'package_id'):
                value = data.get(name)
                version[name] = _id(value) if value is not None else None
            related = data.get('related_skill_ids', [])
            if not isinstance(related, list):
                raise ValueError('related_skill_ids must be a list')
            version['related_skill_ids'] = [_id(value) for value in related]
            interval = {}
            for name in ('started_monotonic_ns', 'ended_monotonic_ns'):
                value = data.get(name)
                interval[name] = _int(value) if value is not None else None
            if all(value is not None for value in interval.values()):
                measured = (interval['ended_monotonic_ns'] - interval['started_monotonic_ns']) / 1e9
                if measured < 0 or not math.isclose(measured, elapsed, rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError('Execution actor interval disagrees with elapsed_seconds')
            entry = {'trace_index': index, 'action': data['action'], 'execution_status': data['execution_status'],
                     'elapsed_seconds': elapsed, 'execution_index': len(executions),
                     '_case_key': None, **version, **interval}
            executions.append(entry)
            if version['program_sha256'] is None or version['profile_sha256'] is None:
                unknowns.append({'kind': 'missing_execution_version', 'trace_index': index})
            if context is not None and after is not None and _identity_key(after) != start_key:
                crossed_keys.update((start_key, _identity_key(after)))
                straddled.append({'trace_index': index, 'case': context, 'case_after': after})
                unknowns.append({'kind': 'straddled_execution', 'trace_index': index})
            elif row is not None and after is not None:
                entry['_case_key'] = start_key
                row['_executions'].append(entry)
            else:
                unassigned_execution = True
                unassigned.append(index)
                unknowns.append({'kind': 'unassigned_execution', 'trace_index': index})
    report['model_usage'] = _sum_usage(models)
    report['model_usage']['case_attribution_complete'] = bool(models) and not unassigned_model
    if not any(entry.get('category') != 'context_operation' for entry in models):
        unknowns.append({'kind': 'no_model_turn_records'})
    report['execution'] = {**_sum_executions(executions, attribution_complete=not unassigned_execution and not straddled),
                           'records': [{key: value for key, value in entry.items() if not key.startswith('_')}
                                       for entry in executions],
                           'straddled': straddled, 'unassigned_trace_indices': unassigned}
    for row in report['cases']:
        row['model_usage'] = _sum_usage(row['_model'], attribution_complete=not unassigned_model)
        row['execution'] = _sum_executions(row['_executions'], attribution_complete=(
            not unassigned_execution and _identity_key(row) not in crossed_keys))
        row['versions'] = [{name: entry[name] for name in ('trace_index', 'program_sha256', 'profile_sha256',
                           'candidate_id', 'package_id', 'related_skill_ids')} for entry in row['_executions']]
        row['segments']['model_seconds'] = row['model_usage']['elapsed_seconds']
        if 'context_operations' in row['model_usage']:
            row['segments'].update(model_seconds_scope='decision plus context-operation attempt records',
                decision_seconds=row['model_usage']['decision_elapsed_seconds'],
                context_operation_seconds=row['model_usage']['context_operation_elapsed_seconds'])
        row['segments']['execution_seconds'] = row['execution']['elapsed_seconds']
        row['segments']['inter_segment_wait_seconds'] = row['execution']['inter_segment_wait_seconds']
        row['normalized'] = _normalized([row], elapsed_seconds=row['elapsed_seconds'],
            model_usage=row['model_usage'], elapsed_basis='case_window')


def build_report(events, trace_events=()):
    """Score supplied editor events; malformed ordering raises ``ValueError``.

    Truncated but structurally valid logs produce incomplete rows and unknowns.
    Character coverage counts scheduled occurrence demand, then presented,
    physically attempted, and independently correct demand separately.
    """
    events = list(events)
    rows, unknowns, planned, batches = [], [], {}, []
    active, run_id, focused = None, None, None
    previous_ns, previous_utc, ended = None, None, False
    previous_protocol, platform_codes = None, None
    batch_order, batch_position = [], 0
    current_batch, completed = None, set()
    subset_datasets = []
    seen = set()
    for index, event in enumerate(events, 1):
        _fields(event, {'protocol', 'run_id', 'seq', 'event', 'utc_ns', 'monotonic_ns', 'data'})
        if (event['protocol'] not in (EVENT_PROTOCOL, EVENT_PROTOCOL_V2)
                or previous_protocol == EVENT_PROTOCOL_V2 and event['protocol'] != EVENT_PROTOCOL_V2
                or _int(event['seq'], 1) != index):
            raise ValueError('Invalid protocol or noncontiguous event sequence')
        previous_protocol = event['protocol']
        _id(event['run_id'])
        if run_id is not None and run_id != event['run_id']:
            raise ValueError('Mixed run IDs')
        run_id = event['run_id']
        _int(event['utc_ns'])
        stamp = _int(event['monotonic_ns'])
        if previous_ns is not None and stamp < previous_ns:
            raise ValueError('Out-of-order monotonic time')
        if previous_utc is not None and event['utc_ns'] < previous_utc:
            unknowns.append({'kind': 'utc_clock_regression', 'seq': index})
        previous_ns, previous_utc = stamp, event['utc_ns']
        kind, data = event['event'], event['data']
        if not isinstance(kind, str) or not isinstance(data, dict) or ended:
            raise ValueError('Invalid event kind/data or event after ended')
        if active:
            active['_last_ns'] = stamp
        if kind == 'batch_begin':
            occurrences = _manifest(data)
            if data['manifest']['protocol'] == 'physical-keyboard-typing-cases-subset-v1':
                subset_datasets.append(_subset_manifest(data['manifest']))
            if active:
                active['diagnostics'].append('batch_changed_without_case_end')
                _finish(active)
                active = None
            if current_batch is not None:
                unfinished = [key for key in batch_order if key not in completed]
                if unfinished:
                    unknowns.append({'kind': 'batch_replaced_before_completion',
                        'batch_id': current_batch[0], 'manifest_sha256': current_batch[1],
                        'unfinished_occurrence_ids': [planned[key]['occurrence_id'] for key in unfinished]})
            ref = (data['batch_id'], data['manifest_sha256'])
            if ref in batches:
                raise ValueError('Duplicate batch_begin')
            batches.append(ref)
            current_batch = ref
            batch_order, batch_position = [_identity_key(value) for value in occurrences.values()], 0
            for value in occurrences.values():
                planned[_identity_key(value)] = value
        elif kind == 'case_begin':
            if active:
                _finish(active)
            active = _row(data, event, focused, platform_codes)
            key = _identity_key(active)
            if key in seen:
                raise ValueError('Duplicate case occurrence')
            seen.add(key)
            expected = planned.get(key)
            if expected is None:
                active['diagnostics'].append('missing_or_unregistered_manifest_occurrence')
            elif any(data[name] != expected[name] for name in data):
                raise ValueError('Case metadata disagrees with manifest')
            if (data['batch_id'], data['manifest_sha256']) != current_batch:
                active['diagnostics'].append('case_outside_active_batch')
            if batch_position >= len(batch_order) or key != batch_order[batch_position]:
                active['diagnostics'].append('unexpected_manifest_order')
            if (data['batch_id'], data['manifest_sha256']) == current_batch:
                batch_position += 1
            rows.append(active)
        elif kind in {'key_down', 'key_up'}:
            _fields(data, {'key', 'code', 'repeated'} if kind == 'key_down' else {'key', 'code'})
            if not isinstance(data['key'], str) or not data['key']:
                raise ValueError('Invalid key name')
            _int(data['code'])
            if kind == 'key_down' and type(data['repeated']) is not bool:
                raise ValueError('repeated must be boolean')
            if platform_codes is not None:
                if kind == 'key_down':
                    platform_codes.add(data['code'])
                else:
                    platform_codes.discard(data['code'])
            if active:
                if active['_platform_keys'] is not None:
                    active['_platform_keys'] = set(platform_codes)
                active['_last_activity_ns'] = stamp
                if kind == 'key_down':
                    active['attempted'] = True
                    active['key_down_count'] += 1
                    active['_snapshot_released_keys'].pop(data['code'], None)
                    if data['code'] in active['_keys'] and not data['repeated']:
                        active['diagnostics'].append('duplicate_unrepeated_key_down')
                    if data['code'] not in active['_keys'] and data['repeated']:
                        active['diagnostics'].append('repeat_without_key_down')
                    active['_keys'][data['code']] = data['key']
                    if data['key'] in {'Return', 'KP_Enter'}:
                        active['_return_down'] = True
                        active['_return_codes'].add(data['code'])
                    if data['key'] == 'BackSpace':
                        active['backspaces'] += 1
                    if data['key'] in {'Delete', 'KP_Delete'}:
                        active['delete_keys'] += 1
                else:
                    down = active['_keys'].pop(data['code'], None)
                    if down is None:
                        down = active['_snapshot_released_keys'].pop(data['code'], None)
                    if down is None:
                        active['diagnostics'].append('key_up_without_key_down')
                    elif down != data['key']:
                        active['diagnostics'].append('key_up_name_mismatch')
                    if data['key'] in {'Return', 'KP_Enter'} and down is not None and active['_submit_ns'] is not None:
                        active['_return_up'] = True
                        active['return_release_evidence'] = 'key_up'
        elif kind == 'key_state_snapshot':
            if event['protocol'] != EVENT_PROTOCOL_V2:
                raise ValueError('key_state_snapshot requires the v2 event protocol')
            _fields(data, {'source', 'pressed_codes'})
            if data['source'] != 'x11_query_keymap':
                raise ValueError('Unsupported key state snapshot source')
            platform_codes = _pressed_codes(data['pressed_codes'])
            if active:
                released = {code: key for code, key in active['_keys'].items() if code not in platform_codes}
                if (not active['key_state_snapshots'] or active['_platform_keys'] != platform_codes or released):
                    active['_last_activity_ns'] = stamp
                active['key_state_snapshots'] += 1
                active['_platform_keys'] = set(platform_codes)
                active['_snapshot_released_keys'] = {
                    code: key for code, key in active['_snapshot_released_keys'].items() if code not in platform_codes}
                active['_snapshot_released_keys'].update(released)
                active['_keys'] = {code: key for code, key in active['_keys'].items() if code in platform_codes}
                if (active['_submit_ns'] is not None and active['_return_down']
                        and not active['_return_codes'] & platform_codes):
                    active['_return_up'] = True
                    if active['return_release_evidence'] is None:
                        active['return_release_evidence'] = 'x11_query_keymap'
        elif kind == 'text_changed':
            _fields(data, {'text', 'source'})
            if not isinstance(data['text'], str) or len(data['text']) > 4096 or type(data['source']) is not str or data['source'] not in {'physical_input', 'training_reset'}:
                raise ValueError('Invalid text_changed')
            if data['source'] == 'training_reset' and data['text']:
                raise ValueError('Training reset cannot inject text')
            if active:
                active['_last_activity_ns'] = stamp
                if active['_submit_ns'] is not None:
                    active['diagnostics'].append('text_changed_after_submit')
                if data['source'] == 'physical_input':
                    active['attempted'] = True
                    if not data['text'].startswith(active['actual_text']):
                        active['correction_text_changes'] += 1
                    if not active['target_text'].startswith(data['text']):
                        active['bad_prefix_snapshots'] += 1
                else:
                    active['_training_reset_texts'] += 1
                active['actual_text'] = data['text']
        elif kind == 'reset':
            _fields(data, {'reason'})
            if data['reason'] != 'explicit_training_reset':
                raise ValueError('Invalid reset reason')
            if active:
                active['reset_count'] += 1
                active['_last_activity_ns'] = stamp
                if active['_submit_ns'] is not None:
                    active['diagnostics'].append('reset_after_submit')
        elif kind == 'focus':
            _fields(data, {'focused'})
            if type(data['focused']) is not bool:
                raise ValueError('focused must be boolean')
            focused = data['focused']
            if active:
                active['_focused'] = focused
                active['_last_activity_ns'] = stamp
                if not focused:
                    active['focus_loss_count'] += 1
        elif kind == 'submit':
            _fields(data, {'text'})
            if not isinstance(data['text'], str) or len(data['text']) > 4096:
                raise ValueError('Invalid submitted text')
            if active:
                active['attempted'] = True
                if active['_submit_ns'] is not None:
                    active['diagnostics'].append('duplicate_submit')
                else:
                    active['submitted_text'] = data['text']
                    active['_submit_ns'] = stamp
                    active['_submit_text_matches_observed'] = data['text'] == active['actual_text']
                if not any(key in {'Return', 'KP_Enter'} for key in active['_keys'].values()):
                    active['diagnostics'].append('submit_without_return_down')
            else:
                unknowns.append({'kind': 'submit_outside_case', 'seq': index})
        elif kind == 'case_end':
            if active:
                _finish(active, event)
                if active['complete']:
                    completed.add(_identity_key(active))
                active = None
            else:
                raise ValueError('case_end without case_begin')
        elif kind == 'ended':
            _fields(data, {'reason'})
            if not isinstance(data['reason'], str):
                raise ValueError('Invalid ended reason')
            if active:
                _finish(active)
                active = None
            ended = True
        else:
            raise ValueError('Unknown typing event kind')
    if active:
        _finish(active)
    if not events:
        unknowns.append({'kind': 'empty_event_log'})
    if not ended:
        unknowns.append({'kind': 'missing_ended_event'})
    missing = [value['occurrence_id'] for key, value in planned.items() if key not in seen]
    for row in rows:
        for diagnostic in row['diagnostics']:
            unknowns.append({'kind': diagnostic, 'occurrence_id': row['occurrence_id']})
    if missing:
        unknowns.append({'kind': 'unpresented_planned_occurrences', 'occurrence_ids': missing})
    report = {'protocol': REPORT_PROTOCOL, 'run_id': run_id, 'cases': rows, 'unknowns': unknowns,
              'source': {'event_count': len(events), 'ended': ended,
                         'batches': [{'batch_id': b, 'manifest_sha256': sha} for b, sha in batches],
                         'planned_occurrences': len(planned), 'unpresented_occurrence_ids': missing}}
    _attach_traces(trace_events, report, unknowns)
    report['coverage'] = _coverage(planned.values(), rows)
    if subset_datasets:
        report['source']['subset_datasets'] = subset_datasets
    wall_seconds = (events[-1]['monotonic_ns'] - events[0]['monotonic_ns']) / 1e9 if events else None
    report['summary'] = {**_quality(rows, model_usage=report['model_usage'], execution=report['execution'],
        elapsed_seconds=wall_seconds, elapsed_basis='run_event_window'), 'by_length_bin': {}, 'by_content_type': {},
        'quality_complete': bool(batches) and ended and not missing and all(row['complete'] for row in rows),
        'wall_elapsed_seconds': wall_seconds,
        'home_returns': None, 'robot_release': 'unknown',
        'timing_note': 'Case wall, model, and execution durations can overlap; do not add or subtract them.'}
    for name in _BINS:
        report['summary']['by_length_bin'][name] = _quality([r for r in rows if r['length_bin'] == name])
    for name in sorted({r['content_type'] for r in rows}):
        report['summary']['by_content_type'][name] = _quality([r for r in rows if r['content_type'] == name])
    report['repeats'] = _repeats(rows)
    for row in rows:
        for name in list(row):
            if name.startswith('_'):
                del row[name]
    return report


def compare_reports(before, after):
    """Pair stable case ID + repeat role + target, with no causal attribution."""
    for report in (before, after):
        if not isinstance(report, dict) or report.get('protocol') != REPORT_PROTOCOL or not isinstance(report.get('cases'), list):
            raise ValueError('Expected typing reports')
    def index(report):
        result = {}
        for row in report['cases']:
            key = (row['case_id'], row['role'], row['target_text'])
            result.setdefault(key, []).append(row)
        return result
    old, new = index(before), index(after)
    matched, unmatched_before, unmatched_after, ambiguous = [], [], [], []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key, []), new.get(key, [])
        if len(a) > 1 or len(b) > 1:
            ambiguous.append({'case_id': key[0], 'role': key[1], 'target_text': key[2],
                              'before_count': len(a), 'after_count': len(b)})
        elif a and b:
            matched.append({'case_id': key[0], 'role': key[1], 'target_text': key[2],
                'before_occurrence_id': a[0]['occurrence_id'], 'after_occurrence_id': b[0]['occurrence_id'],
                'before': _measure(a[0]), 'after': _measure(b[0]), 'delta': _delta(a[0], b[0]),
                'before_versions': deepcopy(a[0]['versions']), 'after_versions': deepcopy(b[0]['versions'])})
        elif a:
            unmatched_before.append(a[0]['occurrence_id'])
        else:
            unmatched_after.append(b[0]['occurrence_id'])
    def distribution(report):
        return {field: dict(sorted(Counter(row[field] for row in report['cases']).items()))
                for field in ('length_bin', 'content_type', 'role', 'target_text')}
    d_before, d_after = distribution(before), distribution(after)
    means = {}
    for name in _measure(before['cases'][0]) if before['cases'] else ():
        values = [pair['delta'][name] for pair in matched if pair['delta'][name] is not None]
        means[name] = {'mean_delta': sum(values) / len(values) if values else None, 'known_pairs': len(values)}
    return {'protocol': COMPARISON_PROTOCOL, 'before_run_id': before.get('run_id'), 'after_run_id': after.get('run_id'),
            'matched': matched, 'paired_summary': means,
            'unmatched_before': unmatched_before, 'unmatched_after': unmatched_after,
            'ambiguous': ambiguous, 'distributions': {'before': d_before, 'after': d_after,
            'changed': d_before != d_after or before['coverage']['overall']['requested'] != after['coverage']['overall']['requested']},
            'unknowns': {'before': deepcopy(before['unknowns']), 'after': deepcopy(after['unknowns'])},
            'interpretation': 'Descriptive paired comparison only; exposure, task distribution, and versions may differ. No causal RSI claim.'}
