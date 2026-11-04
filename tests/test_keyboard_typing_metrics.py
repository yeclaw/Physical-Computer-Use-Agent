"""Evaluation-only evidence: corrections, incomplete logs, and explicit IDs."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from astra_robotics_agent.physical_keyboard.records.typing_metrics import (
    EVENT_PROTOCOL, IDENTITY_PROTOCOL, build_report, compare_reports,
    validate_case_identity,
)


def manifest(target='aa', case_id='case-a', roles=('initial', 'immediate', 'delayed')):
    length = len(target)
    length_bin = next(name for name, lo, hi in (
        ('tiny', 1, 3), ('short', 4, 10), ('medium', 11, 25), ('long', 26, 60)) if lo <= length <= hi)
    value = {'protocol': 'physical-keyboard-typing-cases-v1',
             'config': {'protocol': 'physical-keyboard-typing-case-config-v1', 'seed': 0, 'cases_per_bin': 4},
             'batch_id': 'batch-a', 'base_cases': [{'case_id': case_id, 'target_text': target,
                'length_bin': length_bin, 'content_type': 'alternating'}],
             'occurrences': [{'case_id': case_id, 'occurrence_id': 'occ-' + role,
                'repeat_index': {'initial': 1, 'immediate': 2, 'delayed': 3}[role], 'role': role}
                for role in roles], 'coverage': {}}
    sha = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return value, sha


class Log:
    def __init__(self, target='aa', run_id='run-a', case_id='case-a', roles=('initial', 'immediate', 'delayed')):
        self.manifest, self.sha = manifest(target, case_id, roles)
        self.run_id, self.events, self.now = run_id, [], 0
        self.add('batch_begin', manifest=self.manifest, batch_id='batch-a', manifest_sha256=self.sha)

    def add(self, event, **data):
        self.now += 100_000_000
        self.events.append({'protocol': EVENT_PROTOCOL, 'run_id': self.run_id,
            'seq': len(self.events) + 1, 'event': event, 'utc_ns': 1_000_000_000 + self.now,
            'monotonic_ns': self.now, 'data': data})

    def begin(self, role='initial'):
        occurrence = next(value for value in self.manifest['occurrences'] if value['role'] == role)
        self.add('case_begin', **self.manifest['base_cases'][0], **{key: value for key, value in occurrence.items() if key != 'case_id'},
                 batch_id=self.manifest['batch_id'], manifest_sha256=self.sha)
        self.add('focus', focused=True)

    def key(self, name, code=1, text=None):
        self.add('key_down', key=name, code=code, repeated=False)
        if text is not None:
            self.add('text_changed', text=text, source='physical_input')
        self.add('key_up', key=name, code=code)

    def submit(self, text, finish=True, released=True):
        self.add('key_down', key='Return', code=36, repeated=False)
        self.add('submit', text=text)
        if released:
            self.add('key_up', key='Return', code=36)
        if finish:
            self.now += 500_000_000
            self.add('case_end', status='submitted', keyboard_released=released,
                     reason='return_released', return_released=released,
                     pressed_codes=[] if released else [36], focused=True, settled_ns=500_000_000)

    def correct(self, role='initial'):
        self.begin(role)
        text = self.manifest['base_cases'][0]['target_text']
        for index, char in enumerate(text, 1):
            self.key(char if char != ' ' else 'space', text=text[:index])
        self.submit(text)

    def end(self):
        self.add('ended', reason='operator_stop')
        return self.events

    def identity(self, role='initial', phase='typing'):
        return {'protocol': IDENTITY_PROTOCOL, 'run_id': self.run_id, 'batch_id': self.manifest['batch_id'],
                'manifest_sha256': self.sha, 'case_id': self.manifest['base_cases'][0]['case_id'],
                'occurrence_id': 'occ-' + role, 'repeat_index': {'initial': 1, 'immediate': 2, 'delayed': 3}[role],
                'role': role, 'phase': phase}


def model(identity, usage=None, elapsed=2.5, turn=1):
    return {'event': 'typing_lab_model_turn', 'utc_ns': 1,
            'data': {'case': identity, 'turn_index': turn, 'usage': usage, 'elapsed_seconds': elapsed}}


def execution(identity, after=None, seconds=3.0, program='a' * 64, action='run_skill'):
    return {'event': 'typing_lab_execution', 'utc_ns': 1,
            'data': {'case': identity, 'case_after': identity if after is None else after,
                     'action': action, 'candidate_id': 'candidate-v1', 'package_id': 'package-v1',
                     'program_sha256': program, 'profile_sha256': 'b' * 64,
                     'related_skill_ids': ['skill-a-v1'], 'elapsed_seconds': seconds, 'execution_status': 'ok'}}


def timed_execution(identity, start, end, after=None):
    event = execution(identity, after, seconds=(end - start) / 1e9)
    event['data'].update(started_monotonic_ns=start, ended_monotonic_ns=end)
    return event


class IdentityTests(unittest.TestCase):
    def test_strict_copy_and_run_role_protocol_validation(self):
        identity = Log().identity()
        self.assertEqual(validate_case_identity(identity, 'run-a'), identity)
        self.assertIsNot(validate_case_identity(identity), identity)
        for change in ({'actual_text': 'private'}, {'target_text': 'private'}, {'score': 1},
                       {'run_id': 'run-other'}, {'protocol': 'wrong'}, {'repeat_index': True},
                       {'repeat_index': 2}, {'phase': 'other'}, {'manifest_sha256': 'A' * 64}):
            value = {**identity, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_case_identity(value, 'run-a')
        for field in identity:
            value = dict(identity)
            del value[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_case_identity(value)

    def test_identity_bad_types_fail_closed_and_colon_identifier_bounds_match_engine(self):
        identity = Log().identity()
        for field in identity:
            for wrong in (None, [], {}, True, 5):
                with self.subTest(field=field, wrong=wrong), self.assertRaises(ValueError):
                    validate_case_identity({**identity, field: wrong})
        value = {**identity, 'run_id': 'r:' + 'a' * 158}
        self.assertEqual(validate_case_identity(value)['run_id'], value['run_id'])
        with self.assertRaises(ValueError):
            validate_case_identity({**value, 'run_id': value['run_id'] + 'x'})


class MetricsTests(unittest.TestCase):
    def test_correction_is_final_success_but_never_first_pass(self):
        log = Log()
        log.begin()
        log.key('a', text='a')
        log.key('s', text='as')
        log.key('BackSpace', text='a')
        log.key('a', text='aa')
        log.submit('aa')
        report = build_report(log.end())
        row = report['cases'][0]
        self.assertTrue(row['exact_correct'])
        self.assertFalse(row['first_pass_correct'])
        self.assertEqual((row['backspaces'], row['reset_count'], row['edit_distance']), (1, 0, 0))
        self.assertEqual(row['bad_prefix_snapshots'], 1)
        self.assertGreater(row['elapsed_seconds'], 0)
        self.assertEqual(report['coverage']['overall']['requested']['a'], 6)
        self.assertEqual(report['coverage']['overall']['attempted']['a'], 2)
        self.assertEqual(report['coverage']['overall']['correct']['a'], 2)
        self.assertIsNone(report['summary']['home_returns'])
        self.assertEqual(report['summary']['robot_release'], 'unknown')
        self.assertIsNone(row['model_usage']['total_input_plus_output'])
        json.dumps(report, allow_nan=False)

    def test_all_repeat_roles_pair_metrics_and_exact_version_references(self):
        log = Log()
        log.begin()
        log.key('s', text='as')
        log.key('BackSpace', text='a')
        log.key('a', text='aa')
        log.submit('aa')
        log.correct('immediate')
        log.correct('delayed')
        trace = [execution(log.identity(), program='c' * 64), execution(log.identity('immediate'))]
        report = build_report(log.end(), trace)
        repeat = report['repeats'][0]
        self.assertTrue(repeat['complete'])
        self.assertEqual(set(repeat['roles']), {'initial', 'immediate', 'delayed'})
        self.assertEqual(repeat['deltas']['immediate_minus_initial']['backspaces'], -1)
        self.assertEqual(repeat['deltas']['immediate_minus_initial']['first_pass_correct'], 1)
        self.assertEqual(repeat['roles']['initial']['versions'][0]['program_sha256'], 'c' * 64)
        self.assertEqual(repeat['roles']['immediate']['versions'][0]['program_sha256'], 'a' * 64)
        self.assertEqual(report['source']['unpresented_occurrence_ids'], [])

    def test_reset_is_distinct_from_physical_input_and_prevents_first_pass(self):
        log = Log()
        log.begin()
        log.key('s', text='s')
        log.add('reset', reason='explicit_training_reset')
        log.add('text_changed', text='', source='training_reset')
        log.key('a', text='a')
        log.key('a', text='aa')
        log.submit('aa')
        row = build_report(log.end())['cases'][0]
        self.assertTrue(row['exact_correct'])
        self.assertFalse(row['first_pass_correct'])
        self.assertEqual(row['reset_count'], 1)
        self.assertEqual(row['backspaces'], 0)

    def test_prefix_deletion_is_a_correction_even_without_backspace(self):
        log = Log()
        log.begin()
        log.key('a', text='a')
        log.key('Delete', text='')
        log.key('a', text='aa')
        log.submit('aa')
        row = build_report(log.end())['cases'][0]
        self.assertTrue(row['exact_correct'])
        self.assertFalse(row['first_pass_correct'])
        self.assertEqual(row['correction_text_changes'], 1)
        self.assertEqual(row['delete_keys'], 1)

    def test_out_of_manifest_order_is_incomplete_instead_of_pass(self):
        log = Log()
        log.correct('delayed')
        report = build_report(log.end())
        self.assertIsNone(report['cases'][0]['exact_correct'])
        self.assertIn('unexpected_manifest_order', report['cases'][0]['diagnostics'])
        self.assertFalse(report['summary']['quality_complete'])

    def test_skipped_middle_or_reordered_roles_never_form_complete_repeat_comparison(self):
        for order in (('initial', 'delayed'), ('initial', 'delayed', 'immediate')):
            log = Log()
            for role in order:
                log.correct(role)
            report = build_report(log.end())
            self.assertTrue(report['cases'][0]['exact_correct'])
            for row in report['cases'][1:]:
                self.assertIsNone(row['exact_correct'])
                self.assertIn('unexpected_manifest_order', row['diagnostics'])
            self.assertFalse(report['summary']['quality_complete'])
            self.assertFalse(report['repeats'][0]['complete'])
            self.assertIsNone(report['repeats'][0]['deltas']['delayed_minus_initial']['exact_correct'])

    def test_current_batch_identity_disambiguates_shared_case_and_occurrence_ids(self):
        log = Log()
        newer = deepcopy(log.manifest)
        newer['batch_id'] = 'batch-newer'
        newer['config']['seed'] = 34
        newer_sha = hashlib.sha256(json.dumps(newer, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False).encode()).hexdigest()
        log.add('batch_begin', manifest=newer, batch_id=newer['batch_id'], manifest_sha256=newer_sha)
        # A stale case has IDs also scheduled in B, but belongs to the old A.
        log.correct()
        report = build_report(log.end())
        row = report['cases'][0]
        self.assertFalse(row['complete'])
        self.assertIsNone(row['exact_correct'])
        self.assertIn('case_outside_active_batch', row['diagnostics'])
        self.assertIn('unexpected_manifest_order', row['diagnostics'])
        switched = [item for item in report['unknowns'] if item['kind'] == 'batch_replaced_before_completion']
        self.assertEqual(len(switched), 1)
        self.assertEqual(switched[0]['batch_id'], 'batch-a')
        self.assertEqual(switched[0]['manifest_sha256'], log.sha)
        self.assertEqual(switched[0]['unfinished_occurrence_ids'],
                         ['occ-initial', 'occ-immediate', 'occ-delayed'])

    def test_completed_batch_can_be_followed_by_same_case_ids_in_a_new_manifest(self):
        log = Log()
        for role in ('initial', 'immediate', 'delayed'):
            log.correct(role)
        newer = deepcopy(log.manifest)
        newer['batch_id'] = 'batch-newer'
        newer['config']['seed'] = 34
        newer_sha = hashlib.sha256(json.dumps(newer, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False).encode()).hexdigest()
        log.add('batch_begin', manifest=newer, batch_id=newer['batch_id'], manifest_sha256=newer_sha)
        log.manifest, log.sha = newer, newer_sha
        for role in ('initial', 'immediate', 'delayed'):
            log.correct(role)
        report = build_report(log.end())
        self.assertTrue(report['summary']['quality_complete'])
        self.assertEqual(report['summary']['exact_correct'], 6)
        self.assertEqual(len(report['repeats']), 2)
        self.assertTrue(all(group['complete'] for group in report['repeats']))
        self.assertNotIn('batch_replaced_before_completion', {u['kind'] for u in report['unknowns']})

    def test_switching_with_last_case_active_marks_old_plan_unfinished(self):
        log = Log()
        log.correct('initial')
        log.correct('immediate')
        log.begin('delayed')
        newer = deepcopy(log.manifest)
        newer['batch_id'] = 'batch-newer'
        newer_sha = hashlib.sha256(json.dumps(newer, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False).encode()).hexdigest()
        log.add('batch_begin', manifest=newer, batch_id=newer['batch_id'], manifest_sha256=newer_sha)
        report = build_report(log.end())
        self.assertIn('batch_changed_without_case_end', report['cases'][-1]['diagnostics'])
        unknown = next(item for item in report['unknowns'] if item['kind'] == 'batch_replaced_before_completion')
        self.assertEqual(unknown['unfinished_occurrence_ids'], ['occ-delayed'])

    def test_aborted_and_truncated_correct_text_never_pass(self):
        for aborted in (True, False):
            log = Log()
            log.begin()
            log.key('a', text='aa')
            log.submit('aa', finish=False)
            if aborted:
                log.add('case_end', status='aborted', keyboard_released=None, reason='operator_stop')
                log.end()
            report = build_report(log.events)
            row = report['cases'][0]
            self.assertIsNone(row['exact_correct'])
            self.assertFalse(row['complete'])
            self.assertTrue(row['final_text_exact'])
            self.assertEqual(report['summary']['exact_correct'], 0)
            self.assertGreater(report['summary']['elapsed_seconds'], 0)
            self.assertIn('aborted' if aborted else 'missing_case_end', row['diagnostics'])

    def test_wrong_submission_is_complete_failure(self):
        log = Log()
        log.begin()
        log.key('s', text='as')
        log.submit('as')
        row = build_report(log.end())['cases'][0]
        self.assertTrue(row['complete'])
        self.assertFalse(row['exact_correct'])
        self.assertFalse(row['first_pass_correct'])
        self.assertEqual(row['edit_distance'], 1)

    def test_release_focus_and_settle_proofs_are_required(self):
        for mutation, diagnostic in (
            (lambda e: e['data'].update(keyboard_released=False), 'unconfirmed_keyboard_release'),
            (lambda e: e['data'].update(return_released=False), 'unconfirmed_return_release'),
            (lambda e: e['data'].update(focused=False), 'focus_unconfirmed'),
            (lambda e: e['data'].update(pressed_codes=[36]), 'keys_still_down'),
            (lambda e: e['data'].update(settled_ns=0), 'unconfirmed_settle'),
        ):
            log = Log()
            log.correct()
            mutation(log.events[-1])
            row = build_report(log.end())['cases'][0]
            self.assertIsNone(row['exact_correct'])
            self.assertIn(diagnostic, row['diagnostics'])
        log = Log()
        log.correct()
        del log.events[-2]
        for index, event in enumerate(log.events, 1):
            event['seq'] = index
        row = build_report(log.end())['cases'][0]
        self.assertIsNone(row['exact_correct'])
        self.assertIn('missing_return_release', row['diagnostics'])

    def test_repeated_return_is_one_submission(self):
        log = Log()
        log.begin()
        log.key('a', text='aa')
        log.add('key_down', key='Return', code=36, repeated=False)
        log.add('submit', text='aa')
        log.add('key_down', key='Return', code=36, repeated=True)
        log.add('key_up', key='Return', code=36)
        log.now += 500_000_000
        log.add('case_end', status='submitted', keyboard_released=True, reason='return_released')
        report = build_report(log.end())
        self.assertEqual(len(report['cases']), 1)
        self.assertTrue(report['cases'][0]['first_pass_correct'])

    def test_missing_or_mismatched_text_history_is_not_success(self):
        log = Log()
        log.begin()
        log.submit('aa')
        row = build_report(log.end())['cases'][0]
        self.assertTrue(row['final_text_exact'])
        self.assertIsNone(row['exact_correct'])
        self.assertIn('submit_text_mismatch', row['diagnostics'])

    def test_idle_unattempted_cases_do_not_become_attempted_coverage(self):
        log = Log()
        log.begin()
        log.add('case_end', status='aborted', keyboard_released=None, reason='operator_stop')
        report = build_report(log.end())
        self.assertEqual(report['coverage']['overall']['presented']['a'], 2)
        self.assertEqual(report['coverage']['overall']['attempted']['a'], 0)
        self.assertIn('a', report['coverage']['overall']['requested_unattempted'])

    def test_coverage_and_quality_keep_length_and_content_strata(self):
        log = Log(target='ab 12')
        log.correct()
        report = build_report(log.end())
        self.assertEqual(report['summary']['by_length_bin']['short']['exact_correct'], 1)
        self.assertEqual(report['summary']['by_length_bin']['tiny']['presented'], 0)
        self.assertEqual(report['coverage']['by_content_type']['alternating']['correct'][' '], 1)
        self.assertEqual(report['coverage']['by_length_bin']['short']['requested']['1'], 3)

    def test_strict_sequence_monotonic_malformed_and_manifest_checks(self):
        log = Log()
        log.correct()
        log.end()
        changes = [lambda e: e[1].update(seq=1), lambda e: e[1].update(seq=3),
                   lambda e: e[1].update(monotonic_ns=0),
                   lambda e: e[1].update(run_id='other'),
                   lambda e: e[0]['data'].update(manifest_sha256='0' * 64),
                   lambda e: e[1]['data'].update(target_text='a1')]
        for change in changes:
            events = deepcopy(log.events)
            change(events)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_report(events)

    def test_utc_regression_is_diagnostic_but_monotonic_is_duration_authority(self):
        log = Log()
        log.correct()
        log.events[2]['utc_ns'] = 0
        report = build_report(log.end())
        self.assertTrue(report['cases'][0]['exact_correct'])
        self.assertIn('utc_clock_regression', {unknown['kind'] for unknown in report['unknowns']})

    def test_empty_log_is_unknown_and_json_serializable(self):
        report = build_report([])
        self.assertIsNone(report['run_id'])
        self.assertEqual(report['cases'], [])
        self.assertIn('empty_event_log', {unknown['kind'] for unknown in report['unknowns']})
        self.assertIsNone(report['model_usage']['input_tokens'])
        json.dumps(report, allow_nan=False)


class TraceTests(unittest.TestCase):
    def test_usage_parts_not_double_counted_and_raw_missing_usage_stays_unknown(self):
        log = Log()
        log.correct()
        trace = [model(log.identity(), {'input_tokens': 100, 'output_tokens': 20,
                    'input_tokens_details': {'cached_tokens': 80}, 'output_tokens_details': {'reasoning_tokens': 12}}),
                 model(log.identity(), None, turn=2)]
        report = build_report(log.end(), trace)
        usage = report['model_usage']
        self.assertEqual(usage['calls'], 2)
        self.assertEqual(usage['observed'], {'input_tokens': 100, 'output_tokens': 20,
                         'cached_input_tokens': 80, 'reasoning_output_tokens': 12})
        self.assertIsNone(usage['total_input_plus_output'])
        self.assertEqual(usage['unknown_calls']['input_tokens'], 1)
        self.assertEqual(report['cases'][0]['model_usage']['elapsed_seconds'], 5)
        self.assertEqual(usage['trace_indices'], [0, 1])

    def test_complete_usage_with_explicit_zero_subsets_has_total_input_plus_output(self):
        log = Log()
        log.correct()
        trace = [model(log.identity(), dict(input_tokens=10, output_tokens=2,
                                           cached_input_tokens=0, reasoning_output_tokens=1))]
        report = build_report(log.end(), trace)
        self.assertEqual(report['model_usage']['total_input_plus_output'], 12)
        self.assertTrue(report['model_usage']['complete'])

    def test_unassigned_call_prevents_falsely_complete_per_case_usage(self):
        log = Log()
        log.correct()
        usage = dict(input_tokens=10, output_tokens=2, cached_input_tokens=0, reasoning_output_tokens=1)
        report = build_report(log.end(), [model(log.identity(), usage), model(None, usage)])
        self.assertEqual(report['model_usage']['total_input_plus_output'], 24)
        self.assertEqual(report['cases'][0]['model_usage']['observed']['input_tokens'], 10)
        self.assertIsNone(report['cases'][0]['model_usage']['total_input_plus_output'])
        self.assertFalse(report['cases'][0]['model_usage']['attribution_complete'])
        self.assertIsNone(report['summary']['normalized']['tokens_per_correct_character'])
        self.assertIsNone(report['summary']['normalized']['calls_per_correct_character'])

    def test_actor_intervals_measure_only_adjacent_same_case_wait(self):
        log = Log()
        log.correct()
        trace = [timed_execution(log.identity(), 1_000_000_000_000, 1_003_000_000_000),
                 model(log.identity(), {}),
                 timed_execution(log.identity(), 1_005_000_000_000, 1_007_000_000_000)]
        report = build_report(log.end(), trace)
        row = report['cases'][0]
        self.assertEqual(row['execution']['inter_segment_wait_seconds'], 2)
        self.assertEqual(row['execution']['inter_segment_wait_pairs'], 1)
        self.assertEqual(row['execution']['intervals'][0]['started_monotonic_ns'], 1_000_000_000_000)
        self.assertEqual(report['execution']['records'][1]['ended_monotonic_ns'], 1_007_000_000_000)
        self.assertEqual(report['summary']['by_length_bin']['tiny']['execution']['inter_segment_wait_seconds'], 2)
        self.assertNotIn('_case_key', json.dumps(report))

    def test_legacy_partial_or_overlapping_intervals_remain_unknown(self):
        for mode in ('legacy', 'partial', 'overlap'):
            log = Log()
            log.correct()
            first = timed_execution(log.identity(), 0, 3_000_000_000)
            second = execution(log.identity())
            if mode == 'partial':
                second['data']['started_monotonic_ns'] = 4_000_000_000
            elif mode == 'overlap':
                second = timed_execution(log.identity(), 2_000_000_000, 5_000_000_000)
            report = build_report(log.end(), [first, second])
            value = report['cases'][0]['execution']
            self.assertIsNone(value['inter_segment_wait_seconds'])
            self.assertEqual(value['inter_segment_wait_pairs'], 0)
            self.assertEqual(value['unknown_inter_segment_wait_pairs'], 1)
            self.assertEqual(value['observed_inter_segment_wait_seconds'], 0)

    def test_cross_case_and_nonadjacent_segments_never_invent_wait(self):
        log = Log()
        log.correct()
        log.correct('immediate')
        trace = [timed_execution(log.identity(), 0, 3_000_000_000),
                 timed_execution(log.identity('immediate'), 4_000_000_000, 5_000_000_000),
                 timed_execution(log.identity(), 6_000_000_000, 7_000_000_000)]
        report = build_report(log.end(), trace)
        self.assertIsNone(report['execution']['inter_segment_wait_seconds'])
        self.assertEqual(report['execution']['unknown_inter_segment_wait_pairs'], 2)
        first = report['cases'][0]['execution']
        self.assertIsNone(first['inter_segment_wait_seconds'])
        self.assertEqual(first['inter_segment_wait_unknown_reasons'], {'nonadjacent_segments': 1})

    def test_execution_interval_types_order_and_elapsed_are_validated(self):
        log = Log()
        log.correct()
        for change in ({'started_monotonic_ns': True}, {'ended_monotonic_ns': -1},
                       {'ended_monotonic_ns': 0}, {'elapsed_seconds': 9}):
            event = timed_execution(log.identity(), 1_000_000_000, 4_000_000_000)
            event['data'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_report(log.events, [event])
        rounded = timed_execution(log.identity(), 0, 2_000_000_123)
        rounded['data']['elapsed_seconds'] = 2.0
        self.assertEqual(build_report(log.events, [rounded])['execution']['segments'], 1)

    def test_null_wrong_run_and_text_contaminated_identity_are_never_clock_matched(self):
        log = Log()
        log.correct()
        usage = dict(input_tokens=10, output_tokens=2, cached_input_tokens=0, reasoning_output_tokens=1)
        trace = [model(None, usage), model({**log.identity(), 'run_id': 'other'}, usage),
                 model({**log.identity(), 'actual_text': 'PRIVATE-CONTAMINATION'}, usage)]
        report = build_report(log.end(), trace)
        self.assertEqual(report['model_usage']['calls'], 3)
        self.assertEqual(report['model_usage']['total_input_plus_output'], 36)
        self.assertEqual(report['cases'][0]['model_usage']['calls'], 0)
        self.assertEqual(sum(u['kind'] == 'unassigned_model_call' for u in report['unknowns']), 3)
        self.assertNotIn('PRIVATE-CONTAMINATION', json.dumps(report))

    def test_cross_occurrence_execution_only_contributes_global_and_never_fabricates_quality(self):
        log = Log()
        log.begin()
        log.add('case_end', status='aborted', keyboard_released=None, reason='operator_stop')
        trace = [execution(log.identity(), log.identity('immediate')),
                 execution(log.identity(), seconds=2)]
        report = build_report(log.end(), trace)
        self.assertEqual(report['execution']['segments'], 2)
        self.assertEqual(report['execution']['elapsed_seconds'], 5)
        self.assertEqual(report['cases'][0]['execution']['segments'], 1)
        self.assertEqual(len(report['execution']['straddled']), 1)
        self.assertIsNone(report['cases'][0]['exact_correct'])
        self.assertEqual(report['cases'][0]['versions'][0]['related_skill_ids'], ['skill-a-v1'])

    def test_missing_case_after_and_terminal_phase_are_unassigned(self):
        log = Log()
        log.correct()
        event = execution(log.identity())
        event['data']['case_after'] = None
        report = build_report(log.end(), [event, model(log.identity(phase='closed'), {})])
        self.assertEqual(report['cases'][0]['execution']['segments'], 0)
        self.assertEqual(report['cases'][0]['model_usage']['calls'], 0)
        self.assertIn('unassigned_execution', {unknown['kind'] for unknown in report['unknowns']})

    def test_unknown_unrelated_learning_events_are_ignored_without_copying_payload(self):
        log = Log()
        log.correct()
        report = build_report(log.end(), [{'event': 'model_decision', 'data': {'private': 'not-a-typing-trace'}}])
        self.assertNotIn('not-a-typing-trace', json.dumps(report))
        self.assertEqual(report['execution']['segments'], 0)

    def test_usage_and_elapsed_reject_impossible_values(self):
        log = Log()
        log.correct()
        for trace in ([model(log.identity(), {'input_tokens': -1})],
                      [model(log.identity(), {'input_tokens': 1, 'cached_input_tokens': 2})],
                      [model(log.identity(), {}, elapsed=float('nan'))]):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                build_report(log.events, trace)


class ComparisonTests(unittest.TestCase):
    def test_pairs_require_same_id_role_target_and_report_distribution_shift(self):
        before = Log(run_id='before')
        before.begin()
        before.key('s', text='as')
        before.submit('as')
        old = build_report(before.end(), [execution(before.identity(), program='c' * 64)])
        after = Log(run_id='after')
        after.correct()
        after.correct('immediate')
        new = build_report(after.end(), [execution(after.identity())])
        report = compare_reports(old, new)
        self.assertEqual(len(report['matched']), 1)
        self.assertEqual(report['matched'][0]['delta']['exact_correct'], 1)
        self.assertEqual(report['matched'][0]['delta']['edit_distance'], -1)
        self.assertEqual(report['unmatched_after'], ['occ-immediate'])
        self.assertTrue(report['distributions']['changed'])
        self.assertEqual(report['matched'][0]['before_versions'][0]['program_sha256'], 'c' * 64)
        changed_target = Log(target='ab', run_id='changed')
        changed_target.correct()
        comparison = compare_reports(old, build_report(changed_target.end()))
        self.assertEqual(comparison['matched'], [])
        self.assertEqual(comparison['unmatched_before'], ['occ-initial'])

    def test_ambiguous_stable_ids_are_not_arbitrarily_paired(self):
        log = Log()
        log.correct()
        report = build_report(log.end())
        duplicate = deepcopy(report)
        duplicate['cases'].append(deepcopy(duplicate['cases'][0]))
        comparison = compare_reports(report, duplicate)
        self.assertEqual(comparison['matched'], [])
        self.assertEqual(comparison['ambiguous'][0]['after_count'], 2)


class NormalizedMetricsTests(unittest.TestCase):
    def test_rates_include_failed_case_time_and_cost_in_overall_and_strata(self):
        log = Log()
        log.begin()
        log.key('s', text='as')
        log.key('BackSpace', text='a')
        log.add('reset', reason='explicit_training_reset')
        log.add('text_changed', text='', source='training_reset')
        log.key('a', text='aa')
        log.submit('aa')
        log.begin('immediate')
        log.key('s', text='as')
        log.submit('as')
        trace = [model(log.identity(), dict(input_tokens=100, output_tokens=20,
                    cached_input_tokens=0, reasoning_output_tokens=1)),
                 model(log.identity('immediate'), dict(input_tokens=30, output_tokens=10,
                    cached_input_tokens=0, reasoning_output_tokens=1), turn=2)]
        report = build_report(log.end(), trace)
        first, failed = report['cases']
        norm = first['normalized']
        self.assertEqual(norm['correct_submitted_characters'], 2)
        self.assertAlmostEqual(norm['correct_characters_per_minute'], 120 / first['elapsed_seconds'])
        self.assertEqual(norm['backspaces_per_100_target_characters'], 50)
        self.assertEqual(norm['resets_per_100_target_characters'], 50)
        self.assertEqual(norm['tokens_per_correct_character'], 60)
        self.assertEqual(norm['calls_per_correct_character'], .5)
        self.assertEqual(failed['normalized']['correct_characters_per_minute'], 0)
        self.assertIsNone(failed['normalized']['tokens_per_correct_character'])
        overall = report['summary']['normalized']
        self.assertEqual(overall['target_characters'], 4)
        self.assertEqual(overall['backspaces_per_100_target_characters'], 25)
        self.assertEqual(overall['resets_per_100_target_characters'], 25)
        self.assertEqual(overall['tokens_per_correct_character'], 80)
        self.assertEqual(overall['calls_per_correct_character'], 1)
        self.assertEqual(overall['elapsed_basis'], 'run_event_window')
        self.assertAlmostEqual(overall['correct_characters_per_minute'], 120 / report['summary']['wall_elapsed_seconds'])
        for group in (report['summary']['by_length_bin']['tiny'],
                      report['summary']['by_content_type']['alternating']):
            self.assertEqual(group['normalized']['tokens_per_correct_character'], 80)
            self.assertEqual(group['normalized']['elapsed_basis'], 'sum_case_windows')
            self.assertAlmostEqual(group['normalized']['correct_characters_per_minute'], 120 / group['elapsed_seconds'])

    def test_zero_denominators_and_missing_cost_stay_null(self):
        empty = build_report([])['summary']['normalized']
        for name in ('correct_characters_per_minute', 'backspaces_per_100_target_characters',
                     'resets_per_100_target_characters', 'tokens_per_correct_character', 'calls_per_correct_character'):
            self.assertIsNone(empty[name])
        log = Log()
        log.begin()
        report = build_report(log.events[:2])
        self.assertEqual(report['cases'][0]['elapsed_seconds'], 0)
        self.assertIsNone(report['cases'][0]['normalized']['correct_characters_per_minute'])
        log = Log()
        log.correct()
        report = build_report(log.end())
        self.assertIsNone(report['cases'][0]['normalized']['tokens_per_correct_character'])
        self.assertIsNone(report['cases'][0]['normalized']['calls_per_correct_character'])

    def test_normalized_and_measured_wait_deltas_are_available_for_repeats_and_runs(self):
        log = Log()
        log.correct()
        log.correct('immediate')
        usage = dict(input_tokens=100, output_tokens=20, cached_input_tokens=0, reasoning_output_tokens=1)
        trace = [model(log.identity(), usage), model(log.identity('immediate'), usage, turn=2),
                 timed_execution(log.identity(), 0, 1_000_000_000),
                 timed_execution(log.identity(), 3_000_000_000, 4_000_000_000),
                 timed_execution(log.identity('immediate'), 5_000_000_000, 6_000_000_000),
                 timed_execution(log.identity('immediate'), 7_000_000_000, 8_000_000_000)]
        before = build_report(log.end(), trace)
        repeat = before['repeats'][0]['deltas']['immediate_minus_initial']
        self.assertEqual(repeat['inter_segment_wait_seconds'], -1)
        for field in ('correct_characters_per_minute', 'backspaces_per_100_target_characters',
                      'resets_per_100_target_characters', 'tokens_per_correct_character', 'calls_per_correct_character'):
            self.assertIn(field, repeat)
        new_trace = deepcopy(trace)
        new_trace[0]['data']['usage']['input_tokens'] = 40
        after = build_report(log.events, new_trace)
        pair = compare_reports(before, after)['matched'][0]
        self.assertEqual(pair['delta']['tokens_per_correct_character'], -30)
        self.assertEqual(pair['delta']['calls_per_correct_character'], 0)


class IsolationTests(unittest.TestCase):
    def test_metrics_imports_are_stdlib_or_records_usage_and_inputs_are_not_mutated(self):
        path = Path(__file__).parents[1] / 'src/astra_robotics_agent/physical_keyboard/records/typing_metrics.py'
        tree = ast.parse(path.read_text())
        modules = [node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertLessEqual(set(modules), {'collections', 'copy', 'hashlib', 'json', 'math', 're', 'usage'})
        local = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]
        self.assertEqual([(node.level, node.module) for node in local], [(1, 'usage')])
        log = Log()
        log.correct()
        events, trace = log.end(), [model(log.identity(), {})]
        frozen_events, frozen_trace = deepcopy(events), deepcopy(trace)
        report = build_report(events, trace)
        self.assertEqual(events, frozen_events)
        self.assertEqual(trace, frozen_trace)
        comparison = compare_reports(report, report)
        self.assertEqual(comparison['matched'][0]['delta']['exact_correct'], 0)


if __name__ == '__main__':
    unittest.main()
