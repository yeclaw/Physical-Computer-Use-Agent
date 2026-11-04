"""Strict token measurements and read-only replay of the selected 24 cases."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from astra_robotics_agent.physical_keyboard.records.typing_metrics import build_report
from astra_robotics_agent.physical_keyboard.records.usage import (
    USAGE_FIELDS, UsageNormalizationError, normalize_usage,
)


SNAKE = dict(input_tokens=100, cached_input_tokens=80, output_tokens=20,
             reasoning_output_tokens=12, total_tokens=120)
CAMEL = dict(inputTokens=100, cachedInputTokens=80, outputTokens=20,
             reasoningOutputTokens=12, totalTokens=120)


def trace(usage, index=1, status='completed'):
    return {'event': 'typing_lab_model_turn', 'utc_ns': 1,
            'data': {'case': None, 'turn_index': index, 'usage': usage,
                     'elapsed_seconds': 1, 'status': status}}


class UsageNormalizationTests(unittest.TestCase):
    def test_flat_snake_camel_and_equal_mixed_aliases_are_equivalent(self):
        for raw in (SNAKE, CAMEL, {**SNAKE, **CAMEL}):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_usage(raw), SNAKE)

    def test_historical_nested_details_and_flat_counts_are_compatible(self):
        for raw in (
            {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
             'input_tokens_details': {'cached_tokens': 80},
             'output_tokens_details': {'reasoning_tokens': 12}},
            {'inputTokens': 100, 'outputTokens': 20, 'totalTokens': 120,
             'inputTokensDetails': {'cachedTokens': 80},
             'outputTokensDetails': {'reasoningTokens': 12}},
            {**SNAKE, 'input_tokens_details': {'cached_tokens': 80},
             'outputTokensDetails': {'reasoningTokens': 12}},
        ):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_usage(raw), SNAKE)

    def test_all_missing_fields_remain_unknown_and_no_total_is_inferred(self):
        for raw in (None, {}, {'status': 'cancelled'}, dict.fromkeys(USAGE_FIELDS)):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_usage(raw), dict.fromkeys(USAGE_FIELDS))
        for field in USAGE_FIELDS:
            raw = {field: 0}
            self.assertEqual(normalize_usage(raw), {**dict.fromkeys(USAGE_FIELDS), field: 0})
        self.assertIsNone(normalize_usage({'input_tokens': 5, 'output_tokens': 2})['total_tokens'])

    def test_null_alias_is_unknown_and_another_measurement_can_supply_it(self):
        raw = {'input_tokens': None, **CAMEL, 'cached_input_tokens': None,
               'input_tokens_details': {'cached_tokens': 80},
               'output_tokens_details': None}
        self.assertEqual(normalize_usage(raw), SNAKE)

    def test_disagreeing_flat_or_nested_measurements_are_rejected(self):
        for field, alias in zip(USAGE_FIELDS, CAMEL):
            with self.subTest(field=field), self.assertRaisesRegex(UsageNormalizationError, 'Conflicting'):
                normalize_usage({**SNAKE, alias: SNAKE[field] + 1})
        for extra in (
            {'input_tokens_details': {'cached_tokens': 79}},
            {'outputTokensDetails': {'reasoningTokens': 11}},
            {'input_tokens_details': {'cached_tokens': 80, 'cachedTokens': 79}},
            {'output_tokens_details': {'reasoning_tokens': 12},
             'outputTokensDetails': {'reasoningTokens': 11}},
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(UsageNormalizationError, 'Conflicting'):
                normalize_usage({**SNAKE, **extra})

    def test_counts_reject_booleans_strings_floats_negative_and_nonfinite(self):
        invalid = (True, False, -1, 1.0, float('inf'), float('-inf'), float('nan'), '1', [], {})
        for field in (*USAGE_FIELDS, *CAMEL):
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(UsageNormalizationError):
                    normalize_usage({field: value})
        for container, field in (('input_tokens_details', 'cached_tokens'),
                                  ('outputTokensDetails', 'reasoningTokens')):
            for value in invalid:
                with self.subTest(container=container, value=value), self.assertRaises(UsageNormalizationError):
                    normalize_usage({container: {field: value}})

    def test_invalid_envelopes_and_detail_containers_fail_closed(self):
        for value in (False, 0, '', [], (), 'usage'):
            with self.subTest(value=value), self.assertRaises(UsageNormalizationError):
                normalize_usage(value)
        for name in ('input_tokens_details', 'inputTokensDetails',
                     'output_tokens_details', 'outputTokensDetails'):
            for value in (False, 0, '', [], 'details'):
                with self.subTest(name=name, value=value), self.assertRaises(UsageNormalizationError):
                    normalize_usage({name: value})

    def test_subsets_and_provided_total_cannot_contradict_parents(self):
        for raw in (
            {'input_tokens': 1, 'cachedInputTokens': 2},
            {'outputTokens': 1, 'reasoning_output_tokens': 2},
            {'inputTokens': 2, 'total_tokens': 1},
            {'output_tokens': 2, 'totalTokens': 1},
            {'cached_input_tokens': 2, 'total_tokens': 1},
            {'reasoningOutputTokens': 2, 'totalTokens': 1},
            {'input_tokens': 5, 'output_tokens': 2, 'total_tokens': 8},
            {'inputTokens': 5, 'outputTokens': 2, 'totalTokens': 6},
        ):
            with self.subTest(raw=raw), self.assertRaises(UsageNormalizationError):
                normalize_usage(raw)
        # A missing parent cannot erase a measured child or be inferred from it.
        self.assertEqual(normalize_usage({'cachedInputTokens': 2})['cached_input_tokens'], 2)
        self.assertIsNone(normalize_usage({'cachedInputTokens': 2})['input_tokens'])

    def test_inputs_are_not_mutated_unknown_metadata_is_not_republished(self):
        raw = {**CAMEL, 'input_tokens_details': {'cached_tokens': 80},
               'provider_extra': {'private': [1, 2]}}
        frozen = deepcopy(raw)
        normalized = normalize_usage(raw)
        self.assertEqual(raw, frozen)
        self.assertEqual(set(normalized), set(USAGE_FIELDS))
        normalized['input_tokens'] = 0
        self.assertEqual(raw, frozen)
        path = Path(__file__).parents[1] / 'src/astra_robotics_agent/physical_keyboard/records/usage.py'
        tree = ast.parse(path.read_text())
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)))


class TypingUsageCompatibilityTests(unittest.TestCase):
    def test_report_shape_and_totals_ignore_cached_and_reasoning_subsets(self):
        snake = build_report([], [trace(SNAKE)])
        camel = build_report([], [trace(CAMEL)])
        nested = {key: value for key, value in SNAKE.items()
                  if key not in ('cached_input_tokens', 'reasoning_output_tokens')}
        nested.update(input_tokens_details={'cached_tokens': 80},
                      output_tokens_details={'reasoning_tokens': 12})
        self.assertEqual(snake, camel)
        self.assertEqual(snake, build_report([], [trace(nested)]))
        usage = snake['model_usage']
        self.assertTrue(usage['complete'])
        self.assertEqual(usage['total_input_plus_output'], 120)
        self.assertEqual(usage['observed'], {key: value for key, value in SNAKE.items() if key != 'total_tokens'})

    def test_cancelled_missing_usage_remains_unknown_including_all_observed_cost(self):
        report = build_report([], [trace(CAMEL), trace(None, 2, status='cancelled')])
        usage = report['model_usage']
        self.assertEqual(usage['calls'], 2)
        self.assertFalse(usage['complete'])
        self.assertEqual(usage['observed']['input_tokens'], 100)
        self.assertEqual(set(usage['unknown_calls'].values()), {1})
        self.assertIsNone(usage['total_input_plus_output'])
        self.assertEqual(sum(row['kind'] == 'incomplete_model_usage' for row in report['unknowns']), 1)

    def test_missing_subsets_stay_unknown_without_hiding_known_input_output(self):
        usage = build_report([], [trace({'inputTokens': 100, 'outputTokens': 20})])['model_usage']
        self.assertFalse(usage['complete'])
        self.assertIsNone(usage['cached_input_tokens'])
        self.assertIsNone(usage['reasoning_output_tokens'])
        self.assertEqual(usage['total_input_plus_output'], 120)

    def test_conflicting_trace_fails_instead_of_silently_choosing_one_alias(self):
        with self.assertRaises(UsageNormalizationError):
            build_report([], [trace({**CAMEL, 'input_tokens': 1})])


class HistoricalTypingUsageTests(unittest.TestCase):
    def test_original_camel_traces_match_frozen_derived_report_and_keep_four_unknown(self):
        root = Path(__file__).parents[1]
        review = root / 'artifacts/typing-batch-24-review'
        provenance = review / 'case-analysis-provenance.json'
        if not provenance.is_file():
            self.skipTest('Preserved typing-batch-24-review evidence is not present')
        metadata = json.loads(provenance.read_text())
        source_paths = [root / name for name in metadata['inputs']
                        if name.endswith('learning-events.jsonl')]
        events_path = review / 'inputs/resumed-a02-events.jsonl'
        expected_path = review / 'full-derived-report.json'
        protected = [provenance, expected_path, events_path, *source_paths]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        for path in (events_path, *source_paths):
            self.assertEqual(before[path], metadata['inputs'][path.relative_to(root).as_posix()]['sha256'])
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        traces = []
        for path in source_paths:
            with path.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get('event') in ('typing_lab_model_turn', 'typing_lab_execution'):
                        traces.append(event)
        models = [event for event in traces if event['event'] == 'typing_lab_model_turn']
        self.assertEqual(len(models), 583)
        self.assertEqual(sum(event['data'].get('usage') is None for event in models), 4)
        self.assertEqual(sum('inputTokens' in (event['data'].get('usage') or {}) for event in models), 579)
        report = build_report(events, traces)
        self.assertEqual(report, json.loads(expected_path.read_text()))
        selected = report['cases'][:24]
        self.assertEqual((sum(row['complete'] for row in selected),
                          sum(row['exact_correct'] is True for row in selected),
                          sum(row['first_pass_correct'] is True for row in selected)), (24, 23, 14))
        self.assertEqual(report['cases'][24]['key_down_count'], 0)
        usage = report['model_usage']
        self.assertEqual(usage['observed'], dict(input_tokens=98929343,
            cached_input_tokens=84071040, output_tokens=346260, reasoning_output_tokens=124423))
        self.assertEqual(set(usage['unknown_calls'].values()), {4})
        self.assertIsNone(usage['total_input_plus_output'])
        self.assertEqual(sum(row['kind'] == 'incomplete_model_usage' for row in report['unknowns']), 4)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})


if __name__ == '__main__':
    unittest.main()
