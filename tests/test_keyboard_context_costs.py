"""Maintenance token measurements remain separate from inference decisions."""
from copy import deepcopy
import unittest

from astra_robotics_agent.physical_keyboard.records.typing_metrics import build_report
from test_keyboard_typing_metrics import Log, model


DECISION = dict(input_tokens=100, cached_input_tokens=80, output_tokens=20,
                reasoning_output_tokens=12, total_tokens=120)
CONTEXT = dict(input_tokens=5, cached_input_tokens=1, output_tokens=2,
               reasoning_output_tokens=1, total_tokens=7)
ZERO = dict.fromkeys(CONTEXT, 0)


def operation(case, usage, *, method='native_compact', status='completed', complete=True,
              turn=1, seconds=1):
    return dict(event='typing_lab_context_operation', utc_ns=1, data=dict(case=case,
        turn_index=turn, method=method, status=status, usage=usage,
        elapsed_seconds=seconds, usage_complete=complete))


class ContextCostTests(unittest.TestCase):
    def setUp(self):
        self.log = Log()
        self.log.correct()
        self.events = self.log.end()
        self.case = self.log.identity()

    def report(self, *records):
        return build_report(self.events, records)

    def test_complete_cost_includes_native_and_rebuild_once_with_explicit_categories(self):
        report = self.report(model(self.case, DECISION), operation(self.case, CONTEXT),
            operation(self.case, ZERO, method='rebuild', turn=2, seconds=.1))
        usage = report['model_usage']
        self.assertEqual((usage['calls'], usage['decision_calls'], usage['context_operations']), (3, 1, 2))
        self.assertIn('attempt', usage['call_count_scope'])
        self.assertIn('not necessarily provider RPCs', usage['context_operation_scope'])
        self.assertEqual(usage['total_input_plus_output'], 127)
        self.assertEqual(usage['cached_input_tokens'], 81)
        self.assertEqual(usage['reasoning_output_tokens'], 13)
        self.assertTrue(usage['complete'])
        self.assertEqual(usage['by_category']['decision']['total_input_plus_output'], 120)
        self.assertEqual(usage['by_category']['context_operation']['total_input_plus_output'], 7)
        self.assertEqual(usage['context_operation_methods'], {'native_compact': 1, 'rebuild': 1})
        self.assertEqual(usage['context_operation_statuses'], {'completed': 2})
        self.assertEqual(usage['by_category']['context_operation']['trace_indices'], [1, 2])
        self.assertAlmostEqual(usage['elapsed_seconds'], 3.6)
        row = report['cases'][0]
        self.assertEqual(row['model_usage']['total_input_plus_output'], 127)
        self.assertEqual(row['segments']['decision_seconds'], 2.5)
        self.assertAlmostEqual(row['segments']['context_operation_seconds'], 1.1)
        self.assertEqual(row['normalized']['tokens_per_correct_character'], 127 / 2)
        self.assertEqual(row['normalized']['decision_calls_per_correct_character'], .5)
        self.assertEqual(row['normalized']['context_operations_per_correct_character'], 1)
        self.assertEqual(row['normalized']['calls_per_correct_character'], 1.5)
        self.assertEqual(report['repeats'][0]['roles']['initial']['metrics']['model_calls'], 1)
        self.assertEqual(report['summary']['by_length_bin']['tiny']['model_usage']['context_operations'], 2)
        self.assertEqual(report['summary']['by_content_type']['alternating']['model_usage']['context_operations'], 2)

    def test_unsupported_operation_is_counted_but_missing_usage_is_never_free(self):
        report = self.report(model(self.case, DECISION),
            operation(self.case, None, status='unsupported', complete=False))
        usage = report['model_usage']
        self.assertEqual((usage['calls'], usage['decision_calls'], usage['context_operations']), (2, 1, 1))
        self.assertEqual(usage['context_operation_statuses'], {'unsupported': 1})
        self.assertEqual(usage['observed']['input_tokens'], 100)
        self.assertEqual(set(usage['unknown_calls'].values()), {1})
        self.assertEqual(usage['incomplete_context_operations'], 1)
        self.assertFalse(usage['complete'])
        self.assertIsNone(usage['total_input_plus_output'])
        self.assertTrue(usage['by_category']['decision']['complete'])
        self.assertIsNone(usage['by_category']['context_operation']['total_input_plus_output'])
        self.assertIsNone(report['cases'][0]['normalized']['tokens_per_correct_character'])

    def test_partial_native_usage_keeps_measured_lower_bound_without_washing_complete(self):
        partial = dict(input_tokens=10, output_tokens=3, cached_input_tokens=2, reasoning_output_tokens=1)
        report = self.report(model(self.case, DECISION), operation(self.case, partial, complete=False))
        usage = report['model_usage']
        self.assertEqual(usage['observed'], dict(input_tokens=110, output_tokens=23,
            cached_input_tokens=82, reasoning_output_tokens=13))
        self.assertEqual(usage['observed_total_input_plus_output'], 133)
        self.assertEqual(set(usage['unknown_calls'].values()), {1})
        for field in partial:
            self.assertIsNone(usage[field])
        self.assertFalse(usage['complete'])
        self.assertIsNone(usage['total_input_plus_output'])
        context = usage['by_category']['context_operation']
        self.assertEqual(context['observed']['input_tokens'], 10)
        self.assertEqual(context['observed_total_input_plus_output'], 13)
        self.assertIsNone(context['input_tokens'])
        self.assertEqual(sum(row['kind'] == 'incomplete_context_operation_usage' for row in report['unknowns']), 1)

    def test_explicit_measured_zero_can_be_known_even_for_schema_failure_attempt(self):
        report = self.report(operation(self.case, ZERO, method=None, status='failed'))
        usage = report['model_usage']
        self.assertEqual((usage['calls'], usage['decision_calls'], usage['context_operations']), (1, 0, 1))
        self.assertEqual(usage['total_input_plus_output'], 0)
        self.assertTrue(usage['complete'])
        self.assertEqual(usage['context_operation_methods'], {'unknown': 1})
        self.assertEqual(usage['context_operation_statuses'], {'failed': 1})
        self.assertIn('no_model_turn_records', {row['kind'] for row in report['unknowns']})
        self.assertIsNone(usage['by_category']['decision']['total_input_plus_output'])

    def test_failed_attempt_keeps_unknown_and_zero_difference_does_not_prove_completeness(self):
        for raw in (None, ZERO):
            with self.subTest(raw=raw):
                usage = self.report(operation(self.case, raw, method='rebuild', status='failed', complete=False))['model_usage']
                self.assertEqual(usage['context_operations'], 1)
                self.assertFalse(usage['complete'])
                self.assertIsNone(usage['total_input_plus_output'])
                self.assertEqual(set(usage['unknown_calls'].values()), {1})

    def test_unassigned_context_cost_is_global_and_blocks_precise_case_attribution(self):
        report = self.report(model(self.case, DECISION), operation(None, CONTEXT))
        usage, row = report['model_usage'], report['cases'][0]
        self.assertEqual(usage['total_input_plus_output'], 127)
        self.assertFalse(usage['case_attribution_complete'])
        self.assertEqual(row['model_usage']['observed']['input_tokens'], 100)
        self.assertIsNone(row['model_usage']['input_tokens'])
        self.assertIsNone(row['normalized']['tokens_per_correct_character'])
        self.assertIn('unassigned_context_operation', {item['kind'] for item in report['unknowns']})

    def test_known_parent_counts_and_unknown_subsets_remain_independently_visible(self):
        usage = self.report(operation(self.case, {'input_tokens': 5, 'output_tokens': 2}))['model_usage']
        self.assertEqual(usage['total_input_plus_output'], 7)
        self.assertIsNone(usage['cached_input_tokens'])
        self.assertIsNone(usage['reasoning_output_tokens'])
        self.assertFalse(usage['complete'])
        self.assertEqual(usage['incomplete_context_operations'], 1)

    def test_no_context_events_keeps_existing_report_shape_and_model_usage_semantics(self):
        event = model(self.case, DECISION)
        original = self.report(event)
        legacy_extra_flag = deepcopy(event)
        legacy_extra_flag['data']['usage_complete'] = False
        self.assertEqual(original, self.report(legacy_extra_flag))
        usage = original['model_usage']
        self.assertEqual(usage['calls'], 1)
        self.assertNotIn('decision_calls', usage)
        self.assertNotIn('context_operations', usage)
        self.assertNotIn('by_category', usage)
        self.assertEqual(usage['call_count_scope'], 'observed_records_only')
        self.assertEqual(usage['total_input_plus_output'], 120)

    def test_malformed_context_records_fail_closed_with_no_input_mutation(self):
        for change in ({'usage_complete': 1}, {'usage_complete': None}, {'method': {}},
                       {'method': None}, {'status': 'cancelled'}, {'turn_index': True},
                       {'elapsed_seconds': float('nan')}, {'usage': {'input_tokens': -1}},
                       {'extra_field': 'not-in-contract'}):
            event = operation(self.case, CONTEXT)
            event['data'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.report(event)
        trace = [model(self.case, DECISION), operation(self.case, CONTEXT, complete=False)]
        frozen = deepcopy(trace)
        self.report(*trace)
        self.assertEqual(trace, frozen)


if __name__ == '__main__':
    unittest.main()
