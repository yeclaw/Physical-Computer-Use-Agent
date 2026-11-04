"""Pinned prefix decisions with real deterministic manifests; no I/O or SDK."""

import copy
import json
import unittest

from astra_robotics_agent.physical_keyboard.adapters.environments.typing_cases import (
    ALPHABET, DEFAULT_CONFIG, generate_manifest, manifest_sha256,
)
from astra_robotics_agent.physical_keyboard.adapters.environments.typing_schedule import TypingSchedulePrefix


class TypingSchedulePrefixTests(unittest.TestCase):
    def setUp(self):
        self.manifest = generate_manifest(DEFAULT_CONFIG)
        self.pins = dict(expected_run_id='fixture-editor', expected_batch_id=self.manifest['batch_id'],
                         expected_manifest_sha256=manifest_sha256(self.manifest))
        self.schedule = TypingSchedulePrefix(self.manifest, occurrence_limit=24, **self.pins)

    def identity(self, index=0, **updates):
        value = dict(protocol='physical-keyboard-typing-case-identity-v1',
            run_id=self.pins['expected_run_id'], batch_id=self.pins['expected_batch_id'],
            manifest_sha256=self.pins['expected_manifest_sha256'],
            **self.manifest['occurrences'][index], phase='typing')
        return dict(value, **updates)

    def test_selected_prefix_allows_start_current_and_last_occurrences(self):
        for index in (0, 11, 12, 23):
            for phase in ('typing', 'awaiting_release'):
                with self.subTest(index=index, phase=phase):
                    result = self.schedule.check_identity(self.identity(index, phase=phase))
                    self.assertEqual(result, dict(status='inside_prefix', should_stop=False,
                        original_index=index, occurrence_limit=24, original_occurrence_count=48))

    def test_exact_twenty_fifth_occurrence_stops_before_new_model_work(self):
        result = self.schedule.check_identity(self.identity(24))
        self.assertEqual(result['status'], 'next_outside_boundary')
        self.assertTrue(result['should_stop'])
        self.assertEqual(result['original_index'], 24)
        for phase in ('awaiting_release', 'batch_complete', 'closed'):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                self.schedule.check_identity(self.identity(24, phase=phase))

    def test_skipping_exact_boundary_never_means_complete_or_continue(self):
        for index in (25, 30, 47):
            for phase in ('typing', 'awaiting_release', 'batch_complete'):
                with self.subTest(index=index, phase=phase), self.assertRaises(ValueError):
                    self.schedule.check_identity(self.identity(index, phase=phase))

    def test_unknown_missing_foreign_or_mismatched_identity_is_rejected(self):
        invalid = [None, {}, dict(self.identity(), occurrence_id='case-unknown-r1'),
            dict(self.identity(), run_id='other-editor'), dict(self.identity(), batch_id='other-batch'),
            dict(self.identity(), manifest_sha256='0' * 64), dict(self.identity(), case_id='other-case'),
            dict(self.identity(), repeat_index=2, role='immediate'),
            dict(self.identity(), target_text='not permitted'), dict(self.identity(), exact_correct=True)]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.schedule.check_identity(value)

    def test_closed_or_premature_batch_complete_inside_prefix_is_rejected(self):
        for index in (0, 12, 23):
            for phase in ('closed', 'batch_complete'):
                with self.subTest(index=index, phase=phase), self.assertRaises(ValueError):
                    self.schedule.check_identity(self.identity(index, phase=phase))

    def test_limit_is_strict_bounded_integer(self):
        for limit in (False, True, 0, -1, 49, 1.0, '24', None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                TypingSchedulePrefix(self.manifest, occurrence_limit=limit, **self.pins)
        one = TypingSchedulePrefix(self.manifest, occurrence_limit=1, **self.pins)
        self.assertEqual(one.check_identity(self.identity(1))['status'], 'next_outside_boundary')

    def test_constructor_checks_run_batch_and_manifest_pins(self):
        changes = [dict(expected_run_id=None), dict(expected_run_id='bad/run'),
            dict(expected_batch_id='different-batch'), dict(expected_batch_id=[]),
            dict(expected_manifest_sha256='0' * 64), dict(expected_manifest_sha256='invalid')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                TypingSchedulePrefix(self.manifest, occurrence_limit=24, **dict(self.pins, **change))
        altered = copy.deepcopy(self.manifest)
        altered['occurrences'].pop()
        with self.assertRaises(ValueError):
            TypingSchedulePrefix(altered, occurrence_limit=24, **self.pins)

    def test_full_manifest_limit_stops_only_after_last_release(self):
        complete = TypingSchedulePrefix(self.manifest, occurrence_limit=48, **self.pins)
        self.assertFalse(complete.check_identity(self.identity(47))['should_stop'])
        self.assertFalse(complete.check_identity(self.identity(47, phase='awaiting_release'))['should_stop'])
        self.assertEqual(complete.check_identity(self.identity(47, phase='batch_complete'))['status'],
                         'full_manifest_complete')
        with self.assertRaises(ValueError):
            complete.check_identity(self.identity(46, phase='batch_complete'))

    def test_manifest_and_caller_identity_remain_unchanged_and_text_is_not_retained(self):
        before = copy.deepcopy(self.manifest)
        identity = self.identity(12)
        identity_before = dict(identity)
        self.schedule.check_identity(identity)
        descriptor = self.schedule.describe()
        self.assertEqual(self.manifest, before)
        self.assertEqual(identity, identity_before)
        self.assertEqual(descriptor['selected_occurrence_ids'],
                         [item['occurrence_id'] for item in self.manifest['occurrences'][:24]])
        self.assertEqual(descriptor['next_outside_occurrence_id'], self.manifest['occurrences'][24]['occurrence_id'])
        descriptor['selected_occurrence_ids'].clear()
        self.assertEqual(len(self.schedule.describe()['selected_occurrence_ids']), 24)
        serialized = json.dumps(vars(self.schedule))
        self.assertNotIn('target_text', serialized)
        self.assertNotIn('current_text', serialized)
        for case in self.manifest['base_cases']:
            self.assertNotIn('"' + case['target_text'] + '"', serialized)

    def test_prefix_offline_coverage_remains_full_and_keeps_original_repeat_gaps(self):
        bases = {case['case_id']: case for case in self.manifest['base_cases']}
        selected = self.manifest['occurrences'][:24]
        completed_ids = {item['case_id'] for item in selected[:12]}
        remaining_ids = {item['case_id'] for item in selected[12:]}
        def text(case_ids):
            return ''.join(bases[case_id]['target_text'] for case_id in case_ids)
        self.assertEqual(len(set(ALPHABET) & set(text(completed_ids))), 24)
        self.assertEqual(set(ALPHABET) - set(text(remaining_ids)), set())
        self.assertEqual(set(bases[case_id]['length_bin'] for case_id in completed_ids | remaining_ids),
                         {'tiny', 'short', 'medium', 'long'})
        positions = {}
        for index, occurrence in enumerate(selected):
            positions.setdefault(occurrence['case_id'], []).append(index)
        for first, immediate, delayed in positions.values():
            self.assertEqual(immediate, first + 1)
            self.assertTrue(2 <= delayed - immediate - 1 <= 5)


if __name__ == '__main__':
    unittest.main()
