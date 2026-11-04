"""Real task state and evaluator with fixture keys/clocks, no GTK/hardware IO."""
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.adapters.environments.typing_cases import DEFAULT_CONFIG, generate_manifest
from astra_robotics_agent.physical_keyboard.adapters.environments.typing_lab import TrainingSession, read_case_identity
from astra_robotics_agent.physical_keyboard.records.typing_metrics import build_report


class Clock:
    value = 1000000000
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += int(seconds * 1000000000)


class TypingLabSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.events, self.identities = [], []
        self.manifest = generate_manifest(DEFAULT_CONFIG)
        self.session = TrainingSession(self.manifest, run_id='lab:fixture', emit=self.events.append,
            publish_identity=self.identities.append, clock_ns=self.clock, utc_ns=self.clock)

    def input_text(self, text):
        value = ''
        for char in text:
            self.clock.advance(.01)
            code = ord(char)
            self.session.key_down(char, code)
            value += char
            self.session.text_changed(value)
            self.session.key_up(char, code)

    def submit(self):
        value = self.session.key_down('Return', 13)
        self.assertTrue(value['consume'])
        self.assertTrue(value['submitted'])
        self.session.key_up('Return', 13)
        self.clock.advance(.6)
        self.assertTrue(self.session.tick())

    def test_return_repeat_cannot_submit_the_next_case_and_all_keys_must_release(self):
        original = self.session.identity()
        self.input_text(self.session.snapshot()['target_text'])
        self.session.key_down('x', 88)
        self.session.key_down('Return', 13)
        for _ in range(4):
            self.clock.advance(.2)
            self.assertFalse(self.session.key_down('Return', 13)['submitted'])
            self.assertFalse(self.session.tick())
        self.assertEqual(sum(event['event'] == 'submit' for event in self.events), 1)
        self.session.key_up('Return', 13)
        self.clock.advance(1)
        self.assertFalse(self.session.tick())
        self.session.key_up('x', 88)
        self.clock.advance(.6)
        self.assertTrue(self.session.tick())
        self.assertNotEqual(self.session.identity()['occurrence_id'], original['occurrence_id'])
        self.assertEqual(self.session.snapshot()['current_text'], '')
        self.assertEqual(sum(event['event'] == 'submit' for event in self.events), 1)

    def test_focus_loss_does_not_advance_or_fabricate_release(self):
        self.input_text(self.session.snapshot()['target_text'])
        self.session.key_down('Return', 13)
        self.session.key_up('Return', 13)
        self.session.focus_changed(False)
        self.clock.advance(2)
        self.assertFalse(self.session.tick())
        self.session.focus_changed(True)
        self.assertFalse(self.session.tick())
        self.clock.advance(.6)
        self.assertTrue(self.session.tick())

    def test_full_batch_is_independently_scored_and_correction_not_first_pass(self):
        for index in range(len(self.manifest['occurrences'])):
            target = self.session.snapshot()['target_text']
            self.input_text(target)
            if index == 0:
                self.session.key_down('x', 88)
                self.session.text_changed(target + 'x')
                self.session.key_up('x', 88)
                self.session.key_down('BackSpace', 22)
                self.session.text_changed(target)
                self.session.key_up('BackSpace', 22)
            self.submit()
        self.assertEqual(self.session.phase, 'batch_complete')
        self.session.close()
        report = build_report(self.events)
        self.assertEqual(len(report['cases']), 48)
        self.assertTrue(report['cases'][0]['exact_correct'])
        self.assertFalse(report['cases'][0]['first_pass_correct'])
        self.assertEqual(report['cases'][0]['backspaces'], 1)
        self.assertTrue(all(row['exact_correct'] for row in report['cases']))
        self.assertTrue(all(row['first_pass_correct'] for row in report['cases'][1:]))
        self.assertNotIn('target_text', self.identities[0])
        self.assertNotIn('current_text', self.identities[0])

    def test_wrong_submit_is_kept_and_next_occurrence_is_not_a_hidden_retry(self):
        first = self.session.identity()
        self.input_text('wrong')
        self.submit()
        self.session.close()
        report = build_report(self.events)
        self.assertFalse(report['cases'][0]['exact_correct'])
        self.assertEqual(report['cases'][0]['occurrence_id'], first['occurrence_id'])
        self.assertEqual(report['cases'][1]['status'], 'aborted')

    def test_reset_clears_only_input_and_is_recorded_without_home_or_case_change(self):
        before = self.session.identity()
        self.input_text('x')
        self.session.reset_input()
        self.assertEqual(self.session.current_text, '')
        self.assertEqual(self.session.identity(), before)
        self.input_text(self.session.snapshot()['target_text'])
        self.submit()
        self.session.close()
        row = build_report(self.events)['cases'][0]
        self.assertEqual(row['reset_count'], 1)
        self.assertTrue(row['exact_correct'])
        self.assertFalse(row['first_pass_correct'])

    def test_partial_stop_and_clock_failure_are_not_success(self):
        self.input_text(self.session.snapshot()['target_text'])
        self.session.key_down('Return', 13)
        self.session.close()
        self.session.close()
        self.assertEqual(sum(row['event'] == 'ended' for row in self.events), 1)
        self.assertFalse(build_report(self.events)['cases'][0]['exact_correct'])
        with self.assertRaises(ValueError):
            self.session.load_next_manifest(self.manifest)

    def test_next_batch_keeps_sequence_and_identity_but_rejects_active_replacement(self):
        with self.assertRaises(ValueError):
            self.session.load_next_manifest(generate_manifest(dict(DEFAULT_CONFIG, seed=1)))
        for _ in self.manifest['occurrences']:
            self.input_text(self.session.snapshot()['target_text'])
            self.submit()
        self.session.load_next_manifest(generate_manifest(dict(DEFAULT_CONFIG, seed=1)))
        self.assertEqual(self.session.phase, 'typing')
        self.assertEqual([event['seq'] for event in self.events], list(range(1, len(self.events) + 1)))
        self.assertNotEqual(self.identities[0]['batch_id'], self.session.identity()['batch_id'])

    def test_identity_file_is_strict_and_never_exposes_score_or_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / 'identity.json'
            path.write_text(json.dumps(self.session.identity()))
            self.assertEqual(read_case_identity(path, expected_run_id='lab:fixture'), self.session.identity())
            for data in (dict(self.session.identity(), actual_text='private'),
                         dict(self.session.identity(), score=True),
                         dict(self.session.identity(), run_id='different')):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    read_case_identity(path, expected_run_id='lab:fixture')
            path.write_text(json.dumps(self.session.identity()))
            link = root / 'link.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                read_case_identity(link)


if __name__ == '__main__':
    unittest.main()
