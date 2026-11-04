"""Lost UI releases and exact prefix recovery, without GUI/SDK/keyboard IO."""
import copy
import json
import unittest
from unittest.mock import MagicMock

from astra_robotics_agent.physical_keyboard.adapters.environments.typing_cases import DEFAULT_CONFIG, generate_manifest
from astra_robotics_agent.physical_keyboard.adapters.environments.typing_lab import TrainingSession
from astra_robotics_agent.physical_keyboard.adapters.environments.typing_resume import restore_session
from astra_robotics_agent.physical_keyboard.records.typing_metrics import build_report
from astra_robotics_agent.physical_keyboard.adapters.environments.key_state import X11KeyState


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000000000
        self.events = []
        self.manifest = generate_manifest(DEFAULT_CONFIG)
        self.keys = []
        self.session = TrainingSession(self.manifest, run_id='fixture', emit=self.events.append,
            publish_identity=lambda unused: None, clock_ns=lambda: self.now, utc_ns=lambda: self.now)

    def prefix(self):
        return b''.join((json.dumps(e)+'\n').encode() for e in self.events)

    def restore(self, raw=None):
        return restore_session(self.manifest, self.prefix() if raw is None else raw,
            run_id='fixture', emit=self.events.append, publish_identity=lambda unused: None,
            clock_ns=lambda: self.now, utc_ns=lambda: self.now, key_state_provider=lambda: self.keys)

    def test_lost_modifier_release_recovers_but_wrong_commit_survives_reset(self):
        s = self.session
        s.key_down('Super_R', 134)
        s.text_changed('wrong')
        s.key_down('Return', 36)
        s.key_up('Return', 36)
        s.reset_input()
        before = self.prefix()
        s = self.restore()
        self.assertEqual(self.prefix(), before)
        self.assertEqual(s.current_text, '')
        self.assertFalse(s.tick())
        self.now += 600000000
        self.assertTrue(s.tick())
        report = build_report(self.events)
        self.assertFalse(report['cases'][0]['exact_correct'])
        self.assertEqual(report['cases'][0]['submitted_text'], 'wrong')
        self.assertEqual(sum(e['event'] == 'submit' for e in self.events), 1)
        self.assertEqual(sum(e['event'] == 'key_up' for e in self.events), 1)
        self.assertEqual(s._index, 1)

    def test_unseen_os_key_focus_and_fresh_settle_all_block(self):
        self.session.key_down('Return', 36)
        self.session.key_up('Return', 36)
        s = self.restore()
        self.keys = [200]
        s.tick()
        self.now += 1000000000
        self.assertFalse(s.tick())
        self.keys = []
        self.assertFalse(s.tick())
        self.now += 600000000
        s.focus_changed(False)
        self.assertFalse(s.tick())
        s.focus_changed(True)
        self.assertFalse(s.tick())
        self.now += 600000000
        self.assertTrue(s.tick())

    def test_provider_failure_never_means_released(self):
        self.session.key_down('Return', 36)
        s = self.restore()
        def fail():
            raise OSError('no display')
        s._key_state_provider = fail
        with self.assertRaises(OSError):
            s.tick()
        self.assertEqual(s.phase, 'awaiting_release')
        self.assertIsNone(s._platform_pressed)

    def test_replays_prior_completed_case_and_preserves_active_text(self):
        self.session.text_changed(self.session.snapshot()['target_text'])
        self.session.key_down('Return', 36)
        self.session.key_up('Return', 36)
        self.now += 600000000
        self.session.tick()
        self.session.text_changed('draft')
        s = self.restore()
        self.assertEqual(s._index, 1)
        self.assertEqual(s.current_text, 'draft')
        self.assertEqual(s.phase, 'typing')

    def test_rejects_changed_manifest_terminated_bad_prefix_and_new_boot(self):
        raw = self.prefix()
        bad = copy.deepcopy(self.manifest)
        bad['base_cases'][0]['target_text'] = 'tampered'
        with self.assertRaises(ValueError):
            restore_session(bad, raw, run_id='fixture', emit=lambda e: None, publish_identity=lambda e: None)
        with self.assertRaises(ValueError):
            self.restore(raw[:-1])
        self.session.close()
        with self.assertRaises(ValueError):
            self.restore()
        self.now = 0
        with self.assertRaises(ValueError):
            self.restore(raw)

    def test_v2_can_be_resumed_again_without_duplicate_os_events(self):
        self.session.key_down('Return', 36)
        s = self.restore()
        s.tick()
        n = len(self.events)
        s.tick()
        self.assertEqual(len(self.events), n)
        s = self.restore()
        self.assertEqual(s.phase, 'awaiting_release')
        self.assertTrue(s._return_released)

    def test_native_v2_prefix_can_resume(self):
        events = []
        TrainingSession(self.manifest, run_id='fixture', emit=events.append,
            publish_identity=lambda e: None, clock_ns=lambda: self.now, utc_ns=lambda: self.now,
            key_state_provider=lambda: [])
        s = self.restore(b''.join((json.dumps(e)+'\n').encode() for e in events))
        self.assertEqual(s.phase, 'typing')


class KeyStateTests(unittest.TestCase):
    def test_read_only_bitmap_includes_modifier_and_unknown_key(self):
        lib = MagicMock()
        lib.XOpenDisplay.return_value = 10
        def query(display, keys):
            self.assertEqual(display, 10)
            for code in (36, 134, 200):
                keys[code // 8] |= 1 << (code % 8)
            return 1
        lib.XQueryKeymap.side_effect = query
        provider = X11KeyState(library=lib)
        self.assertEqual(provider(), [36, 134, 200])
        provider.close()
        provider.close()
        lib.XCloseDisplay.assert_called_once_with(10)
        with self.assertRaises(RuntimeError):
            provider()

    def test_unavailable_display_or_failed_query_never_produces_empty_map(self):
        lib = MagicMock()
        lib.XOpenDisplay.return_value = None
        with self.assertRaises(RuntimeError):
            X11KeyState(library=lib)
        lib.XOpenDisplay.return_value = 10
        lib.XQueryKeymap.return_value = 0
        provider = X11KeyState(library=lib)
        with self.assertRaises(RuntimeError):
            provider()
        provider.close()



if __name__ == '__main__':
    unittest.main()
