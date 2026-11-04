"""No model calls: boundary/usage checks for the actual-call bridge."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.astra_bridge import (
    AstraSkillBridge, MODEL, EFFORT, decode_candidate, native_usage)
from astra_robotics_agent.physical_keyboard.contracts import EvaluationPlan
from astra_robotics_agent.physical_keyboard.ledger import Ledger


class FakeSession:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.thread_id = 'thread-synthetic-test'
        self.total_usage = {}
        self.closed = False
        self.prompts = []
        self.failure = False
        self.text = json.dumps({'summary': 'test', 'skill_source': 'def main(robot, profile):\n    return {}\n'})

    def start(self):
        return {'model': MODEL, 'effort': EFFORT, 'service_tier': 'default'}

    def run_turn(self, prompt, handler):
        self.prompts.append(prompt)
        self.total_usage = {'inputTokens': 10, 'cachedInputTokens': 4,
                            'outputTokens': 7, 'reasoningOutputTokens': 5}
        if self.failure:
            raise TimeoutError('synthetic failed turn')
        return {'final_text': self.text, 'thread_id': self.thread_id, 'turn_id': 't1',
                'model': MODEL, 'effort': EFFORT, 'tool_calls': 0, 'usage': self.total_usage}

    def close(self):
        self.closed = True


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def bridge(self):
        result = AstraSkillBridge(self.root, self.root / 'calls', session_factory=FakeSession)
        self.addCleanup(result.close)
        result.start()
        return result

    def test_explicit_model_effort_standard_and_persistent_feedback(self):
        bridge = self.bridge()
        self.assertEqual(bridge.session.options['model'], 'gpt-6-astra')
        self.assertEqual(bridge.session.options['effort'], 'xhigh')
        self.assertEqual(bridge.session.options['service_tier'], 'default')
        self.assertEqual(bridge.session.options['tools'], [])
        first = bridge.propose({'target': 'ab'})
        second = bridge.propose({'actual_result': '', 'previous_source_sha256': first['source_sha256']})
        self.assertEqual(len(bridge.session.prompts), 2)
        self.assertEqual(second['call_number'], 2)
        # Cached/reasoning are subsets, not additive extra costs.
        self.assertEqual(bridge.usage['total_input_plus_output'], 34)
        lines = [json.loads(v) for v in (bridge.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual([v['event'] for v in lines], ['call_started', 'call_finished'] * 2)
        self.assertEqual(lines[-1]['cumulative_usage']['cached_input_tokens'], 8)
        self.assertTrue((bridge.root / 'call-001/skill.py').is_file())

    def test_invalid_output_retains_actual_usage_and_raw_reply(self):
        bridge = self.bridge()
        bridge.session.text = 'not JSON'
        with self.assertRaises(ValueError):
            bridge.propose({'attempt': 1})
        self.assertEqual(bridge.calls[0]['status'], 'failed')
        self.assertEqual(bridge.usage['total_input_plus_output'], 17)
        self.assertTrue((bridge.root / 'call-001/reply.json').is_file())
        self.assertTrue((bridge.root / 'call-001/error.json').is_file())

    def test_failed_transport_keeps_last_counters_but_final_cost_unknown(self):
        bridge = self.bridge()
        bridge.session.failure = True
        with self.assertRaises(TimeoutError):
            bridge.propose({'attempt': 1})
        self.assertIsNone(bridge.usage['total_input_plus_output'])
        self.assertFalse(bridge.usage['complete'])
        self.assertEqual(bridge.calls[0]['last_observed_session_native_usage']['inputTokens'], 10)

    def test_unknown_native_fields_are_not_zero_and_subsets_validated(self):
        self.assertFalse(native_usage(None).complete)
        self.assertIsNone(native_usage({'inputTokens': 3}).output_tokens)
        for bad in [True, -1, 1.2]:
            with self.assertRaises(ValueError):
                native_usage({'inputTokens': bad})
        with self.assertRaises(ValueError):
            native_usage({'inputTokens': 1, 'cachedInputTokens': 2})

    def test_duplicate_or_extra_candidate_keys_and_markdown_rejected(self):
        for text in ['{"summary":"a","summary":"b","skill_source":"pass"}',
                     '{"summary":"a","skill_source":"pass","evaluator":"fake"}',
                     '```json\n{}\n```', '{"summary":"a","skill_source":42}',
                     '{"summary":"","skill_source":"pass"}']:
            with self.assertRaises(ValueError):
                decode_candidate(text)

    def test_unlimited_plan_serializes_explicit_null(self):
        plan = EvaluationPlan({'ab': 'ab'}, 2, 1, None)
        self.assertIsNone(asdict(plan)['max_total_tokens'])
        for limit in [True, -1, 2.5]:
            with self.assertRaises(ValueError):
                EvaluationPlan({'ab': 'ab'}, 1, 1, limit)

    def test_unlimited_comparison_still_requires_known_cost(self):
        def comparison(limit, usage):
            return Ledger._comparison({
                'plan': asdict(EvaluationPlan({'ab': 'ab'}, 1, 1, limit)),
                'generation_usage': usage, 'metadata': {'evidence_mode': 'synthetic'},
                'trials': {v: {'variant': v, 'case_id': 'ab', 'repetition': 0, 'condition_id': 'same',
                               'result': {'status': 'observed', 'observed_text': 'ab' if v == 'candidate' else '',
                                          'usage': asdict(native_usage({'inputTokens': 0, 'cachedInputTokens': 0,
                                                                      'outputTokens': 0, 'reasoningOutputTokens': 0}))}}
                           for v in ('parent', 'candidate')}})
        measured = asdict(native_usage({'inputTokens': 10000, 'cachedInputTokens': 0,
                                       'outputTokens': 10000, 'reasoningOutputTokens': 0}))
        self.assertEqual(comparison(None, measured)['verdict'], 'accepted')
        self.assertIn('token_budget_exceeded', comparison(100, measured)['reasons'])
        self.assertIn('unknown_usage', comparison(None, asdict(native_usage(None)))['reasons'])


if __name__ == '__main__':
    unittest.main()
