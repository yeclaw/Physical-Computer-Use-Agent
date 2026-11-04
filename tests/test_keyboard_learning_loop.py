import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.learning_loop import run_episode, parse_decision, validate_finish


SCOPE = dict(task_id='view', device_id='robot', layout_id='layout-a')


def finish(skills=(), lessons=()):
    return dict(action='finish', result=dict(status='calibrated_partial', observed_text=None,
                device_hypotheses=[], completed_probes=[], next_needed=['Full row still hidden'],
                reflection=dict(summary='Changed probe using actual failure evidence.',
                                skills_saved=list(skills), lessons_saved=list(lessons),
                                limitations=['Unpaired diagnostic; no typing success.'])))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.store = KnowledgeStore(root / 'store', SCOPE, 'test')
        self.output = root / 'output'; self.output.mkdir()
        self.observations = self.executions = 0
        self.statuses = ['blocked', 'ok']

    def evidence(self, name, value):
        path = self.root / (name + '.json')
        path.write_text(json.dumps(value))
        self.store.register_evidence(name, [path])
        return name

    def observe(self):
        self.observations += 1
        evidence_id = self.evidence('obs-%04d' % self.observations, {'frame': self.observations})
        return dict(evidence_id=evidence_id, contentItems=[
            {'type': 'inputText', 'text': 'Actual fixture observation ' + evidence_id},
            {'type': 'inputImage', 'imageUrl': 'data:image/jpeg;base64,fixture'}])

    def execute(self, arguments):
        self.executions += 1
        response = dict(status=self.statuses[self.executions-1], result={'receipts': []})
        evidence_id = self.evidence('exec-%04d' % self.executions, dict(arguments=arguments, response=response))
        return dict(response=response, evidence_id=evidence_id)

    def run(self, session, **kwargs):
        kwargs.setdefault('max_decisions', 12)
        return run_episode(session=session, observe=self.observe, execute=self.execute,
                           store=self.store, output=self.output, **kwargs)


class Session:
    def __init__(self, choose):
        self.choose = choose
        self.prompts = []; self.images = []

    def run_turn(self, prompt, callback, images):
        index = len(self.prompts)
        self.prompts.append(prompt); self.images.append(images)
        decision = self.choose(index, json.loads(prompt.split('\n')[0]))
        return dict(turn_id=str(index), usage={'input_tokens': 1, 'output_tokens': 1},
                    final_text=decision if isinstance(decision, str) else json.dumps(decision))


def skill(evidence, revision):
    return dict(action='save_skill', arguments=dict(id='adaptive_view',
        source_lines=['def main(robot, profile):', '    return ' + str(revision)],
        profile={'revision': revision}, applicability=['Current robot/layout'],
        limitations=['Only a diagnostic'], evidence_ids=evidence))


class LearningLoopTests(unittest.TestCase):
    def test_invalid_label_is_rejected_locally_then_model_selects_corrected_program(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            original = dict(action='run_skill', arguments=dict(label='Elevated downward-view probe',
                source_lines=['def main(robot, profile):', '    return robot.observe()'], profile={}),
                hypothesis='Measure the current state')
            def choose(index, prompt):
                if index == 0:
                    return original
                if index == 1:
                    self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
                    self.assertIn('label', prompt['feedback']['error'])
                    self.assertEqual(fixture.executions, 0)
                    corrected = json.loads(json.dumps(original))
                    corrected['arguments']['label'] = 'view_probe'
                    return corrected
                return finish()
            result = fixture.run(Session(choose))
            self.assertTrue(result['finished_by_model'])
            self.assertEqual(fixture.executions, 1)
            self.assertEqual(original['arguments']['label'], 'Elevated downward-view probe')

    def test_missing_hypothesis_feedback_names_missing_field(self):
        with self.assertRaisesRegex(ValueError, 'missing=hypothesis'):
            parse_decision(json.dumps(dict(action='run_skill', arguments=dict(
                label='probe', source_lines=['def main(robot, profile):', '    return 1'], profile={}))))

    def test_documented_minimal_finish_needs_no_undisclosed_legacy_fields(self):
        decision = finish()
        for key in ('observed_text', 'device_hypotheses', 'completed_probes', 'next_needed'):
            decision['result'].pop(key)
        decision['result']['device_hypothesis'] = {'observation': ['Actual view remains partial']}
        parsed = parse_decision(json.dumps(decision))
        validate_finish(parsed['result'], {})
        self.assertEqual(parsed, decision)  # Preserve model content, no repair/defaults.
        decision['result']['completed_probes'] = 'wrong type'
        with self.assertRaisesRegex(ValueError, 'legacy'):
            parse_decision(json.dumps(decision))

    def test_model_owns_save_verify_failure_reflect_revise_and_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            ids = {}
            def choose(index, prompt):
                feedback = prompt['feedback']
                if index == 0:
                    return skill([prompt['current_observation_evidence_id']], 1)
                if index == 1:
                    ids['first'] = feedback['candidate']['candidate_id']
                    return dict(action='verify_skill', arguments={'candidate_id': ids['first']}, hypothesis='First candidate may be rejected')
                if index == 2:
                    self.assertEqual(feedback['execution']['status'], 'blocked')
                    return dict(action='save_lesson', arguments=dict(id='blocked_diagnostic', claim='Host rejected first candidate; target not verified.',
                        applicability=['Current setup'], limitations=['No causal visual result'], evidence_ids=feedback['evidence_ids']))
                if index == 3:
                    ids['lesson'] = feedback['candidate']['candidate_id']
                    return skill([prompt['current_observation_evidence_id']], 2)
                if index == 4:
                    ids['second'] = feedback['candidate']['candidate_id']
                    return dict(action='inspect_skill', arguments={'candidate_id': ids['second']})
                if index == 5:
                    self.assertIn('return 2', feedback['skill']['source'])
                    return dict(action='verify_skill', arguments={'candidate_id': ids['second']}, hypothesis='Second program should execute')
                self.assertEqual(feedback['execution']['status'], 'ok')
                return finish([ids['first'], ids['second']], [ids['lesson']])
            session = Session(choose)
            result = fixture.run(session)
            self.assertTrue(result['finished_by_model'])
            self.assertEqual(fixture.executions, 2)
            self.assertEqual(fixture.observations, 3)
            self.assertEqual([bool(x) for x in session.images], [True, False, True, False, False, False, True])
            self.assertEqual([v['status'] for v in result['verifications']], ['diagnostic_failed', 'verified_execution_only'])
            self.assertFalse(result['stable_promoted'])
            self.assertNotEqual(ids['first'], ids['second'])
            reopened = KnowledgeStore(fixture.root / 'store', SCOPE, 'another-run')
            self.assertIn('Host rejected', reopened.context_text())
            self.assertIn('return 2', reopened.get_skill(ids['second'])['source'])

    def test_unknown_stops_before_any_new_observation_or_model_turn(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve()); fixture.statuses = ['unknown']
            def choose(index, prompt):
                if index == 0: return skill([prompt['current_observation_evidence_id']], 1)
                return dict(action='verify_skill', arguments={'candidate_id': prompt['feedback']['candidate']['candidate_id']}, hypothesis='Check once')
            session = Session(choose)
            result = fixture.run(session)
            self.assertEqual(result['stopped_reason'], 'unknown_motion_no_retry')
            self.assertEqual((fixture.executions, fixture.observations, len(session.prompts)), (1, 1, 2))
            self.assertEqual(result['verifications'][0]['status'], 'verification_unknown')

    def test_bad_evidence_or_python_rejected_without_execution_then_model_finishes(self):
        for forged in (True, False):
            with self.subTest(forged=forged), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder).resolve())
                def choose(index, prompt):
                    if index:
                        self.assertEqual(prompt['feedback']['phase'], 'request_rejected')
                        return finish()
                    candidate = skill(['imaginary'] if forged else [prompt['current_observation_evidence_id']], 1)
                    if not forged: candidate['arguments']['source_lines'] = ['import os', 'def main(robot, profile):', '    return 0']
                    return candidate
                fixture.run(Session(choose))
                self.assertEqual(fixture.executions, 0)
                self.assertFalse(list((fixture.store.namespace / 'candidates').iterdir()))

    def test_after_dispatch_validation_error_is_fatal_not_a_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            def choose(index, prompt):
                return dict(action='run_skill', arguments={'label': 'probe', 'source_lines': ['def main(robot, profile):', '    return 1'], 'profile': {}}, hypothesis='One probe')
            def execute(arguments):
                fixture.executions += 1
                raise ValueError('post-dispatch receipt error')
            session = Session(choose)
            with self.assertRaisesRegex(ValueError, 'post-dispatch'):
                run_episode(session=session, observe=fixture.observe, execute=execute, store=fixture.store,
                            output=fixture.output, max_decisions=10)
            self.assertEqual((fixture.executions, len(session.prompts)), (1, 1))

    def test_requested_observation_hash_failure_is_fatal_not_old_frame_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            def observe():
                if fixture.observations:
                    raise ValueError('Incomplete/cross-run camera evidence')
                return fixture.observe()
            session = Session(lambda *_: dict(action='observe', arguments={}))
            with self.assertRaisesRegex(ValueError, 'cross-run'):
                run_episode(session=session, observe=observe, execute=fixture.execute,
                            store=fixture.store, output=fixture.output, max_decisions=10)
            self.assertEqual((fixture.executions, len(session.prompts)), (0, 1))

    def test_partial_cannot_claim_view_ready_or_fabricate_saved_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            def choose(index, prompt):
                if index == 0:
                    candidate = finish(); candidate['result']['status'] = 'view_ready'
                    candidate['result']['view_assessment'] = dict(visible_key_labels=list('YUIOP'), petal_tip_visible=True, nearby_surfaces_described='Only right part visible')
                    return candidate
                if index == 1: return finish(['invented'])
                return finish()
            result = fixture.run(Session(choose), legacy_view_setup=True)
            self.assertEqual(len(result['turns']), 3)
            self.assertEqual(result['reply']['status'], 'calibrated_partial')
            self.assertEqual(fixture.executions, 0)

    def test_normal_task_finish_does_not_require_full_row_or_promote_visual_report(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            decision = finish()
            decision['result'].update(status='completed', observed_text='a')
            result = fixture.run(Session(lambda *_: decision))
            self.assertTrue(result['finished_by_model'])
            self.assertEqual(result['reply']['observed_text'], 'a')
            self.assertFalse(result['stable_promoted'])
            self.assertTrue(result['diagnostic_only'])
            with self.assertRaisesRegex(ValueError, 'non-contact'):
                validate_finish(decision['result'], {}, legacy_view_setup=True)

    def test_default_loop_can_continue_past_twenty_decisions_and_save_then_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            candidate_id = None
            def choose(index, prompt):
                nonlocal candidate_id
                self.assertIsNone(prompt['remaining_decisions'])
                if index < 21:
                    return dict(action='observe', arguments={})
                if index == 21:
                    return dict(action='save_lesson', arguments=dict(
                        id='persistent_learning', claim='Fixture observation sequence exceeded twenty turns.',
                        applicability=['This synthetic fixture only'], limitations=['No physical result'],
                        evidence_ids=[prompt['current_observation_evidence_id']]))
                candidate_id = prompt['feedback']['candidate']['candidate_id']
                decision = finish(lessons=[candidate_id])
                decision['result']['status'] = 'partial'
                return decision
            session = Session(choose)
            # Omit max_decisions to exercise the actual normal default.
            result = run_episode(session=session, observe=fixture.observe, execute=fixture.execute,
                                 store=fixture.store, output=fixture.output)
            self.assertEqual(len(session.prompts), 23)
            self.assertTrue(result['finished_by_model'])
            self.assertIsNone(result['max_decisions'])
            self.assertEqual(result['reply']['reflection']['lessons_saved'], [candidate_id])
            self.assertEqual((fixture.executions, fixture.observations), (0, 22))

    def test_explicit_decision_budget_still_works_and_invalid_budget_has_no_callbacks(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            session = Session(lambda *_: dict(action='observe', arguments={}))
            for invalid in (0, -1, True, 1.5):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    fixture.run(session, max_decisions=invalid)
            self.assertEqual((fixture.observations, fixture.executions, len(session.prompts)), (0, 0, 0))
            result = fixture.run(session, max_decisions=2)
            self.assertEqual(len(session.prompts), 2)
            self.assertTrue(result['decision_limit_reached'])
            self.assertFalse(result.get('finished_by_model', False))
            self.assertEqual([json.loads(x.split('\n')[0])['remaining_decisions'] for x in session.prompts], [2, 1])

    def test_nonfinite_exponent_and_multiline_or_unknown_action_rejected(self):
        for raw in ('{"action":"observe","arguments":{"x":1e999}}',
                    '{"action":"promote","arguments":{}}',
                    '{"action":"reset","arguments":{}}'):
            with self.assertRaises(ValueError): parse_decision(raw)

    def test_unstructured_error_after_dispatch_is_unknown_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder).resolve())
            def choose(index, prompt):
                return dict(action='run_skill', arguments={'label': 'probe', 'source_lines': ['def main(robot, profile):', '    return 1'], 'profile': {}}, hypothesis='One probe')
            def execute(arguments):
                fixture.executions += 1
                return dict(evidence_id=fixture.evidence('exec-0001', {'error': 'audit failed'}),
                            response={'status': 'error', 'error': {'type': 'OSError'}})
            session = Session(choose)
            result = run_episode(session=session, observe=fixture.observe, execute=execute, store=fixture.store,
                                 output=fixture.output, max_decisions=10)
            self.assertEqual(result['stopped_reason'], 'unknown_motion_no_retry')
            self.assertEqual((fixture.executions, fixture.observations, len(session.prompts)), (1, 1, 1))


if __name__ == '__main__': unittest.main()
