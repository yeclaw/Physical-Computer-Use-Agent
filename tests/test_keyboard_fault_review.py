"""Offline fault review fixtures: no model service, controller or real knowledge."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.agent.fault_review import run_fault_review
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore


SCOPE = dict(task_id='fault-fixture', device_id='fixture-device', layout_id='fixture-layout')
SOURCE = ['def main(robot, profile):',
          '    return robot.move_tcp(xyz_m=[0, 0, 0.2], quaternion_xyzw=[0, 0, 0, 1])']


def finish(skills=(), lessons=(), action='finish'):
    return dict(action=action, result=dict(status='partial', reflection=dict(
        summary='Fixture review; effectiveness remains unknown.', skills_saved=list(skills),
        lessons_saved=list(lessons), limitations=['No post-fault observation or physical verification.'])))


class Model:
    def __init__(self, choose, native=False):
        self.choose, self.native = choose, native
        self.prompts, self.contexts, self.images, self.native_replies = [], [], [], []

    def run_turn(self, prompt, callback, images):
        index = len(self.prompts)
        context = json.loads(prompt.split('\n')[0])
        self.prompts.append(prompt)
        self.contexts.append(context)
        self.images.append(list(images))
        if self.native:
            for name in ('observe', 'execute', 'run_skill', 'verify_skill', 'call_skill', 'reset', 'reconnect'):
                self.native_replies.append(callback(name, {}))
        decision = self.choose(index, context)
        return dict(turn_id='fixture-' + str(index),
                    usage=dict(input_tokens=7, output_tokens=3, cached_input_tokens=2,
                               reasoning_output_tokens=1, cost_usd=0.001),
                    final_text=decision if type(decision) is str else json.dumps(decision),
                    internal_reasoning='PRIVATE_FIXTURE_MUST_NOT_BE_PERSISTED')


class FaultReviewTests(unittest.TestCase):
    def test_new_read_only_evidence_preserves_unknown_state_and_denies_motion(self):
        diagnostic_calls, camera_calls = [], []
        def register(folder, name, value):
            folder.mkdir()
            path = folder / 'result.json'
            path.write_text(json.dumps(value))
            self.store.register_evidence(name, [path])
            return name
        def diagnostic(fault, folder):
            diagnostic_calls.append(fault)
            identifier = register(folder, 'diag-' + str(len(diagnostic_calls)), {'source_location': 'fixture_bridge'})
            return dict(success=True, status='diagnostic_available', evidence_id=identifier,
                        summary=dict(source_location='fixture_bridge', physical_cause='unknown'))
        def camera(folder):
            camera_calls.append(folder)
            identifier = register(folder, 'cached-camera', {'robot_state': 'unknown'})
            return dict(success=True, evidence_id=identifier, robot_state='unknown', motion_available=False,
                contentItems=[dict(type='inputImage', imageUrl='data:image/jpeg;base64,fixture-passive')])
        def choose(index, context):
            self.assertIsNone(context['current_observation_evidence_id'])
            self.assertFalse(context['motion_available'])
            self.assertEqual(context['current_state'], 'unknown')
            self.assertEqual(context['failure_diagnostics']['summary']['physical_cause'], 'unknown')
            if index == 0:
                return dict(action='inspect_recorder', arguments={})
            if index == 1:
                self.assertFalse(context['feedback']['motion_available'])
                return dict(action='refresh_fault_diagnostics', arguments={})
            if index == 2:
                return dict(action='observe', arguments={})
            self.assertEqual(context['feedback']['phase'], 'request_rejected')
            return finish()
        model = Model(choose)
        report = self.run_review(model, diagnostics_provider=diagnostic, passive_observe=camera)
        self.assertEqual(len(diagnostic_calls), 2)
        self.assertEqual(len(camera_calls), 1)
        self.assertEqual(model.images[1], ['data:image/jpeg;base64,fixture-passive'])
        self.assertFalse(report['task_completed'])
        self.assertEqual(len(report['diagnostics']), 2)

    def test_unavailable_passive_reader_returns_feedback_without_ending_review(self):
        def unavailable(folder):
            raise OSError('fixture recorder cache unavailable')
        def choose(index, context):
            if index == 0:
                return dict(action='inspect_recorder', arguments={})
            self.assertEqual(context['feedback']['result']['status'], 'passive_recorder_unavailable')
            self.assertIn('fixture recorder', context['feedback']['result']['error'])
            self.assertIsNone(context['current_observation_evidence_id'])
            return finish()
        report = self.run_review(Model(choose), passive_observe=unavailable)
        self.assertTrue(report['finished_by_model'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fault-review-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = KnowledgeStore(self.root / 'store', SCOPE, 'fixture-run')
        self.library = SkillDevelopment(self.store)
        self.output = self.root / 'review'
        image = self.root / 'old.jpg'
        image.write_bytes(b'\xff\xd8\xffsynthetic-old-image')
        self.store.register_evidence('before-execution', [image])
        execution = self.root / 'uncertain.json'
        execution.write_text('{"status":"unknown","fixture":true}')
        self.store.register_evidence('failed-execution', [execution])
        self.fault = dict(stopped_reason='unknown_motion_no_retry',
            error_type='FixtureTrackingError', error='Synthetic tracking threshold exceeded',
            execution_status='unknown', last_observation=dict(evidence_id='before-execution'),
            feedback=dict(execution=dict(status='unknown', result=dict(motion_outcome_uncertain=True)),
                          evidence_ids=['before-execution', 'failed-execution']),
            attempted_program=dict(source='def main(robot, profile):\n    return "UNIQUE_ATTEMPTED_SOURCE"\n',
                                   profile=dict(value='UNIQUE_ATTEMPTED_PROFILE'), label='fixture_attempt'))

    def run_review(self, model, **kwargs):
        return run_fault_review(session=model, store=self.store, output=self.output,
                                fault=self.fault, **kwargs)

    def package_arguments(self, evidence_id):
        return dict(id='candidate_repair', source_lines=SOURCE, profile={},
            contract=dict(description='Model-chosen fixture candidate',
                          input_schema=dict(type='object', properties={}, additionalProperties=False),
                          preconditions=['State must be established later.'],
                          postconditions=['Effect requires fresh verification.']),
            applicability=['Only this fixture.'], limitations=['Not executed or independently verified.'],
            evidence_ids=[evidence_id])

    def test_motion_requests_are_refused_then_model_can_record_issue_save_candidate_and_finish(self):
        motions = [dict(action='run_skill', arguments=dict(label='probe', source_lines=SOURCE, profile={}),
                        hypothesis='Fixture motion request must be rejected.'),
                   dict(action='observe', arguments={}), dict(action='reset', arguments={}),
                   dict(action='verify_skill', arguments=dict(candidate_id='missing'), hypothesis='No verification.'),
                   dict(action='call_skill', arguments=dict(package_id='missing', arguments={}), hypothesis='No call.')]
        observed = {}
        def choose(index, context):
            self.assertIsNone(context['current_observation_evidence_id'])
            self.assertEqual(context['current_state'], 'unknown')
            if index < len(motions):
                if index:
                    self.assertEqual(context['feedback']['phase'], 'request_rejected')
                return motions[index]
            if index == 5:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
                return dict(action='record_improvement', arguments=dict(id='model_chosen_issue',
                    summary='Fixture tracking fault leaves outcome unknown.',
                    rationale='Reusable transition warrants later investigation.', status='open',
                    priority=83, revisit_when=['After an independent current-state check.'],
                    skill_ids=[], evidence_ids=[context['fault_evidence_id']]))
            if index == 6:
                observed['issue'] = context['feedback']['result']['record_id']
                args = self.package_arguments(context['fault_evidence_id'])
                args['improvement_ids'] = [observed['issue']]
                return dict(action='save_skill_package', arguments=args)
            observed['candidate'] = context['feedback']['result']['candidate_id']
            return finish(skills=[observed['candidate']])
        model = Model(choose, native=True)
        with patch.object(self.store, 'record_verification', side_effect=AssertionError('No verification')) as verify, \
             patch('astra_robotics_agent.physical_keyboard.agent.development.DevelopmentActions.resolve_call',
                   side_effect=AssertionError('No call resolution')) as call:
            report = self.run_review(model)
        verify.assert_not_called()
        call.assert_not_called()
        self.assertTrue(all(reply['success'] is False for reply in model.native_replies))
        self.assertEqual(len(model.contexts), 8)
        self.assertTrue(report['finished_by_model'])
        self.assertEqual(report['stopped_reason'], self.fault['stopped_reason'])
        self.assertFalse(report['task_completed'])
        self.assertFalse(report['completed'])
        self.assertFalse(report['stable_promoted'])
        self.assertEqual(report['verifications'], [])
        self.assertEqual(len(report['saved_candidates']), 1)
        self.assertEqual(report['saved_candidates'][0]['candidate_id'], observed['candidate'])
        self.assertEqual(report['saved_candidates'][0]['status'], 'candidate_not_promoted')
        issue = self.library.inspect(observed['issue'])
        self.assertEqual((issue['priority'], issue['status']), (83, 'open'))
        self.assertFalse(issue['independently_verified'])
        self.assertEqual(self.library.search('', 'outcome')['results'], [])
        self.assertEqual(json.loads((self.output / 'fault.json').read_text()), self.fault)
        evidence = self.store._evidence(report['fault_evidence_id'])
        self.assertEqual([Path(item['source_path']).name for item in evidence['files']],
                         ['fault.json', 'attempted-source.py', 'attempted-profile.json'])
        self.assertNotIn('UNIQUE_ATTEMPTED_SOURCE', model.prompts[0])
        self.assertNotIn('UNIQUE_ATTEMPTED_PROFILE', model.prompts[0])
        self.assertIn('BEFORE the failed execution', model.prompts[0])
        self.assertEqual(model.images, [[]] * 8)
        self.assertNotIn('PRIVATE_FIXTURE_MUST_NOT_BE_PERSISTED',
                         (self.output / 'turn-00/reply.json').read_text())
        records = [json.loads(line) for line in (self.output / 'learning-events.jsonl').read_text().splitlines()]
        for kind in ('fault_review_model_request', 'fault_review_model_reply',
                     'typing_lab_model_turn', 'fault_review_turn_evidence'):
            self.assertEqual(sum(record['event'] == kind for record in records), 8)

    def test_historical_images_are_opt_in_and_cannot_be_used_for_task_outcome_or_completion(self):
        def choose(index, context):
            if index == 0:
                self.assertEqual(context['last_observation']['timing'], 'pre_execution_not_current')
                self.assertTrue(context['last_observation']['historical'])
                return dict(action='inspect_evidence', arguments=dict(evidence_id='before-execution'))
            if index == 1:
                self.assertTrue(context['feedback']['result']['historical'])
                return dict(action='inspect_evidence', arguments=dict(evidence_id='before-execution',
                                                                     file_indices=[0], include_images=True))
            if index == 2:
                return dict(action='record_skill_outcome', arguments=dict(id='fake_success', skill_ids=[],
                    evidence_ids=['before-execution'], observation_evidence_id='before-execution',
                    task_result='success', correction_performed=False, remaining_issues=[], summary='Invalid old image.'))
            if index == 3:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
                decision = finish()
                decision['result'].update(status='completed', observed_text='fabricated post-fault input')
                return decision
            if index == 4:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
                return finish(action='checkpoint')
            self.assertEqual(context['feedback']['action'], 'checkpoint')
            return finish()
        model = Model(choose)
        report = self.run_review(model)
        self.assertEqual([len(items) for items in model.images], [0, 0, 1, 0, 0, 0])
        self.assertIn('HISTORICAL evidence before-execution', model.prompts[2])
        self.assertTrue(all(c['current_observation_evidence_id'] is None for c in model.contexts))
        self.assertFalse(report['task_completed'])
        self.assertEqual(report['improvements'], [])
        self.assertEqual(self.library.search('', 'outcome')['results'], [])

    def test_legacy_save_and_full_source_read_remain_inert(self):
        ids = {}
        def choose(index, context):
            if index == 0:
                return dict(action='inspect_evidence', arguments=dict(
                    evidence_id=context['fault_evidence_id'], file_indices=[1, 2]))
            if index == 1:
                files = context['feedback']['result']['files']
                self.assertIn('UNIQUE_ATTEMPTED_SOURCE', files[1]['text'])
                self.assertIn('UNIQUE_ATTEMPTED_PROFILE', files[2]['text'])
                return dict(action='save_lesson', arguments=dict(id='fault_limit',
                    claim='Fixture outcome is unknown; no diagnosis established.',
                    applicability=['This fixture.'], limitations=['Historical evidence only.'],
                    evidence_ids=[context['fault_evidence_id']]))
            if index == 2:
                ids['lesson'] = context['feedback']['candidate']['candidate_id']
                return dict(action='save_skill', arguments=dict(id='legacy_candidate',
                    source_lines=SOURCE, profile={}, applicability=['This fixture.'],
                    limitations=['Not executed.'], evidence_ids=[context['fault_evidence_id']]))
            if index == 3:
                ids['skill'] = context['feedback']['candidate']['candidate_id']
                return dict(action='inspect_skill', arguments=dict(candidate_id=ids['skill']))
            self.assertIn('robot.move_tcp', context['feedback']['skill']['source'])
            return finish(skills=[ids['skill']], lessons=[ids['lesson']])
        report = self.run_review(Model(choose))
        self.assertEqual(len(report['saved_candidates']), 2)
        self.assertEqual(list((self.store.namespace / 'verifications').iterdir()), [])

    def test_foreign_scope_and_malformed_schema_are_rejected_without_writes(self):
        foreign = KnowledgeStore(self.root / 'store', dict(SCOPE, layout_id='foreign'), 'foreign')
        path = self.root / 'foreign.json'
        path.write_text('{}')
        foreign.register_evidence('foreign-evidence', [path])
        foreign_skill = foreign.propose('skill', dict(id='foreign_skill', source_lines=SOURCE,
            profile={}, applicability=['Foreign fixture.'], limitations=['Not run.'],
            evidence_ids=['foreign-evidence']))['candidate_id']
        invalid = [
            dict(action='inspect_evidence', arguments=dict(evidence_id='foreign-evidence')),
            dict(action='inspect_skill', arguments=dict(candidate_id=foreign_skill)),
            dict(action='save_lesson', arguments=dict(id='foreign', claim='Invalid scope.',
                applicability=['Fixture.'], limitations=['Unverified.'], evidence_ids=['foreign-evidence'])),
            dict(action='record_improvement', arguments=dict(id='bad_priority', summary='Fixture.',
                rationale='Fixture.', status='open', skill_ids=[], evidence_ids=['before-execution'], priority=True)),
            dict(action='search_knowledge', arguments=dict(query='', extra='invalid')),
            '{"action":"finish","action":"observe","arguments":{}}',
        ]
        def choose(index, context):
            if index:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
            return invalid[index] if index < len(invalid) else finish()
        report = self.run_review(Model(choose))
        self.assertTrue(report['finished_by_model'])
        self.assertEqual(report['saved_candidates'], [])
        self.assertEqual(report['improvements'], [])
        self.assertEqual(list((self.store.namespace / 'candidates').iterdir()), [])
        self.assertEqual(self.library.search('', 'all')['results'], [])

    def test_none_budget_continues_past_twenty_turns_and_preserves_usage_cost_trace(self):
        case = dict(protocol='physical-keyboard-typing-case-identity-v1', run_id='fixture-run',
            batch_id='batch-1', manifest_sha256='a' * 64, case_id='case-1', occurrence_id='occurrence-1',
            repeat_index=1, role='initial', phase='typing')
        trace = []
        def choose(index, context):
            self.assertIsNone(context['remaining_decisions'])
            return dict(action='list_improvements', arguments={}) if index < 22 else finish()
        report = self.run_review(Model(choose), case_identity=case,
                                 trace_event=lambda kind, data: trace.append((kind, data)))
        self.assertIsNone(report['max_decisions'])
        self.assertEqual(len(report['turns']), 23)
        self.assertNotIn('decision_limit_reached', report)
        self.assertEqual(len(trace), 23)
        for index, (kind, data) in enumerate(trace):
            self.assertEqual(kind, 'typing_lab_model_turn')
            self.assertEqual((data['case'], data['turn_index']), (case, index))
            self.assertEqual(data['usage'], report['turns'][index]['usage'])
            self.assertEqual(data['usage']['cost_usd'], 0.001)
            self.assertGreaterEqual(data['elapsed_seconds'], 0)
        self.assertAlmostEqual(sum(turn['usage']['cost_usd'] for turn in report['turns']), .023)

    def test_explicit_budget_retains_original_fault_and_invalid_budget_has_no_model_turn(self):
        model = Model(lambda *_: dict(action='list_improvements', arguments={}))
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_review(model, max_decisions=invalid)
        self.assertFalse(self.output.exists())
        self.assertEqual(model.prompts, [])
        report = self.run_review(model, max_decisions=2)
        self.assertEqual(len(report['turns']), 2)
        self.assertTrue(report['decision_limit_reached'])
        self.assertEqual(report['stopped_reason'], 'unknown_motion_no_retry')
        self.assertFalse(report['finished_by_model'])
        self.assertFalse(report['task_completed'])
        self.assertEqual([c['remaining_decisions'] for c in model.contexts], [2, 1])

    def test_operator_stop_before_inference_preserves_evidence_and_model_exception_traces_missing_usage(self):
        model = Model(lambda *_: finish())
        def stop(progress):
            self.assertEqual(progress['phase'], 'fault_review_model_inference')
            raise KeyboardInterrupt('Fixture operator stop')
        with self.assertRaises(KeyboardInterrupt):
            self.run_review(model, status=stop)
        self.assertEqual(model.prompts, [])
        report = json.loads((self.output / 'report.json').read_text())
        self.assertEqual(report['stopped_reason'], self.fault['stopped_reason'])
        self.store._evidence(report['fault_evidence_id'])
        self.output = self.root / 'model-error-review'
        trace = []
        class BrokenModel:
            def run_turn(self, *args, **kwargs):
                raise RuntimeError('Fixture model failure')
        with self.assertRaisesRegex(RuntimeError, 'Fixture model failure'):
            self.run_review(BrokenModel(), trace_event=lambda kind, data: trace.append((kind, data)))
        self.assertEqual(len(trace), 1)
        self.assertIsNone(trace[0][1]['case'])
        self.assertIsNone(trace[0][1]['usage'])
        report = json.loads((self.output / 'report.json').read_text())
        self.assertEqual(report['stopped_reason'], self.fault['stopped_reason'])
        self.assertFalse(report['task_completed'])


if __name__ == '__main__':
    unittest.main()
