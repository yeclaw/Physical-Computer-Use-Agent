"""Same-session fault suspension/resumption; all effects and model turns are fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.agent.campaign import load_task_contract
from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from test_keyboard_skill_development_loop import Environment, Model, SCOPE, EMIT, GLYPHS, finish

ROOT = Path(__file__).resolve().parents[1]


def emit(payload):
    return dict(action='run_skill', arguments=dict(label='fixture_' + payload, source_lines=EMIT,
                profile=dict(payload=payload, glyphs=GLYPHS)), hypothesis='Fixture input only.')


class PersistentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='persistent-loop-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = KnowledgeStore(self.root / 'store', SCOPE, 'fixture')
        self.env = Environment(self.root, self.store, 'fixture')
        self.output = self.root / 'output'
        self.output.mkdir()

    def run_model(self, choose, *, handler, **kwargs):
        model = Model(choose)
        report = kwargs.pop('report', {})
        options = dict(session=model, observe=self.env.observe, execute=self.env.execute,
            store=self.store, output=self.output, report=report, max_decisions=8,
            skill_development=True, capture_fault=True, fault_handler=handler)
        options.update(kwargs)
        run_episode(**options)
        return model, report

    def contract(self):
        document = json.loads((ROOT / 'configs/physical_keyboard/skill_development_contract.json').read_text())
        document['scope'] = SCOPE
        path = self.root / 'task.json'
        path.write_text(json.dumps(document))
        return load_task_contract(path, SCOPE)

    def events(self):
        return [json.loads(line) for line in (self.output / 'learning-events.jsonl').read_text().splitlines()]

    def test_unknown_terminal_and_transport_exception_resume_without_replaying_completed_prefix(self):
        # Each subtest has an isolated store; none reads actual robot state.
        for outcome in ('unknown', 'terminal', 'exception'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                store = KnowledgeStore(root / 'store', SCOPE, 'fixture')
                env = Environment(root, store, 'fixture')
                output = root / 'output'; output.mkdir()
                dispatched, faults = [], []
                def execute(arguments):
                    dispatched.append(arguments['profile']['payload'])
                    actual = env.execute(arguments)
                    if len(dispatched) > 1:
                        return actual
                    if outcome == 'exception':
                        raise RuntimeError('fixture EOF after dispatch')
                    actual['response'].update(status='unknown' if outcome == 'unknown' else 'error',
                        result=dict(receipts=[], motion_outcome_uncertain=outcome == 'unknown',
                                    controller_terminal=outcome == 'terminal'))
                    return actual
                def handler(fault):
                    faults.append(fault)
                    self.assertEqual(env.text, 'a')
                    return dict(observation=env.observe(), recovery=dict(generation=2))
                def choose(index, context, prompt):
                    if index == 0:
                        return emit('a')
                    if index == 1:
                        self.assertEqual(context['feedback']['phase'], 'execution_resumed')
                        self.assertFalse(context['feedback']['replayed'])
                        self.assertIn('"fixture_visible_text": "a"', prompt)
                        return emit('b')
                    return finish()
                model = Model(choose)
                report = run_episode(session=model, observe=env.observe, execute=execute, store=store,
                    output=output, skill_development=True, capture_fault=True, fault_handler=handler,
                    max_decisions=5)
                self.assertEqual(dispatched, ['a', 'b'])
                self.assertEqual(env.text, 'ab')
                self.assertEqual(len(model.requests), 3)
                self.assertEqual([row['index'] for row in report['turns']], [0, 1, 2])
                self.assertEqual(len(faults), 1)
                self.assertEqual(report['fault_history'][0]['status'], 'resumed')
                self.assertNotIn('stopped_reason', report)
                self.assertNotIn('fault_snapshot', report)
                self.assertEqual(len(list(output.glob('turn-*'))), 3)
                events = [json.loads(line) for line in (output / 'learning-events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(row['event'] == 'fault_suspended' for row in events), 1)
                self.assertEqual(sum(row['event'] == 'fault_resumed' for row in events), 1)

    def test_initial_observation_failure_has_no_invented_old_evidence(self):
        faults = []
        def observe():
            raise RuntimeError('fixture initial camera unavailable')
        def handler(fault):
            faults.append(fault)
            self.assertIsNone(fault['last_observation']['evidence_id'])
            self.assertEqual(fault['last_observation']['timing'], 'unavailable')
            self.assertIsNone(fault['attempted_program'])
            return dict(observation=self.env.observe(), recovery=dict(generation=2))
        model, report = self.run_model(lambda *unused: finish(), handler=handler, observe=observe)
        self.assertEqual(len(faults), 1)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.contexts[0]['feedback']['phase'], 'execution_resumed')
        self.assertEqual(len(self.env.executions), 0)
        self.assertEqual(report['fault_history'][0]['status'], 'resumed')

    def test_post_execution_observation_failure_does_not_repeat_or_verify_completed_action(self):
        count, faults = [], []
        def observe():
            count.append(1)
            if len(count) == 2:
                raise RuntimeError('fixture camera lost after completed motion')
            return self.env.observe()
        def handler(fault):
            faults.append(fault)
            self.assertEqual(fault['feedback']['execution']['status'], 'ok')
            self.assertEqual(fault['attempted_program']['label'], 'fixture_a')
            return dict(observation=self.env.observe(), recovery=dict(generation=2))
        model, report = self.run_model(lambda i, *unused: emit('a') if i == 0 else finish(),
            handler=handler, observe=observe)
        self.assertEqual(self.env.text, 'a')
        self.assertEqual(len(self.env.executions), 1)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(len(faults), 1)
        self.assertEqual(report['verifications'], [])
        self.assertEqual(model.contexts[1]['feedback']['phase'], 'execution_resumed')

    def test_reset_unknown_exception_and_post_reset_observation_failure_do_not_repeat_reset(self):
        from test_keyboard_training_reset import Fixture, Session, arguments, finished
        for outcome in ('unknown', 'exception', 'post_observation'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                calls, faults, observations = [], [], []
                def observe():
                    observations.append(1)
                    if outcome == 'post_observation' and len(observations) == 2:
                        raise RuntimeError('fixture post-reset camera unavailable')
                    return fixture.observe()
                def reset(args, recovery):
                    calls.append(recovery)
                    if outcome == 'exception':
                        raise RuntimeError('fixture reset EOF')
                    response = dict(status='unknown' if outcome == 'unknown' else 'ok',
                                    result=dict(reset_outcome_uncertain=outcome == 'unknown',
                                                motion_outcome_uncertain=outcome == 'unknown'))
                    return dict(response=response, evidence_id=fixture.evidence('reset-response', response))
                def handler(fault):
                    faults.append(fault)
                    return dict(observation=fixture.observe(), recovery=dict(generation=2))
                session = Session(lambda i, *unused: dict(action='reset', arguments=arguments()) if i == 0 else finished())
                report = run_episode(session=session, observe=observe, execute=fixture.execute, reset=reset,
                    store=fixture.store, output=fixture.output, skill_development=True, capture_fault=True,
                    fault_handler=handler, max_decisions=4)
                self.assertEqual((len(calls), len(faults), len(session.prompts)), (1, 1, 2))
                self.assertEqual(report['fault_history'][0]['status'], 'resumed')
                self.assertNotIn('fault_snapshot', report)
                if outcome != 'exception':
                    self.assertEqual(len(report['reset_attempts']), 1)

    def test_phase_history_candidates_and_review_saved_candidates_survive_recovery(self):
        saved = {}
        def handler(fault):
            candidate = self.store.propose('lesson', dict(id='review_fact', claim='Fixture fault outcome remains unverified.',
                applicability=['Fixture.'], limitations=['No actual hardware.'],
                evidence_ids=[fault['last_observation']['evidence_id']]))
            saved['review'] = candidate
            return dict(observation=self.env.observe(), recovery=dict(generation=2, saved_candidates=[candidate]))
        def execute(arguments):
            actual = self.env.execute(arguments)
            actual['response'] = dict(status='unknown', result=dict(motion_outcome_uncertain=True))
            return actual
        def choose(index, context, unused):
            if index == 0:
                return dict(action='save_lesson', arguments=dict(id='before_fault', claim='Fixture prior knowledge.',
                    applicability=['Fixture.'], limitations=['No hardware.'],
                    evidence_ids=[context['current_observation_evidence_id']]))
            if index == 1:
                saved['before'] = context['feedback']['candidate']['candidate_id']
                return finish(status='completed')
            if index == 2:
                return emit('a')
            result = finish(status='completed')
            result['result']['reflection']['lessons_saved'] = [saved['before'], saved['review']['candidate_id']]
            return result
        model, report = self.run_model(choose, handler=handler, execute=execute,
            task_contract=self.contract(), max_decisions=4)
        self.assertEqual(len(model.requests), 4)
        self.assertEqual(len(report['development_phase_records']), 2)
        self.assertEqual(len(report['saved_candidates']), 2)
        self.assertEqual(len(report['turns']), 4)
        self.assertFalse(report['task_completed'])
        self.assertEqual(len(self.env.executions), 1)
        self.assertEqual(report['stopped_reason'], 'bounded_episode_decision_limit')

    def test_handler_stop_and_failure_propagate_without_new_model_or_action(self):
        for error in (OperatorStop('fixture STOP'), RuntimeError('fixture restore failed'), ValueError('fixture bad restore')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); output = root / 'out'; output.mkdir()
                store = KnowledgeStore(root / 'store', SCOPE, 'fixture')
                env = Environment(root, store, 'fixture'); calls = []
                def execute(arguments):
                    calls.append(arguments)
                    return dict(response=dict(status='unknown'), evidence_id=env.evidence('exec', {}))
                def handler(unused):
                    raise error
                model = Model(lambda *unused: emit('a'))
                report = {}
                with self.assertRaises(type(error)) as raised:
                    run_episode(session=model, observe=env.observe, execute=execute, store=store, output=output,
                        skill_development=True, capture_fault=True, fault_handler=handler, report=report, max_decisions=4)
                self.assertIs(raised.exception, error)
                self.assertEqual((len(calls), len(model.requests), env.observations), (1, 1, 1))
                self.assertEqual(report['fault_history'][0]['status'], 'suspended')

    def test_operator_stop_in_observe_or_execute_is_not_sent_to_fault_handler(self):
        for stage in ('observe', 'execute'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve(); output=root/'out'; output.mkdir()
                store=KnowledgeStore(root/'store',SCOPE,'fixture'); env=Environment(root,store,'fixture')
                def stop(*unused):
                    raise OperatorStop('fixture STOP')
                model=Model(lambda *unused: emit('a'))
                with self.assertRaises(OperatorStop):
                    run_episode(session=model, observe=stop if stage=='observe' else env.observe,
                        execute=stop if stage=='execute' else env.execute, store=store, output=output,
                        skill_development=True, capture_fault=True,
                        fault_handler=lambda unused:self.fail('STOP must not enter recovery'), max_decisions=3)

    def test_stale_or_unregistered_recovery_observation_is_rejected(self):
        for invalid in ('stale', 'unregistered'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve(); output=root/'out'; output.mkdir()
                store=KnowledgeStore(root/'store',SCOPE,'fixture'); env=Environment(root,store,'fixture')
                initial=[]
                def observe():
                    value=env.observe(); initial.append(value); return value
                def handler(unused):
                    value=initial[0] if invalid=='stale' else dict(evidence_id='unregistered',contentItems=[])
                    return dict(observation=value,recovery={})
                model=Model(lambda *unused:emit('a'))
                with self.assertRaises((RuntimeError,ValueError)):
                    run_episode(session=model,observe=observe,execute=lambda unused:dict(
                        response=dict(status='unknown'),evidence_id=env.evidence('exec',{})),store=store,
                        output=output,skill_development=True,capture_fault=True,fault_handler=handler,max_decisions=3)
                self.assertEqual(len(model.requests),1)

    def test_recoverable_program_error_is_returned_to_model_without_controller_recovery(self):
        def execute(arguments):
            actual=self.env.execute(arguments)
            actual['response']=dict(status='error',result=dict(receipts=[],motion_outcome_uncertain=False,
                recoverable_program_error=True,error=dict(type='SkillError',message='Fixture post-tool code error')))
            return actual
        model,report=self.run_model(lambda i,*unused:emit('a') if i==0 else finish(),execute=execute,
            handler=lambda unused:self.fail('A healthy controller must not enter recovery'))
        self.assertEqual(self.env.text,'a')
        self.assertEqual(report['fault_history'],[])
        self.assertEqual(model.contexts[1]['feedback']['execution_disposition'],'partial_program_error')
        self.assertFalse(model.contexts[1]['feedback']['replayed'])

    def test_training_assistance_waits_then_continues_same_task_without_execution(self):
        requests=[]
        def wait(request):
            requests.append(request)
            return self.env.observe()
        def choose(i,context,unused):
            if i==0:
                return dict(action='request_assistance',arguments=dict(kind='operator_action',reason='Fixture lost focus.',
                    required_human_action='Restore fixture focus.',observation_evidence_id=context['current_observation_evidence_id']))
            self.assertEqual(context['feedback']['phase'],'assistance_update')
            return finish(status='completed')
        model,report=self.run_model(choose,handler=lambda unused:self.fail('No controller fault'),
            assistance_wait=wait,task_contract=self.contract(),max_decisions=2)
        self.assertEqual((len(requests),len(model.requests),len(self.env.executions)),(1,2,0))
        self.assertFalse(report['assistance_history'][0]['terminal'])
        self.assertEqual(len(report['development_phase_records']),1)
        self.assertNotIn('assistance_requested',report)

    def test_persistent_callbacks_require_explicit_development_and_capture(self):
        for kwargs in [dict(fault_handler=lambda unused:None),
                       dict(skill_development=True,fault_handler=lambda unused:None),
                       dict(skill_development=True,capture_fault=True,fault_handler=True),
                       dict(skill_development=True,capture_fault=True,assistance_wait=lambda unused:None)]:
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):
                run_episode(session=None,observe=lambda:self.fail('No observation'),execute=None,store=None,
                            output=None,**kwargs)


if __name__=='__main__':
    unittest.main()
