"""Real episode/store/analysis paths with declared fake model and robot effects.

Every artifact lives in a resolved temporary directory. These fixtures verify
software boundaries and compatibility, not physical or model learning efficacy.
"""
import base64
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.adapters.skill_sources.learning_sources import LearningSources
from astra_robotics_agent.physical_keyboard.agent.campaign import load_task_contract
from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
from astra_robotics_agent.physical_keyboard.agent.learning import ContinualLearning
from astra_robotics_agent.physical_keyboard.agent.learning_loop import parse_decision, run_episode
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS
from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore


ROOT = Path(__file__).parents[1]
SCOPE = dict(task_id='continual-integration', device_id='fixture-device', layout_id='fixture-layout')


def partial():
    return dict(action='finish', result=dict(status='partial', reflection=dict(
        summary='Fixture declaration only; the task remains unverified.', skills_saved=[],
        lessons_saved=[], limitations=['No physical or real model validation.'])))


def package(evidence_id, **updates):
    result = dict(id='fixture-package', source_lines=['def main(robot, profile):',
        '    return {"text": profile["text"], "exact": profile["exact"]}'],
        profile={'text': 'old-profile', 'exact': 1},
        contract=dict(description='A model-authored fixture parameter interface',
            input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
            preconditions=['Inspect the fixture start'], postconditions=['Read fresh fixture evidence']),
        applicability=['Only the fake scope'], limitations=['No physical success evidence'],
        evidence_ids=[evidence_id])
    result.update(updates)
    return result


class Model:
    model, effort, service_tier = 'fixture-model', 'high', 'default'
    thread_id, total_usage_complete = 'fixture-thread', True

    def __init__(self, choose):
        self.choose, self.contexts, self.prompts, self.images = choose, [], [], []
        self.total_usage = {}

    def run_turn(self, prompt, callback, images):
        context = json.loads(prompt.split('\n')[0])
        index = len(self.contexts)
        self.contexts.append(context); self.prompts.append(prompt); self.images.append(list(images))
        decision = self.choose(index, context)
        usage = dict(inputTokens=11, cachedInputTokens=0, outputTokens=2,
                     reasoningOutputTokens=0, totalTokens=13)
        for key, value in usage.items():
            self.total_usage[key] = self.total_usage.get(key, 0) + value
        return dict(turn_id='fixture-' + str(index), final_text=json.dumps(decision), usage=usage)


class Environment:
    def __init__(self, root):
        self.root = root
        self.access = LearningAccess(SCOPE)
        self.store = KnowledgeStore(root / 'knowledge', SCOPE, 'fixture-run', learning_access=self.access)
        self.sources = LearningSources(SCOPE, access=self.access)
        self.output = root / 'episode'; self.output.mkdir()
        self.observations, self.executions = 0, []

    def observe(self):
        self.observations += 1
        evidence_id = 'observation-' + str(self.observations)
        folder = self.root / evidence_id; folder.mkdir()
        response = folder / 'response.json'
        response.write_text(json.dumps(dict(state=dict(joint=[.12345678912345678]),
                                           snapshot=dict(fixture=True))))
        image = folder / 'right_wrist.jpg'; image.write_bytes(b'\xff\xd8\xfffixture-historical-jpeg')
        self.store.register_evidence(evidence_id, [response, image])
        return dict(evidence_id=evidence_id, contentItems=[
            dict(type='inputText', text='Trusted fake observation ' + evidence_id),
            dict(type='inputImage', imageUrl='data:image/jpeg;base64,' + base64.b64encode(image.read_bytes()).decode())])

    def execute(self, arguments):
        # The real bounded AST interpreter runs without any physical callbacks.
        result = run_skill([('fixture.py', arguments['source'])], tools={}, profile=arguments['profile'])
        self.executions.append(dict(arguments=arguments, result=result))
        evidence_id = 'execution-' + str(len(self.executions))
        folder = self.root / evidence_id; folder.mkdir()
        response = dict(status='ok', result=dict(skill_result=result, receipts=[], motion_outcome_uncertain=False))
        contents = {'request.json': json.dumps(arguments), 'skill.py': arguments['source'],
                    'profile.json': json.dumps(arguments['profile']), 'response.json': json.dumps({'response': response})}
        files = []
        for name, content in contents.items():
            path = folder / name; path.write_text(content); files.append(path)
        self.store.register_evidence(evidence_id, files)
        return dict(response=response, evidence_id=evidence_id)

    def run(self, model, *, enabled=True, sources=True, **kwargs):
        learning = (dict(learning_sources=self.sources if sources else None,
                         learning_access=self.access) if enabled else {})
        return run_episode(session=model, observe=self.observe, execute=self.execute,
            store=self.store, output=self.output, skill_development=True,
            continual_learning=enabled, **learning, **kwargs)


class ContinualLoopIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='continual-loop-fixture-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.environment = Environment(self.root)

    def test_disabled_tools_reject_explicitly_without_added_roundtrips_or_learning_namespace(self):
        decision = dict(action='knowledge_directory', arguments={})
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(decision), skill_development=True, continual_learning=False)
        self.assertEqual(parse_decision(json.dumps(decision), skill_development=True, continual_learning=True), decision)
        model = Model(lambda *_: partial())
        result = self.environment.run(model, enabled=False)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(len(model.contexts), 1)
        self.assertIsNone(result['max_decisions'])
        self.assertFalse((self.environment.store.namespace / 'learning').exists())
        self.assertEqual(self.environment.executions, [])

    def test_no_demo_own_evidence_is_available_only_when_explicitly_requested(self):
        def choose(index, context):
            if index == 0:
                self.assertNotIn('demo_trajectories', context)
                return dict(action='list_learning_sources', arguments={'origin': 'evidence'})
            if index == 1:
                page = context['feedback']['result']['registered_evidence']
                self.assertEqual(page['items'][0]['source_id'], 'evidence:observation-1')
                return dict(action='inspect_learning_sources', arguments={'source_id': 'evidence:observation-1'})
            self.assertTrue(context['feedback']['observation_unchanged'])
            self.assertTrue(context['feedback']['result']['provenance']['historical'])
            return partial()
        model = Model(choose)
        result = self.environment.run(model, sources=False, max_decisions=5)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(result['reply']['status'], 'partial')
        self.assertEqual((self.environment.observations, len(self.environment.executions)), (1, 0))

    def test_partial_with_continuous_contract_remains_nonterminal_and_can_read_knowledge(self):
        document = json.loads((ROOT / 'configs/physical_keyboard/task_contract.json').read_text())
        document['scope'] = SCOPE
        path = self.root / 'task.json'; path.write_text(json.dumps(document))
        contract = load_task_contract(path, SCOPE)
        def choose(index, context):
            if index == 0:
                return partial()
            self.assertEqual(context['task_state']['checkpoints_completed'], 1)
            return dict(action='knowledge_directory', arguments={})
        model = Model(choose)
        result = self.environment.run(model, sources=False, task_contract=contract, max_decisions=2)
        self.assertFalse(result['task_completed'])
        self.assertFalse(result['finished_by_model'])
        self.assertEqual(len(result['phase_checkpoints']), 1)
        self.assertEqual(self.environment.executions, [])

    def test_source_analysis_commit_materializes_exact_profile_for_original_call_skill(self):
        self.environment.sources.register('numeric-fixture', kind='robot_attempt',
            data={'samples': [{'q': [.12345678912345678]}]})
        state = {}
        def choose(index, context):
            feedback = context.get('feedback') or {}
            if index == 0:
                return dict(action='inspect_learning_sources', arguments={'source_id': 'numeric-fixture'})
            if index == 1:
                self.assertEqual(feedback['result']['data']['samples']['type'], 'array')
                return dict(action='analyze_learning_sources', arguments=dict(id='exact-profile',
                    source_lines=['def analyze(data, parameters):',
                        '    return {"text": "computed", "exact": data["sample"]["samples"][0]["q"][0] + parameters["delta"]}'],
                    inputs={'sample': {'source_id': 'numeric-fixture'}}, parameters={'delta': .1}))
            if index == 2:
                state['resource_id'] = feedback['result']['resource']['resource_id']
                declaration = package(context['current_observation_evidence_id'],
                    profile_resource_id=state['resource_id'], local_ref='p')
                del declaration['profile']
                return dict(action='commit_learning', arguments={'packages': [declaration]})
            if index == 3:
                state['package_id'] = feedback['result']['bindings']['p']
            if len(self.environment.executions) == 0:
                return dict(action='call_skill', arguments={'package_id': state['package_id'], 'arguments': {}},
                            hypothesis='The explicit fixture result should preserve the computed value')
            return partial()
        result = self.environment.run(Model(choose), max_decisions=9)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(len(self.environment.executions), 1)
        actual = self.environment.executions[0]
        self.assertEqual(actual['arguments']['profile']['exact'], .12345678912345678 + .1)
        self.assertEqual(actual['result'], {'text': 'computed', 'exact': .12345678912345678 + .1})
        self.assertFalse(result['stable_promoted'])

    def test_half_commit_is_hidden_from_old_search_inspect_verify_call_and_dependency_save(self):
        observation = self.environment.observe()
        development = DevelopmentActions(self.environment.store)
        facade = ContinualLearning(development, Model(lambda *_: partial()),
            output=self.root / 'prepared', access=self.environment.access)
        declaration = package(observation['evidence_id'], id='pending-package')
        plan = facade.changes.preflight({'packages': [declaration]})
        def fail(stage):
            if stage == 'before_final_commit': raise RuntimeError('Fixture interrupted publication')
        facade.changes.failure_hook = fail
        with self.assertRaises(RuntimeError):
            facade.changes.commit({'packages': [declaration]})
        pending = plan['operations'][0]['identifier']
        candidate = development.library.inspect(pending)['candidate_id']
        decisions = [
            dict(action='search_knowledge', arguments={'query': 'pending-package', 'kind': 'all', 'limit': 8}),
            dict(action='inspect_knowledge', arguments={'identifier': pending}),
            dict(action='verify_skill', arguments={'candidate_id': candidate}, hypothesis='Fixture gate must reject'),
            dict(action='call_skill', arguments={'package_id': pending, 'arguments': {}}, hypothesis='Fixture gate must reject'),
            dict(action='save_skill_package', arguments=package('observation-2', id='child', dependencies={'part': pending})),
            partial(),
        ]
        def choose(index, context):
            if index == 1:
                self.assertNotIn(pending, json.dumps(context['feedback']['result']))
            if 2 <= index <= 5:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
            return decisions[index]
        result = self.environment.run(Model(choose), max_decisions=8)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(self.environment.executions, [])

    def test_explicit_partial_demo_policy_filters_directory_and_rejects_hidden_source_analysis(self):
        # The supplied source service deliberately retains a broader catalog.
        # The loop boundary still enforces the explicitly selected subset.
        self.environment.sources.register('allowed-demo', kind='human_demo', data={'samples': [{'q': [1]}]})
        self.environment.sources.register('hidden-demo', kind='human_demo',
            data={'samples': [{'q': [999], 'secret_fixture': 'never-show-hidden-parameters'}]})
        access = LearningAccess(SCOPE, dict(protocol=PROTOCOLS['access'], scope=SCOPE,
            allowed_source_ids=['allowed-demo'], allowed_knowledge_ids=[],
            allowed_evidence_ids=[], allow_own_experience=True))
        self.environment.access = access
        self.environment.store.learning_access = access
        def choose(index, context):
            if index == 0:
                return dict(action='list_learning_sources', arguments={'origin': 'sources', 'kind': 'human_demo'})
            if index == 1:
                self.assertEqual([row['source_id'] for row in context['feedback']['result']['sources']], ['allowed-demo'])
                return dict(action='inspect_learning_sources', arguments={'source_id': 'hidden-demo'})
            if index == 2:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
                return dict(action='analyze_learning_sources', arguments=dict(id='forbidden-analysis',
                    source_lines=['def analyze(data, parameters):', '    return data'],
                    inputs={'hidden': {'source_id': 'hidden-demo'}}, parameters={}))
            if index == 3:
                self.assertEqual(context['feedback']['phase'], 'request_rejected')
                return dict(action='inspect_learning_sources', arguments={'source_id': 'allowed-demo'})
            self.assertEqual(context['feedback']['result']['source_id'], 'allowed-demo')
            return partial()
        model = Model(choose)
        result = self.environment.run(model, max_decisions=7)
        self.assertTrue(result['finished_by_model'])
        self.assertNotIn('never-show-hidden-parameters', json.dumps(model.contexts))
        self.assertEqual(self.environment.executions, [])


if __name__ == '__main__':
    unittest.main()
