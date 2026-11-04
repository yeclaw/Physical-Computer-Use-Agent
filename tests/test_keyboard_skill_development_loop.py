"""Offline integration of model-owned skill development through the real loop.

All decisions, observations and robot effects are fixtures. The real bounded
interpreter, knowledge store and development library are exercised; this is
not evidence of physical typing, model learning efficacy or stable promotion.
"""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill


ROOT = Path(__file__).resolve().parents[1]
SCOPE = {'task_id': 'development-fixture', 'device_id': 'fake-device', 'layout_id': 'fake-layout'}
GLYPHS = {letter: (index + 1) / 1000 for index, letter in enumerate('abcdefghijklmnopqrstuvwxyz')}
EMIT = [
    'def main(robot, profile):',
    '    for symbol in profile["payload"]:',
    '        x = profile["glyphs"][symbol]',
    '        robot.move_tcp(xyz_m=[x, 0, 0.1], quaternion_xyzw=[0,0,0,1])',
    '        robot.move_tcp(xyz_m=[x, 0, 0.2], quaternion_xyzw=[0,0,0,1])',
    '    return {"released": True}',
]
NOISY = EMIT[:-1] + [
    '    robot.move_tcp(xyz_m=[profile["glyphs"]["s"], 0, 0.1], quaternion_xyzw=[0,0,0,1])',
    '    robot.move_tcp(xyz_m=[profile["glyphs"]["s"], 0, 0.2], quaternion_xyzw=[0,0,0,1])',
    '    return {"released": True}',
]
CORRECT = ['def main(robot, profile):',
           '    robot.move_tcp(xyz_m=[-0.001, 0, 0.1], quaternion_xyzw=[0,0,0,1])',
           '    robot.move_tcp(xyz_m=[-0.001, 0, 0.2], quaternion_xyzw=[0,0,0,1])',
           '    return {"released": True}']


def declaration():
    return {'description': 'Fixture reusable text operation chosen by the mock RSI.',
            'input_schema': {'type': 'object', 'properties': {
                'payload': {'type': 'string', 'minLength': 1, 'maxLength': 20}},
                'required': ['payload'], 'additionalProperties': False},
            'preconditions': ['Only the observed fake layout applies.'],
            'postconditions': ['Inspect the next fixture observation for exact text.'],
            'side_effects': []}


def finish(status='partial', skills=()):
    return {'action': 'finish', 'result': {'status': status, 'reflection': {
        'summary': 'Fixture phase complete; physical reliability remains untested.',
        'skills_saved': list(skills), 'lessons_saved': [],
        'limitations': ['Fake model, robot and visual feedback; no physical result.']}}}


class Model:
    def __init__(self, choose):
        self.choose, self.requests, self.contexts, self.images = choose, [], [], []

    def run_turn(self, prompt, callback, images):
        index = len(self.requests)
        context = json.loads(prompt.split('\n')[0])
        self.requests.append(prompt)
        self.contexts.append(context)
        self.images.append(list(images))
        decision = self.choose(index, context, prompt)
        return {'turn_id': 'fixture-' + str(index), 'usage': {'input_tokens': 1, 'output_tokens': 1},
                'final_text': json.dumps(decision)}


class Environment:
    def __init__(self, root, store, prefix):
        self.root, self.store, self.prefix = root, store, prefix
        self.text, self.observations, self.executions = '', 0, []

    def evidence(self, suffix, value):
        identifier = self.prefix + '-' + suffix
        path = self.root / (identifier + '.json')
        path.write_text(json.dumps(value))
        self.store.register_evidence(identifier, [path])
        return identifier

    def observe(self):
        self.observations += 1
        identifier = self.evidence('obs-' + str(self.observations), {'visible_text': self.text})
        return {'evidence_id': identifier, 'contentItems': [
            {'type': 'inputText', 'text': json.dumps({'fixture_visible_text': self.text})}]}

    def execute(self, arguments):
        moves = []
        def move(xyz_m, quaternion_xyzw):
            moves.append(list(xyz_m))
            if xyz_m[2] == .1:
                if xyz_m[0] == -.001:
                    self.text = self.text[:-1]
                else:
                    self.text += next(key for key, value in GLYPHS.items() if value == xyz_m[0])
            return {'status': 'ok'}
        value = run_skill([('skill.py', arguments['source'])], profile=arguments['profile'],
            tools={'move_tcp': move, 'observe': lambda: {}, 'stop': lambda: {},
                   'wait_bounded': lambda seconds: {}})
        response = {'status': 'ok', 'result': {'skill_result': value, 'receipts': [],
                    'motion_outcome_uncertain': False}}
        self.executions.append({'arguments': arguments, 'moves': moves, 'response': response})
        identifier = self.evidence('exec-' + str(len(self.executions)), self.executions[-1])
        return {'response': response, 'evidence_id': identifier}


class SkillDevelopmentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='skill-development-loop-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'first')
        self.library = SkillDevelopment(self.store)
        evidence = self.root / 'seed.json'
        evidence.write_text('{"fixture":true}')
        self.store.register_evidence('seed-observation', [evidence])

    def old_skill(self, identifier, source, profile):
        return self.store.propose('skill', {'id': identifier, 'source_lines': source, 'profile': profile,
            'applicability': ['Fake layout only.'], 'limitations': ['Unverified fixture operation.'],
            'evidence_ids': ['seed-observation']})['candidate_id']

    def package_args(self, identifier='parameterized'):
        return {'id': identifier, 'source_lines': EMIT,
                'profile': {'glyphs': GLYPHS, 'payload': 'a'}, 'contract': declaration(),
                'applicability': ['Only this fake layout.'], 'limitations': ['No real hardware validation.'],
                'evidence_ids': ['seed-observation']}

    def package(self, logical_id):
        return next(item for item in self.library.search(logical_id, 'package', 32)['results']
                    if item['id'] == logical_id)

    def output(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_model_records_compensated_defect_creates_interfaces_and_reuses_new_combinations(self):
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
        seed = self.old_skill('noisy_seed', NOISY, {'glyphs': GLYPHS, 'payload': 'a'})
        approach = self.old_skill('approach', ['def main(robot, profile):',
            '    return robot.move_tcp(xyz_m=[0,0,0.2], quaternion_xyzw=[0,0,0,1])'], {})
        retreat = self.old_skill('retreat', ['def main(robot, profile):',
            '    return robot.move_tcp(xyz_m=[0,0,0.3], quaternion_xyzw=[0,0,0,1])'], {})
        lesson = self.store.propose('lesson', {'id': 'historical_side_effect',
            'claim': 'Full historical fixture text: a task may be corrected while the primitive remains defective.',
            'applicability': ['All fixture compositions using this primitive.'],
            'limitations': ['Observation-based fixture report, not physical truth.'],
            'evidence_ids': ['seed-observation']})['candidate_id']
        parent_dir = self.store.namespace / 'candidates' / seed
        parent_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent_dir.iterdir()}
        env = Environment(self.root, self.store, 'first')
        observed = {}
        def choose(index, context, raw):
            current = context['current_observation_evidence_id']
            if index == 0:
                return {'action': 'run_skill', 'arguments': {'label': 'noisy', 'source_lines': NOISY,
                    'profile': {'glyphs': GLYPHS, 'payload': 'a'}}, 'hypothesis': 'Fixture should expose the extra symbol.'}
            if index == 1:
                self.assertEqual(env.text, 'as')
                return {'action': 'record_improvement', 'arguments': {'id': 'extra_symbol',
                    'summary': 'The primitive emits an unintended additional symbol.',
                    'rationale': 'Compensation may fix one draft without improving repeated reuse.',
                    'status': 'open', 'skill_ids': [seed], 'evidence_ids': [current]}}
            if index == 2:
                return {'action': 'run_skill', 'arguments': {'label': 'correct_draft',
                    'source_lines': CORRECT, 'profile': {}}, 'hypothesis': 'Correct the draft in this fixture.'}
            if index == 3:
                self.assertEqual(env.text, 'a')
                return {'action': 'record_skill_outcome', 'arguments': {'id': 'compensated_result',
                    'skill_ids': [seed], 'evidence_ids': [current], 'observation_evidence_id': current,
                    'task_result': 'success', 'correction_performed': True,
                    'remaining_issues': ['The seed still emits an additional symbol.'],
                    'summary': 'Current draft corrected; underlying side effect remains.'}}
            if index == 4:
                pending = self.library.context()['pending_improvements']
                self.assertEqual(pending[0]['status'], 'open')
                observed['issue'] = pending[0]['record_id']
                args = self.package_args('revised_primitive')
                args.update(parent_id=seed, evidence_ids=[current], improvement_ids=[observed['issue']])
                return {'action': 'save_skill_package', 'arguments': args}
            if index == 5:
                revised = self.package('revised_primitive')
                self.assertEqual(context['feedback']['result']['package_id'], revised['package_id'])
                self.assertEqual(context['feedback']['result']['candidate_id'], revised['candidate_id'])
                self.assertEqual(context['feedback']['result']['contract'], declaration())
                observed['revised'] = revised
                args = self.package_args('composable_sequence')
                args.update(profile={}, source_lines=['def main(robot, profile):',
                    '    approach(robot, {})', '    press(robot, {"payload": profile["payload"]})',
                    '    return retreat(robot, {})'], dependencies={'approach': approach,
                    'press': revised['package_id'], 'retreat': retreat}, evidence_ids=[current])
                return {'action': 'save_skill_package', 'arguments': args}
            if index in (6, 7):
                composed = self.package('composable_sequence')
                if index == 6:
                    self.assertEqual(context['feedback']['result']['package_id'], composed['package_id'])
                    self.assertEqual(context['feedback']['result']['candidate_id'], composed['candidate_id'])
                else:
                    self.assertEqual(context['feedback']['package_id'], composed['package_id'])
                    self.assertEqual(context['feedback']['task_result_basis'], 'awaiting_model_visual_report')
                return {'action': 'call_skill', 'arguments': {'package_id': self.package('composable_sequence')['package_id'],
                    'arguments': {'payload': 'orbit' if index == 6 else 'delta'}},
                    'hypothesis': 'Reuse one declared interface for a previously unexecuted combination.'}
            if index == 8:
                self.assertEqual(env.text, 'aorbitdelta')
                return {'action': 'record_skill_outcome', 'arguments': {'id': 'new_combinations',
                    'skill_ids': [self.package('composable_sequence')['package_id']],
                    'evidence_ids': [current], 'observation_evidence_id': current,
                    'task_result': 'success', 'correction_performed': False, 'remaining_issues': [],
                    'summary': 'Two new fixture combinations executed without the prior side effect.'}}
            return finish(skills=[observed['revised']['candidate_id'], self.package('composable_sequence')['candidate_id']])
        model = Model(choose)
        result = run_episode(session=model, observe=env.observe, execute=env.execute, store=self.store,
            output=self.output('first-output'), max_decisions=12, skill_development=True)
        self.assertTrue(result['finished_by_model'])
        self.assertEqual(len(env.executions), 4)
        for execution in env.executions[2:]:
            self.assertEqual(execution['moves'][0], [0, 0, .2])
            self.assertEqual(execution['moves'][-1], [0, 0, .3])
            self.assertIn('approach', execution['arguments']['source'])
        self.assertFalse(result['stable_promoted'])
        self.assertEqual(self.library.context()['pending_improvements'][0]['status'], 'open')
        outcomes = self.library.search('', 'outcome', 8)['results']
        self.assertEqual(len(outcomes), 2)
        for outcome in outcomes:
            self.assertEqual(outcome['basis'], 'model_report')
            self.assertFalse(self.library.inspect(outcome['record_id'])['independently_verified'])
            self.assertFalse(outcome['stable'])
        self.assertEqual(parent_hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent_dir.iterdir()})

        later_store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'later')
        later = Environment(self.root, later_store, 'later')
        def later_choose(index, context, raw):
            if index == 0:
                return {'action': 'search_knowledge', 'arguments': {'query': 'extra symbol', 'kind': 'improvement'}}
            if index == 1:
                self.assertIn(observed['issue'], raw)
                return {'action': 'inspect_knowledge', 'arguments': {'identifier': lesson}}
            if index == 2:
                self.assertIn('Full historical fixture text', raw)
                return {'action': 'inspect_knowledge', 'arguments': {'identifier': self.package('composable_sequence')['package_id']}}
            if index == 3:
                self.assertIn('Only the observed fake layout applies.', raw)
                return {'action': 'call_skill', 'arguments': {'package_id': self.package('composable_sequence')['package_id'],
                    'arguments': {'payload': 'trial'}}, 'hypothesis': 'Reuse the stored interface in a new run.'}
            return finish()
        run_episode(session=Model(later_choose), observe=later.observe, execute=later.execute,
            store=later_store, output=self.output('later-output'), max_decisions=7, skill_development=True)
        self.assertEqual(later.text, 'trial')
        self.assertEqual(len(later.executions), 1)

    def test_bad_parameters_and_disabled_actions_cannot_dispatch(self):
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import parse_decision, run_episode
        package = self.library.save_package(self.package_args())
        decision = {'action': 'call_skill', 'arguments': {'package_id': package['package_id'],
                    'arguments': {'payload': 3}}, 'hypothesis': 'Malformed input must not move.'}
        for enabled in (False, True):
            env = Environment(self.root, self.store, 'reject-' + str(enabled))
            model = Model(lambda index, context, raw: decision if index == 0 else finish())
            run_episode(session=model, observe=env.observe, execute=env.execute, store=self.store,
                output=self.output('reject-' + str(enabled)), max_decisions=3, skill_development=enabled)
            self.assertEqual(env.executions, [])
            self.assertTrue((self.root / ('reject-' + str(enabled)) / 'turn-00/protocol-rejection.json').is_file())
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(decision), skill_development=False)
        for action in ('search_knowledge', 'inspect_knowledge', 'save_skill_package',
                       'record_improvement', 'record_skill_outcome', 'inspect_evidence'):
            with self.subTest(disabled_action=action), self.assertRaises(ValueError):
                parse_decision(json.dumps({'action': action, 'arguments': {}}), skill_development=False)
        no_hypothesis = {key: value for key, value in decision.items() if key != 'hypothesis'}
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(no_hypothesis), skill_development=True)

    def test_unknown_call_stops_after_one_dispatch_without_new_observation_or_retry(self):
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
        package = self.library.save_package(self.package_args())
        env = Environment(self.root, self.store, 'unknown')
        sent = []
        def unknown(arguments):
            sent.append(arguments)
            identifier = env.evidence('unknown-exec', {'status': 'unknown'})
            return {'response': {'status': 'unknown', 'result': {'motion_outcome_uncertain': True}},
                    'evidence_id': identifier}
        model = Model(lambda *unused: {'action': 'call_skill', 'arguments': {
            'package_id': package['package_id'], 'arguments': {'payload': 'orbit'}},
            'hypothesis': 'An uncertain dispatch must stop, never retry.'})
        report = run_episode(session=model, observe=env.observe, execute=unknown, store=self.store,
            output=self.output('unknown-output'), max_decisions=5, skill_development=True)
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(env.observations, 1)
        self.assertEqual(report['stopped_reason'], 'unknown_motion_no_retry')

    def test_outcome_for_a_registered_but_stale_observation_is_rejected(self):
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
        package = self.library.save_package(self.package_args())
        env = Environment(self.root, self.store, 'fresh')
        args = {'id': 'stale', 'skill_ids': [package['package_id']], 'evidence_ids': ['seed-observation'],
                'observation_evidence_id': 'seed-observation', 'task_result': 'success',
                'correction_performed': False, 'remaining_issues': [], 'summary': 'Stale report.'}
        model = Model(lambda index, context, raw: {'action': 'record_skill_outcome', 'arguments': args}
                      if index == 0 else finish())
        output = self.output('stale-output')
        run_episode(session=model, observe=env.observe, execute=env.execute, store=self.store,
            output=output, max_decisions=3, skill_development=True)
        self.assertEqual(self.library.search('', 'outcome')['results'], [])
        self.assertIn('current', (output / 'turn-00/protocol-rejection.json').read_text().lower())

    def test_historical_image_is_opt_in_and_does_not_replace_the_current_observation(self):
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
        image = self.root / 'historical.jpg'
        image.write_bytes(b'\xff\xd8\xfffixture-historical-image')
        self.store.register_evidence('old-image', [image])
        env = Environment(self.root, self.store, 'live')
        def choose(index, context, raw):
            if index == 0:
                return {'action': 'inspect_evidence', 'arguments': {'evidence_id': 'old-image'}}
            if index == 1:
                self.assertNotIn('image_base64', raw)
                return {'action': 'inspect_evidence', 'arguments': {'evidence_id': 'old-image',
                    'file_indices': [0], 'include_images': True}}
            self.assertIn('historical', raw.lower())
            self.assertEqual(context['current_observation_evidence_id'], 'live-obs-1')
            return finish()
        model = Model(choose)
        run_episode(session=model, observe=env.observe, execute=env.execute, store=self.store,
            output=self.output('image-output'), max_decisions=4, skill_development=True)
        self.assertEqual(model.images[:2], [[], []])
        self.assertEqual(model.images[2], ['data:image/jpeg;base64,' + base64.b64encode(image.read_bytes()).decode()])
        self.assertEqual(env.observations, 1)

    def test_training_completed_phase_continues_without_chat_waiter_or_fixed_target_word(self):
        from astra_robotics_agent.physical_keyboard.agent.campaign import load_task_contract
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
        document = json.loads((ROOT / 'configs/physical_keyboard/skill_development_contract.json').read_text())
        document['scope'] = SCOPE
        self.assertNotIn('target_text', document)
        path = self.root / 'contract.json'
        path.write_text(json.dumps(document))
        task = load_task_contract(path, SCOPE)
        self.assertTrue(task.is_skill_development_training)
        env = Environment(self.root, self.store, 'continuous')
        model = Model(lambda index, context, raw: finish('completed') if index == 0
                      else {'action': 'observe', 'arguments': {}} if index == 1 else finish())
        report = run_episode(session=model, observe=env.observe, execute=env.execute, store=self.store,
            output=self.output('continuous-output'), max_decisions=3, skill_development=True,
            task_contract=task)
        self.assertEqual(len(model.requests), 3)
        self.assertFalse(report['task_completed'])
        self.assertEqual(report['stopped_reason'], 'bounded_episode_decision_limit')
        self.assertGreaterEqual(env.observations, 2)


if __name__ == '__main__':
    unittest.main()
