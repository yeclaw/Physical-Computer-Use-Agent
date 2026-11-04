"""Offline Return repair/storage/reuse regression through the real loop and AST.

The model decisions, robot movement and visual outcome are fixtures. This checks
the framework's failure-to-revision-to-reuse plumbing, not physical learning,
contact safety, independently verified text success or stable-skill promotion.
All coordinates belong to the fake test environment, not the Shanghai keyboard.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


SCOPE = dict(task_id='return-fixture', device_id='fake-robot', layout_id='fake-layout-a')
SHALLOW_SOURCE = '''def main(robot, profile):
    robot.move_tcp(xyz_m=profile['shallow_xyz_m'], quaternion_xyzw=profile['quaternion_xyzw'])
    robot.move_tcp(xyz_m=profile['hover_xyz_m'], quaternion_xyzw=profile['quaternion_xyzw'])
    return {'released_state': robot.observe()}
'''
REVISED_SOURCE = '''def main(robot, profile):
    state = robot.observe()
    pose = state['observation']['pose']
    for i in range(3):
        if abs(pose['xyz_m'][i] - profile['hover_xyz_m'][i]) > profile['start_tolerance_m']:
            return {'status': 'start_pose_mismatch', 'state': state}
    for xyz in profile['press_release_xyz_m']:
        robot.move_tcp(xyz_m=xyz, quaternion_xyzw=profile['quaternion_xyzw'])
    return {'released_state': robot.observe()}
'''
PROFILE = dict(hover_xyz_m=[.1, 0., .2], shallow_xyz_m=[.1, 0., .16],
               press_release_xyz_m=[[.1, 0., .14], [.1, 0., .12], [.1, 0., .2]],
               quaternion_xyzw=[0., 0., 0., 1.], start_tolerance_m=.001)


def finish(skills=(), lessons=(), **extra):
    reflection = dict(summary='Offline fixture completed the repair/reuse sequence.',
                      skills_saved=list(skills), lessons_saved=list(lessons),
                      limitations=['Mock robot and visual outcomes; no physical success claim.'])
    return dict(action='finish', result=dict(status='partial', reflection=reflection, **extra))


def run_request(source):
    return dict(action='run_skill', arguments=dict(label='return_attempt',
                source_lines=source.splitlines(), profile=PROFILE),
                hypothesis='Fake visual feedback should show whether Return registered and released.')


class ModelFixture:
    def __init__(self, choose):
        self.choose = choose
        self.requests = []

    def run_turn(self, prompt, callback, images):
        index = len(self.requests)
        self.requests.append(prompt)
        context = json.loads(prompt.split('\n')[0])
        decision = self.choose(index, context, prompt)
        return dict(turn_id='fixture-%02d' % index,
                    usage={'input_tokens': 1, 'output_tokens': 1},
                    final_text=json.dumps(decision))


class ReturnEnvironment:
    """Only the action/observation boundary is fake; AST and storage are real."""
    def __init__(self, root, store, prefix):
        self.root = root
        self.store = store
        self.prefix = prefix
        self.pose = list(PROFILE['hover_xyz_m'])
        self.sent = False
        self.executions = []
        self.observations = 0

    def evidence(self, suffix, value):
        evidence_id = self.prefix + '-' + suffix
        path = self.root / (evidence_id + '.json')
        path.write_text(json.dumps(value))
        self.store.register_evidence(evidence_id, [path])
        return evidence_id

    def observe(self):
        self.observations += 1
        value = dict(outgoing_visible=self.sent, compose_empty=self.sent,
                     released=self.pose[2] >= .19, evidence_mode='fixture')
        evidence_id = self.evidence('obs-%02d' % self.observations, value)
        return dict(evidence_id=evidence_id, contentItems=[
            dict(type='inputText', text=json.dumps({'fixture_visual_outcome': value}))])

    def execute(self, arguments):
        from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill
        moves = []
        def state():
            return {'observation': {'pose': {'xyz_m': list(self.pose),
                                           'quaternion_xyzw': PROFILE['quaternion_xyzw']}}}
        def move(xyz_m, quaternion_xyzw):
            self.pose = list(xyz_m)
            moves.append(list(xyz_m))
            if xyz_m[2] <= .13:
                self.sent = True
            return dict(status='ok', actual_xyz_m=list(self.pose))
        result = run_skill([('skill.py', arguments['source'])],
            tools=dict(observe=state, move_tcp=move, wait_bounded=lambda seconds: {}, stop=lambda: {}),
            profile=arguments['profile'])
        response = dict(status='ok', result=dict(skill_result=result, receipts=[],
                        evidence_mode='fixture', physical_text_success=None))
        self.executions.append(dict(source=arguments['source'], profile=arguments['profile'],
                                    moves=moves, response=response))
        evidence_id = self.evidence('exec-%02d' % len(self.executions), self.executions[-1])
        return dict(response=response, evidence_id=evidence_id)


class EnterSkillReuseTests(unittest.TestCase):
    def test_failed_seed_is_revised_saved_and_reused_from_a_fresh_store_session(self):
        from astra_robotics_agent.physical_keyboard.agent.learning_loop import run_episode
        from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
        with tempfile.TemporaryDirectory(prefix='return-repair-') as folder:
            root = Path(folder).resolve()
            store = KnowledgeStore(root / 'knowledge', SCOPE, 'repair-run')
            env = ReturnEnvironment(root, store, 'repair')
            output = root / 'repair-output'
            output.mkdir()
            ids = {}
            def choose(index, context, raw):
                feedback = context['feedback']
                if index == 0:
                    return run_request(SHALLOW_SOURCE)
                if index == 1:
                    self.assertEqual(feedback['execution']['status'], 'ok')
                    self.assertIn('"outgoing_visible": false', raw)
                    self.assertFalse(env.sent)
                    return dict(action='save_lesson', arguments=dict(id='return_seed_missed',
                        claim='Fixture motion returned ok but the next observation showed no sent message.',
                        applicability=['Fake Return environment only'],
                        limitations=['No physical hardware or text-ground-truth validation'],
                        evidence_ids=feedback['evidence_ids']))
                if index == 2:
                    ids['lesson'] = feedback['candidate']['candidate_id']
                    return run_request(REVISED_SOURCE)
                if index == 3:
                    self.assertIn('"outgoing_visible": true', raw)
                    self.assertIn('"released": true', raw)
                    return dict(action='save_skill', arguments=dict(id='return_local_press_release',
                        source_lines=REVISED_SOURCE.splitlines(), profile=PROFILE,
                        applicability=['Fake Return hover start only; approach and retreat are separate'],
                        limitations=['Fixture result only; not a stable physical key skill'],
                        evidence_ids=feedback['evidence_ids']))
                ids['skill'] = feedback['candidate']['candidate_id']
                return finish(skills=[ids['skill']], lessons=[ids['lesson']])
            result = run_episode(session=ModelFixture(choose), observe=env.observe,
                execute=env.execute, store=store, output=output, max_decisions=8)
            self.assertTrue(result['finished_by_model'])
            self.assertEqual(len(env.executions), 2)
            self.assertNotEqual(env.executions[0]['source'], env.executions[1]['source'])
            self.assertEqual([len(x['moves']) for x in env.executions], [2, 3])
            self.assertFalse(result['stable_promoted'])
            self.assertEqual(len(result['saved_candidates']), 2)
            original_objects = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in store.namespace.rglob('*') if p.is_file()}
            original_index = (store.namespace / 'index.jsonl').read_bytes()

            reopened = KnowledgeStore(root / 'knowledge', SCOPE, 'reuse-run')
            self.assertIn('return_seed_missed', reopened.context_text())
            self.assertIn('hover start only', reopened.context_text())
            reuse = ReturnEnvironment(root, reopened, 'reuse')
            reuse_output = root / 'reuse-output'
            reuse_output.mkdir()
            def reuse_choose(index, context, raw):
                if index == 0:
                    return dict(action='inspect_skill', arguments={'candidate_id': ids['skill']})
                if index == 1:
                    skill = context['feedback']['skill']
                    self.assertEqual(skill['source'], REVISED_SOURCE)
                    self.assertEqual(skill['profile'], PROFILE)
                    return dict(action='verify_skill', arguments={'candidate_id': ids['skill']},
                        hypothesis='Reused frozen hover-only program should register and release in this fixture.')
                self.assertIn('"outgoing_visible": true', raw)
                return finish(reused_skill=ids['skill'])
            reused = run_episode(session=ModelFixture(reuse_choose), observe=reuse.observe,
                execute=reuse.execute, store=reopened, output=reuse_output, max_decisions=5)
            self.assertEqual(len(reuse.executions), 1)
            self.assertEqual(reuse.executions[0]['source'], REVISED_SOURCE)
            self.assertEqual(reuse.executions[0]['moves'], PROFILE['press_release_xyz_m'])
            self.assertEqual(reused['saved_candidates'], [])
            self.assertEqual(reused['verifications'][0]['status'], 'verified_execution_only')
            self.assertFalse(reused['verifications'][0]['stable'])
            self.assertEqual(reused['verifications'][0]['text_success'], 'unknown')
            self.assertFalse(reused['stable_promoted'])
            for path, digest in original_objects.items():
                if path.name == 'index.jsonl':
                    continue  # Evidence/verification append to the index; originals remain immutable.
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertTrue((store.namespace / 'index.jsonl').read_bytes().startswith(original_index))
            events = [json.loads(line) for line in (output / 'learning-events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event'] == 'model_decision' for e in events), 5)
            self.assertTrue(any(e['event'] == 'framework_feedback' and
                                e['data'].get('action') == 'save_skill' for e in events))


if __name__ == '__main__':
    unittest.main()
