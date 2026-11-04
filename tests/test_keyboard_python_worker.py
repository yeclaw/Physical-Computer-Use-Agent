"""Frozen executable contracts and real bounded-Python worker subprocesses."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.contracts import PROTOCOL, Scope, canonical, digest
from astra_robotics_agent.physical_keyboard.executable import resolve_execution
from astra_robotics_agent.physical_keyboard.transport import JsonProcess


def descriptor(entry='skill.py', includes=None):
    return canonical(dict(protocol='physical-keyboard-executable-v1', kind='bounded-python-v1',
                          entry=entry, includes=[] if includes is None else includes))


class ExecutableTests(unittest.TestCase):
    def bundle(self, files):
        objects = {digest(content): content for content in files.values()}
        bundle = {'files': {name: {'sha256': digest(content), 'bytes': len(content)} for name, content in files.items()}}
        return bundle, objects.__getitem__

    def test_legacy_program_and_profile_remain_opaque(self):
        files = {'actions.json': b'bad program', 'profile.json': b'legacy profile bytes'}
        bundle, read = self.bundle(files)
        value = resolve_execution(bundle, read)
        self.assertEqual(value['kind'], 'declarative-v1')
        self.assertEqual(value['program_sha256'], digest(files['actions.json']))
        self.assertIsNone(value['profile'])
        self.assertEqual(value['sources'], files)

    def test_python_sources_profile_and_descriptor_are_frozen(self):
        files = {'execution.json': descriptor(includes=['lib/helpers.py']), 'profile.json': canonical({'depth': .01}),
                 'skill.py': b'def main(robot, profile):\n    return helper(profile)\n',
                 'lib/helpers.py': b'def helper(profile):\n    return profile["depth"]\n'}
        bundle, read = self.bundle(files)
        value = resolve_execution(bundle, read)
        self.assertEqual(value['sources'], files)
        self.assertEqual(value['profile'], {'depth': .01})
        self.assertEqual(value['file_name'], 'execution.json')
        self.assertEqual(value['program_sha256'], digest(files['execution.json']))

    def test_descriptor_unsafe_duplicate_and_nonpython_sources_rejected(self):
        for entry, includes in [('../escape.py', []), ('/escape.py', []), ('a//b.py', []),
                                ('skill.py', ['skill.py']), ('skill.py', ['helper.py', 'helper.py']),
                                ('skill.txt', []), ('skill.py', ['helper.txt']),
                                ('skill.py', ['h' + str(i) + '.py' for i in range(16)])]:
            with self.subTest(entry=entry, includes=includes):
                bundle, read = self.bundle({'execution.json': descriptor(entry, includes), 'profile.json': b'{}',
                                           'skill.py': b'def main(robot, profile):\n    return None\n'})
                with self.assertRaises(ValueError):
                    resolve_execution(bundle, read)

    def test_source_byte_and_sha_and_size_limits(self):
        files = {'execution.json': descriptor(), 'profile.json': b'{}', 'skill.py': b'x' * (65536 + 1)}
        bundle, read = self.bundle(files)
        with self.assertRaises(ValueError):
            resolve_execution(bundle, read)
        files['skill.py'] = b'normal bytes'
        bundle, read = self.bundle(files)
        with self.assertRaises(ValueError):
            resolve_execution(bundle, lambda _: b'changed bytes')
        bundle['files']['skill.py']['bytes'] += 1
        with self.assertRaises(ValueError):
            resolve_execution(bundle, read)

    def test_python_metadata_has_consistent_valueerror_contract(self):
        for profile in [b'[]', b'null', b'{"x":1,"x":2}', b'{"x":NaN}']:
            with self.subTest(profile=profile):
                bundle, read = self.bundle({'execution.json': descriptor(), 'profile.json': profile,
                                           'skill.py': b'def main(robot, profile):\n    return None\n'})
                with self.assertRaises(ValueError):
                    resolve_execution(bundle, read)


class PythonWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope = Scope('synthetic', 'python-fixture', 'keyboard-fixture',
                           *[digest(value.encode()) for value in ['config', 'protocol', 'evaluator', 'goal', 'safety']])

    def materialize(self, source, *, helpers=None, profile=None, descriptor_bytes=None):
        helpers = helpers or {}
        files = {'execution.json': descriptor_bytes or descriptor(includes=list(helpers)),
                 'profile.json': canonical(profile or {'press': [0, 0, .19], 'release': [0, 0, .2]}),
                 'skill.py': source.encode() if isinstance(source, str) else source, **helpers}
        bundle = dict(protocol=PROTOCOL, label='synthetic-python', kind='seed', scope=asdict(self.scope),
                      files={name: {'sha256': digest(content), 'bytes': len(content)} for name, content in files.items()},
                      provenance={'origin': 'original', 'evidence_mode': 'synthetic'}, parent=None)
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        manifest = canonical(bundle)
        (self.root / 'bundle.json').write_bytes(manifest)
        bundle_id = digest(manifest)
        job = dict(protocol='physical-keyboard-python-worker-job-v1', scope_key=self.scope.key,
                   profile_id=digest(files['profile.json']), permit_id='synthetic-permit', trial_id='synthetic-trial',
                   observation_id='initial-observation', bundle_id=bundle_id)
        (self.root / 'job.json').write_bytes(canonical(job))
        return bundle_id

    def invoke(self, bundle_id, reply=None, on_ready=None):
        worker = JsonProcess([sys.executable, '-m', 'astra_robotics_agent.physical_keyboard.python_worker',
                              '--manifest', str(self.root / 'bundle.json'), '--sha256', bundle_id,
                              '--job', str(self.root / 'job.json')], self.root / 'worker.stderr')
        self.addCleanup(worker.close)
        messages, calls, done = [], [], None
        for _ in range(140):
            message = worker.receive(3)
            messages.append(message)
            if message['type'] == 'ready' and on_ready:
                on_ready(message)
            elif message['type'] == 'tool_call':
                self.assertEqual(message['index'], len(calls))
                calls.append(message)
                result = reply(message) if reply else self.ok(message)
                worker.send({'type': 'tool_result', 'index': message['index'], 'result': result})
            elif message['type'] == 'done':
                done = message
                break
        self.assertIsNotNone(done)
        worker.process.wait(timeout=2)
        exit_report = worker.close()
        self.assertTrue(exit_report['waited'])
        return calls, done, exit_report, messages

    @staticmethod
    def ok(message):
        result = {'status': 'ok', 'evidence_mode': 'synthetic'}
        if message['method'] == 'observe':
            result['observation'] = {'observation_id': 'observation-' + str(message['index']),
                                     'pose': {'xyz_m': [0, 0, .2], 'quaternion_xyzw': [0, 0, 0, 1]}}
        return result

    def assert_preflight_error(self, bundle_id):
        calls, done, exited, _ = self.invoke(bundle_id)
        self.assertEqual(calls, [])
        self.assertEqual(done['status'], 'error')
        self.assertEqual(exited['returncode'], 2)

    def test_helpers_loops_and_real_receipt_feedback(self):
        source = '''def main(robot, profile):
    count = 0
    for target in profile["targets"]:
        state = robot.observe()
        if state["observation"]["pose"]["xyz_m"][2] != target[2]:
            move(robot, target)
            count += 1
    return {"moved": count}
'''
        bundle_id = self.materialize(source,
            helpers={'helpers.py': b'def move(robot, target):\n    return robot.move_tcp(target, [0, 0, 0, 1])\n'},
            profile={'targets': [[0, 0, .19], [0, 0, .18]]})
        calls, done, exited, messages = self.invoke(bundle_id)
        self.assertEqual([call['method'] for call in calls], ['observe', 'move_tcp', 'observe', 'move_tcp'])
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(done['skill_result'], {'moved': 2})
        self.assertEqual(exited['returncode'], 0)
        self.assertEqual(messages[0]['bundle_id'], bundle_id)

    def test_successful_move_invalidates_observation_and_next_move_refreshes(self):
        bundle_id = self.materialize('''def main(robot, profile):
    robot.move_tcp(profile["press"], [0, 0, 0, 1])
    robot.move_tcp(profile["release"], [0, 0, 0, 1])
    return {"program_completed": True}
''')
        calls, done, _, _ = self.invoke(bundle_id)
        self.assertEqual([call['method'] for call in calls], ['move_tcp', 'observe', 'move_tcp'])
        self.assertEqual(calls[0]['arguments']['observation_id'], 'initial-observation')
        self.assertEqual(calls[2]['arguments']['observation_id'], 'observation-1')
        self.assertEqual(done['status'], 'completed')
        self.assertNotIn('observed_text', done)

    def test_rejected_tool_finally_releases_but_cannot_claim_success(self):
        bundle_id = self.materialize('''def main(robot, profile):
    recovered = False
    try:
        robot.move_tcp(profile["press"], [0, 0, 0, 1])
    except ToolError:
        recovered = True
    finally:
        robot.move_tcp(profile["release"], [0, 0, 0, 1])
    return {"success": True, "recovered": recovered}
''')
        def reply(message):
            if message['index'] == 0:
                return {'status': 'rejected', 'reason': 'synthetic_press_limit', 'executed': False}
            return self.ok(message)
        calls, done, exited, _ = self.invoke(bundle_id, reply)
        self.assertEqual([call['method'] for call in calls], ['move_tcp', 'move_tcp'])
        self.assertEqual(done['status'], 'blocked')
        self.assertEqual(done['tool_failure'], 'rejected')
        self.assertTrue(done['skill_result']['success'])
        self.assertEqual(exited['returncode'], 0)

    def test_unknown_cannot_replay_or_mask_failure_and_only_stop_remains(self):
        bundle_id = self.materialize('''def main(robot, profile):
    try:
        robot.move_tcp(profile["press"], [0, 0, 0, 1])
    except ToolError:
        pass
    try:
        robot.move_tcp(profile["release"], [0, 0, 0, 1])
    except ToolError:
        pass
    try:
        robot.wait_bounded(0.01)
    except ToolError:
        pass
    robot.stop()
    return {"success": True}
''')
        def reply(message):
            if message['index'] == 0:
                return {'status': 'unknown', 'reason': 'synthetic_unknown_dispatch', 'executed': None}
            return self.ok(message)
        calls, done, exited, _ = self.invoke(bundle_id, reply)
        self.assertEqual([call['method'] for call in calls], ['move_tcp', 'stop'])
        self.assertEqual(done['status'], 'blocked')
        self.assertEqual(done['tool_failure'], 'unknown')
        self.assertEqual(exited['returncode'], 0)

    def test_import_reflection_and_bad_helper_rejected_before_actions(self):
        variants = ['import os\ndef main(robot, profile):\n    robot.stop()\n',
                    'def main(robot, profile):\n    return robot.__class__\n']
        for index, source in enumerate(variants):
            with self.subTest(index=index):
                bundle_id = self.materialize(source)
                self.assert_preflight_error(bundle_id)
        bundle_id = self.materialize('def main(robot, profile):\n    robot.stop()\n',
                                     helpers={'bad.py': b'def bad():\n    import os\n'})
        self.assert_preflight_error(bundle_id)

    def test_changed_helper_hash_rejected_before_actions(self):
        bundle_id = self.materialize('def main(robot, profile):\n    robot.stop()\n',
                                     helpers={'helper.py': b'def helper():\n    return 1\n'})
        (self.root / 'helper.py').write_bytes(b'def helper():\n    return 2\n')
        self.assert_preflight_error(bundle_id)

    def test_wrong_manifest_hash_and_job_profile_scope_are_rejected(self):
        bundle_id = self.materialize('def main(robot, profile):\n    robot.stop()\n')
        self.assert_preflight_error('0' * 64)
        original = (self.root / 'job.json').read_bytes()
        for field in ['scope_key', 'profile_id', 'bundle_id']:
            with self.subTest(field=field):
                job = json.loads(original)
                job[field] = '0' * 64
                (self.root / 'job.json').write_bytes(canonical(job))
                self.assert_preflight_error(bundle_id)

    def test_symlink_source_and_descriptor_path_escape_are_rejected(self):
        bundle_id = self.materialize('def main(robot, profile):\n    robot.stop()\n',
                                     helpers={'helper.py': b'def helper():\n    return 1\n'})
        external = self.root / 'outside.txt'
        external.write_bytes((self.root / 'helper.py').read_bytes())
        (self.root / 'helper.py').unlink()
        (self.root / 'helper.py').symlink_to(external)
        self.assert_preflight_error(bundle_id)
        (self.root / 'helper.py').unlink()
        bundle_id = self.materialize('def main(robot, profile):\n    robot.stop()\n',
                                     descriptor_bytes=descriptor('../outside.py'))
        self.assert_preflight_error(bundle_id)

    def test_malformed_tool_response_is_fatal_and_not_retried(self):
        bundle_id = self.materialize('''def main(robot, profile):
    try:
        robot.observe()
    except ToolError:
        robot.observe()
    return {"success": True}
''')
        calls, done, exited, _ = self.invoke(bundle_id, lambda _: {'status': 'invalid'})
        self.assertEqual(len(calls), 1)
        self.assertEqual(done['status'], 'error')
        self.assertEqual(exited['returncode'], 2)

    def test_unbounded_tool_loop_stops_without_over_budget_cleanup_action(self):
        bundle_id = self.materialize('''def main(robot, profile):
    try:
        while True:
            robot.observe()
    finally:
        robot.stop()
''')
        calls, done, exited, _ = self.invoke(bundle_id)
        self.assertLessEqual(len(calls), 64)
        self.assertEqual(done['status'], 'error')
        self.assertEqual(exited['returncode'], 2)


if __name__ == '__main__':
    unittest.main()
