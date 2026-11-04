"""Real local pipe/worker checks, without hardware, network, models, or Python skills."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from astra_robotics_agent.physical_keyboard.transport import (
    MAX_LINE_BYTES, JsonProcess, ProtocolError, TransportError, encode_line, strict_loads)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.children = []

    def child(self, code):
        child = JsonProcess([sys.executable, '-c', code], self.root / f'child-{len(self.children)}.stderr')
        self.children.append(child)
        self.addCleanup(child.close)
        return child

    def test_strict_json_rejects_ambiguous_and_nonfinite_values(self):
        for content in ['{"x":1,"x":2}', '{"x":{"y":1,"y":2}}',
                        '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
                        '{"x":1e999}', '[]', 'null', '{}{}', b'{"x":"\xff"}',
                        '{"x":"\\ud800"}', 'x' * (MAX_LINE_BYTES + 1)]:
            with self.subTest(content=str(content)[:80]), self.assertRaises(ProtocolError):
                strict_loads(content)
        self.assertEqual(strict_loads('{"ok":"你好","n":1.25}'), {'ok': '你好', 'n': 1.25})

    def test_encode_rejects_invalid_or_oversized_messages(self):
        for value in [[], {'x': float('nan')}, {'x': float('inf')}, {1: 'coerced'},
                      {'x': '\ud800'}, {'x': 'a' * MAX_LINE_BYTES}]:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ProtocolError):
                encode_line(value)
        self.assertEqual(encode_line({'x': 1}), b'{"x":1}\n')

    def test_frame_boundary_always_reserves_newline(self):
        value = {'x': 'a' * (MAX_LINE_BYTES - 9)}
        self.assertEqual(len(encode_line(value)), MAX_LINE_BYTES)
        self.assertEqual(strict_loads(encode_line(value)), value)
        self.assertEqual(strict_loads(encode_line(value)[:-1]), value)
        with self.assertRaises(ProtocolError):
            encode_line({'x': value['x'] + 'a'})
        with self.assertRaises(ProtocolError):
            strict_loads(encode_line(value)[:-1] + b' ')

    def test_partial_line_timeout_preserves_bytes(self):
        child = self.child("import sys,time;sys.stdout.write('{\"x\":');sys.stdout.flush();"
                           "time.sleep(.25);sys.stdout.write('1}\\n');sys.stdout.flush()")
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            child.receive(0.05)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(child.receive(2), {'x': 1})

    def test_multiple_buffered_messages_and_clean_eof(self):
        child = self.child("import sys;sys.stdout.write('{\"x\":1}\\n{\"x\":2}\\n');sys.stdout.flush()")
        self.assertEqual(child.receive(2), {'x': 1})
        self.assertEqual(child.receive(2), {'x': 2})
        with self.assertRaises(EOFError):
            child.receive(2)

    def test_partial_eof_and_oversized_line_fail_closed(self):
        for code in ["import sys;sys.stdout.write('{\"x\":1}');sys.stdout.flush()",
                     "import sys;sys.stdout.write('x'*65537);sys.stdout.flush()"]:
            with self.subTest(code=code):
                child = self.child(code)
                with self.assertRaises(ProtocolError):
                    child.receive(2)
                with self.assertRaises(ProtocolError):
                    child.receive(0)

    def test_invalid_wire_lines_fail_closed(self):
        for content in ['{"x":NaN}\n', '{"x":1,"x":2}\n', '[]\n']:
            with self.subTest(content=content):
                child = self.child('import sys;sys.stdout.write(' + repr(content) + ');sys.stdout.flush()')
                with self.assertRaises(ProtocolError):
                    child.receive(2)

    def test_send_echo_and_stderr_retained(self):
        child = self.child("import sys;sys.stderr.write('retained\\n');sys.stderr.flush();"
                           "sys.stdout.buffer.write(sys.stdin.buffer.readline());sys.stdout.flush()")
        child.send({'unicode': '英文', 'nested': {'x': 2}})
        self.assertEqual(child.receive(2), {'unicode': '英文', 'nested': {'x': 2}})
        child.process.wait(timeout=2)
        report = child.close()
        self.assertEqual(report['returncode'], 0)
        self.assertTrue(report['waited'])
        self.assertEqual(child.stderr_path.read_text(), 'retained\n')
        self.assertEqual(report, child.close())
        with self.assertRaises(TransportError):
            child.send({})

    def test_serialization_failure_keeps_connection_but_send_timeout_breaks_it(self):
        echo = self.child("import sys;sys.stdout.buffer.write(sys.stdin.buffer.readline());sys.stdout.flush()")
        with self.assertRaises(ProtocolError):
            echo.send({'x': float('nan')}, timeout_s=0.2)
        echo.send({'still': 'usable'}, timeout_s=0.2)
        self.assertEqual(echo.receive(2), {'still': 'usable'})
        stalled = self.child("import time;print('{\"ready\":true}',flush=True);time.sleep(60)")
        self.assertEqual(stalled.receive(2), {'ready': True})
        timed_out = False
        started = time.monotonic()
        for _ in range(100):
            try:
                stalled.send({'x': 'a' * 16000}, timeout_s=0.03)
            except TimeoutError:
                timed_out = True
                break
        self.assertTrue(timed_out, 'Unread pipe must eventually reject a bounded send')
        self.assertLess(time.monotonic() - started, 4)
        with self.assertRaises(ProtocolError):
            stalled.send({}, timeout_s=0.1)

    def test_cleanup_escalates_and_waits_owned_process(self):
        child = self.child("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                           "print('{\"ready\":true}',flush=True);time.sleep(60)")
        self.assertEqual(child.receive(2), {'ready': True})
        started = time.monotonic()
        report = child.close()
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(report['terminate_sent'])
        self.assertTrue(report['kill_sent'])
        self.assertTrue(report['waited'])
        self.assertIsNotNone(child.process.poll())
        self.assertEqual(report['cleanup_errors'], [])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0
        self.job = {'protocol': 'physical-keyboard-worker-job-v1', 'scope_key': '0' * 64,
                    'profile_id': 'profile-1', 'permit_id': 'permit-1', 'trial_id': 'trial-1',
                    'observation_id': None}

    def launch(self, actions, *, job=None, expected_sha=None, raw_program=None):
        self.count += 1
        prefix = self.root / str(self.count)
        prefix.mkdir()
        content = (json.dumps({'protocol': 'physical-keyboard-action-plan-v1', 'actions': actions}).encode()
                   if raw_program is None else raw_program)
        program = prefix / 'program.json'
        program.write_bytes(content)
        job_path = prefix / 'job.json'
        job_path.write_text(json.dumps(self.job if job is None else job))
        program_sha = hashlib.sha256(content).hexdigest()
        child = JsonProcess([sys.executable, '-m', 'astra_robotics_agent.physical_keyboard.worker',
                             '--program', str(program), '--sha256', expected_sha or program_sha,
                             '--job', str(job_path)], prefix / 'worker.stderr')
        self.addCleanup(child.close)
        return child, program_sha

    @staticmethod
    def action(method, **arguments):
        return {'method': method, 'arguments': arguments}

    def assert_error(self, child):
        message = child.receive(2)
        self.assertEqual(message['type'], 'done')
        self.assertEqual(message['status'], 'error')
        self.assertIsInstance(message['error'], str)
        self.assertEqual(child.process.wait(timeout=2), 2)

    def test_valid_real_worker_flow(self):
        actions = [self.action('observe'), self.action('move_tcp', xyz_m=[0, 0.1, 0.2],
                                                    quaternion_xyzw=[0, 0, 0, 1]),
                   self.action('wait_bounded', seconds=0.01), self.action('stop')]
        child, program_sha = self.launch(actions)
        self.assertEqual(child.receive(2), {'type': 'ready', 'program_sha256': program_sha})
        for index, action in enumerate(actions):
            message = child.receive(2)
            self.assertEqual(message['type'], 'tool_call')
            self.assertEqual((message['index'], message['method']), (index, action['method']))
            arguments = dict(action['arguments'])
            if action['method'] == 'move_tcp':
                arguments['observation_id'] = 'observation-1'
            self.assertEqual(message['arguments'], arguments)
            result = {'status': 'ok'}
            if action['method'] == 'observe':
                result['observation'] = {'observation_id': 'observation-1'}
            child.send({'type': 'tool_result', 'index': index, 'result': result})
        self.assertEqual(child.receive(2), {'type': 'done', 'status': 'completed'})
        self.assertEqual(child.process.wait(timeout=2), 0)

    def test_move_consumes_observation_and_missing_is_explicit(self):
        move = self.action('move_tcp', xyz_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1])
        child, _ = self.launch([move, move], job={**self.job, 'observation_id': 'initial-observation'})
        self.assertEqual(child.receive(2)['type'], 'ready')
        for index, observation in [(0, 'initial-observation'), (1, None)]:
            call = child.receive(2)
            self.assertEqual(call['arguments']['observation_id'], observation)
            child.send({'type': 'tool_result', 'index': index, 'result': {'status': 'ok'}})
        self.assertEqual(child.receive(2)['status'], 'completed')

    def test_non_ok_receipt_stops_without_next_action(self):
        child, _ = self.launch([self.action('observe'), self.action('stop')])
        self.assertEqual(child.receive(2)['type'], 'ready')
        self.assertEqual(child.receive(2)['method'], 'observe')
        child.send({'type': 'tool_result', 'index': 0, 'result': {'status': 'unknown'}})
        self.assertEqual(child.receive(2), {'type': 'done', 'status': 'blocked'})
        self.assertEqual(child.process.wait(timeout=2), 0)

    def test_invalid_later_action_rejected_before_any_tool(self):
        invalid = [self.action('observe'), {'method': 'grant_trial', 'arguments': {}}]
        child, _ = self.launch(invalid)
        self.assert_error(child)

    def test_schema_vectors_authority_and_bounds_rejected_upfront(self):
        invalid = [self.action('observe', permit_id='forged'),
                   self.action('move_tcp', xyz_m=[True, 0, 0], quaternion_xyzw=[0, 0, 0, 1]),
                   self.action('move_tcp', xyz_m=[0, 0], quaternion_xyzw=[0, 0, 0, 1]),
                   self.action('move_tcp', xyz_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 0]),
                   self.action('move_tcp', xyz_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1], observation_id='forged'),
                   self.action('wait_bounded', seconds=-1), self.action('wait_bounded', seconds=True)]
        for action in invalid:
            with self.subTest(action=action):
                child, _ = self.launch([action])
                self.assert_error(child)
        child, _ = self.launch([self.action('stop')] * 65)
        self.assert_error(child)

    def test_program_digest_and_non_json_python_rejected_without_execution(self):
        child, _ = self.launch([], expected_sha='f' * 64)
        self.assert_error(child)
        sentinel = self.root / 'must-not-exist'
        content = ('open(' + repr(str(sentinel)) + ',"w").write("executed")').encode()
        child, _ = self.launch([], raw_program=content)
        self.assert_error(child)
        self.assertFalse(sentinel.exists())

    def test_job_schema_rejects_extra_authority_and_bad_protocol(self):
        for job in [{**self.job, 'ledger_path': '/not-authorized'},
                    {**self.job, 'protocol': 'wrong'}, {**self.job, 'scope_key': 'not-a-hash'}]:
            with self.subTest(job=job):
                child, _ = self.launch([], job=job)
                self.assert_error(child)

    def test_response_schema_mismatch_and_missing_observation_rejected(self):
        for response in [
            {'type': 'grant_trial', 'index': 0, 'result': {'status': 'ok'}},
            {'type': 'tool_result', 'index': True, 'result': {'status': 'ok'}},
            {'type': 'tool_result', 'index': 1, 'result': {'status': 'ok'}},
            {'type': 'tool_result', 'index': 0, 'result': {'status': 'ok'}, 'extra': True},
            {'type': 'tool_result', 'index': 0, 'result': []},
            {'type': 'tool_result', 'index': 0, 'result': {}},
            {'type': 'tool_result', 'index': 0, 'result': {'status': 'ok'}},
        ]:
            with self.subTest(response=response):
                child, _ = self.launch([self.action('observe')])
                self.assertEqual(child.receive(2)['type'], 'ready')
                self.assertEqual(child.receive(2)['type'], 'tool_call')
                child.send(response)
                self.assert_error(child)


if __name__ == '__main__':
    unittest.main()
