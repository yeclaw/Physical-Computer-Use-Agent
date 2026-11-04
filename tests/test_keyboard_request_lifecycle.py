"""Request lifecycle/EOF cancellation with local threads and pipes; no SDK."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from astra_robotics_agent.physical_keyboard.runtime.request_lifecycle import (
    MAX_LINE_BYTES, PROTOCOL, ProtocolEmitter, RequestInput, RequestInputError,
    RequestLifecycle, descriptor,
)


def frame(method='state', identifier='request-1'):
    return (json.dumps(dict(id=identifier, method=method, args={})) + '\n').encode()


class ProtocolEmitterTests(unittest.TestCase):
    def test_large_partial_frames_and_real_audit_remain_whole_across_threads(self):
        from astra_robotics_agent.physical_keyboard.runtime.control_protocol import Audit, encoded
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.jsonl'
            audit = Audit(path, 'emitter-no-sdk')
            chunks = []
            count = 4
            barrier = threading.Barrier(count)
            failures = []
            def write(data):
                # Force many GIL-release opportunities between partial writes:
                # only the production emitter's complete-frame lock can keep
                # heartbeat-sized and large result frames from interleaving.
                n = min(997, len(data))
                chunks.append(data[:n])
                time.sleep(.00001)
                return n
            emitter = ProtocolEmitter(write, encoded,
                audit=lambda value: audit('protocol_response', response=value))
            expected = {}
            for worker in range(count):
                for index in range(4):
                    key = f'{worker}-{index}'
                    expected[key] = dict(id=key, payload='大' * (22000 if index % 2 else 11))
            def producer(worker):
                try:
                    barrier.wait(timeout=2)
                    for index in range(4):
                        emitter(expected[f'{worker}-{index}'])
                except BaseException as exc:
                    failures.append(exc)
            workers = [threading.Thread(target=producer, args=(i,), daemon=True) for i in range(count)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=5)
                self.assertFalse(any(worker.is_alive() for worker in workers), 'framing/audit deadlock')
                self.assertEqual(failures, [])
            finally:
                audit.close()
            lines = b''.join(chunks).splitlines()
            self.assertEqual(len(lines), len(expected))
            decoded = []
            for line in lines:
                self.assertTrue(line.startswith(b'@@PCU@@'))
                decoded.append(json.loads(line[len(b'@@PCU@@'):]))
            self.assertEqual({value['id']: value for value in decoded}, expected)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), len(expected))
            self.assertEqual([row['response'] for row in rows], decoded)

    def test_write_and_audit_failures_preserve_original_exception(self):
        original = BrokenPipeError('physical protocol pipe closed')
        def write(_data):
            raise original
        emitter = ProtocolEmitter(write, lambda _value: b'{}')
        with self.assertRaises(BrokenPipeError) as raised:
            emitter({})
        self.assertIs(raised.exception, original)
        original = OSError('audit disk unavailable')
        written = []
        def audit(_value):
            raise original
        emitter = ProtocolEmitter(lambda data: written.append(data) or len(data),
                                   lambda _value: b'{}', audit=audit)
        with self.assertRaises(OSError) as raised:
            emitter({})
        self.assertIs(raised.exception, original)
        self.assertEqual(written, [])

    def test_invalid_byte_counts_fail_without_infinite_write_loop(self):
        for returned in (0, -1, True, None, 3.5, 999):
            writes = []
            def write(data):
                writes.append(data)
                return returned
            with self.subTest(returned=returned):
                emitter = ProtocolEmitter(write, lambda _value: b'{}')
                with self.assertRaises(OSError):
                    emitter({})
                self.assertEqual(len(writes), 1)

    def test_encoder_error_prevents_audit_and_write_and_reentry_is_supported(self):
        original = ValueError('unencodable value')
        calls = []
        def encode(_value):
            raise original
        emitter = ProtocolEmitter(lambda _data: calls.append('write'), encode,
                                   audit=lambda _value: calls.append('audit'))
        with self.assertRaises(ValueError) as raised:
            emitter({})
        self.assertIs(raised.exception, original)
        self.assertEqual(calls, [])
        chunks = []
        def audit(value):
            if value['id'] == 'outer':
                emitter({'id': 'inner'})
        emitter = ProtocolEmitter(lambda data: chunks.append(data) or len(data),
            lambda value: json.dumps(value).encode(), audit=audit)
        emitter({'id': 'outer'})
        self.assertEqual([json.loads(line[len(b'@@PCU@@'):])['id']
                          for line in b''.join(chunks).splitlines()], ['inner', 'outer'])


class LifecycleTests(unittest.TestCase):
    def test_descriptor_has_no_total_deadline_and_separates_liveness(self):
        self.assertEqual(descriptor(), dict(protocol=PROTOCOL, heartbeat_interval_s=5.0,
            total_duration_limit_s=None,
            heartbeat_meaning='process_liveness_not_physical_progress',
            disconnect_cancels=True))
        d = descriptor()
        d['protocol'] = 'changed'
        self.assertEqual(descriptor()['protocol'], PROTOCOL)

    def test_accepted_sync_then_monotonic_heartbeat_without_invented_progress(self):
        rows = []
        heartbeat = threading.Event()
        def emit(row):
            rows.append(row)
            if row['phase'] == 'running':
                heartbeat.set()
        with RequestLifecycle('run-1', emit, interval_s=.005):
            self.assertEqual(rows[0]['phase'], 'accepted')
            self.assertTrue(heartbeat.wait(1))
        count = len(rows)
        time.sleep(.015)
        self.assertEqual(len(rows), count)
        self.assertEqual([r['sequence'] for r in rows], list(range(1, count + 1)))
        for row in rows:
            self.assertEqual(row['id'], 'run-1')
            self.assertEqual(row['protocol'], PROTOCOL)
            self.assertEqual(row['type'], 'request_update')
            self.assertEqual(row['progress_seq'], 0)
            self.assertIsNone(row['progress_kind'])
            self.assertIsNone(row['progress_anchor'])

    def test_only_execution_events_change_progress_and_fields_cannot_forge_time(self):
        rows = []
        progressed = threading.Event()
        def emit(row):
            rows.append(row)
            if row['progress_seq'] == 2:
                progressed.set()
        with RequestLifecycle('run-1', emit, interval_s=.005) as lifecycle:
            self.assertFalse(lifecycle.note('request_heartbeat', {}))
            self.assertTrue(lifecycle.note('tool_started', {'utc_ns': 1}))
            self.assertTrue(lifecycle.note('trajectory_point_dispatched', {'index': 7}))
            self.assertTrue(progressed.wait(1))
        final = rows[-1]
        self.assertEqual(final['progress_seq'], 2)
        self.assertEqual(final['progress_kind'], 'trajectory_point_dispatched')
        self.assertGreater(final['progress_anchor']['utc_ns'], 1)
        self.assertGreater(final['progress_anchor']['monotonic_ns'], 0)

    def test_emitter_can_reenter_note_without_state_lock_deadlock(self):
        lifecycle = None
        completed = threading.Event()
        def emit(row):
            if row['phase'] == 'running':
                lifecycle.note('tool_ended', {})
                completed.set()
        with RequestLifecycle('run-1', emit, interval_s=.005) as value:
            lifecycle = value
            self.assertTrue(completed.wait(1))

    def test_synchronous_emit_failure_keeps_original_exception(self):
        failure = BrokenPipeError('accepted failed')
        def emit(_row):
            raise failure
        lifecycle = RequestLifecycle('run-1', emit)
        with self.assertRaises(BrokenPipeError) as raised:
            with lifecycle:
                self.fail('execution must not start')
        self.assertIs(raised.exception, failure)
        self.assertIs(lifecycle.error, failure)
        self.assertIsNone(lifecycle._thread)

    def test_background_emit_failure_reaches_execution_thread_on_exit(self):
        failure = BrokenPipeError('running failed')
        attempted = threading.Event()
        def emit(row):
            if row['phase'] == 'running':
                attempted.set()
                raise failure
        lifecycle = RequestLifecycle('run-1', emit, interval_s=.005)
        with self.assertRaises(BrokenPipeError) as raised:
            with lifecycle:
                self.assertTrue(attempted.wait(1))
        self.assertIs(raised.exception, failure)
        self.assertFalse(lifecycle._thread.is_alive())

    def test_original_main_thread_failure_not_replaced_by_emit_failure(self):
        original = RuntimeError('SDK original')
        attempted = threading.Event()
        def emit(row):
            if row['phase'] == 'running':
                attempted.set()
                raise BrokenPipeError('also unavailable')
        lifecycle = RequestLifecycle('run-1', emit, interval_s=.005)
        with self.assertRaises(RuntimeError) as raised:
            with lifecycle:
                self.assertTrue(attempted.wait(1))
                raise original
        self.assertIs(raised.exception, original)
        self.assertIsInstance(lifecycle.error, BrokenPipeError)

    def test_blocked_writer_does_not_deadlock_main_thread_cleanup(self):
        started = threading.Event()
        release = threading.Event()
        def emit(row):
            if row['phase'] == 'running':
                started.set()
                release.wait(5)
        lifecycle = RequestLifecycle('run-1', emit, interval_s=.005)
        try:
            with self.assertRaisesRegex(RuntimeError, 'writer did not stop'):
                with lifecycle:
                    self.assertTrue(started.wait(1))
        finally:
            release.set()
            lifecycle._thread.join(timeout=1)
        self.assertFalse(lifecycle._thread.is_alive())

    def test_invalid_config_and_reentry_fail_before_execution(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(interval=value), self.assertRaises(ValueError):
                RequestLifecycle('run-1', lambda _x: None, interval_s=value)
        for value in ('', 'a' * 129, None, 4):
            with self.subTest(identifier=value), self.assertRaises(ValueError):
                RequestLifecycle(value, lambda _x: None)
        lifecycle = RequestLifecycle('run-1', lambda _x: None)
        with lifecycle:
            pass
        with self.assertRaises(RuntimeError):
            with lifecycle:
                pass


class InputTests(unittest.TestCase):
    def pipe(self, callback=None):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, 'rb', buffering=0)
        writer = os.fdopen(write_fd, 'wb', buffering=0)
        source = RequestInput(stream, callback)
        def cleanup():
            source.close()
            writer.close()
            source._thread.join(timeout=1)
            stream.close()
        self.addCleanup(cleanup)
        return source, writer

    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertTrue(predicate())

    def test_normal_multiple_request_roundtrips_and_idle_eof(self):
        called = []
        source, writer = self.pipe(lambda: called.append('cancel'))
        for index in range(3):
            line = frame(identifier='run-' + str(index))
            writer.write(line)
            self.assertEqual(source.readline(), line)
            source.begin()
            source.check()
            source.end()
        writer.close()
        self.assertEqual(source.readline(), b'')
        self.assertEqual(called, [])

    def test_eof_after_frame_before_begin_prevents_dispatch(self):
        called = []
        source, writer = self.pipe(lambda: called.append('cancel'))
        writer.write(frame())
        self.assertEqual(source.readline(), frame())
        writer.close()
        self.wait_for(lambda: source._eof)
        with self.assertRaisesRegex(RequestInputError, 'before execution'):
            source.begin()
        self.assertEqual(called, [])

    def test_eof_during_execution_cancels_once_and_delivers_no_new_motion(self):
        called = threading.Event()
        source, writer = self.pipe(called.set)
        writer.write(frame())
        source.readline()
        source.begin()
        writer.close()
        self.assertTrue(called.wait(1))
        with self.assertRaisesRegex(RequestInputError, 'EOF'):
            source.check()
        source.end()
        with self.assertRaises(RequestInputError):
            source.readline()

    def test_close_during_execution_cancels_instead_of_queueing(self):
        called = threading.Event()
        source, writer = self.pipe(called.set)
        writer.write(frame('run_skill'))
        source.readline()
        source.begin()
        writer.write(frame('close', 'close-2'))
        self.assertTrue(called.wait(1))
        with self.assertRaisesRegex(RequestInputError, 'close requested'):
            source.readline()
        source.end()

    def test_concurrent_motion_is_rejected_and_not_queued(self):
        called = threading.Event()
        source, writer = self.pipe(called.set)
        writer.write(frame('run_skill'))
        source.readline()
        source.begin()
        writer.write(frame('run_skill', 'run-2'))
        self.assertTrue(called.wait(1))
        self.assertFalse(source._lines)
        with self.assertRaisesRegex(RequestInputError, 'concurrent'):
            source.check()
        source.end()

    def test_second_frame_before_begin_rejects_all_without_callback(self):
        called = []
        source, writer = self.pipe(lambda: called.append('cancel'))
        writer.write(frame('run_skill'))
        source.readline()
        writer.write(frame('run_skill', 'run-2'))
        self.wait_for(lambda: source.error is not None)
        with self.assertRaisesRegex(RequestInputError, 'concurrent'):
            source.begin()
        self.assertEqual(called, [])

    def test_cancel_callback_is_outside_input_lock_and_failure_is_retained(self):
        observed = threading.Event()
        source, writer = self.pipe()
        failure = OSError('signal unavailable')
        def callback():
            source.end()
            observed.set()
            raise failure
        source.cancel_callback = callback
        writer.write(frame('run_skill'))
        source.readline()
        source.begin()
        writer.close()
        self.assertTrue(observed.wait(1))
        self.wait_for(lambda: source.cancel_error is not None)
        self.assertIs(source.cancel_error, failure)
        with self.assertRaises(RequestInputError):
            source.check()

    def test_partial_frame_and_oversize_are_rejected_before_dispatch(self):
        for line in (b'{"id":"partial"}', b'x' * (MAX_LINE_BYTES + 1) + b'\n'):
            with self.subTest(size=len(line)):
                class Stream:
                    def readline(self, limit):
                        self.limit = limit
                        return line
                stream = Stream()
                source = RequestInput(stream)
                source._thread.join(timeout=1)
                self.assertEqual(stream.limit, MAX_LINE_BYTES + 2)
                with self.assertRaisesRegex(RequestInputError, 'frame'):
                    source.readline()
                source.close()

    def test_stream_read_failure_is_preserved(self):
        failure = OSError('input failed')
        class Stream:
            def readline(self, _limit):
                raise failure
        source = RequestInput(Stream())
        source._thread.join(timeout=1)
        with self.assertRaises(OSError) as raised:
            source.readline()
        self.assertIs(raised.exception, failure)
        source.close()

    def test_begin_needs_delivered_frame_and_local_close_does_not_own_stream(self):
        source, writer = self.pipe()
        with self.assertRaises(RequestInputError):
            source.begin()
        source.close()
        self.assertFalse(source.stream.closed)
        self.assertEqual(source.readline(), b'')

    def test_real_buffered_input_close_joins_without_closing_or_reading_python_buffer(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, 'rb')
        writer = os.fdopen(write_fd, 'wb', buffering=0)
        class DescriptorOnly:
            def fileno(self):
                return stream.fileno()
            def readline(self, _limit):
                raise AssertionError('must not acquire buffered stdin read lock')
        source = RequestInput(DescriptorOnly())
        try:
            writer.write(frame())
            self.assertEqual(source.readline(), frame())
            source.begin()
            source.end()
            started = time.monotonic()
            source.close()
            self.assertLess(time.monotonic() - started, .3)
            self.assertFalse(source._thread.is_alive())
            self.assertFalse(stream.closed)
            self.assertFalse(source._eof)
        finally:
            source.close()
            writer.close()
            stream.close()

    def test_split_os_frame_and_idle_reader_close_preserve_byte_boundaries(self):
        source, writer = self.pipe()
        line = frame()
        writer.write(line[:9])
        writer.write(line[9:])
        self.assertEqual(source.readline(), line)
        source.begin()
        source.end()
        source.close()
        self.assertFalse(source._thread.is_alive())

    def test_fd_oversize_without_newline_is_bounded_and_cancels(self):
        source, writer = self.pipe()
        # Sending from a thread avoids blocking the test runner on pipe capacity.
        def send():
            try:
                writer.write(b'x' * (MAX_LINE_BYTES + 2))
            except (OSError, ValueError):
                pass
        producer = threading.Thread(target=send, daemon=True)
        producer.start()
        try:
            with self.assertRaisesRegex(RequestInputError, 'frame'):
                source.readline()
            producer.join(timeout=1)
            self.assertFalse(producer.is_alive())
            self.assertLessEqual(len(source._read_buffer), MAX_LINE_BYTES + 2)
        finally:
            source.close()


if __name__ == '__main__':
    unittest.main()
