"""Non-terminal request updates and input cancellation; no SDK access.

Heartbeats describe transport/process liveness. Progress only records execution
events provided by the owning thread and never attests physical/task success.
There is no total execution-duration limit in this protocol.
"""
from collections import deque
import json
import math
import os
import select
import threading
import time


PROTOCOL = 'physical-keyboard-request-lifecycle-v1'
MAX_LINE_BYTES = 1024 * 1024
_PROGRESS_EVENTS = frozenset({
    'tool_started', 'tool_ended', 'trajectory_point_dispatched',
    'native_timed_prepared', 'native_timed_dispatched',
    'trajectory_started', 'trajectory_completed', 'trajectory_asset_frozen',
    'skill_frozen', 'skill_ended', 'recording_guard',
})


def descriptor():
    return dict(protocol=PROTOCOL, heartbeat_interval_s=5.0,
                total_duration_limit_s=None,
                heartbeat_meaning='process_liveness_not_physical_progress',
                disconnect_cancels=True)


def idle_heartbeat_descriptor():
    """Application-channel liveness only; no SDK, sensor or task observation."""
    return dict(protocol='physical-keyboard-idle-heartbeat-v1', method='ping',
                motion_requested=False, sdk_state_queried=False)


class ProtocolEmitter:
    """Serialize one complete audited frame across all partial OS writes.

    Both execution and lifecycle threads use this same instance. The injected
    callbacks make the production framing/audit path testable without hardware.
    """
    def __init__(self, write, encode, audit=None, prefix='@@PCU@@'):
        if not callable(write) or not callable(encode):
            raise ValueError('callable byte writer and encoder required')
        if audit is not None and not callable(audit):
            raise ValueError('callable audit hook required')
        if type(prefix) is not str or '\n' in prefix or '\r' in prefix:
            raise ValueError('protocol prefix must be a single-line string')
        self.write, self.encode, self.audit = write, encode, audit
        self._prefix = prefix.encode('utf-8')
        self._lock = threading.RLock()

    def __call__(self, value):
        with self._lock:
            payload = self.encode(value)
            if type(payload) is not bytes:
                raise TypeError('protocol encoder must return bytes')
            data = self._prefix + payload + b'\n'
            if self.audit is not None:
                self.audit(value)
            offset = 0
            while offset < len(data):
                written = self.write(data[offset:])
                if type(written) is not int or not 0 < written <= len(data) - offset:
                    raise OSError('protocol writer returned invalid byte count')
                offset += written


class RequestLifecycle:
    """Emit accepted/running updates around one synchronous request.

    The caller owns final-result emission, after this context has exited. The
    emitter must serialize complete frames with other protocol writers. Never
    hold our state lock across emit: an emitter may itself acquire an audit
    lock which invokes note().
    """
    def __init__(self, request_id, emit, interval_s=5):
        if type(request_id) is not str or not 0 < len(request_id) <= 128:
            raise ValueError('nonempty request id <=128 characters required')
        if (type(interval_s) not in (int, float)
                or not math.isfinite(interval_s) or interval_s <= 0):
            raise ValueError('positive finite heartbeat interval required')
        if not callable(emit):
            raise ValueError('callable request update emitter required')
        self.request_id, self.emit, self.interval_s = request_id, emit, interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._entered = False
        self._closed = False
        self._sequence = 0
        self._progress_seq = 0
        self._progress_kind = None
        self._progress_anchor = None
        self.error = None

    def _update(self, phase):
        with self._lock:
            self._sequence += 1
            message = dict(type='request_update', protocol=PROTOCOL,
                id=self.request_id, sequence=self._sequence, phase=phase,
                progress_seq=self._progress_seq,
                progress_kind=self._progress_kind,
                progress_anchor=(None if self._progress_anchor is None
                                 else dict(self._progress_anchor)))
        # In particular, do not move this call inside the state lock.
        try:
            self.emit(message)
        except BaseException as exc:
            with self._lock:
                if self.error is None:
                    self.error = exc
            self._stop.set()
            raise

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                self._update('running')
            except BaseException:
                # The original error is re-raised by check()/context exit on
                # the execution thread, never treated as a healthy heartbeat.
                return

    def check(self):
        with self._lock:
            error = self.error
        if error is not None:
            raise error

    def note(self, kind, fields=None):
        """Record trusted execution-event progress, ignoring unrelated audit."""
        self.check()
        if kind not in _PROGRESS_EVENTS:
            return False
        with self._lock:
            if self._closed:
                return False
            self._progress_seq += 1
            self._progress_kind = kind
            # This is the event observation time; fields cannot forge it or
            # smuggle model-selected data into a progress/liveness attestation.
            self._progress_anchor = dict(utc_ns=time.time_ns(),
                                         monotonic_ns=time.monotonic_ns())
        return True

    def __enter__(self):
        if self._entered:
            raise RuntimeError('request lifecycle cannot be reused')
        self._entered = True
        self._update('accepted')
        self._thread = threading.Thread(target=self._run, daemon=True,
                                       name='request-lifecycle-' + self.request_id)
        self._thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._stop.set()
        if self._thread is not None:
            # A blocked protocol writer must not lock execution-thread cleanup
            # forever. If it cannot join, no normal terminal may be emitted.
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                with self._lock:
                    if self.error is None:
                        self.error = RuntimeError('request update writer did not stop')
        with self._lock:
            self._closed = True
        if exception_type is None:
            self.check()
        return False


class RequestInputError(ValueError):
    """No further command may be dispatched from this input stream."""


class RequestInput:
    """Read one bounded frame ahead while execution remains on the main thread.

    EOF or a second command while active requests cancellation through the
    injected callback. It may signal the owning process; it must not touch SDK
    state from this reader thread. begin() is the final pre-dispatch EOF gate.
    The stream remains caller-owned and is never closed here.
    """
    def __init__(self, stream, cancel_callback=None):
        if not callable(getattr(stream, 'readline', None)):
            raise ValueError('readline-capable binary input required')
        if cancel_callback is not None and not callable(cancel_callback):
            raise ValueError('callable cancellation hook required')
        self.stream, self.cancel_callback = stream, cancel_callback
        self._condition = threading.Condition()
        self._lines = deque()
        self._queued = False
        self._delivered = False
        self._active = False
        self._eof = False
        self._closed = False
        self._cancelled = False
        self.error = None
        self.cancel_error = None
        self._read_buffer = bytearray()
        try:
            descriptor_number = stream.fileno()
            self._fd = descriptor_number if type(descriptor_number) is int and descriptor_number >= 0 else None
        except (AttributeError, OSError, ValueError):
            self._fd = None
        self._thread = threading.Thread(target=self._read, daemon=True,
                                        name='request-input')
        self._thread.start()

    def _fail(self, error, *, eof=False):
        with self._condition:
            if self.error is None:
                self.error = error
            if eof:
                self._eof = True
            should_cancel = self._active and not self._cancelled
            self._cancelled = True
            self._lines.clear()
            self._condition.notify_all()
        # A signal may immediately interrupt a main thread currently in begin,
        # end, or check: never invoke the hook while holding our state lock.
        if should_cancel and self.cancel_callback is not None:
            try:
                self.cancel_callback()
            except BaseException as exc:
                with self._condition:
                    self.cancel_error = exc
                    self._condition.notify_all()

    def _readline(self):
        if self._fd is None:
            # Injectable in-memory streams have no OS descriptor. Their caller
            # owns cancellation/unblocking; real stdin always uses the path
            # below and never holds a Python buffered-reader lock at shutdown.
            return self.stream.readline(MAX_LINE_BYTES + 2)
        limit = MAX_LINE_BYTES + 2
        while True:
            with self._condition:
                if self._closed:
                    return b''
            newline = self._read_buffer.find(b'\n')
            if newline >= 0:
                line = bytes(self._read_buffer[:newline + 1])
                del self._read_buffer[:newline + 1]
                return line
            if len(self._read_buffer) >= limit:
                line = bytes(self._read_buffer)
                self._read_buffer.clear()
                return line
            try:
                readable, _, _ = select.select([self._fd], [], [], .05)
                if not readable:
                    continue
                chunk = os.read(self._fd, min(65536, limit - len(self._read_buffer)))
            except InterruptedError:
                continue
            if not chunk:
                line = bytes(self._read_buffer)
                self._read_buffer.clear()
                return line
            self._read_buffer.extend(chunk)

    def _read(self):
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                line = self._readline()
                if not line:
                    with self._condition:
                        if self._closed:
                            return
                        active = self._active
                        self._eof = True
                        self._condition.notify_all()
                    if active:
                        self._fail(RequestInputError('request input EOF during execution'), eof=True)
                    return
                if (type(line) is not bytes or len(line) > MAX_LINE_BYTES
                        or not line.endswith(b'\n')):
                    self._fail(RequestInputError('oversize or incomplete binary request frame'))
                    return
                with self._condition:
                    if self._closed:
                        return
                    concurrent = self._active or self._queued
                    if not concurrent:
                        self._lines.append(line)
                        self._queued = True
                        self._condition.notify_all()
                if concurrent:
                    # Decode only to improve the error message. The main
                    # thread remains the sole strict protocol validator.
                    try:
                        value = json.loads(line)
                        close = type(value) is dict and value.get('method') == 'close'
                    except (ValueError, UnicodeError, RecursionError):
                        close = False
                    self._fail(RequestInputError('close requested during execution' if close
                                                 else 'concurrent request rejected; no dispatch'))
                    return
        except BaseException as exc:
            self._fail(exc)

    def check(self):
        with self._condition:
            error = self.error
        if error is not None:
            raise error

    def readline(self, _limit=None):
        with self._condition:
            while not self._lines and not self._eof and not self._closed and self.error is None:
                self._condition.wait()
            if self.error is not None:
                raise self.error
            if self._lines:
                line = self._lines.popleft()
                self._delivered = True
                return line
            return b''

    def begin(self):
        with self._condition:
            if self.error is not None:
                raise self.error
            if self._eof or self._closed or self._cancelled:
                raise RequestInputError('request input ended before execution; no dispatch')
            if self._active or not self._delivered:
                raise RequestInputError('request execution requires one delivered frame')
            self._active = True
            self._queued = False
            self._delivered = False

    def end(self):
        """Mark dispatch ended without replacing an in-flight SDK exception."""
        with self._condition:
            self._active = False
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            # Real descriptor readers wake at most 50ms later; do not close the
            # caller's stdin descriptor just to interrupt a buffered read.
            self._thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
        return False
