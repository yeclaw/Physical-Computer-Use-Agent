"""Bounded JSON Lines over pipes to an owned local process; never retries requests.

This is transport and process cleanup, not a security sandbox. Each wire line,
including its newline, is limited to 64 KiB. Sending accepts the caller's
remaining deadline (five seconds by default); requests are never retried.
"""

import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


MAX_LINE_BYTES = 64 * 1024
SEND_TIMEOUT_S = 5.0


class TransportError(RuntimeError):
    pass


class ProtocolError(TransportError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _constant(value):
    raise ProtocolError('Non-finite JSON number: ' + value)


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError('Non-finite JSON number')
    return number


def _unicode_and_keys(value):
    if isinstance(value, str):
        value.encode('utf-8')
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProtocolError('JSON keys must be strings')
            _unicode_and_keys(key)
            _unicode_and_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _unicode_and_keys(child)


def strict_loads(content):
    """Read exactly one bounded JSON object, rejecting duplicate and NaN values."""
    if not isinstance(content, (str, bytes)):
        raise ProtocolError('JSON input must be text or bytes')
    try:
        raw = content.encode('utf-8') if isinstance(content, str) else content
        if len(raw) + (0 if raw.endswith(b'\n') else 1) > MAX_LINE_BYTES:
            raise ProtocolError('JSON line exceeds 64 KiB')
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=_constant, parse_float=_float)
        _unicode_and_keys(value)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError('Invalid JSON object') from exc
    if not isinstance(value, dict):
        raise ProtocolError('JSON message must be an object')
    return value


def encode_line(value):
    if not isinstance(value, dict):
        raise ProtocolError('JSON message must be an object')
    try:
        _unicode_and_keys(value)
        content = json.dumps(value, separators=(',', ':'), ensure_ascii=False,
                             allow_nan=False).encode('utf-8') + b'\n'
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError('Message is not finite JSON') from exc
    if len(content) > MAX_LINE_BYTES:
        raise ProtocolError('JSON line exceeds 64 KiB')
    strict_loads(content)
    return content


def _duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Timeout must be a finite nonnegative number')
    return value


class JsonProcess:
    """Own one subprocess/session, with bounded reads and explicit final wait."""

    def __init__(self, command, stderr_path, env=None):
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError('Command must be a nonempty list of strings')
        self._closed = False
        self._failure = None
        self._buffer = bytearray()
        self._close_report = None
        self.stderr_path = Path(stderr_path)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open('ab', buffering=0)
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self._stderr, env=env, bufsize=0,
                                            start_new_session=True)
        except BaseException:
            self._stderr.close()
            raise
        self.pid = self.process.pid
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _open(self):
        if self._closed:
            raise TransportError('Process transport is closed')
        if self._failure is not None:
            raise ProtocolError('Process transport already failed: ' + self._failure)

    def send(self, message, timeout_s=SEND_TIMEOUT_S):
        self._open()
        deadline = time.monotonic() + _duration(timeout_s)
        content = memoryview(encode_line(message))
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin, selectors.EVENT_WRITE)
            while content:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._failure = 'send timeout; delivery may be partial'
                    raise TimeoutError(self._failure)
                if not writable.select(remaining):
                    continue
                try:
                    count = os.write(self.process.stdin.fileno(), content)
                except BlockingIOError:
                    continue
                except (BrokenPipeError, OSError) as exc:
                    self._failure = 'send failed; delivery may be partial'
                    raise TransportError(self._failure) from exc
                if count <= 0:
                    self._failure = 'send made no progress'
                    raise TransportError(self._failure)
                content = content[count:]

    def receive(self, timeout_s):
        self._open()
        deadline = time.monotonic() + _duration(timeout_s)
        try:
            while True:
                newline = self._buffer.find(b'\n')
                if newline >= 0:
                    if newline + 1 > MAX_LINE_BYTES:
                        raise ProtocolError('JSON line exceeds 64 KiB')
                    line = bytes(self._buffer[:newline + 1])
                    del self._buffer[:newline + 1]
                    return strict_loads(line)
                if len(self._buffer) >= MAX_LINE_BYTES:
                    raise ProtocolError('JSON line exceeds 64 KiB')
                remaining = max(0.0, deadline - time.monotonic())
                events = self._selector.select(remaining)
                if not events:
                    raise TimeoutError('Timed out waiting for a complete JSON line')
                try:
                    chunk = os.read(self.process.stdout.fileno(), 4096)
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Timed out waiting for a complete JSON line')
                    continue
                if not chunk:
                    if self._buffer:
                        raise ProtocolError('EOF in partial JSON line')
                    raise EOFError('Process stdout closed')
                self._buffer.extend(chunk)
                if time.monotonic() >= deadline and b'\n' not in self._buffer:
                    raise TimeoutError('Timed out waiting for a complete JSON line')
        except ProtocolError as exc:
            self._failure = str(exc)
            raise

    def close(self):
        """Terminate only this owned session, escalate if needed, and reap its child."""
        if self._close_report is not None:
            return {**self._close_report, 'cleanup_errors': list(self._close_report['cleanup_errors'])}
        self._closed = True
        report = {'pid': self.pid, 'returncode': None, 'terminate_sent': False,
                  'kill_sent': False, 'waited': False, 'cleanup_errors': []}
        try:
            try:
                self.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.pid, signal.SIGTERM)
                report['terminate_sent'] = True
            except ProcessLookupError:
                pass
            except OSError as exc:
                report['cleanup_errors'].append('SIGTERM: ' + str(exc))
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                    report['kill_sent'] = True
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2.0)
            report['waited'] = True
            # A session leader can exit before its owned descendants. No global
            # process search is needed; the process group is the ownership boundary.
            try:
                os.killpg(self.pid, signal.SIGKILL)
                report['kill_sent'] = True
            except ProcessLookupError:
                pass
            except OSError as exc:
                report['cleanup_errors'].append('group cleanup: ' + str(exc))
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['cleanup_errors'].append(str(exc))
        finally:
            report['returncode'] = self.process.poll()
            try:
                self._selector.close()
            except OSError as exc:
                report['cleanup_errors'].append('selector cleanup: ' + str(exc))
            for stream in (self.process.stdin, self.process.stdout, self._stderr):
                try:
                    stream.close()
                except OSError as exc:
                    report['cleanup_errors'].append(str(exc))
            self._close_report = report
        return {**report, 'cleanup_errors': list(report['cleanup_errors'])}
