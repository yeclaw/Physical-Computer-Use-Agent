"""Owned remote-host transport; request IDs and unknown outcomes are never replayed."""
from pathlib import Path
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time

PREFIX = "@@PCU@@"

def anchor():
    return dict(utc_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())

def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()

def digest(value):
    return hashlib.sha256(value).hexdigest()

class RemoteHost:
    def __init__(self, command, output: Path, *, timeout=180, wait_check=None, require_lifecycle=False,
                 require_idle_heartbeat=False, heartbeat_interval=30, heartbeat_timeout=15):
        for value in (heartbeat_interval, heartbeat_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite idle heartbeat interval/timeout required')
        self.output, self.timeout, self.counter = output, timeout, 0
        self._rpc_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._heartbeat_failure = None
        self._closing = False
        self.heartbeat_interval, self.heartbeat_timeout = heartbeat_interval, heartbeat_timeout
        self._last_activity = time.monotonic()
        self._last_heartbeat_response = None
        self.wait_check = wait_check
        self._wait_interrupted = False
        self._closed_result = None
        self._lifecycle_enabled = False
        self._updates = {}
        self.last_failure = None
        self._pending_request = None
        self._last_successful_response = None
        self.incoming = queue.Queue()
        self.stderr = (output / 'ssh-stderr.log').open('xb')
        self.events = (output / 'transport.jsonl').open('x', buffering=1)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.ready = self.receive()
            if self.ready.get('type') != 'ready':
                raise RuntimeError('Host not ready: ' + str(self.ready))
            descriptor = self.ready.get('request_lifecycle')
            if require_lifecycle and descriptor is None:
                raise RuntimeError('Host request lifecycle capability missing')
            if descriptor is not None:
                from .request_lifecycle import descriptor as lifecycle_descriptor
                if descriptor != lifecycle_descriptor():
                    raise RuntimeError('Host request lifecycle contract mismatch')
                self._lifecycle_enabled = True
            self._remember_response(self.ready)
            from .request_lifecycle import idle_heartbeat_descriptor
            capability = self.ready.get('idle_heartbeat')
            if require_idle_heartbeat and capability is None:
                raise RuntimeError('Host idle heartbeat capability missing')
            if capability is not None:
                if capability != idle_heartbeat_descriptor():
                    raise RuntimeError('Host idle heartbeat contract mismatch')
                self._heartbeat_thread = threading.Thread(target=self._heartbeat,
                    daemon=True, name='remote-idle-heartbeat')
                self._heartbeat_thread.start()
        except BaseException:
            # Constructor failure must also close stdin and reap the SSH child.
            # Never send a motion or replay an ambiguous startup.
            self.close()
            raise

    def log(self, event, **fields):
        self.events.write(canonical(dict(event=event, anchor=anchor(), **fields)).decode() + '\n')
        self.events.flush()
        os.fsync(self.events.fileno())

    def _read(self):
        try:
            for line in self.process.stdout:
                if not line.startswith(PREFIX):
                    self.incoming.put(RuntimeError('Unexpected remote stdout; no replay'))
                    return
                self.incoming.put(json.loads(line[len(PREFIX):]))
        except BaseException as exc:
            self.incoming.put(exc)
        finally:
            self.incoming.put(EOFError('Remote host EOF; dispatched outcomes may be unknown'))

    def _remember_response(self, response):
        request = self._pending_request or {}
        remembered = dict(
            request_id=request.get('id'), method=request.get('method'),
            response_id=response.get('id'), response_type=response.get('type'),
            response_status=response.get('status'), anchor=anchor())
        if request.get('method') == 'ping':
            self._last_heartbeat_response = remembered
        else:
            self._last_successful_response = remembered
        self._last_activity = time.monotonic()

    def _heartbeat(self):
        while not self._heartbeat_stop.wait(min(self.heartbeat_interval, 1)):
            if not self._rpc_lock.acquire(blocking=False):
                continue  # run_skill already has its own lifecycle updates.
            try:
                if self._closing or self._pending_request is not None:
                    continue
                if time.monotonic() - self._last_activity < self.heartbeat_interval:
                    continue
                try:
                    self._request_locked('ping', {}, _cleanup=True,
                        _deadline=time.monotonic() + self.heartbeat_timeout)
                except BaseException as exc:
                    self._heartbeat_failure = exc
                    self._record_failure('idle_heartbeat_failed', 'idle_heartbeat', exc)
                    self._heartbeat_stop.set()
                    return
            finally:
                self._rpc_lock.release()

    def _record_failure(self, event, stage, error):
        # Diagnostics must not replace the original exception or infer whether
        # a partially written request reached the remote process. In particular,
        # logging "sending" above is not a delivery acknowledgement.
        request = self._pending_request or {}
        failure = dict(
            request_id=request.get('id'), method=request.get('method'), stage=stage,
            error_type=type(error).__name__, error=str(error)[:2048],
            process_returncode=None,
            last_successful_response=self._last_successful_response,
            last_heartbeat_response=getattr(self, '_last_heartbeat_response', None),
            automatic_retry=False, automatic_reconnect=False,
            request_delivery='unknown', stderr_path=str(self.output / 'ssh-stderr.log'),
            anchor=anchor())
        self.last_failure = failure
        if request.get('method') == 'ping':
            failure.update(motion_requested=False, sdk_state_queried=False)
        try:
            failure['process_returncode'] = self.process.poll()
            self.log(event, failure=failure)
        except Exception as diagnostic_error:
            # The caller must still receive the original BrokenPipe/EOF/timeout
            # if, for example, the diagnostic filesystem is also unavailable.
            failure['diagnostic_recording_error'] = repr(diagnostic_error)[:2048]

    def receive(self, *, deadline=None, cleanup=False):
        deadline = time.monotonic() + self.timeout if deadline is None else deadline
        while True:
            if not cleanup and getattr(self, 'wait_check', None) is not None:
                try:
                    self.wait_check()
                except BaseException:
                    self._wait_interrupted = True
                    raise
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty()
                result = self.incoming.get(timeout=min(.1, remaining))
                break
            except queue.Empty as exc:
                if time.monotonic() < deadline:
                    continue
                if not cleanup:
                    self._record_failure('receive_timeout', 'receive', exc)
                raise
        if isinstance(result, BaseException):
            if not cleanup:
                self._record_failure('receive_eof' if isinstance(result, EOFError) else 'receive_failed',
                                     'receive', result)
            raise result
        self.log('received', response=result)
        return result

    def _receive_terminal(self, request, *, cleanup=False, deadline=None):
        """Liveness and actual execution progress are independent, never success."""
        from .request_lifecycle import PROTOCOL
        updates = getattr(self, '_updates', {})
        last_seq, progress_seq = updates.get(request['id'], (0, 0))
        idle_until = progress_until = time.monotonic() + self.timeout
        while True:
            wait_until = min(idle_until, progress_until)
            if deadline is not None:
                wait_until = min(wait_until, deadline)
            response = self.receive(deadline=wait_until, cleanup=cleanup)
            if type(response) is not dict or response.get('id') != request['id']:
                error = RuntimeError('Remote response ID mismatch; no retry')
                self._record_failure('response_mismatch', 'response_id', error)
                raise error
            if response.get('type') != 'request_update':
                return response
            sequence, progress = response.get('sequence'), response.get('progress_seq')
            phase = response.get('phase')
            if (not getattr(self, '_lifecycle_enabled', False)
                    or request['method'] != 'run_skill' or response.get('protocol') != PROTOCOL
                    or type(sequence) is not int or sequence <= last_seq
                    or type(progress) is not int or progress < progress_seq
                    or phase != ('accepted' if last_seq == 0 else 'running')
                    or (last_seq == 0 and (sequence != 1 or progress != 0))):
                error = RuntimeError('Invalid remote request lifecycle update; no retry')
                self._record_failure('invalid_request_update', 'request_update', error)
                raise error
            now = time.monotonic()
            idle_until = now + self.timeout
            if progress > progress_seq:
                progress_until = now + self.timeout
            # Mere heartbeat cannot renew the independent real-progress window.
            last_seq, progress_seq = sequence, progress
            self._updates[request['id']] = (last_seq, progress_seq)

    def request(self, method, args, *, _cleanup=False):
        with self._rpc_lock:
            if not _cleanup and self._closing:
                raise RuntimeError('Remote host is closing; no new dispatch')
            if not _cleanup and self._heartbeat_failure is not None:
                raise self._heartbeat_failure
            return self._request_locked(method, args, _cleanup=_cleanup)

    def _request_locked(self, method, args, *, _cleanup=False, _deadline=None):
        if self._pending_request is not None:
            raise RuntimeError('Previous remote request is unresolved; no new dispatch')
        self.counter += 1
        request = dict(id='actor-%06d' % self.counter, method=method, args=args)
        self._pending_request = request
        self.log('sending', request=request)
        try:
            self.process.stdin.write(canonical(request).decode() + '\n')
        except Exception as exc:
            self._record_failure('sending_failed', 'write', exc)
            raise
        try:
            self.process.stdin.flush()
        except Exception as exc:
            self._record_failure('sending_failed', 'flush', exc)
            raise
        try:
            response = self._receive_terminal(request, cleanup=_cleanup, deadline=_deadline)
        except KeyboardInterrupt:
            self._wait_interrupted = True
            raise
        if method == 'ping':
            from .request_lifecycle import idle_heartbeat_descriptor
            result = response.get('result')
            if (response.get('status') != 'ok' or type(result) is not dict
                    or set(result) != set(idle_heartbeat_descriptor()) | {'anchor'}
                    or {k: result[k] for k in idle_heartbeat_descriptor()} != idle_heartbeat_descriptor()
                    or type(result['anchor']) is not dict
                    or not {'utc_ns', 'monotonic_ns'} <= set(result['anchor'])
                    or set(result['anchor']) - {'utc_ns', 'monotonic_ns', 'sampling_span_ns'}
                    or any(type(result['anchor'][k]) is not int or result['anchor'][k] <= 0
                           for k in ('utc_ns', 'monotonic_ns'))
                    or ('sampling_span_ns' in result['anchor']
                        and (type(result['anchor']['sampling_span_ns']) is not int
                             or result['anchor']['sampling_span_ns'] < 0))):
                raise RuntimeError('Invalid idle heartbeat acknowledgement; no retry')
        self._remember_response(response)
        self._pending_request = None
        getattr(self, '_updates', {}).pop(request['id'], None)
        return response

    def close(self):
        self._heartbeat_stop.set()
        with self._rpc_lock:
            self._closing = True
            result = self._close_locked()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        return result

    def _close_locked(self):
        if getattr(self, '_closed_result', None) is not None:
            return self._closed_result
        result = {'attempted': True}
        try:
            if self.process.poll() is None:
                if self._pending_request is not None and not getattr(self, '_wait_interrupted', False):
                    # Reconcile the OLD ID before issuing a separate close ID.
                    pending = self._pending_request
                    try:
                        late = self._receive_terminal(pending, cleanup=True,
                            deadline=time.monotonic() + min(self.timeout, 20))
                        result['late_response'] = late
                        self.log('late_response_reconciled', request_id=pending['id'], response=late,
                                 current_physical_state='unverified')
                        self._remember_response(late)
                        self._pending_request = None
                    except BaseException as exc:
                        result['pending_error'] = repr(exc)
                if self._pending_request is None:
                    result['response'] = self.request('close', {}, _cleanup=True)
                else:
                    result['unresolved_request_id'] = self._pending_request['id']
                    result['physical_outcome'] = 'unknown_no_replay'
                    # EOF is an out-of-band cancellation at the new host. Do
                    # not queue close behind an unresolved physical program.
        except BaseException as exc:
            result['error'] = repr(exc)
        finally:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                result['returncode'] = self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # Terminate SSH; the remote host handles EOF/HUP best-effort.
                # No guarantee of physical release; separately inspect remote ended.json.
                self.process.terminate()
                try:
                    result['returncode'] = self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    result['returncode'] = self.process.wait()
                result['forced_transport_shutdown'] = True
            try:
                self.log('closed', **result)
            finally:
                self.process.stdout.close()
                self.stderr.close()
                self.events.close()
        self._closed_result = result
        return result
