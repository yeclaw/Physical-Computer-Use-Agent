"""Explicit reset-only IPC for the training editor; never reads its text.

One connection carries one bounded JSON line, followed by write-side EOF.
There are no retries. Receipts
are idempotent only for the lifetime of this explicitly scoped editor service;
the cache is never evicted to permit a request ID to execute again.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import threading
import time


PROTOCOL = "physical-notepad-reset-v1"
MAX_MESSAGE_BYTES = 4096
RESET_RESULT = {"cleared": True, "focus_requested": True,
                "ime_reset_requested": True, "buffer_read": False}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")


class NotepadResetError(RuntimeError):
    """A reset was not dispatched, or a request/receipt was invalid."""


class NotepadResetUncertain(NotepadResetError):
    """A reset may have executed; never automatically send it again."""


def _identifier(value):
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise NotepadResetError("timeout must be a finite positive number")
    return float(value)


def _path(value):
    path = Path(value)
    if not path.is_absolute() or not path.name or path.name in (".", ".."):
        raise NotepadResetError("an absolute explicit socket path is required")
    try:
        parent = path.parent.resolve(strict=True)
        info = parent.stat()
    except OSError as exc:
        raise NotepadResetError("socket parent is unavailable") from exc
    if parent != path.parent or not stat.S_ISDIR(info.st_mode):
        raise NotepadResetError("socket parent must be a canonical directory")
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise NotepadResetError("socket parent must be owned by this user and not group/world writable")
    return path


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def constant(unused):
        raise ValueError("non-finite JSON")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _encode(value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("message too large")
    return raw


def _receive(connection, *, timeout):
    raw = bytearray()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("reset message deadline elapsed")
        connection.settimeout(remaining)
        part = connection.recv(min(1024, MAX_MESSAGE_BYTES + 1 - len(raw)))
        if not part:
            raise ValueError("incomplete JSON line")
        raw.extend(part)
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
        if b"\n" in raw:
            if raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
                raise ValueError("exactly one JSON line is required")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout("reset message deadline elapsed")
            connection.settimeout(remaining)
            if connection.recv(1):
                raise ValueError("extra data after JSON line")
            return _decode(bytes(raw[:-1]))


def _error(request, run_id, code, message, *, uncertain=False):
    request = request if isinstance(request, dict) else {}
    return {"protocol": PROTOCOL,
            "request_id": request.get("request_id") if _identifier(request.get("request_id")) else None,
            "run_id": request.get("run_id") if _identifier(request.get("run_id")) else run_id,
            "status": "error",
            "error": {"code": code, "message": message, "uncertain": uncertain}}


def _valid_result(value):
    return (isinstance(value, dict) and set(value) == set(RESET_RESULT)
            and all(type(value[key]) is bool and value[key] == expected
                    for key, expected in RESET_RESULT.items()))


def _validate_receipt(reply, request):
    if not isinstance(reply, dict):
        raise ValueError("receipt is not an object")
    for key in ("protocol", "request_id", "run_id"):
        if reply.get(key) != request[key]:
            raise ValueError("receipt scope/identity mismatch")
    if reply.get("status") == "ok":
        if set(reply) != {"protocol", "request_id", "run_id", "status", "result"} or not _valid_result(reply["result"]):
            raise ValueError("invalid success receipt")
    elif reply.get("status") == "error":
        if set(reply) != {"protocol", "request_id", "run_id", "status", "error"}:
            raise ValueError("invalid error receipt")
        error = reply["error"]
        if (not isinstance(error, dict) or set(error) != {"code", "message", "uncertain"}
                or not _identifier(error["code"]) or not isinstance(error["message"], str)
                or not error["message"] or len(error["message"]) > 256
                or type(error["uncertain"]) is not bool):
            raise ValueError("invalid error payload")
    else:
        raise ValueError("unknown receipt status")


def parse_receipt(raw: bytes, *, run_id, request_id):
    """Validate a remote helper's single stdout line without any I/O/actions.

    Invalid receipts raise NotepadResetError. A caller that has already sent a
    reset must treat that validation failure as an uncertain execution outcome.
    """
    if not _identifier(run_id) or not _identifier(request_id):
        raise NotepadResetError("run_id/request_id must be bounded identifiers")
    if (not isinstance(raw, bytes) or len(raw) > MAX_MESSAGE_BYTES
            or raw.count(b"\n") != 1 or not raw.endswith(b"\n")):
        raise NotepadResetError("receipt must be one bounded complete JSON line")
    try:
        reply = _decode(raw[:-1])
        _validate_receipt(reply, {"protocol": PROTOCOL, "run_id": run_id, "request_id": request_id})
    except (ValueError, RecursionError) as exc:
        raise NotepadResetError("receipt has invalid JSON, identity, or schema") from exc
    return reply


def request_reset(socket_path, *, run_id, request_id, timeout=5.0):
    """Send once and return a validated ok/error receipt; never reads text.

    A pre-connect failure raises NotepadResetError. Any transport or receipt
    failure after sending starts raises NotepadResetUncertain. Neither case is
    retried here. An editor error is returned with its explicit uncertain flag.
    """
    if not _identifier(run_id) or not _identifier(request_id):
        raise NotepadResetError("run_id/request_id must be bounded identifiers")
    timeout = _timeout(timeout)
    path = _path(socket_path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise NotepadResetError("reset socket is unavailable") from exc
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise NotepadResetError("reset endpoint must be this user's owner-only socket")
    request = {"protocol": PROTOCOL, "request_id": request_id, "run_id": run_id, "action": "reset"}
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dispatched = False
    try:
        connection.settimeout(timeout)
        connection.connect(str(path))
        dispatched = True
        connection.sendall(_encode(request))
        connection.shutdown(socket.SHUT_WR)
        reply = _receive(connection, timeout=timeout)
        _validate_receipt(reply, request)
        return reply
    except (OSError, ValueError, RecursionError) as exc:
        error = NotepadResetUncertain if dispatched else NotepadResetError
        raise error("reset receipt unavailable or invalid; do not automatically retry") from exc
    finally:
        connection.close()


class GtkMainThreadReset:
    """GI-independent adapter marshaling a reset onto a GUI's idle queue.

    A timed-out queued callback is cancelled before it can mutate the editor.
    If the callback had started, its outcome is explicitly uncertain instead.
    Call close() on window destruction to cancel callbacks not yet started.
    """

    def __init__(self, reset_callback, idle_add, *, timeout=5.0):
        self._callback = reset_callback
        self._idle_add = idle_add
        self._timeout = _timeout(timeout)
        self._lock = threading.Lock()
        self._pending = []
        self._closed = False

    def __call__(self):
        state = {"done": threading.Event(), "started": False, "cancelled": False}
        with self._lock:
            if self._closed:
                raise NotepadResetError("editor is closed")
            self._pending.append(state)

        def run():
            with self._lock:
                if state["cancelled"] or self._closed:
                    state["cancelled"] = True
                    state["done"].set()
                    return False
                state["started"] = True
            try:
                state["result"] = self._callback()
            except Exception:
                # Do not expose a callback exception, which might contain text.
                state["failed"] = True
            finally:
                state["done"].set()
            return False

        try:
            self._idle_add(run)
            finished = state["done"].wait(self._timeout)
            with self._lock:
                if not finished:
                    state["cancelled"] = True
                    if state["started"]:
                        raise NotepadResetUncertain("editor reset started but did not finish in time")
                    raise NotepadResetError("editor did not dispatch reset before timeout")
                if state["cancelled"]:
                    raise NotepadResetError("editor closed before reset was dispatched")
            if state.get("failed") or not _valid_result(state.get("result")):
                raise NotepadResetUncertain("editor reset callback did not confirm completion")
            return state["result"]
        finally:
            with self._lock:
                self._pending.remove(state)

    def close(self):
        with self._lock:
            self._closed = True
            for state in self._pending:
                if not state["started"]:
                    state["cancelled"] = True
                    state["done"].set()


class NotepadResetServer:
    """Single-owner, reset-only worker; it never executes GUI work directly."""

    def __init__(self, socket_path, *, run_id, reset_action, action_timeout=5.0, max_receipts=None):
        if not _identifier(run_id):
            raise NotepadResetError("invalid editor run_id")
        if not callable(reset_action):
            raise NotepadResetError("reset_action must be callable")
        if max_receipts is not None and (isinstance(max_receipts, bool) or not isinstance(max_receipts, int) or max_receipts < 1):
            raise NotepadResetError("max_receipts must be null or a positive integer")
        self.socket_path = _path(socket_path)
        self.run_id = run_id
        self._action = reset_action
        self._timeout = _timeout(action_timeout)
        self._max_receipts = max_receipts
        self._receipts = {}
        self._listener = None
        self._identity = None
        self._thread = None
        self._closed = threading.Event()

    def start(self):
        if self._listener is not None or self._closed.is_set():
            raise NotepadResetError("reset service cannot be restarted")
        if os.path.lexists(self.socket_path):
            raise NotepadResetError("reset socket path already exists; refusing to overwrite")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            info = self.socket_path.lstat()
            self._identity = (info.st_dev, info.st_ino)
            # Set access before listen: no peer can connect during chmod.
            os.chmod(self.socket_path, 0o600)
            listener.listen(8)
            listener.settimeout(0.2)
        except OSError as exc:
            listener.close()
            self._unlink_owned_socket()
            raise NotepadResetError("could not start explicit reset socket") from exc
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="notepad-reset", daemon=True)
        self._thread.start()
        return self

    def _unlink_owned_socket(self):
        try:
            info = self.socket_path.lstat()
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self._identity:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _handle(self, request):
        if (not isinstance(request, dict) or set(request) != {"protocol", "request_id", "run_id", "action"}
                or not _identifier(request.get("request_id")) or not _identifier(request.get("run_id"))):
            return _error(request, self.run_id, "invalid_request", "exact reset-only request schema required")
        request_id = request["request_id"]
        cached = self._receipts.get(request_id)
        if cached:
            if request != cached[0]:
                return _error(request, self.run_id, "request_id_conflict", "request_id was already used for a different request")
            return copy.deepcopy(cached[1])
        if request["protocol"] != PROTOCOL or request["run_id"] != self.run_id:
            return _error(request, self.run_id, "scope_mismatch", "request does not match this editor protocol/run")
        if request["action"] != "reset":
            return _error(request, self.run_id, "unsupported_action", "only reset is supported")
        if self._max_receipts is not None and len(self._receipts) >= self._max_receipts:
            return _error(request, self.run_id, "receipt_capacity", "receipt capacity reached; reset was not dispatched")
        if self._closed.is_set():
            return _error(request, self.run_id, "service_closed", "reset service is closing; reset was not dispatched")
        try:
            result = self._action()
            if not _valid_result(result):
                raise NotepadResetUncertain("reset action did not confirm completion")
            reply = {"protocol": PROTOCOL, "request_id": request_id, "run_id": self.run_id,
                     "status": "ok", "result": copy.deepcopy(result)}
        except NotepadResetUncertain:
            reply = _error(request, self.run_id, "reset_uncertain", "reset completion is uncertain; do not retry", uncertain=True)
        except NotepadResetError:
            reply = _error(request, self.run_id, "reset_not_dispatched", "editor reset was not dispatched")
        except Exception:
            reply = _error(request, self.run_id, "reset_uncertain", "reset completion is uncertain; do not retry", uncertain=True)
        self._receipts[request_id] = (copy.deepcopy(request), copy.deepcopy(reply))
        return reply

    def _serve(self):
        while not self._closed.is_set():
            try:
                connection, unused_address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(self._timeout)
                try:
                    # Linux additionally verifies the peer UID; socket/parent
                    # permissions provide the portable owner-only boundary.
                    if hasattr(socket, "SO_PEERCRED"):
                        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                        if struct.unpack("3i", credentials)[1] != os.getuid():
                            continue
                    request = _receive(connection, timeout=self._timeout)
                    reply = self._handle(request)
                except (OSError, ValueError, RecursionError):
                    reply = _error(None, self.run_id, "malformed_request", "bounded single-line JSON request required")
                try:
                    connection.sendall(_encode(reply))
                except OSError:
                    pass  # Receipt stays cached even if its caller disconnects.

    def close(self, *, wait=True):
        self._closed.set()
        if self._listener is not None:
            self._listener.close()
        if wait and self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(self._timeout + 0.5)
        self._unlink_owned_socket()


def main():
    """Remote-helper entry point; stdout is one receipt, never editor text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    try:
        receipt = request_reset(args.socket, run_id=args.run_id, request_id=args.request_id, timeout=args.timeout)
    except NotepadResetError as exc:
        receipt = _error({"request_id": args.request_id, "run_id": args.run_id}, args.run_id,
                         "transport_uncertain" if isinstance(exc, NotepadResetUncertain) else "not_dispatched",
                         "reset transport did not produce a trusted receipt; do not retry",
                         uncertain=isinstance(exc, NotepadResetUncertain))
    print(json.dumps(receipt, separators=(",", ":"), allow_nan=False))
    return 0 if receipt["status"] == "ok" else (3 if receipt["error"]["uncertain"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
