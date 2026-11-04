"""Real local IPC and GUI-thread scheduling; no GUI, model, or hardware."""

import ast
import errno
import json
import os
from pathlib import Path
import queue
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from astra_robotics_agent.physical_keyboard.notepad_reset import (
    GtkMainThreadReset, MAX_MESSAGE_BYTES, NotepadResetError,
    NotepadResetServer, NotepadResetUncertain, PROTOCOL, RESET_RESULT,
    parse_receipt, request_reset,
)


class NotepadResetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nr-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "editor.sock"
        self.calls = []

    def action(self):
        self.calls.append("reset")
        return dict(RESET_RESULT)

    def server(self, action=None, **kwargs):
        service = NotepadResetServer(self.path, run_id="editor-a", reset_action=action or self.action, **kwargs).start()
        self.addCleanup(service.close)
        return service

    def request(self, request_id="one", **kwargs):
        return request_reset(self.path, run_id="editor-a", request_id=request_id, **kwargs)

    def raw(self, raw):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(self.path))
            connection.sendall(raw)
            try:
                connection.shutdown(socket.SHUT_WR)
            except OSError as exc:
                # A malformed request can already have been rejected/closed.
                # Still require the actual structured error receipt below.
                if exc.errno != errno.ENOTCONN:
                    raise
            response = bytearray()
            while not response.endswith(b"\n"):
                part = connection.recv(4096)
                if not part:
                    break
                response.extend(part)
            return json.loads(response)

    def test_success_is_scoped_owner_only_and_same_id_executes_once(self):
        self.server()
        self.assertEqual(stat.S_IMODE(self.path.lstat().st_mode), 0o600)
        first = self.request()
        second = self.request()
        self.assertEqual(first, second)
        self.assertEqual(first, {"protocol": PROTOCOL, "request_id": "one", "run_id": "editor-a",
                                 "status": "ok", "result": RESET_RESULT})
        self.assertEqual(self.calls, ["reset"])
        self.request("two")
        self.assertEqual(self.calls, ["reset", "reset"])

    def test_scope_rejects_other_editor_or_protocol_and_conflicting_id(self):
        self.server()
        bad = request_reset(self.path, run_id="other-editor", request_id="bad-scope")
        self.assertEqual(bad["error"]["code"], "scope_mismatch")
        request = {"protocol": "other-protocol", "request_id": "bad-protocol", "run_id": "editor-a", "action": "reset"}
        self.assertEqual(self.raw(json.dumps(request).encode() + b"\n")["error"]["code"], "scope_mismatch")
        self.assertEqual(self.calls, [])
        self.request()
        request.update(protocol=PROTOCOL, request_id="one", action="get_text")
        self.assertEqual(self.raw(json.dumps(request).encode() + b"\n")["error"]["code"], "request_id_conflict")
        self.assertEqual(self.calls, ["reset"])

    def test_no_text_read_injection_or_arbitrary_control_actions(self):
        self.server()
        base = {"protocol": PROTOCOL, "request_id": "one", "run_id": "editor-a", "action": "reset"}
        for request in [dict(base, action="get_text"), dict(base, action="type"),
                        dict(base, action="focus"), dict(base, text="secret"),
                        dict(base, command="launch"), {k: v for k, v in base.items() if k != "action"}]:
            with self.subTest(request=request):
                reply = self.raw(json.dumps(request).encode() + b"\n")
                self.assertEqual(reply["status"], "error")
                self.assertNotIn("secret", json.dumps(reply))
        self.assertEqual(self.calls, [])

    def test_malformed_duplicate_fields_nonfinite_and_oversize_never_execute(self):
        self.server()
        for raw in [b"not-json\n", b"[]\n", b"\xff\n", b"{}\n{}\n", b"{}", b"{\"a\":NaN}\n",
                    b'{"protocol":"physical-notepad-reset-v1","request_id":"one","request_id":"two","run_id":"editor-a","action":"reset"}\n',
                    b" " * (MAX_MESSAGE_BYTES + 1) + b"\n"]:
            with self.subTest(prefix=raw[:32]):
                self.assertEqual(self.raw(raw)["status"], "error")
        self.assertEqual(self.calls, [])

    def test_capacity_does_not_evict_receipt_and_repeat_execution(self):
        self.server(max_receipts=1)
        first = self.request("one")
        self.assertEqual(self.request("two")["error"]["code"], "receipt_capacity")
        self.assertEqual(self.request("one"), first)
        self.assertEqual(self.calls, ["reset"])

    def test_simultaneous_duplicate_requests_share_one_execution(self):
        self.server()
        replies = queue.Queue()
        threads = [threading.Thread(target=lambda: replies.put(self.request())) for unused in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(replies.qsize(), 5)
        self.assertTrue(all(replies.get()["status"] == "ok" for unused in range(5)))
        self.assertEqual(self.calls, ["reset"])

    def test_callback_failure_is_uncertain_cached_and_does_not_leak_exception(self):
        def failed():
            self.calls.append("partial")
            raise RuntimeError("private editor text must not leak")
        self.server(failed)
        first = self.request()
        self.assertEqual(first["status"], "error")
        self.assertTrue(first["error"]["uncertain"])
        self.assertNotIn("private", json.dumps(first))
        self.assertEqual(self.request(), first)
        self.assertEqual(self.calls, ["partial"])

    def test_timeout_after_dispatch_is_uncertain_and_never_automatically_retried(self):
        started, finish, completed = threading.Event(), threading.Event(), threading.Event()

        def slow():
            self.calls.append("reset")
            started.set()
            finish.wait(2)
            completed.set()
            return dict(RESET_RESULT)

        self.server(slow)
        try:
            with self.assertRaises(NotepadResetUncertain):
                self.request(timeout=0.03)
            self.assertTrue(started.is_set())
            self.assertEqual(self.calls, ["reset"])
        finally:
            finish.set()
        self.assertTrue(completed.wait(1))

    def test_missing_socket_and_invalid_identifiers_are_not_dispatched(self):
        with self.assertRaises(NotepadResetError) as result:
            self.request()
        self.assertNotIsInstance(result.exception, NotepadResetUncertain)
        self.server()
        for values in [{"run_id": "editor-a", "request_id": "bad id"},
                       {"run_id": "editor-a", "request_id": "x" * 161},
                       {"run_id": "editor-a", "request_id": "one", "timeout": float("nan")}]:
            with self.subTest(values=values), self.assertRaises(NotepadResetError):
                request_reset(self.path, **values)
        self.assertEqual(self.calls, [])

    def test_existing_file_symlink_or_socket_are_never_overwritten(self):
        self.path.write_text("keep")
        with self.assertRaises(NotepadResetError):
            self.server()
        self.assertEqual(self.path.read_text(), "keep")
        self.path.unlink()
        target = self.root / "target"
        target.write_text("keep target")
        self.path.symlink_to(target)
        with self.assertRaises(NotepadResetError):
            self.server()
        self.assertTrue(self.path.is_symlink())
        self.assertEqual(target.read_text(), "keep target")
        self.path.unlink()
        self.server()
        with self.assertRaises(NotepadResetError):
            NotepadResetServer(self.path, run_id="other", reset_action=self.action).start()

    def test_client_refuses_symlink_or_nonprivate_socket_and_shared_parent(self):
        self.server()
        link = self.root / "alias.sock"
        link.symlink_to(self.path)
        with self.assertRaises(NotepadResetError):
            request_reset(link, run_id="editor-a", request_id="one")
        self.path.chmod(0o666)
        with self.assertRaises(NotepadResetError):
            self.request()
        self.path.chmod(0o600)
        shared = self.root / "shared"
        shared.mkdir(mode=0o777)
        shared.chmod(0o777)
        with self.assertRaises(NotepadResetError):
            NotepadResetServer(shared / "new.sock", run_id="a", reset_action=self.action)
        self.assertEqual(self.calls, [])

    def test_close_removes_only_own_socket_and_never_replacement_file(self):
        service = self.server()
        service.close()
        self.assertFalse(self.path.exists())
        with self.assertRaises(NotepadResetError):
            service.start()
        other = self.server()
        self.path.unlink()
        self.path.write_text("replacement")
        other.close()
        self.assertEqual(self.path.read_text(), "replacement")

    def test_cli_emits_only_machine_readable_receipt_and_error_exit_codes(self):
        self.server()
        argv = [sys.executable, "-m", "astra_robotics_agent.physical_keyboard.notepad_reset",
                "--socket", str(self.path), "--run-id", "editor-a", "--request-id", "cli"]
        success = subprocess.run(argv, capture_output=True, text=True, timeout=4)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["result"], RESET_RESULT)
        self.assertEqual(success.stderr, "")
        bad = subprocess.run(argv[:-3] + ["other", "--request-id", "cli-other"], capture_output=True, text=True, timeout=4)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(json.loads(bad.stdout)["error"]["uncertain"], False)

    def test_remote_receipt_parser_rejects_extra_frames_duplicates_and_scope(self):
        self.server()
        reply = self.request()
        raw = json.dumps(reply).encode() + b"\n"
        self.assertEqual(parse_receipt(raw, run_id="editor-a", request_id="one"), reply)
        duplicate = raw.replace(b'"status": "ok"', b'"status": "error", "status": "ok"')
        corrupt = dict(reply, result=dict(RESET_RESULT, buffer_read=True))
        for invalid in [raw + b"{}\n", raw[:-1], duplicate, b" " * (MAX_MESSAGE_BYTES + 1),
                        json.dumps(corrupt).encode() + b"\n", raw.replace(b'"one"', b'"other"'),
                        raw.replace(b'"editor-a"', b'"editor-b"'), raw.replace(b"true", b"1", 1)]:
            with self.subTest(raw=invalid[:40]), self.assertRaises(NotepadResetError):
                parse_receipt(invalid, run_id="editor-a", request_id="one")
        self.assertEqual(self.calls, ["reset"])

    def test_separate_late_frame_is_rejected_before_action(self):
        self.server()
        raw = json.dumps({"protocol": PROTOCOL, "request_id": "one", "run_id": "editor-a", "action": "reset"}).encode() + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(self.path))
            connection.sendall(raw)
            time.sleep(0.01)
            connection.sendall(b"{}\n")
            connection.shutdown(socket.SHUT_WR)
            reply = json.loads(connection.recv(4096))
        self.assertEqual(reply["status"], "error")
        self.assertEqual(self.calls, [])

    def test_cli_uncertain_exit_is_distinct_and_error_remains_parseable(self):
        def failed():
            raise RuntimeError("partial reset")
        self.server(failed)
        result = subprocess.run([sys.executable, "-m", "astra_robotics_agent.physical_keyboard.notepad_reset",
                                 "--socket", str(self.path), "--run-id", "editor-a", "--request-id", "cli"],
                                capture_output=True, timeout=4)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stderr, b"")
        reply = parse_receipt(result.stdout, run_id="editor-a", request_id="cli")
        self.assertEqual(reply["status"], "error")
        self.assertTrue(reply["error"]["uncertain"])


class GuiResetDispatchTests(unittest.TestCase):
    def setUp(self):
        self.scheduled = queue.Queue()
        self.results = queue.Queue()

    def invoke(self, adapter):
        def worker():
            try:
                self.results.put(adapter())
            except Exception as exc:
                self.results.put(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        self.addCleanup(lambda: thread.join(2))
        return thread

    def test_callback_runs_on_pumped_gui_thread_and_returns_exact_receipt(self):
        called = []

        def reset():
            called.append(threading.get_ident())
            return dict(RESET_RESULT)

        adapter = GtkMainThreadReset(reset, self.scheduled.put, timeout=1)
        thread = self.invoke(adapter)
        callback = self.scheduled.get(timeout=1)
        self.assertEqual(called, [])
        self.assertIs(callback(), False)
        thread.join(1)
        self.assertEqual(called, [threading.get_ident()])
        self.assertEqual(self.results.get(), RESET_RESULT)

    def test_timeout_before_gui_dispatch_cancels_late_mutation(self):
        calls = []
        adapter = GtkMainThreadReset(lambda: calls.append("reset"), self.scheduled.put, timeout=0.02)
        thread = self.invoke(adapter)
        callback = self.scheduled.get(timeout=1)
        thread.join(1)
        error = self.results.get(timeout=1)
        self.assertIsInstance(error, NotepadResetError)
        self.assertNotIsInstance(error, NotepadResetUncertain)
        callback()
        self.assertEqual(calls, [])

    def test_editor_close_cancels_queued_work_without_blocking_gui(self):
        calls = []
        adapter = GtkMainThreadReset(lambda: calls.append("reset"), self.scheduled.put, timeout=1)
        thread = self.invoke(adapter)
        callback = self.scheduled.get(timeout=1)
        adapter.close()
        thread.join(1)
        self.assertIsInstance(self.results.get(timeout=1), NotepadResetError)
        callback()
        self.assertEqual(calls, [])
        with self.assertRaises(NotepadResetError):
            adapter()

    def test_started_callback_timeout_is_uncertain_and_not_scheduled_again(self):
        started, release = threading.Event(), threading.Event()

        def reset():
            started.set()
            release.wait(1)
            return dict(RESET_RESULT)

        adapter = GtkMainThreadReset(reset, self.scheduled.put, timeout=0.03)
        thread = self.invoke(adapter)
        callback = self.scheduled.get(timeout=1)
        fake_gui = threading.Thread(target=callback)
        fake_gui.start()
        try:
            self.assertTrue(started.wait(1))
            thread.join(1)
            self.assertIsInstance(self.results.get(timeout=1), NotepadResetUncertain)
            self.assertTrue(self.scheduled.empty())
        finally:
            release.set()
            fake_gui.join(1)



if __name__ == "__main__":
    unittest.main()
