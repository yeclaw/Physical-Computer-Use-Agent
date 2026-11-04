"""Codex app-server client with explicit thread rotation and manual compaction.

The session is synchronous and must have one owner. Transport events are read on
a daemon thread so a stalled subprocess cannot defeat the turn deadline.
Legacy calls retain the CLI 0.154.0 protocol shape. Explicit service tier and
execution-tool restrictions use CLI 0.162.1. Feature disabling is defense in
depth, not an exhaustive tool allowlist or an OS sandbox: an unexpected builtin
item aborts only after its notification and cannot undo an executed operation.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from collections import deque
from typing import Callable

from .physical_keyboard.records.redaction import redact_text, sensitive_key


class CodexError(RuntimeError):
    """A failed or ambiguous runtime operation."""


class CodexTimeout(CodexError):
    pass


class _CodexRequestRejected(CodexError):
    """A correlated RPC rejection with selected, credential-redacted metadata."""

    def __init__(self, method: str, error):
        self.method = method
        self.error = _selected_error(error)
        super().__init__(f"{method} rejected (code {self.error.get('code')})")


def _native_usage_complete(total: dict | None) -> bool:
    """Completeness of observed protocol counters, never a billing guarantee."""
    required = {"inputTokens", "cachedInputTokens", "outputTokens",
                "reasoningOutputTokens", "totalTokens"}
    return isinstance(total, dict) and required.issubset(total) and all(
        type(total[key]) is int and total[key] >= 0 for key in required)


def _error_text(value: str) -> str:
    """Bound selected diagnostic text without retaining credential material."""
    return redact_text(value)


def _error_detail(value, depth: int = 0):
    """Copy bounded error metadata, never raw frames, items or account payloads."""
    if depth > 6:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return _error_text(value)
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_error_detail(item, depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        blocked = {"account", "email", "auth", "authorization", "accesstoken", "refreshtoken",
                   "apikey", "password", "secret", "credentials", "credential", "token", "cookie",
                   "cookies", "sessionid", "userid", "organizationid", "prompt", "input", "output", "reasoning", "items",
                   "messages", "content", "text"}
        return {_error_text(str(key))[:100]: _error_detail(item, depth + 1)
                for key, item in list(value.items())[:40]
                if re.sub(r"[^a-z]", "", str(key).lower()) not in blocked
                and not sensitive_key(key)}
    return "[UNSUPPORTED]"


def _selected_error(value) -> dict:
    if not isinstance(value, dict):
        return {"message": _error_text(value)} if isinstance(value, str) else {}
    fields = {"message", "code", "type", "codexErrorInfo", "additionalDetails", "httpStatus",
              "httpStatusCode", "status", "retryAfterSeconds", "willRetry"}
    return {key: _error_detail(item) for key, item in value.items() if key in fields}


def _permanent_failure(error: dict, notifications: list[dict]) -> bool:
    """Only terminal model failures reach here; permanent requests must not loop."""
    text = json.dumps([error, notifications], ensure_ascii=True).lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    markers = ("authentication", "unauthorized", "unauthenticated", "invalidapikey", "notloggedin",
               "tokenexpired", "permissiondenied", "accessdenied", "forbidden", "insufficientquota",
               "quotaexceeded", "usagelimit", "creditbalance", "billing", "payment",
               "contextwindow", "contextlength", "maximumcontext", "inputtoolong", "prompttoolong",
               "modelnotfound", "modelnotavailable", "modelunavailable", "unsupportedmodel",
               "invalidrequest", "badrequest", "sandbox", "approvalrequired")
    if any(marker in compact for marker in markers):
        return True
    def forbidden_status(value):
        if isinstance(value, dict):
            return any((key in {"httpStatus", "httpStatusCode"} and item in (400, 401, 402, 403, 404, 413))
                       or forbidden_status(item) for key, item in value.items())
        return isinstance(value, list) and any(forbidden_status(item) for item in value)
    return forbidden_status([error, notifications])


def _retry_after_seconds(value) -> float:
    """Honor a server delay when the selected protocol error supplies one."""
    if isinstance(value, dict):
        hint = value.get('retryAfterSeconds')
        own = (float(hint) if type(hint) in (int, float) and math.isfinite(hint)
               and hint >= 0 else 0.0)
        return max([own] + [_retry_after_seconds(item) for item in value.values()])
    if isinstance(value, list):
        return max([0.0] + [_retry_after_seconds(item) for item in value])
    return 0.0


class CodexTurnFailed(CodexError):
    """A known terminal failed turn; only guarded inference sessions can retry it."""

    def __init__(self, *, thread_id: str, turn_id: str, error: dict,
                 error_notifications: list[dict], usage: dict | None, retryable: bool,
                 attempt: int, tool_activity_seen: bool, usage_complete: bool | None = None):
        self.thread_id, self.turn_id = thread_id, turn_id
        self.error = copy.deepcopy(error)
        self.error_notifications = copy.deepcopy(error_notifications)
        self.usage = copy.deepcopy(usage)
        self.retryable, self.attempt = retryable, attempt
        self.details = {"thread_id": thread_id, "turn_id": turn_id, "status": "failed",
                        "error": self.error, "error_notifications": self.error_notifications,
                        "usage": self.usage, "usage_available": usage is not None,
                        "usage_complete": (_native_usage_complete(usage) if usage_complete is None
                                           else usage_complete), "retryable": retryable,
                        "attempt": attempt, "tool_activity_seen": tool_activity_seen,
                        "failed_at": time.time()}
        message = self.error.get("message") or next((n.get("error", {}).get("message")
                    for n in reversed(self.error_notifications) if n.get("error", {}).get("message")), None)
        super().__init__("Codex turn ended with status failed" + (f": {message}" if message else ""))


class CodexSession:
    def __init__(self, cwd: Path, tools: list[dict], instructions: str,
                 model: str = "gpt-6-astra", effort: str = "high",
                 log_dir: Path | None = None, timeout: float = 600, *,
                 service_tier: str | None = None, disable_execution_tools: bool = False,
                 inference_retries: int = 0, retry_delay_seconds: float = 1.0,
                 read_isolation=None):
        self.cwd = Path(cwd).resolve()
        if not self.cwd.is_dir() or timeout <= 0:
            raise ValueError("cwd must exist and timeout must be positive")
        if service_tier is not None and (not isinstance(service_tier, str) or not re.fullmatch(
                r"[a-z][a-z0-9_-]{0,63}", service_tier)):
            raise ValueError("service_tier must be an explicit tier ID or None to inherit")
        if type(disable_execution_tools) is not bool:
            raise ValueError("disable_execution_tools must be boolean")
        if type(inference_retries) is not int or not 0 <= inference_retries <= 3:
            raise ValueError("inference_retries must be an integer from 0 to 3")
        if (type(retry_delay_seconds) not in (int, float) or not math.isfinite(retry_delay_seconds)
                or retry_delay_seconds < 0):
            raise ValueError("retry_delay_seconds must be finite and nonnegative")
        if inference_retries and (not disable_execution_tools or tools != []):
            raise ValueError("Inference retries require disabled execution tools and an empty tool list")
        self.tools = copy.deepcopy(tools)
        self.instructions, self.model, self.effort = instructions, model, effort
        self.service_tier = service_tier
        self.disable_execution_tools = disable_execution_tools
        self.read_isolation = read_isolation
        if read_isolation is not None:
            if not disable_execution_tools or tools != []:
                raise ValueError('read isolation requires inference-only mode')
            read_isolation.prefix(self.cwd)
        self._isolation_config = {} if read_isolation is None else read_isolation.config_overrides()
        self.inference_retries, self.retry_delay_seconds = inference_retries, retry_delay_seconds
        self.timeout = timeout
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._proc = None
        self._incoming: queue.Queue = queue.Queue()
        self._pending: deque = deque()
        self._next_id = 0
        self.thread_id: str | None = None
        self.total_usage: dict = {}
        self.total_usage_complete = True
        self.thread_usage: dict[str, dict] = {}
        self._operation_lock = threading.Lock()
        self.metadata: dict = {}
        self._active = False
        self._closed = False
        self._current_turn_id: str | None = None
        self._current_turn_usage_available = False
        self._known_turn_ids: set[str] = set()
        self._tool_names: set[str] = set()
        for tool in self.tools:
            # Accept the old flat spelling but emit the installed tagged schema.
            tool.setdefault("type", "function")
            if tool["type"] != "function" or not all(
                k in tool for k in ("name", "description", "inputSchema")
            ):
                raise ValueError("tools must be function specs with name/description/inputSchema")
            if tool["name"] in self._tool_names:
                raise ValueError("duplicate dynamic tool name")
            self._tool_names.add(tool["name"])

    def _explicit_config(self) -> dict:
        """Opt-in overrides only; these are requests, not an isolation guarantee.

        CLI 0.162.1 has no global builtin whitelist. In particular a local
        features listing can keep unified_exec enabled despite its override.
        Custom MCP configuration and server-internal builtins are not governed
        by a dynamicTools list; callers must use a disposable empty workspace.
        """
        config = dict(self._isolation_config)
        if self.service_tier is not None:
            config["service_tier"] = self.service_tier
        if self.disable_execution_tools:
            config.update({"features." + feature: False for feature in (
                "shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins",
                "browser_use", "computer_use", "view_image", "image_generation",
                "multi_agent", "multi_agent_v2", "code_mode", "code_mode_host",
                "hooks", "workspace_dependencies", "skill_search", "tool_suggest",
                "goals", "sleep_tool")})
            config["web_search"] = "disabled"
        return config

    def _check_service_tier(self, value) -> None:
        if self.service_tier is None:
            return
        # The standard tier can be represented by the absence of a priority
        # tier. Every turn still sends an explicit non-inheriting override.
        expected = "priority" if self.service_tier == "fast" else self.service_tier
        observed = "priority" if value == "fast" else value
        if observed != expected and not (expected == "default" and observed is None):
            raise CodexError("App-server changed requested service tier")

    def _check_item(self, item: dict, turn_id: str | None) -> None:
        if not self.disable_execution_tools:
            return
        kind = item.get("type")
        if kind not in {"userMessage", "agentMessage", "reasoning", "contextCompaction",
                        "functionCallOutput", "dynamicToolCall"}:
            self._log("unexpected_builtin_item", thread_id=self.thread_id,
                      turn_id=turn_id, item_type=kind, prevention_guaranteed=False)
            raise CodexError("Unexpected builtin item; execution may already have occurred")

    def _log(self, event: str, **fields) -> None:
        # Explicit allowlisted fields at call sites: no raw protocol, auth,
        # prompts, tool bodies, shell outputs, or model reasoning.
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "codex_runtime.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": event, "time": time.time(), **fields}) + "\n")

    def _read_stdout(self) -> None:
        try:
            for line in self._proc.stdout:
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError()
                except (ValueError, TypeError):
                    self._incoming.put(CodexError("Invalid app-server JSON frame"))
                    return
                self._incoming.put(message)
        except (OSError, ValueError):
            pass
        finally:
            self._incoming.put(CodexError("Codex app-server reached EOF"))

    def _send(self, frame: dict) -> None:
        try:
            self._proc.stdin.write(json.dumps(frame) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise CodexError("Codex app-server input closed") from exc

    def _receive(self, deadline: float, *, pending: bool = True) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexTimeout("Codex operation exceeded its deadline")
        if pending and self._pending:
            return self._pending.popleft()
        try:
            result = self._incoming.get(timeout=remaining)
        except queue.Empty as exc:
            raise CodexTimeout("Codex operation exceeded its deadline") from exc
        if isinstance(result, Exception):
            raise result
        return result

    def _request(self, method: str, params: dict, deadline: float) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params})
        while True:
            frame = self._receive(deadline, pending=False)
            if frame.get("id") == request_id and "method" not in frame:
                if "error" in frame:
                    if method in {"turn/start", "thread/compact/start"}:
                        self._log("turn_request_rejected" if method == "turn/start"
                                  else "context_compaction_request_rejected", thread_id=self.thread_id,
                                  error=_selected_error(frame["error"]))
                    # Server error text may include user paths/prompts; don't expose it.
                    raise _CodexRequestRejected(method, frame["error"])
                return frame.get("result", {})
            self._pending.append(frame)

    def start(self) -> dict:
        if self._closed:
            raise CodexError("Session has been closed")
        if self.thread_id is not None:
            return copy.deepcopy(self.metadata)
        deadline = time.monotonic() + self.timeout
        try:
            command = ["codex", "app-server", "--stdio"]
            for key, value in self._explicit_config().items():
                command.extend(["-c", key + "=" + json.dumps(value)])
            if self.read_isolation is not None:
                command = self.read_isolation.prefix(self.cwd) + command
            self._proc = subprocess.Popen(
                command, cwd=str(self.cwd),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            initialized = self._request("initialize", {
                "clientInfo": {"name": "astra_robotics_agent", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            }, deadline)
            self._send({"method": "initialized", "params": {}})
            account = self._request("account/read", {"refreshToken": False}, deadline)
            if (account.get("account") or {}).get("type") != "chatgpt":
                raise CodexError("An existing ChatGPT login is required; API-key fallback is disabled")
            # Never retain or log the email, plan, tokens, or full account payload.
            del account
            cursor = None
            selected = None
            seen_cursors = set()
            while True:
                params = {"includeHidden": True, "limit": 100}
                if cursor is not None:
                    params["cursor"] = cursor
                catalog = self._request("model/list", params, deadline)
                selected = next((m for m in catalog.get("data", [])
                                 if m.get("model") == self.model), None)
                if selected is not None:
                    break
                cursor = catalog.get("nextCursor")
                if not cursor or cursor in seen_cursors:
                    raise CodexError(f"Requested model unavailable: {self.model}")
                seen_cursors.add(cursor)
            efforts = {x.get("reasoningEffort") for x in selected.get("supportedReasoningEfforts", [])}
            if self.effort not in efforts:
                raise CodexError(f"Requested reasoning effort unavailable: {self.effort}")
            if "image" not in selected.get("inputModalities", []):
                raise CodexError("Requested model does not advertise image input")
            self.metadata = {"model": self.model, "effort": self.effort,
                             "cwd": str(self.cwd), "user_agent": initialized.get("userAgent"),
                             "ephemeral": True, "auth_type": "chatgpt",
                             "context_strategy": "persistent_thread", "accounting_scope": "episode"}
            if self.inference_retries:
                self.metadata.update(inference_retries=self.inference_retries,
                                     retry_delay_seconds=self.retry_delay_seconds,
                                     recovery_policy="terminal_failed_inference_only_same_thread")
            self._start_thread(deadline, "session_start")
            self._log("session_started", **self.metadata)
            return copy.deepcopy(self.metadata)
        except BaseException:
            self.close()
            raise

    def _start_thread(self, deadline: float, reason: str) -> None:
        params = {
            "model": self.model, "allowProviderModelFallback": False,
            "cwd": str(self.cwd), "runtimeWorkspaceRoots": [str(self.cwd)],
            "sandbox": "workspace-write", "approvalPolicy": "never",
            "config": {"model_reasoning_effort": self.effort,
                       "sandbox_workspace_write.writable_roots": [str(self.cwd)],
                       "sandbox_workspace_write.network_access": False,
                       "sandbox_workspace_write.exclude_tmpdir_env_var": True,
                       "sandbox_workspace_write.exclude_slash_tmp": True},
            "developerInstructions": self.instructions,
            "dynamicTools": self.tools, "ephemeral": True,
            "experimentalRawEvents": False,
        }
        params["config"].update(self._explicit_config())
        if self.service_tier is not None:
            params["serviceTier"] = self.service_tier
        result = self._request("thread/start", params, deadline)
        if result.get("model") != self.model or result.get("reasoningEffort") != self.effort:
            raise CodexError("App-server changed requested model or reasoning effort")
        self._check_service_tier(result.get("serviceTier"))
        sandbox = result.get("sandbox", {})
        if sandbox.get("type") != "workspaceWrite" or sandbox.get("networkAccess") is not False:
            raise CodexError("App-server did not apply the workspace sandbox")
        if any(not Path(root).resolve().is_relative_to(self.cwd)
               for root in sandbox.get("writableRoots", [])):
            raise CodexError("App-server added writable roots outside the session workspace")
        thread_id = result.get("thread", {}).get("id")
        if not isinstance(thread_id, str) or not thread_id or thread_id in self.thread_usage:
            raise CodexError("App-server did not return a unique new thread ID")
        previous = self.thread_id
        self.thread_id = thread_id
        self.thread_usage[thread_id] = {}
        self.metadata.update(thread_id=thread_id, thread_count=len(self.thread_usage))
        if self.service_tier is not None:
            self.metadata.update(service_tier=self.service_tier,
                                 thread_service_tier=result.get("serviceTier"),
                                 service_tier_policy="explicit_per_turn")
        if self.disable_execution_tools:
            self.metadata.update(execution_tool_restrictions="requested_feature_disables_and_event_abort",
                                 builtin_isolation_verified=False)
        if previous is not None:
            self.metadata["context_strategy"] = "fresh_thread_per_observation"
        self._log("thread_started", **self.metadata, thread_index=len(self.thread_usage),
                  previous_thread_id=previous, reason=reason)
        if previous is not None:
            self._log("thread_rotated", thread_id=thread_id, previous_thread_id=previous,
                      thread_index=len(self.thread_usage), reason=reason,
                      context_strategy=self.metadata["context_strategy"], accounting_scope="episode")

    def rotate_thread(self, reason: str = "new_observation") -> dict:
        """Start an empty ephemeral conversation on the existing connection.

        Call once per new observation, never between repair turns for the same
        event. Native totals remain in thread_usage; total_usage is episode
        cumulative. Ambiguous protocol failures close the session.
        """
        if self._closed or not self.thread_id:
            raise CodexError("Call start() before rotate_thread()")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Thread rotation requires a nonempty reason")
        if not self._operation_lock.acquire(blocking=False):
            raise CodexError("Cannot rotate during an unfinished turn or rotation")
        try:
            deadline = time.monotonic() + self.timeout
            account = self._request("account/read", {"refreshToken": False}, deadline)
            if (account.get("account") or {}).get("type") != "chatgpt":
                raise CodexError("An existing ChatGPT login is required; API-key fallback is disabled")
            del account
            self._start_thread(deadline, reason)
            return copy.deepcopy(self.metadata)
        except BaseException:
            self.close()
            raise
        finally:
            self._operation_lock.release()

    def context_capabilities(self) -> dict:
        """Describe explicit interfaces without probing the service or model.

        The method/schema was checked against CLI 0.162.1. This declaration is
        not evidence that the running server or account accepts compaction.
        External action/RPC/STOP/fault boundaries remain the caller's authority.
        """
        return {"protocol": "codex-context-capabilities-v1",
                "native_compact": {"method": "thread/compact/start",
                                   "schema_version": "0.162.1",
                                   "runtime_verified": False},
                "fresh_thread": True, "client_automatic_compaction_requests": False,
                "provider_internal_compaction": "not_controlled_or_verified",
                "physical_boundary_checked": False}

    def compact_context(self, timeout_s: float | None = None) -> dict:
        """Explicitly compact the current thread at a caller-checked boundary.

        The shared lock covers acceptance and the entire compact lifecycle.
        An empty RPC acknowledgement is not completion. No tool is dispatched,
        and failures never retry, rotate, or replay a previous request. This
        lock establishes only model-client exclusivity, not physical safety.
        """
        if self._closed or not self.thread_id:
            raise CodexError("Call start() before compact_context()")
        timeout = self.timeout if timeout_s is None else timeout_s
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Compaction timeout must be finite and positive")
        if not self._operation_lock.acquire(blocking=False):
            raise CodexError("Cannot compact during an unfinished turn or context operation")
        self._active = True
        before = dict(self.total_usage)
        thread_id, turn_id = self.thread_id, None
        usage_seen, usage_complete = False, True
        started_items, completed_items = set(), set()
        errors = []
        try:
            deadline = time.monotonic() + timeout
            self._log("context_compaction_requested", thread_id=thread_id)
            try:
                self._request("thread/compact/start", {"threadId": thread_id}, deadline)
            except _CodexRequestRejected as exc:
                # Only a known absent RPC with no observed new/ambiguous work
                # can return to the same thread. Buffered native counters still
                # belong in the lower bound before any uncertain turn closes.
                unstarted = True
                for frame in self._pending:
                    if "id" in frame:
                        unstarted = False
                    method, params = frame.get("method"), frame.get("params", {})
                    if params.get("threadId") not in (None, thread_id):
                        continue
                    event_turn = (params.get("turn", {}).get("id") if method in {
                        "turn/started", "turn/completed"} else params.get("turnId"))
                    if event_turn in self._known_turn_ids:
                        continue
                    if (method in {"turn/started", "turn/completed", "item/started", "item/completed",
                                   "thread/tokenUsage/updated", "error", "turn/error", "thread/error",
                                   "thread/status/changed"} or event_turn is not None):
                        unstarted = False
                    if method == "thread/tokenUsage/updated" and params.get("threadId") == thread_id:
                        total = params.get("tokenUsage", {}).get("total", {})
                        self._record_usage(total)
                        usage_seen |= bool(total)
                        usage_complete &= _native_usage_complete(total)
                if exc.error.get("code") == -32601 and unstarted:
                    receipt = {"operation": "native_compact", "status": "unsupported",
                               "thread_id": thread_id, "previous_thread_id": thread_id,
                               "turn_id": None, "item_ids": [],
                               "error": copy.deepcopy(exc.error), "provider_operation_started": False,
                               "usage": None, "usage_available": False, "usage_complete": False,
                               "total_usage": dict(self.total_usage),
                               "total_usage_complete": self.total_usage_complete,
                               "thread_total_usage": dict(self.thread_usage[thread_id]),
                               "thread_count": len(self.thread_usage),
                               "accounting_scope": "episode", "fresh_observation_required": False,
                               "requires_explicit_rebuild_choice": True}
                    self._log("context_compaction_unsupported", **receipt)
                    return receipt
                raise
            self._log("context_compaction_accepted", thread_id=thread_id)
            while True:
                frame = self._receive(deadline)
                method, params = frame.get("method"), frame.get("params", {})
                if "id" in frame and method:
                    self._send({"id": frame["id"], "error": {
                        "code": -32601, "message": "Tools are unavailable during compaction"}})
                    raise CodexError("Unexpected interactive request during compaction")
                if params.get("threadId") != thread_id and not (
                        method in {"error", "turn/error", "thread/error"}
                        and params.get("threadId") is None):
                    continue
                event_turn = (params.get("turn", {}).get("id") if method in {
                    "turn/started", "turn/completed"} else params.get("turnId"))
                if event_turn in self._known_turn_ids and event_turn != turn_id:
                    continue
                if method == "turn/started":
                    if not isinstance(event_turn, str) or not event_turn:
                        raise CodexError("Compaction turn has no valid ID")
                    if turn_id is not None and event_turn != turn_id:
                        raise CodexError("Unexpected concurrent compaction turn")
                    turn_id = event_turn
                    self._known_turn_ids.add(turn_id)
                    turn = params.get("turn", {})
                    if "model" in turn and turn["model"] != self.model:
                        raise CodexError("App-server changed requested model during compaction")
                    for field in ("effort", "reasoningEffort"):
                        if field in turn and turn[field] != self.effort:
                            raise CodexError("App-server changed requested reasoning effort during compaction")
                    if "serviceTier" in turn:
                        self._check_service_tier(turn["serviceTier"])
                    for item in turn.get("items", []):
                        if item.get("type") not in {
                                "contextCompaction", "reasoning", "userMessage", "agentMessage"}:
                            raise CodexError("Unexpected tool or builtin item during compaction")
                elif event_turn not in (None, turn_id):
                    raise CodexError("Compaction event has no matching started turn")
                if method in {"thread/tokenUsage/updated", "item/started", "item/completed"} and (
                        turn_id is None or event_turn != turn_id):
                    raise CodexError("Compaction item or usage has no matching started turn")
                if method == "model/rerouted" and params.get("toModel") != self.model:
                    raise CodexError("Model rerouting is forbidden during compaction")
                if method in {"error", "turn/error", "thread/error"}:
                    error = _selected_error(params.get("error", params))
                    errors.append({"method": method, "error": error})
                    self._log("context_compaction_error_notification", thread_id=thread_id,
                              turn_id=turn_id, error=error)
                    raise CodexError("Provider reported a compaction error")
                if method == "thread/tokenUsage/updated":
                    total = params.get("tokenUsage", {}).get("total", {})
                    self._record_usage(total)
                    usage_seen |= bool(total)
                    usage_complete &= _native_usage_complete(total)
                elif method in {"item/started", "item/completed"}:
                    item = params.get("item", {})
                    if item.get("type") not in {
                            "contextCompaction", "reasoning", "userMessage", "agentMessage"}:
                        raise CodexError("Unexpected tool or builtin item during compaction")
                    if item.get("type") == "contextCompaction":
                        item_id = item.get("id")
                        if turn_id is None or not isinstance(item_id, str) or not item_id:
                            raise CodexError("Compaction item has no matching turn or ID")
                        if method == "item/started":
                            started_items.add(item_id)
                        elif item_id in started_items:
                            completed_items.add(item_id)
                        else:
                            raise CodexError("Compaction item completed without a matching start")
                elif method == "turn/completed":
                    if turn_id is None or event_turn != turn_id:
                        raise CodexError("Compaction completed without a matching started turn")
                    turn = params.get("turn", {})
                    for item in turn.get("items", []):
                        if item.get("type") not in {
                                "contextCompaction", "reasoning", "userMessage", "agentMessage"}:
                            raise CodexError("Unexpected tool or builtin item during compaction")
                    if not usage_seen or not usage_complete:
                        self.total_usage_complete = False
                    usage = ({key: value - before.get(key, 0)
                              for key, value in self.total_usage.items()} if usage_seen else None)
                    if turn.get("status") == "failed":
                        raise CodexTurnFailed(
                            thread_id=thread_id, turn_id=turn_id,
                            error=_selected_error(turn.get("error")), error_notifications=errors,
                            usage=usage, usage_complete=bool(usage_seen and usage_complete),
                            retryable=False, attempt=1, tool_activity_seen=False)
                    if turn.get("status") != "completed":
                        raise CodexError("Compaction turn did not complete")
                    if not completed_items or started_items != completed_items:
                        raise CodexError("Compaction turn lacks a completed contextCompaction lifecycle")
                    receipt = {"operation": "native_compact", "status": "completed",
                               "thread_id": thread_id, "previous_thread_id": thread_id,
                               "turn_id": turn_id, "item_ids": sorted(completed_items),
                               "usage": usage, "usage_available": usage_seen,
                               "usage_complete": bool(usage_seen and usage_complete),
                               "total_usage": dict(self.total_usage),
                               "total_usage_complete": self.total_usage_complete,
                               "thread_total_usage": dict(self.thread_usage[thread_id]),
                               "thread_count": len(self.thread_usage), "accounting_scope": "episode",
                               "fresh_observation_required": True}
                    self._log("context_compaction_completed", **receipt)
                    return receipt
        except BaseException as exc:
            if not isinstance(exc, CodexTurnFailed):
                self.total_usage_complete = False
            else:
                exc.details.update(operation="native_compact", total_usage=dict(self.total_usage),
                                   total_usage_complete=self.total_usage_complete)
            self._log("context_compaction_aborted", thread_id=thread_id, turn_id=turn_id,
                      error_type=type(exc).__name__, usage_available=usage_seen,
                      usage_complete=(isinstance(exc, CodexTurnFailed)
                                      and exc.details["usage_complete"]),
                      total_usage=dict(self.total_usage),
                      total_usage_complete=self.total_usage_complete)
            self.close()
            raise
        finally:
            self._active = False
            self._operation_lock.release()

    def _record_usage(self, total: dict) -> None:
        if not isinstance(total, dict) or any(type(v) is not int or v < 0 for v in total.values()):
            raise CodexError("Invalid native thread token usage")
        previous = self.thread_usage[self.thread_id]
        if any(k not in total or total[k] < v for k, v in previous.items()):
            raise CodexError("Native thread token usage decreased or lost fields")
        for key, value in total.items():
            self.total_usage[key] = self.total_usage.get(key, 0) + value - previous.get(key, 0)
        self.thread_usage[self.thread_id] = dict(total)
        if not _native_usage_complete(total):
            self.total_usage_complete = False
        self._log("usage_updated", thread_id=self.thread_id,
                  total_usage=dict(self.total_usage), thread_total_usage=dict(total),
                  total_usage_complete=self.total_usage_complete, accounting_scope="episode")

    @staticmethod
    def _tool_result(result: dict) -> dict:
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            raise CodexError("Tool result requires boolean success")
        if not isinstance(result.get("contentItems"), list):
            raise CodexError("Tool result requires contentItems")
        for item in result["contentItems"]:
            kind = item.get("type")
            field = {"inputText": "text", "inputImage": "imageUrl"}.get(kind)
            if field is None or not isinstance(item.get(field), str):
                raise CodexError("Tool content must be inputText or inputImage")
        return result

    def _call_tool(self, handler, name, arguments, deadline):
        # A deadline may stop the session, but cannot roll back an already
        # executing simulation callback. The owner must stop its executor too.
        completed: queue.Queue = queue.Queue()
        def invoke():
            try:
                completed.put((True, self._tool_result(handler(name, arguments))))
            except BaseException as exc:
                completed.put((False, exc))
        threading.Thread(target=invoke, daemon=True).start()
        try:
            ok, result = completed.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise CodexTimeout("Dynamic tool callback exceeded turn deadline") from exc
        if not ok:
            raise CodexError("Dynamic tool callback failed; session aborted without retry") from result
        return result

    def run_turn(self, text: str, tool_handler: Callable[[str, dict], dict],
                 images: list[str] = ()) -> dict:
        """Run one logical request; opt-in retries never dispatch execution tools.

        The lock includes every attempt and backoff. Only a known terminal failed
        model turn on a healthy connection is retried, never an ambiguous send,
        deadline, EOF, tool event, cancellation or configuration/auth failure.
        """
        if self._closed or not self.thread_id:
            raise CodexError("Call start() before run_turn()")
        if not self._operation_lock.acquire(blocking=False):
            raise CodexError("Concurrent turns or rotations are not supported")
        self._active = True
        before = dict(self.total_usage)
        attempts = []
        try:
            for attempt in range(1, self.inference_retries + 2):
                try:
                    reply = self._run_turn_once(text, tool_handler, images, attempt=attempt)
                except CodexTurnFailed as exc:
                    attempts.append(copy.deepcopy(exc.details))
                    if not exc.retryable or attempt > self.inference_retries:
                        exc.details.update(attempts=copy.deepcopy(attempts), attempt_count=attempt,
                                           failed_turn_ids=[item["turn_id"] for item in attempts],
                                           total_usage=dict(self.total_usage),
                                           total_usage_complete=self.total_usage_complete)
                        exc.details["retry_exhausted"] = bool(exc.retryable and attempt > self.inference_retries)
                        raise
                    delay = max(self.retry_delay_seconds * 2 ** (attempt - 1),
                                _retry_after_seconds([exc.error, exc.error_notifications]))
                    self._log("inference_retry_scheduled", thread_id=self.thread_id,
                              failed_turn_id=exc.turn_id, attempt=attempt,
                              next_attempt=attempt + 1, retry_delay_seconds=delay)
                    time.sleep(delay)
                    continue
                attempts.append({"attempt": attempt, "thread_id": self.thread_id,
                                 "turn_id": reply["turn_id"], "status": "completed",
                                 "usage": copy.deepcopy(reply["usage"]),
                                 "usage_available": reply["usage_available"],
                                 "usage_complete": reply["usage_complete"],
                                 "error_notifications": copy.deepcopy(reply["error_notifications"])})
                observed = any(item["usage_available"] for item in attempts)
                reply.update(usage=({key: value - before.get(key, 0)
                                    for key, value in self.total_usage.items()} if observed else None),
                             usage_available=observed,
                             usage_complete=all(item["usage_complete"] for item in attempts),
                             total_usage_complete=self.total_usage_complete,
                             attempts=attempts, attempt_count=attempt,
                             failed_turn_ids=[item["turn_id"] for item in attempts
                                              if item["status"] == "failed"])
                self._log("logical_turn_completed", thread_id=self.thread_id,
                          turn_id=reply["turn_id"], attempt_count=attempt,
                          failed_turn_ids=reply["failed_turn_ids"], usage=reply["usage"],
                          usage_complete=reply["usage_complete"],
                          total_usage_complete=self.total_usage_complete)
                return reply
        except BaseException as exc:
            self._log("turn_aborted", thread_id=self.thread_id,
                      turn_id=getattr(exc, "turn_id", None) or self._current_turn_id,
                      error_type=type(exc).__name__,
                      usage_available=self._current_turn_usage_available,
                      usage_complete=(isinstance(exc, CodexTurnFailed)
                                      and exc.details["usage_complete"]),
                      total_usage=dict(self.total_usage),
                      total_usage_complete=self.total_usage_complete)
            self.close()
            raise
        finally:
            self._active = False
            self._current_turn_id = None
            self._current_turn_usage_available = False
            self._operation_lock.release()

    def _run_turn_once(self, text: str, tool_handler: Callable[[str, dict], dict],
                       images: list[str], *, attempt: int) -> dict:
        deadline = time.monotonic() + self.timeout
        before = dict(self.total_usage)
        usage_seen = False
        usage_complete = True
        self._current_turn_usage_available = False
        tool_activity_seen = False
        error_notifications = []
        replies: dict = {}
        messages: dict = {}
        turn_id = None
        try:
            inputs = [{"type": "text", "text": text, "text_elements": []}]
            for image in images:
                if image.startswith(("data:", "https://", "http://")):
                    inputs.append({"type": "image", "url": image})
                else:
                    path = Path(image).resolve(strict=True)
                    if not path.is_relative_to(self.cwd):
                        raise ValueError("Local input images must be inside session cwd")
                    inputs.append({"type": "localImage", "path": str(path)})
            params = {
                "threadId": self.thread_id, "input": inputs,
                "model": self.model, "effort": self.effort, "summary": "none",
                "sandboxPolicy": {"type": "workspaceWrite", "writableRoots": [str(self.cwd)],
                                  "networkAccess": False, "excludeTmpdirEnvVar": True,
                                  "excludeSlashTmp": True},
            }
            if self.service_tier is not None:
                params["serviceTierForTurn"] = self.service_tier
            result = self._request("turn/start", params, deadline)
            turn_id = result["turn"]["id"]
            self._current_turn_id = turn_id
            self._known_turn_ids.add(turn_id)
            if "model" in result["turn"] and result["turn"]["model"] != self.model:
                raise CodexError("App-server changed requested model")
            if "effort" in result["turn"] and result["turn"]["effort"] != self.effort:
                raise CodexError("App-server changed requested reasoning effort")
            if "serviceTier" in result["turn"]:
                self._check_service_tier(result["turn"]["serviceTier"])
            for item in result["turn"].get("items", []):
                tool_activity_seen |= item.get("type") in {"dynamicToolCall", "functionCallOutput"}
                self._check_item(item, turn_id)
            self._log("turn_started", thread_id=self.thread_id, turn_id=turn_id, attempt=attempt)
            while True:
                frame = self._receive(deadline)
                method, params = frame.get("method"), frame.get("params", {})
                if "id" in frame and method:
                    tool_activity_seen = True
                    if method != "item/tool/call":
                        self._send({"id": frame["id"], "error": {
                            "code": -32601, "message": "Interactive requests are not supported"}})
                        raise CodexError("Unexpected interactive app-server request")
                    if params.get("threadId") != self.thread_id or params.get("turnId") != turn_id:
                        raise CodexError("Dynamic tool request belongs to another thread or turn")
                    name, args = params.get("tool"), params.get("arguments")
                    if name not in self._tool_names or params.get("namespace") or not isinstance(args, dict):
                        raise CodexError("Unexpected dynamic tool or malformed arguments")
                    call_id = params.get("callId")
                    if not isinstance(call_id, str) or not call_id:
                        raise CodexError("Dynamic tool call has no callId")
                    signature = json.dumps([name, args], sort_keys=True)
                    if call_id in replies:
                        old_signature, output = replies[call_id]
                        if signature != old_signature:
                            raise CodexError("Duplicate callId changed arguments")
                    else:
                        output = self._call_tool(tool_handler, name, args, deadline)
                        replies[call_id] = (signature, output)
                        self._log("tool_completed", thread_id=self.thread_id, turn_id=turn_id, tool=name,
                                  call_id=call_id, success=output["success"])
                    self._send({"id": frame["id"], "result": output})
                    continue
                if params.get("threadId") not in (None, self.thread_id):
                    continue
                if method == "model/rerouted" and params.get("toModel") != self.model:
                    raise CodexError("Model rerouting is forbidden for this experiment")
                if params.get("turnId") not in (None, turn_id):
                    continue
                if method in {"error", "turn/error", "thread/error"}:
                    notification = {"method": method, "error": _selected_error(params.get("error", params)),
                                    "willRetry": params.get("willRetry") if type(params.get("willRetry")) is bool else None,
                                    "observed_at": time.time()}
                    error_notifications.append(notification)
                    self._log("turn_error_notification", thread_id=self.thread_id, turn_id=turn_id,
                              attempt=attempt, **notification)
                    continue
                if method in ("item/started", "item/completed"):
                    tool_activity_seen |= params.get("item", {}).get("type") in {
                        "dynamicToolCall", "functionCallOutput"}
                    self._check_item(params.get("item", {}), turn_id)
                if method == "thread/tokenUsage/updated":
                    total = params.get("tokenUsage", {}).get("total", {})
                    self._record_usage(total)
                    usage_seen |= bool(total)
                    self._current_turn_usage_available = usage_seen
                    usage_complete &= _native_usage_complete(total)
                elif method == "item/completed":
                    item = params.get("item", {})
                    if item.get("type") == "agentMessage":
                        messages[item["id"]] = item
                elif method == "turn/completed":
                    turn = params.get("turn", {})
                    if turn.get("id") != turn_id:
                        continue
                    for item in turn.get("items", []):
                        tool_activity_seen |= item.get("type") in {"dynamicToolCall", "functionCallOutput"}
                        self._check_item(item, turn_id)
                        if item.get("type") == "agentMessage":
                            messages[item["id"]] = item
                    if turn.get("status") == "failed":
                        error = _selected_error(turn.get("error"))
                        usage = ({k: v - before.get(k, 0) for k, v in self.total_usage.items()}
                                 if usage_seen else None)
                        if not usage_seen or not usage_complete:
                            self.total_usage_complete = False
                        failure = CodexTurnFailed(
                            thread_id=self.thread_id, turn_id=turn_id, error=error,
                            error_notifications=error_notifications, usage=usage,
                            usage_complete=bool(usage_seen and usage_complete),
                            retryable=bool(self.inference_retries and not tool_activity_seen and not replies
                                           and not _permanent_failure(error, error_notifications)
                                           and self._proc.poll() is None and not self._closed),
                            attempt=attempt, tool_activity_seen=tool_activity_seen)
                        self._log("turn_failed", **failure.details)
                        raise failure
                    if turn.get("status") != "completed":
                        raise CodexError(f"Codex turn ended with status {turn.get('status')}")
                    break
            final = [m.get("text", "") for m in messages.values()
                     if m.get("phase") == "final_answer"]
            if not final:
                final = [m.get("text", "") for m in messages.values() if m.get("phase") is None]
            reply_text = "\n".join(final)
            usage = ({k: max(0, v - before.get(k, 0)) for k, v in self.total_usage.items()}
                     if usage_seen else None)
            if not usage_seen or not usage_complete:
                self.total_usage_complete = False
            reply = {"text": reply_text, "final_text": reply_text,
                     "thread_id": self.thread_id, "turn_id": turn_id,
                     "model": self.model, "effort": self.effort,
                     "usage": usage, "total_usage": dict(self.total_usage),
                     "usage_available": usage_seen,
                     "usage_complete": bool(usage_seen and usage_complete), "tool_calls": len(replies),
                     "error_notifications": copy.deepcopy(error_notifications),
                     "thread_total_usage": dict(self.thread_usage[self.thread_id]),
                     "thread_count": len(self.thread_usage),
                     "context_strategy": self.metadata["context_strategy"],
                     "accounting_scope": "episode"}
            if self.service_tier is not None:
                reply["service_tier"] = self.service_tier
            if self.disable_execution_tools:
                reply["builtin_isolation_verified"] = False
            self._log("turn_completed", thread_id=self.thread_id, turn_id=turn_id, usage=usage,
                      total_usage=self.total_usage, thread_total_usage=reply["thread_total_usage"],
                      accounting_scope="episode", tool_calls=len(replies))
            return reply
        except CodexTurnFailed:
            raise
        except BaseException:
            # A partial snapshot before cancellation/EOF/timeout is a lower
            # bound. Later snapshots must not erase this unknown interval.
            self.total_usage_complete = False
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
            for stream in (self._proc.stdin, self._proc.stdout):
                if stream is not None:
                    stream.close()
        self._log("session_closed", thread_id=self.thread_id,
                  total_usage=dict(self.total_usage),
                  total_usage_complete=self.total_usage_complete)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
