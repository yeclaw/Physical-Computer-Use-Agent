"""Public device seam for the program runtime; imports no hardware libraries.

Factories are trusted, operator-configured Python extensions, never code chosen
by the task model. Adapters own device initialization and physical constraints.
``prepare`` must have no physical side effects. Only it may use
``AdmissionRejected`` to certify that an invalid request was not dispatched.
An exception from ``dispatch`` does not establish what the device did.

Frozen dataclasses protect their attributes, not nested dictionaries. The host
must snapshot JSON fields before retaining them as execution evidence. Tokens
are private adapter objects and must not enter model context or evidence JSON.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
import re
from typing import Any, Protocol, runtime_checkable


_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def json_copy(value: Any) -> Any:
    """Return independent finite JSON data without coercing keys or types."""
    def validate(item, depth=0):
        if depth > 32:
            raise ValueError("adapter JSON nesting exceeds 32 levels")
        if type(item) in (type(None), bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("adapter JSON numbers must be finite")
            return
        if type(item) is list:
            for child in item:
                validate(child, depth + 1)
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("adapter JSON object keys must be strings")
            for child in item.values():
                validate(child, depth + 1)
            return
        raise ValueError("adapter values must be JSON data")

    validate(value)
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


@dataclass(frozen=True)
class AdapterDescription:
    id: str
    coordinate_frame: str
    views: tuple[str, ...]
    evidence_mode: str = "physical"

    def __post_init__(self):
        if type(self.id) is not str or not _ID.fullmatch(self.id):
            raise ValueError("adapter id requires 1..64 letters, digits, underscores or hyphens")
        if (type(self.coordinate_frame) is not str or not self.coordinate_frame.strip()
                or len(self.coordinate_frame) > 256 or "\x00" in self.coordinate_frame):
            raise ValueError("adapter coordinate_frame must name the actual command frame")
        if (type(self.views) is not tuple or len(self.views) > 64
                or any(type(view) is not str or not _ID.fullmatch(view) for view in self.views)
                or len(set(self.views)) != len(self.views)):
            raise ValueError("adapter views require unique declared view identifiers")
        if self.evidence_mode not in {"physical", "synthetic"}:
            raise ValueError("adapter evidence_mode must distinguish physical and synthetic evidence")

    def as_dict(self):
        return json_copy(dict(id=self.id, coordinate_frame=self.coordinate_frame,
                              views=list(self.views), evidence_mode=self.evidence_mode))


@dataclass(frozen=True)
class Observation:
    state: dict
    content_items: list[dict]

    def __post_init__(self):
        if type(self.state) is not dict:
            raise ValueError("observation state must be a JSON object")
        if type(self.content_items) is not list or any(type(item) is not dict for item in self.content_items):
            raise ValueError("observation content_items must be a list of JSON objects")
        json_copy(self.state)
        json_copy(self.content_items)


@dataclass(frozen=True)
class PreparedCall:
    method: str
    arguments: dict
    token: Any = None

    def __post_init__(self):
        if type(self.method) is not str or not _ID.fullmatch(self.method):
            raise ValueError("prepared method must be an explicit tool identifier")
        if type(self.arguments) is not dict:
            raise ValueError("prepared arguments must be a JSON object")
        json_copy(self.arguments)


@dataclass(frozen=True)
class CallReceipt:
    status: str
    dispatch_state: str
    result: dict

    def __post_init__(self):
        valid = {("ok", "completed"), ("rejected", "not_started"), ("unknown", "unknown")}
        if type(self.status) is not str or type(self.dispatch_state) is not str or (
                self.status, self.dispatch_state) not in valid:
            raise ValueError("receipt must distinguish completed, rejected-before-dispatch and unknown")
        if type(self.result) is not dict:
            raise ValueError("receipt result must be a JSON object")
        json_copy(self.result)

    def as_dict(self):
        return json_copy(dict(status=self.status, dispatch_state=self.dispatch_state,
                              result=self.result))


class AdmissionRejected(ValueError):
    """Trusted ``prepare`` rejection: this call made no physical side effects."""


@runtime_checkable
class DeviceAdapter(Protocol):
    def describe(self) -> AdapterDescription:
        """Declare identity/frame/views without starting physical execution."""
        ...

    def observe(self) -> Observation:
        """Return current device evidence; it is distinct from a command target."""
        ...

    def prepare(self, method: str, arguments: dict) -> PreparedCall:
        """Validate a complete call without side effects, or raise AdmissionRejected."""
        ...

    def dispatch(self, prepared: PreparedCall) -> CallReceipt:
        """Execute once; exceptions and unknown receipts must never be auto retried."""
        ...

    def stop(self) -> CallReceipt:
        """Attempt device stop; report its outcome without claiming physical release."""
        ...

    def close(self) -> None | dict:
        """Release owned resources; failures remain observable to the host."""
        ...


def load_factory(spec: str):
    """Resolve an explicit ``package.module:function`` trusted extension."""
    if type(spec) is not str or spec.count(":") != 1:
        raise ValueError("factory requires package.module:function")
    module, name = spec.split(":")
    if (not module or any(not part.isidentifier() for part in module.split("."))
            or not name.isidentifier() or name.startswith("_")):
        raise ValueError("factory requires an importable module and public function name")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError("configured factory must be callable")
    return factory


def load_adapter(factory: str, settings: dict) -> DeviceAdapter:
    """Create and validate one explicitly configured adapter; no SDK fallback."""
    if type(settings) is not dict:
        raise ValueError("adapter settings must be a JSON object")
    adapter = load_factory(factory)(json_copy(settings))
    try:
        for name in ("describe", "observe", "prepare", "dispatch", "stop", "close"):
            if not callable(getattr(adapter, name, None)):
                raise TypeError("adapter must implement " + name + "()")
        if not isinstance(adapter.describe(), AdapterDescription):
            raise TypeError("adapter describe() must return AdapterDescription")
    except BaseException as original:
        cleanup = getattr(adapter, "close", None)
        if callable(cleanup):
            try:
                cleanup()
            except BaseException as error:
                original.add_note("adapter cleanup after failed loading also failed: " + repr(error))
        raise
    return adapter
