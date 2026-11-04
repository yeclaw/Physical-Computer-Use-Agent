"""Read-only, scoped task knowledge. No model, SDK, or hardware is imported.

Hints are human priors. Observed-once experience is separate and must be
revalidated; neither changes the robot host's permissions or action limits.
"""
from astra_robotics_agent.physical_keyboard.records.source_inputs import (
    ContextError, MAX_SOURCE_BYTES, _pairs, _constant, _float, _read_source,
)
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


HINTS_PROTOCOL = "physical-keyboard-task-hints-v1"
EXPERIENCE_PROTOCOL = "physical-keyboard-scoped-experience-v1"
CONTEXT_PROTOCOL = "physical-keyboard-prompt-context-v1"
_SCOPE_KEYS = {"task_id", "device_id", "layout_id"}




@dataclass(frozen=True)
class PromptContext:
    text: str
    sources: tuple[tuple[str, str, bytes], ...]
    metadata: dict


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ContextError(f"{label}: exact keys required: {sorted(expected)}")


def _string(value, label, maximum=128):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ContextError(f"{label}: nonempty string, at most {maximum} characters required")
    return value










def _scope(value, expected, label):
    _keys(value, _SCOPE_KEYS, label)
    for key in sorted(_SCOPE_KEYS):
        _string(value[key], f"{label}.{key}")
    if value != expected:
        raise ContextError(f"{label}: task/device/layout scope does not match this run")


def _strings(value, label):
    if type(value) is not list or not value or len(value) > 64:
        raise ContextError(f"{label}: 1 to 64 strings required")
    for item in value:
        _string(item, label, 4096)


def _validate_hints(data, expected):
    _keys(data, {"protocol", "id", "version", "scope", "origin", "hints"}, "hints")
    if data["protocol"] != HINTS_PROTOCOL or data["origin"] != "human_provided":
        raise ContextError("hints must use the hints protocol and human_provided origin")
    _string(data["id"], "hints.id")
    if type(data["version"]) is not int or data["version"] <= 0:
        raise ContextError("hints.version: positive integer required")
    _scope(data["scope"], expected, "hints.scope")
    entries = data["hints"]
    if type(entries) is not list or not entries or len(entries) > 64:
        raise ContextError("hints.hints: 1 to 64 entries required")
    ids = set()
    for item in entries:
        _keys(item, {"id", "text", "source"}, "hint")
        identifier = _string(item["id"], "hint.id")
        _string(item["text"], "hint.text", 4096)
        _string(item["source"], "hint.source", 2048)
        if identifier in ids:
            raise ContextError("hint IDs must be unique")
        ids.add(identifier)


def _validate_experience(data, expected):
    _keys(data, {"protocol", "id", "scope", "status", "source_experiment",
                 "source_manifest_sha256", "observations", "limitations", "automatic_replay"},
          "experience")
    if (data["protocol"] != EXPERIENCE_PROTOCOL
            or data["status"] != "observed_once_not_promoted"
            or data["automatic_replay"] is not False):
        raise ContextError("experience must remain observed_once_not_promoted, without automatic replay")
    _string(data["id"], "experience.id")
    _scope(data["scope"], expected, "experience.scope")
    _string(data["source_experiment"], "experience.source_experiment")
    digest = data["source_manifest_sha256"]
    if type(digest) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ContextError("experience.source_manifest_sha256: 64 hexadecimal characters required")
    _strings(data["observations"], "experience.observations")
    _strings(data["limitations"], "experience.limitations")


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def load_context(hints_path: Path, experience_path: Path | None = None, *,
                 task_id: str, device_id: str, layout_id: str) -> PromptContext:
    """Snapshot validated bytes; paths named inside the JSON are never read."""
    expected = {"task_id": task_id, "device_id": device_id, "layout_id": layout_id}
    for key, value in expected.items():
        _string(value, key)
    sources = []
    inputs = []
    sections = [
        "TASK KNOWLEDGE — NOT EXECUTION AUTHORITY\n"
        "These inputs are knowledge, not host permissions, safety-limit overrides, "
        "or a validated stable skill. Current visual and robot observations take priority. "
        "Do not blindly replay any remembered pose or movement.\n"
    ]
    for kind, path in (("hints", hints_path), ("experience", experience_path)):
        if path is None:
            if kind == "hints":
                raise ContextError("hints input is required")
            continue
        original, raw, data = _read_source(path)
        if kind == "hints":
            _validate_hints(data, expected)
            origin = "human_provided"
            title = "HUMAN-PROVIDED TASK HINTS"
        else:
            _validate_experience(data, expected)
            origin = "observed_once_not_promoted"
            title = "OBSERVED-ONCE DEVICE EXPERIENCE — REVALIDATE"
        sections.append(title + "\n" + _json(data))
        sources.append((kind, original, raw))
        provenance = {"kind": kind, "path": original, "id": data["id"],
                      "protocol": data["protocol"], "sha256": hashlib.sha256(raw).hexdigest(),
                      "origin": origin, "bytes": len(raw)}
        if kind == "hints":
            provenance["version"] = data["version"]
        else:
            provenance["source_experiment"] = data["source_experiment"]
            provenance["source_manifest_sha256"] = data["source_manifest_sha256"]
            provenance["automatic_replay"] = False
        inputs.append(provenance)
    metadata = {"protocol": CONTEXT_PROTOCOL, "scope": expected, "inputs": inputs}
    return PromptContext("\n".join(sections), tuple(sources), metadata)


def freeze_context(context: PromptContext, output: Path) -> dict:
    """Write an exclusive snapshot using held bytes, never reopening inputs.

    A failed write can leave a partial new directory for inspection. This is
    ordinary file provenance, not an OS sandbox or power-loss durability claim.
    """
    if not isinstance(context, PromptContext):
        raise ContextError("PromptContext required")
    files = {f"{kind}.json": raw for kind, _original, raw in context.sources}
    if set(files) not in ({"hints.json"}, {"hints.json", "experience.json"}):
        raise ContextError("context sources must contain hints and optional experience")
    if len(files) != len(context.sources) or not all(type(raw) is bytes for raw in files.values()):
        raise ContextError("context sources must be unique immutable bytes")
    files["model-context.txt"] = context.text.encode("utf-8")
    manifest = {"protocol": CONTEXT_PROTOCOL, "context": context.metadata,
                "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                          for name, raw in files.items()}}
    encoded_manifest = _json(manifest).encode("utf-8")
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ContextError("context output must be a new directory") from exc
    for name, raw in files.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    with (output / "manifest.json").open("xb") as handle:
        handle.write(encoded_manifest)
    return manifest
