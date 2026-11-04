"""Single-owner storage for evidence-backed Astra knowledge candidates.

This stores proposals and diagnostic execution records, never stable skills or
task-success scores. It is not a sandbox or a cross-process transaction system.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccessDenied
from astra_robotics_agent.physical_keyboard.records.knowledge_domains import validate_knowledge_scope

from astra_robotics_agent.physical_keyboard.records.journal import SegmentedJournal
from astra_robotics_agent.physical_keyboard.records.storage import StorageError

from .read_scope import active_snapshot, assert_policy_writable, assert_writable, read_operation, read_snapshot

import astra_robotics_agent.physical_keyboard.runtime.skill_language as language


class KnowledgeError(ValueError):
    pass


_MAX_FILE = 8 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_CANDIDATE = re.compile(r"(?:lesson|skill)-[A-Za-z0-9_-]{1,64}-[0-9a-f]{64}\Z")
_SCOPE = {"task_id", "device_id", "layout_id"}
_AUTHORITY = {"execute", "safety", "safety_limits", "host", "permissions", "limits",
              "service_tier", "speed_percent", "speed_fraction", "control_hz",
              "workspace_min_xyz", "workspace_max_xyz", "max_tool_calls"}


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KnowledgeError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise KnowledgeError("nonfinite JSON")


def _decoded(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise KnowledgeError("invalid stored JSON") from exc


def _text(value, label, maximum=2048):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise KnowledgeError(f"{label}: nonempty string of at most {maximum} characters required")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise KnowledgeError(f"{label}: invalid Unicode") from exc
    return value


def _identifier(value, label="id"):
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise KnowledgeError(f"{label}: 1..64 ASCII letters, digits, '-' or '_' required")
    return value


def _texts(value, label):
    if type(value) is not list or not 1 <= len(value) <= 16:
        raise KnowledgeError(f"{label}: nonempty list of at most 16 strings required")
    return [_text(item, label) for item in value]


def _read(path, maximum=_MAX_FILE):
    """Reject symlinks in every input component, FIFO/device files, and oversize."""
    path = Path(path)
    if ".." in path.parts:
        raise KnowledgeError("parent traversal is forbidden")
    path = Path(os.path.abspath(path))
    snapshot = active_snapshot()
    if snapshot is not None:
        snapshot.watch(path)
    directory = descriptor = None
    try:
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise KnowledgeError("regular file within byte limit required")
        remaining, chunks = maximum + 1, []
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > maximum or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise KnowledgeError("file oversized or changed while read")
        if snapshot is not None:
            snapshot.watch(path)
            snapshot.stats["file_reads"] = snapshot.stats.get("file_reads", 0) + 1
            snapshot.stats["file_bytes"] = snapshot.stats.get("file_bytes", 0) + len(raw)
        return raw
    except OSError as exc:
        raise KnowledgeError(f"cannot read regular file without following symlinks: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _write_new(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)


def _profile(value):
    if type(value) is not dict:
        raise KnowledgeError("profile must be a finite JSON dictionary")
    count = 0

    def visit(item, depth=0):
        nonlocal count
        count += 1
        if depth > 16 or count > 2048:
            raise KnowledgeError("profile exceeds structural budget")
        if type(item) is dict:
            for key, child in item.items():
                _text(key, "profile key", 128)
                normalized = key.strip().casefold().replace("-", "_")
                if (normalized in _AUTHORITY or normalized.startswith(("max_translation", "max_rotation",
                       "max_step_translation", "max_step_rotation", "disable_protection"))):
                    raise KnowledgeError("profile cannot override host authority: " + key)
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str:
            if len(item) > 4096:
                raise KnowledgeError("profile string too long")
        elif type(item) is float:
            if not math.isfinite(item):
                raise KnowledgeError("profile must be finite")
        elif type(item) is int:
            if item.bit_length() > 128:
                raise KnowledgeError("profile integer too large")
        elif type(item) not in (bool, type(None)):
            raise KnowledgeError("profile must contain only JSON values")
    visit(value)
    try:
        raw = _encoded(value)
    except (ValueError, UnicodeError) as exc:
        raise KnowledgeError("invalid profile JSON") from exc
    if len(raw) > 32768:
        raise KnowledgeError("profile exceeds 32 KiB")
    return raw


def _validate_source(source, *, demo_tools=False):
    """Use the interpreter's AST validator without invoking its interpreter.

    There is no public validation-only function in skill_language. These are
    its existing node, top-level and entry checks, including unreachable code.
    """
    limits = language.Limits()
    if len(source.encode("utf-8")) > limits.max_source_bytes:
        raise KnowledgeError("source exceeds interpreter byte budget")
    try:
        tree = ast.parse(source, filename="candidate.py")
        if sum(1 for _ in ast.walk(tree)) > limits.max_ast_nodes:
            raise KnowledgeError("source exceeds AST node budget")
        callables = language._BUILTINS | {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        language._validate(tree, limits, callables, demo_tools=demo_tools)
        names = set()
        main = None
        for statement in tree.body:
            if isinstance(statement, ast.FunctionDef):
                name = statement.name
                if name == "main":
                    main = statement
            elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                  and isinstance(statement.targets[0], ast.Name)
                  and language._constant_expression(statement.value)):
                name = statement.targets[0].id
            else:
                raise KnowledgeError("top level permits only functions and literal constants")
            if name in names or name in language._RESERVED or name in {"robot", "profile"}:
                raise KnowledgeError("duplicate or reserved module name")
            names.add(name)
        if main is None or len(main.args.args) != 2:
            raise KnowledgeError("main must have exactly two parameters")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise KnowledgeError("skill failed static validation: " + str(exc)) from exc


class KnowledgeStore:
    def __init__(self, root: Path, scope: dict, run_id: str, *, demo_tools=False, learning_access=None):
        if type(demo_tools) is not bool:
            raise KnowledgeError('demo_tools must be an explicit boolean')
        self.demo_tools = demo_tools
        if type(scope) is not dict or set(scope) != _SCOPE:
            raise KnowledgeError("exact task/device/layout scope required")
        self._scope = {key: _text(scope[key], key, 128) for key in sorted(_SCOPE)}
        self.learning_access = learning_access
        self.run_id = _text(run_id, "run_id", 128)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.namespace = self.root / "scopes" / _sha(_encoded(self._scope))
        for name in ("objects", "evidence", "candidates", "verifications"):
            (self.namespace / name).mkdir(parents=True, exist_ok=True)

    def read_snapshot(self):
        return read_snapshot(self, KnowledgeError)

    @property
    def learning_access(self):
        return self._learning_access

    @learning_access.setter
    def learning_access(self, value):
        if value is not None and value.scope != self._scope:
            raise KnowledgeError('learning access scope mismatch')
        if getattr(self, '_learning_access', None) is not value:
            assert_policy_writable(self)
        self._learning_access = value

    def _check_knowledge_access(self, identifier, record):
        if self.learning_access is None:
            return []
        access = self.learning_access
        if 'source_ids' in record:
            access.bind_knowledge(identifier, access.check_sources(record['source_ids'], require_provenance=True))
        return access.knowledge_sources(identifier)

    def _evidence_source_ids(self, evidence):
        if self.learning_access is None:
            return []
        sources = set()
        for item in evidence:
            values = self.learning_access.evidence_sources(item['evidence_id'])
            sources.update(self.learning_access.check_sources(values, require_provenance=True))
        return sorted(sources)

    def _append(self, event):
        assert_writable(self)
        try:
            SegmentedJournal(self.namespace / "index.jsonl").append(_encoded(event))
        except StorageError as exc:
            raise KnowledgeError(str(exc)) from exc

    @read_operation
    def _events(self):
        return active_snapshot(self).cached("events", str(self.namespace), self._load_events)

    def _load_events(self):
        path = self.namespace / "index.jsonl"
        if not path.exists() and not path.is_symlink() and not path.with_name(path.stem + ".segments").exists():
            return []
        try:
            events = [_decoded(line) for line in SegmentedJournal(path).lines()]
        except StorageError as exc:
            raise KnowledgeError(str(exc)) from exc
        for event in events:
            if type(event) is not dict or event.get("scope") != self._scope:
                raise KnowledgeError("stored index scope mismatch")
        return events

    def _object(self, raw):
        assert_writable(self)
        digest = _sha(raw)
        path = self.namespace / "objects" / digest
        if path.exists():
            if _read(path) != raw:
                raise KnowledgeError("content-addressed object was corrupted")
        else:
            _write_new(path, raw)
        return digest

    def register_evidence(self, evidence_id: str, paths: list[Path], *, source_ids=None) -> dict:
        assert_writable(self)
        _identifier(evidence_id, "evidence_id")
        if type(paths) is not list or not 1 <= len(paths) <= 16:
            raise KnowledgeError("evidence needs 1..16 actual files")
        files = []
        snapshots = []
        for path in paths:
            raw = _read(path)
            snapshots.append(raw)
            files.append({"source_path": str(Path(os.path.abspath(path))), "sha256": _sha(raw), "bytes": len(raw)})
        output = self.namespace / "evidence" / (evidence_id + ".json")
        if output.exists():
            previous = self._evidence(evidence_id)
            if ([(item["sha256"], item["bytes"]) for item in previous["files"]]
                    != [(item["sha256"], item["bytes"]) for item in files]):
                raise KnowledgeError("evidence ID already refers to different contents")
            return previous
        sources = None
        if self.learning_access is not None:
            if source_ids is None:
                # This is a trusted framework API, never a model action.
                sources = self.learning_access.register_own_evidence(evidence_id)
            else:
                sources = self.learning_access.check_sources(source_ids, require_provenance=True)
                self.learning_access.bind_evidence(evidence_id, sources)
        elif source_ids is not None:
            raise KnowledgeError('explicit evidence sources require a learning access policy')
        for raw in snapshots:
            self._object(raw)
        record = {"protocol": "physical-keyboard-registered-evidence-v1", "evidence_id": evidence_id,
                  "scope": self._scope, "run_id": self.run_id, "files": files}
        if sources is not None:
            record['source_ids'] = sources
        raw = _encoded(record)
        _write_new(output, raw)
        self._append({"type": "evidence_registered", "evidence_id": evidence_id,
                      "scope": self._scope, "sha256": _sha(raw)})
        return dict(record, path=str(output), sha256=_sha(raw))

    @read_operation
    def _evidence(self, evidence_id):
        _identifier(evidence_id, "evidence_id")
        snapshot = active_snapshot(self)
        value = snapshot.cached("evidence", evidence_id, lambda: self._load_evidence(evidence_id))
        return deepcopy(value)

    def _load_evidence(self, evidence_id):
        _identifier(evidence_id, "evidence_id")
        events = [item for item in self._events()
                  if item.get("type") == "evidence_registered" and item.get("evidence_id") == evidence_id]
        if len(events) != 1:
            raise KnowledgeError("evidence ID was not registered by the framework")
        path = self.namespace / "evidence" / (evidence_id + ".json")
        raw = _read(path)
        if _sha(raw) != events[0]["sha256"]:
            raise KnowledgeError("evidence record hash mismatch")
        record = _decoded(raw)
        if record.get("scope") != self._scope or record.get("evidence_id") != evidence_id:
            raise KnowledgeError("evidence scope or identity mismatch")
        for item in record["files"]:
            if re.fullmatch("[0-9a-f]{64}", item["sha256"]) is None:
                raise KnowledgeError("invalid evidence object identity")
            def verify_object():
                obj = _read(self.namespace / "objects" / item["sha256"])
                if _sha(obj) != item["sha256"] or len(obj) != item["bytes"]:
                    raise KnowledgeError("evidence object hash mismatch")
                return True
            snapshot = active_snapshot(self)
            if snapshot is None:
                verify_object()
            else:
                snapshot.cached("verified_object", (item["sha256"], item["bytes"]), verify_object)
        if self.learning_access is not None:
            if 'source_ids' in record:
                self.learning_access.bind_evidence(evidence_id, record['source_ids'])
            self.learning_access.evidence_sources(evidence_id)
        return dict(record, path=str(path), sha256=_sha(raw))

    @read_operation
    def _references(self, evidence_ids):
        if type(evidence_ids) is not list or not 1 <= len(evidence_ids) <= 16:
            raise KnowledgeError("nonempty registered evidence_ids required")
        if any(type(value) is not str for value in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
            raise KnowledgeError("evidence IDs must be unique strings")
        return [self._evidence(value) for value in evidence_ids]

    def propose(self, kind: str, args: dict) -> dict:
        assert_writable(self)
        return self._propose(kind, args)

    def revise_skill(self, candidate_id: str, args: dict, *, reason: str) -> dict:
        assert_writable(self)
        """Operator maintenance only; this is not a model learning action.

        Add a clearly attributed version, then supersede prior versions of the
        same skill in default context. Originals and execution evidence remain
        readable. Like the rest of this store, this requires a single owner.
        """
        parent = self._candidate(candidate_id)
        if parent['kind'] != 'skill' or type(args) is not dict or args.get('id') != parent['id']:
            raise KnowledgeError('revision must retain the registered skill identity')
        reason = _text(reason, 'revision reason')
        family = [item for item in self._events() if item.get('type') == 'candidate_proposed'
                  and item.get('kind') == 'skill' and item.get('id') == parent['id']]
        if parent['version'] != max(item['version'] for item in family):
            raise KnowledgeError('operator revision must start from the latest skill version')
        # Validate all old versions before making any replacement active.
        previous = [self._candidate(item['candidate_id'])['candidate_id'] for item in family]
        result = self._propose('skill', args, revision={'parent_candidate_id': candidate_id, 'reason': reason})
        self._append({'type': 'skill_superseded', 'scope': self._scope,
                      'candidate_ids': previous, 'replacement_candidate_id': result['candidate_id'],
                      'origin': 'human_requested_revision', 'reason': reason})
        return result

    def _propose(self, kind: str, args: dict, *, revision=None, source_ids=None) -> dict:
        assert_writable(self)
        if type(kind) is not str or kind not in {"lesson", "skill"} or type(args) is not dict:
            raise KnowledgeError("lesson or skill dictionary required")
        keys = {"id", "applicability", "limitations", "evidence_ids"} | ({"claim"} if kind == "lesson" else {"source_lines", "profile"})
        if not keys <= set(args) or set(args) - keys - {"knowledge_scope"}:
            raise KnowledgeError("exact candidate fields required")
        identifier = _identifier(args["id"])
        data = {"id": identifier, "applicability": _texts(args["applicability"], "applicability"),
                "limitations": _texts(args["limitations"], "limitations"),
                "evidence_ids": list(args["evidence_ids"]) if type(args["evidence_ids"]) is list else args["evidence_ids"]}
        evidence = self._references(args["evidence_ids"])
        if "knowledge_scope" in args:
            data["knowledge_scope"] = validate_knowledge_scope(args["knowledge_scope"])
        if kind == "lesson":
            data["claim"] = _text(args["claim"], "claim", 4096)
        else:
            lines = args["source_lines"]
            if type(lines) is not list or not 1 <= len(lines) <= 1024 or any(type(line) is not str for line in lines):
                raise KnowledgeError("source_lines must contain 1..1024 strings")
            source = "\n".join(lines) + "\n"
            try:
                _validate_source(source, demo_tools=self.demo_tools)
            except UnicodeError as exc:
                raise KnowledgeError("source must be valid Unicode") from exc
            profile = _profile(args["profile"])
            data.update(source=source, profile=_decoded(profile), source_sha256=_sha(source.encode("utf-8")),
                        profile_sha256=_sha(profile))
        payload = {"protocol": "physical-keyboard-knowledge-candidate-v1", "kind": kind,
                   "scope": self._scope, "run_id": self.run_id,
                   "origin": "human_requested_revision" if revision is not None else "astra_generated",
                   "status": "candidate_not_promoted", "source": "astra_proposal", "evidence": evidence, **data}
        if self.learning_access is not None:
            sources = set(self._evidence_source_ids(evidence))
            if source_ids is not None:
                sources.update(self.learning_access.check_sources(source_ids, require_provenance=True))
            if revision is not None:
                parent = self._candidate(revision['parent_candidate_id'])
                sources.update(self.learning_access.check_sources(
                    self.learning_access.knowledge_sources(parent['candidate_id']), require_provenance=True))
            payload['source_ids'] = self.learning_access.check_sources(sorted(sources), require_provenance=True)
        if revision is not None:
            payload['revision'] = revision
        candidate_id = kind + "-" + identifier + "-" + _sha(_encoded(payload))
        if self.learning_access is not None:
            self.learning_access.bind_knowledge(candidate_id, payload['source_ids'])
        directory = self.namespace / "candidates" / candidate_id
        if directory.exists():
            candidate = self._candidate(candidate_id)
            return self._summary(candidate, directory)
        versions = [item["version"] for item in self._events()
                    if item.get("type") == "candidate_proposed" and item.get("kind") == kind and item.get("id") == identifier]
        payload.update(candidate_id=candidate_id, version=max(versions, default=0) + 1)
        files = {"candidate.json": _encoded(payload)}
        if kind == "skill":
            files.update({"source.py": source.encode("utf-8"), "profile.json": profile})
        manifest = {"protocol": "physical-keyboard-candidate-files-v1", "candidate_id": candidate_id,
                    "files": {name: {"sha256": _sha(raw), "bytes": len(raw)} for name, raw in files.items()}}
        manifest_bytes = _encoded(manifest)
        directory.mkdir()
        for name, raw in files.items():
            _write_new(directory / name, raw)
        _write_new(directory / "manifest.json", manifest_bytes)
        self._append({"type": "candidate_proposed", "candidate_id": candidate_id, "id": identifier,
                      "kind": kind, "version": payload["version"], "scope": self._scope,
                      "manifest_sha256": _sha(manifest_bytes)})
        return self._summary(payload, directory)

    @read_operation
    def _candidate(self, candidate_id):
        if type(candidate_id) is not str or _CANDIDATE.fullmatch(candidate_id) is None:
            raise KnowledgeError("invalid candidate ID")
        snapshot = active_snapshot(self)
        value = snapshot.cached("candidate", candidate_id, lambda: self._load_candidate(candidate_id))
        return deepcopy(value)

    def _load_candidate(self, candidate_id):
        if type(candidate_id) is not str or _CANDIDATE.fullmatch(candidate_id) is None:
            raise KnowledgeError("invalid candidate ID")
        matches = [item for item in self._events()
                   if item.get("type") == "candidate_proposed" and item.get("candidate_id") == candidate_id]
        if len(matches) != 1:
            raise KnowledgeError("candidate is not registered in this scope")
        directory = self.namespace / "candidates" / candidate_id
        raw = _read(directory / "manifest.json")
        if _sha(raw) != matches[0]["manifest_sha256"]:
            raise KnowledgeError("candidate manifest hash mismatch")
        manifest = _decoded(raw)
        if manifest.get("candidate_id") != candidate_id or set(manifest["files"]) not in ({"candidate.json"}, {"candidate.json", "source.py", "profile.json"}):
            raise KnowledgeError("unexpected candidate files")
        contents = {}
        for name, expected in manifest["files"].items():
            contents[name] = _read(directory / name)
            if len(contents[name]) != expected["bytes"] or _sha(contents[name]) != expected["sha256"]:
                raise KnowledgeError("candidate file hash mismatch: " + name)
        candidate = _decoded(contents["candidate.json"])
        if candidate.get("scope") != self._scope or candidate.get("candidate_id") != candidate_id:
            raise KnowledgeError("candidate scope or identity mismatch")
        if self.learning_access is not None:
            self._check_knowledge_access(candidate_id, candidate)
            for evidence in candidate.get('evidence', []):
                actual = self._evidence(evidence['evidence_id'])
                if actual['sha256'] != evidence['sha256']:
                    raise KnowledgeError('candidate evidence hash mismatch')
            parent = candidate.get('revision', {}).get('parent_candidate_id')
            if parent is not None:
                self._candidate(parent)
        if "knowledge_scope" in candidate:
            validate_knowledge_scope(candidate["knowledge_scope"])
        if candidate["kind"] == "skill":
            source = candidate["source"]
            if (contents.get("source.py") != source.encode("utf-8")
                    or _sha(contents["source.py"]) != candidate["source_sha256"]
                    or _sha(contents["profile.json"]) != candidate["profile_sha256"]
                    or contents["profile.json"] != _profile(candidate["profile"])):
                raise KnowledgeError("candidate source/profile mismatch")
            # Loading inert prior knowledge must still work when this optional
            # capability is disconnected. Execution has its own active tool
            # whitelist; reading a saved source never grants that authority.
            _validate_source(source, demo_tools=True)
        return candidate

    @read_operation
    def _summary(self, candidate, directory):
        return {"candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
                "id": candidate["id"], "version": candidate["version"],
                "path": str(directory), "sha256": _sha(_read(directory / "candidate.json")),
                "manifest_sha256": _sha(_read(directory / "manifest.json")),
                "status": "candidate_not_promoted"}

    @read_operation
    def get_skill(self, candidate_id) -> dict:
        candidate = self._candidate(candidate_id)
        if candidate["kind"] != "skill":
            raise KnowledgeError("candidate is not a skill")
        return {"source": candidate["source"], "profile": deepcopy(candidate["profile"]),
                "label": "knowledge-" + _sha(candidate_id.encode("ascii"))[:32], "candidate_id": candidate_id}

    def record_verification(self, candidate_id, evidence_ids: list[str], execution_status: str) -> dict:
        assert_writable(self)
        self._candidate(candidate_id)
        evidence = self._references(evidence_ids)
        statuses = {"ok": "verified_execution_only", "blocked": "diagnostic_failed",
                    "error": "diagnostic_failed", "unknown": "verification_unknown"}
        if type(execution_status) is not str or execution_status not in statuses:
            raise KnowledgeError("explicit recognized SDK execution status required")
        record = {"protocol": "physical-keyboard-diagnostic-verification-v1",
                  "verification_id": uuid.uuid4().hex, "candidate_id": candidate_id,
                  "scope": self._scope, "run_id": self.run_id, "evidence": evidence,
                  "execution_status": execution_status, "status": statuses[execution_status],
                  "text_success": "unknown", "stable": False}
        path = self.namespace / "verifications" / (record["verification_id"] + ".json")
        raw = _encoded(record)
        _write_new(path, raw)
        self._append({"type": "diagnostic_verification", "scope": self._scope,
                      "candidate_id": candidate_id, "verification_id": record["verification_id"],
                      "status": record["status"], "sha256": _sha(raw)})
        return dict(record, path=str(path), sha256=_sha(raw))

    @read_operation
    def _verification_visible(self, event):
        if self.learning_access is None:
            return True
        verification_id = event.get('verification_id')
        if type(verification_id) is not str or re.fullmatch('[a-f0-9]{32}', verification_id) is None:
            raise KnowledgeError('invalid verification identity')
        raw = _read(self.namespace / 'verifications' / (verification_id + '.json'))
        if _sha(raw) != event['sha256']:
            raise KnowledgeError('verification hash mismatch')
        record = _decoded(raw)
        try:
            for item in record['evidence']:
                actual = self._evidence(item['evidence_id'])
                if actual['sha256'] != item['sha256']:
                    raise KnowledgeError('verification evidence hash mismatch')
        except LearningAccessDenied:
            return False
        return True

    @read_operation
    def context_text(self, limit=8) -> str:
        if type(limit) is not int or not 1 <= limit <= 32:
            raise KnowledgeError("context limit must be 1..32")
        events = self._events()
        superseded = {identifier for item in events if item.get('type') == 'skill_superseded'
                      for identifier in item['candidate_ids']}
        entries = [item for item in events if item.get("type") == "candidate_proposed"
                   and item['candidate_id'] not in superseded]
        if self.learning_access is not None:
            visible = []
            for item in entries:
                try:
                    self._candidate(item['candidate_id'])
                except LearningAccessDenied:
                    continue
                visible.append(item)
            entries = visible
        entries = entries[-limit:]
        lines = ["KNOWLEDGE CANDIDATES — NOT STABLE",
                 "Each entry states its origin; human_requested_revision is operator maintenance, not autonomous learning.",
                 "Scope-matched active proposals only; do not replay automatically. Execution verification is not visual task success."]
        for item in entries:
            try:
                candidate = self._candidate(item["candidate_id"])
            except LearningAccessDenied:
                continue
            def excerpt(text):
                text = " ".join(text.split())
                return text if len(text) <= 240 else text[:237] + "..."
            verifications = [event["status"] for event in events
                             if event.get("type") == "diagnostic_verification"
                             and event.get("candidate_id") == candidate["candidate_id"]
                             and self._verification_visible(event)]
            summary = {"candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
                                   "origin": candidate["origin"],
                                   "version": candidate["version"], "claim": excerpt(candidate.get("claim", "")),
                                   "applicability": [excerpt(text) for text in candidate["applicability"][:2]],
                                   "limitations": [excerpt(text) for text in candidate["limitations"][:2]],
                                   "last_execution_verification": verifications[-1] if verifications else "none",
                                   "status": "candidate_not_promoted"}
            if candidate['kind'] == 'skill':
                optional = set(language.tool_specs(demo_tools=True)) - set(language.tool_specs())
                required = sorted({node.attr for node in ast.walk(ast.parse(candidate['source']))
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == 'robot' and node.attr in optional})
                if required:
                    summary['optional_tools_required'] = required
                    summary['optional_tools_enabled'] = self.demo_tools
            lines.append(_encoded(summary).decode("utf-8").strip())
        return "\n".join(lines) + "\n"
