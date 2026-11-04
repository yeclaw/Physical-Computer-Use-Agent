"""Scoped, evidence-backed skill development chosen by the RSI model.

The library stores model declarations and model task reports. It neither picks
an experiment nor promotes a skill. Runtime still receives only a frozen
source/profile/label request through the existing guarded executor.
"""
import base64
import json
import os
from pathlib import Path
import re
import stat
from astra_robotics_agent.physical_keyboard.records.journal import SegmentedJournal
from astra_robotics_agent.physical_keyboard.records.storage import StorageError
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccessDenied
from astra_robotics_agent.physical_keyboard.records.knowledge_domains import (
    knowledge_scope, validate_knowledge_scope, matches_knowledge_scope,
    validate_domain_filter, validate_workflow_steps,
)

from astra_robotics_agent.physical_keyboard.skills.knowledge import (
    KnowledgeError, _decoded, _encoded, _identifier, _profile, _read, _sha,
    _text, _texts, _validate_source, _write_new,
)


from .read_scope import active_snapshot, assert_writable, read_operation


PROTOCOL = 'physical-keyboard-skill-development-v1'
_KINDS = {'package', 'improvement', 'outcome', 'workflow'}
_SEARCH_KINDS = _KINDS | {'all', 'skill', 'lesson'}
_RECORD_ID = re.compile(r'(package|improvement|outcome|workflow)-[A-Za-z0-9_-]{1,64}-[a-f0-9]{64}\Z')
_MAX_RECORD = 512 * 1024


def _copy(value):
    return _decoded(_encoded(value))


def _exact(value, required, optional=()):
    if type(value) is not dict or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise KnowledgeError('exact development fields required: ' + ','.join(sorted(required)))


def _list(value, label, *, minimum=0, maximum=16):
    if (type(value) is not list or not minimum <= len(value) <= maximum
            or any(type(item) is not str for item in value) or len(set(value)) != len(value)):
        raise KnowledgeError(label + ': a bounded unique list of strings is required')
    return [_text(item, label, 256) for item in value]


def _excerpt(value, maximum=240):
    value = ' '.join(str(value).split())
    return value if len(value) <= maximum else value[:maximum - 3] + '...'


def _evidence_json_text(raw, pointer):
    """Select one RFC 6901 value, then serialize deterministic finite JSON.

    The pointer is a JSON string, not its URI-fragment representation. Array
    tokens follow RFC 6901's non-negative, no-leading-zero index syntax.
    """
    if (type(pointer) is not str or len(pointer) > 1024
            or (pointer and not pointer.startswith('/'))):
        raise KnowledgeError('json_pointer must be an RFC 6901 string of at most 1024 characters')
    try:
        pointer.encode('utf-8')
    except UnicodeError as exc:
        raise KnowledgeError('json_pointer must be valid Unicode') from exc
    tokens = pointer.split('/')[1:] if pointer else []
    if len(tokens) > 16 or any(re.search(r'~(?![01])', token) for token in tokens):
        raise KnowledgeError('json_pointer exceeds 16 tokens or contains an invalid escape')
    value = _decoded(raw)
    for token in tokens:
        token = token.replace('~1', '/').replace('~0', '~')
        if type(value) is dict:
            if token not in value:
                raise KnowledgeError('json_pointer object member does not exist')
            value = value[token]
        elif type(value) is list:
            if re.fullmatch(r'0|[1-9][0-9]*', token) is None or int(token) >= len(value):
                raise KnowledgeError('json_pointer array index is invalid or out of range')
            value = value[int(token)]
        else:
            raise KnowledgeError('json_pointer cannot descend through a scalar')
    try:
        return _encoded(value).decode('utf-8').removesuffix('\n')
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise KnowledgeError('selected evidence value cannot be serialized as finite JSON') from exc


def _check_directory(path):
    """Do not introduce a new symlink-following write route."""
    path = Path(os.path.abspath(path))
    for part in reversed((path,) + tuple(path.parents)):
        if part.exists() or part.is_symlink():
            mode = part.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise KnowledgeError('development directory must not follow symlinks')


class SkillDevelopment:
    """Single-owner append-only extension of an existing KnowledgeStore.

    Old candidates and their formats are untouched. A new package also appends
    an ordinary v1 candidate containing the resolved executable program, so
    existing verification and completion lookup remain usable.
    """

    def __init__(self, store):
        self.store = store
        self._scope = _copy(store._scope)
        self.root = store.namespace / 'development'
        _check_directory(self.root)
        self.root.mkdir(exist_ok=True)
        self.objects = self.root / 'objects'
        _check_directory(self.objects)
        self.objects.mkdir(exist_ok=True)
        self.index = self.root / 'index.jsonl'

    @property
    def scope(self):
        return _copy(self._scope)

    def _load_records(self):
        if (not self.index.exists() and not self.index.is_symlink()
                and not self.index.with_name(self.index.stem + ".segments").exists()):
            return []
        try:
            rows = [_decoded(line) for line in SegmentedJournal(self.index).lines()]
        except StorageError as exc:
            raise KnowledgeError(str(exc)) from exc
        records, identifiers, versions = [], set(), {}
        for row in rows:
            _exact(row, {'type', 'scope', 'record_id', 'kind', 'id', 'version', 'sha256', 'bytes'})
            if (row['type'] != 'development_record' or row['scope'] != self.scope
                    or type(row['kind']) is not str or row['kind'] not in _KINDS):
                raise KnowledgeError('development index scope or kind mismatch')
            identifier = row['record_id']
            if type(identifier) is not str or _RECORD_ID.fullmatch(identifier) is None or identifier in identifiers:
                raise KnowledgeError('invalid or duplicate development registration')
            _identifier(row['id'])
            if (type(row['version']) is not int or row['version'] < 1
                    or type(row['sha256']) is not str or re.fullmatch(r'[a-f0-9]{64}', row['sha256']) is None
                    or type(row['bytes']) is not int or not 0 < row['bytes'] <= _MAX_RECORD):
                raise KnowledgeError('invalid development index metadata')
            family = (row['kind'], row['id'])
            if row['version'] != versions.get(family, 0) + 1:
                raise KnowledgeError('development version sequence mismatch')
            raw = _read(self.objects / (row['sha256'] + '.json'), maximum=_MAX_RECORD)
            if len(raw) != row['bytes'] or _sha(raw) != row['sha256']:
                raise KnowledgeError('development object hash mismatch')
            record = _decoded(raw)
            if (type(record) is not dict or record.get('protocol') != PROTOCOL or record.get('scope') != self.scope
                    or any(record.get(key) != row[key] for key in ('record_id', 'kind', 'id', 'version'))
                    or record.get('origin') != 'astra_generated'
                    or record.get('stable') is not False):
                raise KnowledgeError('development object identity or scope mismatch')
            seed = dict(record)
            seed.pop('record_id')
            expected = row['kind'] + '-' + row['id'] + '-' + _sha(_encoded(seed))
            if identifier != expected:
                raise KnowledgeError('development content identity mismatch')
            identifiers.add(identifier)
            versions[family] = row['version']
            records.append(record)
        return records

    @read_operation
    def _records(self, *, include_denied=False):
        snapshot = active_snapshot(self.store)
        if snapshot is None:
            records = self._load_records()
        else:
            records = snapshot.cached("development_index", str(self.index), self._load_records)
        if include_denied:
            return records
        guard = getattr(self, "publication_guard", None)
        key = (str(self.index), id(guard))
        if snapshot is not None:
            return snapshot.cached("development_visible", key, lambda: self._visible_records(records))
        return self._visible_records(records)

    def _visible_records(self, records):
        guard = getattr(self, "publication_guard", None)
        if guard is not None:
            records = [record for record in records if guard(record["record_id"])]
        if self.store.learning_access is None:
            return records
        lookup = {item['record_id']: item for item in records}
        visible = []
        for record in records:
            try:
                self._check_record_access(record, lookup)
            except LearningAccessDenied:
                continue
            visible.append(record)
        return visible

    @read_operation
    def _check_record_access(self, record, lookup=None, visiting=None):
        if self.store.learning_access is None:
            return []
        if lookup is None:
            lookup = self._record_lookup()
        value = record['record_id']
        if visiting is not None and value in visiting:
            raise KnowledgeError('cyclic learning dependency closure')
        snapshot = active_snapshot(self.store)
        if snapshot is None:
            return self._load_record_access(record, lookup, visiting)
        # Keep the lookup alive: an id-only key must never alias a later map
        # after an earlier filtered lookup has been garbage-collected.
        snapshot.memo.setdefault(('lookup_identity', id(lookup)), lookup)
        return list(snapshot.cached("closure", (str(self.index), id(lookup), value),
            lambda: self._load_record_access(record, lookup, visiting)))

    @read_operation
    def _record_lookup(self):
        snapshot = active_snapshot(self.store)
        def collect():
            return {item['record_id']: item for item in self._records(include_denied=True)}
        if snapshot is None:
            return collect()
        return snapshot.cached("development_lookup", str(self.index), collect)

    def _load_record_access(self, record, lookup=None, visiting=None):
        """Check every pinned edge, independently of a summary's own source ID."""
        access = self.store.learning_access
        if access is None:
            return []
        lookup = ({item['record_id']: item for item in self._records(include_denied=True)}
                  if lookup is None else lookup)
        visiting = set() if visiting is None else visiting
        identifier = record['record_id']
        if identifier in visiting:
            raise KnowledgeError('cyclic learning dependency closure')
        visiting = visiting | {identifier}
        sources = set(self.store._check_knowledge_access(identifier, record))
        for evidence_id in record.get('evidence_ids', []):
            actual = self.store._evidence(evidence_id)
            sources.update(access.evidence_sources(evidence_id))
            embedded = [item for item in record.get('evidence', []) if item['evidence_id'] == evidence_id]
            if embedded and embedded[0]['sha256'] != actual['sha256']:
                raise KnowledgeError('development evidence hash mismatch')
        references = list(record.get('dependencies', {}).values())
        references += record.get('skill_ids', []) + record.get('improvement_ids', [])
        for name in ('candidate_id', 'parent_id', 'merged_into'):
            if record.get(name) is not None:
                references.append(record[name])
        references += [pin['candidate_id'] for pin in record.get('dependency_pins', [])]
        references += record.get('lesson_ids', []) + record.get('resource_ids', []) + record.get('knowledge_ids', [])
        for reference in dict.fromkeys(references):
            if reference in lookup:
                sources.update(self._check_record_access(lookup[reference], lookup, visiting))
            elif reference.startswith(('skill-', 'lesson-')):
                self.store._candidate(reference)
                sources.update(access.knowledge_sources(reference))
            elif reference.startswith(('package-', 'improvement-', 'outcome-', 'workflow-')):
                raise KnowledgeError('missing registered development dependency')
            else:
                # New resource readers have their own integrity checks; an
                # explicit unbound reference still cannot evade source policy.
                sources.update(access.knowledge_sources(reference))
        access.check_sources(sorted(sources))
        return sorted(sources)

    def _field_source_ids(self, fields, extra=None):
        """Derive sources mechanically before a new immutable record exists."""
        access = self.store.learning_access
        if access is None:
            return []
        sources = set()
        if extra is not None:
            sources.update(access.check_sources(extra, require_provenance=True))
        for evidence_id in fields.get('evidence_ids', []):
            self.store._evidence(evidence_id)
            sources.update(access.check_sources(access.evidence_sources(evidence_id), require_provenance=True))
        references = list(fields.get('dependencies', {}).values())
        references += fields.get('skill_ids', []) + fields.get('improvement_ids', [])
        references += fields.get('lesson_ids', []) + fields.get('resource_ids', []) + fields.get('knowledge_ids', [])
        references += [pin['candidate_id'] for pin in fields.get('dependency_pins', [])]
        for name in ('candidate_id', 'parent_id', 'merged_into'):
            if fields.get(name) is not None:
                references.append(fields[name])
        for reference in dict.fromkeys(references):
            if reference.startswith(('skill-', 'lesson-')):
                self.store._candidate(reference)
                sources.update(access.check_sources(access.knowledge_sources(reference), require_provenance=True))
            elif reference.startswith(('package-', 'improvement-', 'outcome-', 'workflow-')):
                sources.update(access.check_sources(self._check_record_access(self._record(reference)), require_provenance=True))
            else:
                sources.update(access.check_sources(access.knowledge_sources(reference), require_provenance=True))
        return access.check_sources(sorted(sources), require_provenance=True)

    @read_operation
    def _record(self, identifier):
        if type(identifier) is not str or _RECORD_ID.fullmatch(identifier) is None:
            raise KnowledgeError('invalid development record ID')
        lookup = self._record_lookup()
        if identifier not in lookup:
            raise KnowledgeError('development record is not registered in this scope')
        record = lookup[identifier]
        self._check_record_access(record, lookup)
        return record

    def _append(self, kind, identifier, fields, *, source_ids=None):
        assert_writable(self.store)
        identifier = _identifier(identifier)
        previous = [item for item in self._records(include_denied=True) if item['kind'] == kind and item['id'] == identifier]
        if self.store.learning_access is not None:
            fields = dict(fields, source_ids=self._field_source_ids(fields, source_ids))
        record = dict(protocol=PROTOCOL, kind=kind, id=identifier,
                      version=max((item['version'] for item in previous), default=0) + 1,
                      scope=self.scope, run_id=self.store.run_id, origin='astra_generated',
                      stable=False, **fields)
        record['record_id'] = kind + '-' + identifier + '-' + _sha(_encoded(record))
        if self.store.learning_access is not None:
            self.store.learning_access.bind_knowledge(record['record_id'], record['source_ids'])
        raw = _encoded(record)
        if len(raw) > _MAX_RECORD:
            raise KnowledgeError('development record exceeds byte limit')
        digest = _sha(raw)
        target = self.objects / (digest + '.json')
        if target.exists() or target.is_symlink():
            if _read(target, maximum=_MAX_RECORD) != raw:
                raise KnowledgeError('development content-addressed object was corrupted')
        else:
            _write_new(target, raw)
        row = dict(type='development_record', scope=self.scope, record_id=record['record_id'],
                   kind=kind, id=identifier, version=record['version'], sha256=digest, bytes=len(raw))
        try:
            SegmentedJournal(self.index).append(_encoded(row))
        except StorageError as exc:
            raise KnowledgeError(str(exc)) from exc
        return record

    @read_operation
    def _resolve_skill(self, identifier):
        if type(identifier) is not str:
            raise KnowledgeError('skill reference must be a registered ID')
        if identifier.startswith('package-'):
            record = self._record(identifier)
            if record['kind'] != 'package':
                raise KnowledgeError('package reference must name a skill package')
            candidate = self.store._candidate(record['candidate_id'])
            if (candidate['kind'] != 'skill'
                    or candidate['source_sha256'] != record['compiled_source_sha256']
                    or candidate['profile_sha256'] != record['compiled_profile_sha256']):
                raise KnowledgeError('package executable candidate differs from frozen pin')
            for pin in record['dependency_pins']:
                dependency = self.store._candidate(pin['candidate_id'])
                if (dependency['kind'] != 'skill' or dependency['source_sha256'] != pin['source_sha256']
                        or dependency['profile_sha256'] != pin['profile_sha256']):
                    raise KnowledgeError('package dependency pin mismatch')
            return candidate, record
        candidate = self.store._candidate(identifier)
        if candidate['kind'] != 'skill':
            raise KnowledgeError('skill reference names another knowledge kind')
        return candidate, None

    def _skill_references(self, identifiers, *, minimum=0):
        identifiers = _list(identifiers, 'skill_ids', minimum=minimum)
        for identifier in identifiers:
            self._resolve_skill(identifier)
        return identifiers

    def _knowledge_metadata(self, args):
        fields = {}
        if 'knowledge_scope' in args:
            fields['knowledge_scope'] = validate_knowledge_scope(args['knowledge_scope'])
        if 'knowledge_ids' in args:
            values = _list(args['knowledge_ids'], 'knowledge_ids')
            for identifier in values:
                record = self.inspect(identifier)
                if record['kind'] not in {'skill', 'package', 'workflow', 'lesson'}:
                    raise KnowledgeError('knowledge_ids must name skill, workflow or lesson records')
            fields['knowledge_ids'] = values
        return fields

    def save_application_skill(self, args, *, source_ids=None):
        """Save an inert, versioned recipe interpreted by the same RSI model.

        Steps may reference exact robot programs but have no executable source.
        The agent owns observation, semantic assessment and stage progression.
        """
        assert_writable(self.store)
        required = {'id', 'application_id', 'description', 'applicability', 'limitations',
                    'evidence_ids', 'steps'}
        _exact(args, required, {'parent_id', 'improvement_ids'})
        self._records()
        logical_id = _identifier(args['id'])
        application_id = _text(args['application_id'], 'application_id', 128)
        steps = validate_workflow_steps(args['steps'])
        skill_ids = self._skill_references(list(dict.fromkeys(value for step in steps for value in step['skill_ids'])))
        pins = []
        for reference in skill_ids:
            candidate, _ = self._resolve_skill(reference)
            pins.append(dict(reference_id=reference, **{key: candidate[key] for key in
                ('candidate_id', 'source_sha256', 'profile_sha256')}))
        parent_id = args.get('parent_id')
        if parent_id is not None:
            parent = self.inspect(parent_id)
            if (parent['kind'] != 'workflow' or parent['id'] != logical_id
                    or parent['application_id'] != application_id):
                raise KnowledgeError('workflow parent must retain logical and application identity')
        improvements = _list(args.get('improvement_ids', []), 'improvement_ids')
        for identifier in improvements:
            if self._record(identifier)['kind'] != 'improvement':
                raise KnowledgeError('improvement_ids must name registered improvement records')
        evidence = self.store._references(args['evidence_ids'])
        record = self._append('workflow', logical_id, dict(
            application_id=application_id, knowledge_scope=dict(domain='application', application_id=application_id),
            description=_text(args['description'], 'description'), steps=steps,
            applicability=_texts(args['applicability'], 'applicability'),
            limitations=_texts(args['limitations'], 'limitations'),
            evidence_ids=list(args['evidence_ids']), evidence=evidence,
            skill_ids=skill_ids, skill_pins=pins, parent_id=parent_id,
            improvement_ids=improvements, status='candidate_not_promoted',
            execution='agent_interpreted_workflow_not_robot_program'), source_ids=source_ids)
        return self._summary(record)

    def _summary(self, record):
        if record.get('protocol') != PROTOCOL:
            return {'identifier': record['candidate_id'], 'candidate_id': record['candidate_id'],
                    'kind': record['kind'], 'id': record['id'], 'version': record['version'],
                    'claim': _excerpt(record.get('claim', '')),
                    'applicability': [_excerpt(item) for item in record['applicability'][:2]],
                    'limitations': [_excerpt(item) for item in record['limitations'][:2]],
                    'origin': record['origin'], 'status': record['status'],
                    'superseded': record.get('superseded', False),
                    'evidence_ids': [item['evidence_id'] for item in record['evidence'][:2]],
                    'knowledge_scope': knowledge_scope(record)}
        result = {key: record[key] for key in ('record_id', 'kind', 'id', 'version', 'origin', 'stable')}
        result['identifier'] = record['record_id']
        result['knowledge_scope'] = knowledge_scope(record)
        if 'knowledge_ids' in record:
            result['knowledge_ids'] = record['knowledge_ids'][:8]
            result['knowledge_count'] = len(record['knowledge_ids'])
        if record['kind'] == 'package':
            result.update(package_id=record['record_id'], candidate_id=record['candidate_id'],
                          description=_excerpt(record['contract']['description']),
                          preconditions=[_excerpt(item) for item in record['contract']['preconditions'][:2]],
                          postconditions=[_excerpt(item) for item in record['contract']['postconditions'][:2]],
                          status='candidate_not_promoted', parent_id=record.get('parent_id'))
        elif record['kind'] == 'workflow':
            result.update(application_id=record['application_id'], description=_excerpt(record['description']),
                applicability=record['applicability'][:2], limitations=record['limitations'][:2],
                evidence_ids=record['evidence_ids'][:2], step_count=len(record['steps']),
                skill_ids=record['skill_ids'][:4], parent_id=record.get('parent_id'),
                execution='agent_interpreted_workflow_not_robot_program', status='candidate_not_promoted')
        else:
            result.update(basis='model_report', summary=_excerpt(record['summary']),
                          skill_ids=record['skill_ids'][:4], skill_count=len(record['skill_ids']),
                          evidence_ids=record['evidence_ids'][:2])
            if record['kind'] == 'improvement':
                result.update(status=record['status'], rationale=_excerpt(record['rationale']),
                              priority=record.get('priority', 50),
                              revisit_when=record.get('revisit_when', [])[:2])
                if record.get('merged_into') is not None:
                    result['merged_into'] = record['merged_into']
            else:
                result.update(task_result=record['task_result'], correction_performed=record['correction_performed'],
                              remaining_issues=[_excerpt(item) for item in record['remaining_issues'][:2]])
        return result

    @read_operation
    def search(self, query='', kind='all', limit=8, *, domain=None, application_id=None):
        if type(query) is not str or len(query) > 512 or type(kind) is not str or kind not in _SEARCH_KINDS:
            raise KnowledgeError('bounded query and recognized knowledge kind required')
        try:
            query.encode('utf-8')
        except UnicodeError as exc:
            raise KnowledgeError('query must be valid Unicode') from exc
        if type(limit) is not int or not 1 <= limit <= 32:
            raise KnowledgeError('search limit must be 1..32')
        validate_domain_filter(domain, application_id)
        records = []
        legacy_events = self.store._events()
        superseded = {identifier for event in legacy_events if event.get('type') == 'skill_superseded'
                      for identifier in event['candidate_ids']}
        for event in legacy_events:
            if event.get('type') == 'candidate_proposed':
                try:
                    records.append(dict(self.store._candidate(event['candidate_id']),
                                        superseded=event['candidate_id'] in superseded))
                except LearningAccessDenied:
                    continue
        records.extend(self._records())
        tokens = re.findall(r'[a-z0-9]+|[\u3400-\u9fff]+', query.casefold())
        ranked = []
        for order, record in enumerate(records):
            if kind != 'all' and record['kind'] != kind:
                continue
            if not matches_knowledge_scope(record, domain, application_id):
                continue
            contract = record.get('contract', {})
            text = ' '.join(str(record.get(field, '')) for field in
                            ('id', 'claim', 'summary', 'rationale', 'description', 'application_id', 'applicability', 'limitations',
                             'status', 'task_result', 'remaining_issues'))
            text += ' ' + ' '.join(str(contract.get(field, '')) for field in
                                   ('description', 'preconditions', 'postconditions', 'side_effects', 'input_schema'))
            text = text.casefold()
            score = sum(token in text for token in tokens)
            if tokens and not score:
                continue
            ranked.append((score, order, record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return {'query': query, 'kind': kind, 'results': [self._summary(row[2]) for row in ranked[:limit]],
                'matched_count': len(ranked), 'scope': self.scope, 'ranking': 'lexical_then_registration_order'}

    @read_operation
    def inspect(self, identifier):
        if type(identifier) is not str:
            raise KnowledgeError('registered knowledge ID required')
        if identifier.startswith(('skill-', 'lesson-')):
            superseded = {value for event in self.store._events() if event.get('type') == 'skill_superseded'
                          for value in event['candidate_ids']}
            return _copy(dict(self.store._candidate(identifier), superseded=identifier in superseded))
        record = self._record(identifier)
        if record['kind'] == 'package':
            candidate, _ = self._resolve_skill(identifier)
            record = dict(record, package_id=identifier, source=candidate['source'], profile=candidate['profile'])
        if record['kind'] == 'workflow':
            validate_workflow_steps(record['steps'])
            if knowledge_scope(record) != {'domain': 'application', 'application_id': record['application_id']}:
                raise KnowledgeError('application workflow knowledge scope mismatch')
            for pin in record['skill_pins']:
                candidate, _ = self._resolve_skill(pin['reference_id'])
                if any(candidate[key] != pin[key] for key in ('candidate_id', 'source_sha256', 'profile_sha256')):
                    raise KnowledgeError('application workflow executable pin mismatch')
        return _copy(record)

    def save_package(self, args, *, source_ids=None):
        assert_writable(self.store)
        from astra_robotics_agent.physical_keyboard.skills.programs import compile_program, validate_contract
        required = {'id', 'source_lines', 'profile', 'contract', 'applicability', 'limitations', 'evidence_ids'}
        _exact(args, required, {'parent_id', 'dependencies', 'improvement_ids', 'knowledge_scope'})
        self._records()  # Reject a corrupt extension before appending a v1 candidate.
        identifier = _identifier(args['id'])
        references = self.store._references(args['evidence_ids'])
        applicability = _texts(args['applicability'], 'applicability')
        limitations = _texts(args['limitations'], 'limitations')
        contract = validate_contract(args['contract'])
        parent_id = args.get('parent_id')
        if parent_id is not None:
            self._resolve_skill(parent_id)
        improvements = _list(args.get('improvement_ids', []), 'improvement_ids')
        for value in improvements:
            if self._record(value)['kind'] != 'improvement':
                raise KnowledgeError('improvement_ids must name registered improvement records')
        declared = args.get('dependencies', {})
        if type(declared) is not dict or len(declared) > 8:
            raise KnowledgeError('dependencies must be a bounded alias-to-ID object')
        dependencies = {}
        for alias, reference in declared.items():
            candidate, _ = self._resolve_skill(reference)
            dependencies[alias] = {key: candidate[key] for key in ('source', 'profile', 'candidate_id')}
        program = compile_program(args['source_lines'], args['profile'], dependencies,
                                  demo_tools=self.store.demo_tools)
        # All metadata is validated before the v1 append; no old source is edited.
        original = _copy({'source_lines': args['source_lines'], 'profile': args['profile']})
        fields = dict(contract=contract, source_lines=original['source_lines'],
                      declared_profile=original['profile'], applicability=applicability,
                      limitations=limitations, evidence_ids=list(args['evidence_ids']), evidence=references,
                      parent_id=parent_id, dependencies=_copy(declared), improvement_ids=improvements,
                      entrypoint=_copy(program['entrypoint']), dependency_pins=_copy(program['dependency_pins']),
                      status='candidate_not_promoted')
        if len(_encoded(fields)) + len('\n'.join(program['source_lines']).encode()) > _MAX_RECORD - 4096:
            raise KnowledgeError('package exceeds bounded record size')
        candidate_args = dict(id=identifier, source_lines=program['source_lines'],
            profile=program['profile'], applicability=applicability, limitations=limitations,
            evidence_ids=args['evidence_ids'])
        if 'knowledge_scope' in args:
            fields['knowledge_scope'] = validate_knowledge_scope(args['knowledge_scope'])
            candidate_args['knowledge_scope'] = fields['knowledge_scope']
        candidate = (self.store.propose('skill', candidate_args) if self.store.learning_access is None
                     else self.store._propose('skill', candidate_args, source_ids=self._field_source_ids(fields, source_ids)))
        frozen = self.store._candidate(candidate['candidate_id'])
        fields.update(candidate_id=candidate['candidate_id'], compiled_source_sha256=frozen['source_sha256'],
                      compiled_profile_sha256=frozen['profile_sha256'])
        record = self._append('package', identifier, fields, source_ids=source_ids)
        return dict(self._summary(record), contract=_copy(contract))

    def record_improvement(self, args, *, source_ids=None):
        assert_writable(self.store)
        required = {'id', 'summary', 'rationale', 'status', 'skill_ids', 'evidence_ids'}
        _exact(args, required, {'priority', 'revisit_when', 'merged_into', 'knowledge_ids', 'knowledge_scope'})
        identifier = _identifier(args['id'])
        if type(args['status']) is not str or args['status'] not in {'open', 'deferred', 'resolved', 'reopened', 'merged'}:
            raise KnowledgeError('recognized model-reported improvement status required')
        latest = {item['id']: item for item in self._records() if item['kind'] == 'improvement'}
        previous = latest.get(identifier, {})
        priority = args.get('priority', previous.get('priority', 50))
        if type(priority) is not int or not 0 <= priority <= 100:
            raise KnowledgeError('model-authored priority must be an integer 0..100')
        revisit = _list(args.get('revisit_when', previous.get('revisit_when', [])),
                        'revisit_when', maximum=8)
        merge = {}
        if args['status'] == 'merged':
            target = self._record(args.get('merged_into'))
            if (target['kind'] != 'improvement' or target['id'] == identifier
                    or latest[target['id']]['status'] == 'merged'):
                raise KnowledgeError('merge target must name another current non-merged improvement')
            merge['merged_into'] = target['record_id']
        elif 'merged_into' in args:
            raise KnowledgeError('merged_into is only valid for status merged')
        skills = self._skill_references(args['skill_ids'])
        evidence = self.store._references(args['evidence_ids'])
        record = self._append('improvement', args['id'], dict(
            summary=_text(args['summary'], 'summary'), rationale=_text(args['rationale'], 'rationale'),
            status=args['status'], skill_ids=skills, evidence_ids=list(args['evidence_ids']), evidence=evidence,
            priority=priority, revisit_when=revisit,
            basis='model_report', independently_verified=False, **self._knowledge_metadata(args), **merge), source_ids=source_ids)
        return self._summary(record)

    @read_operation
    def list_improvements(self, status='pending', limit=8, offset=0):
        """Page current model-owned backlog, without mixing superseded versions."""
        statuses = {'open', 'deferred', 'resolved', 'reopened', 'merged'}
        if type(status) is not str or status not in statuses | {'pending', 'all'}:
            raise KnowledgeError('recognized improvement-list status required')
        if (type(limit) is not int or not 1 <= limit <= 16
                or type(offset) is not int or not 0 <= offset <= 1000000):
            raise KnowledgeError('bounded improvement-list limit and offset required')
        latest = {}
        for record in self._records():
            if record['kind'] == 'improvement':
                latest.pop(record['id'], None)
                latest[record['id']] = record
        ordered = [item for _, item in sorted(enumerate(latest.values()),
                   key=lambda pair: (pair[1].get('priority', 50), pair[0]), reverse=True)]
        pending = {'open', 'reopened', 'deferred'}
        rows = [item for item in ordered if status == 'all'
                or item['status'] in (pending if status == 'pending' else {status})]
        result = dict(scope=self.scope, status=status, offset=offset,
                      items=[self._summary(item) for item in rows[offset:offset + limit]],
                      total_count=len(rows), pending_count=sum(item['status'] in pending for item in ordered),
                      selection='model-declared priority then latest registration',
                      basis='model_report_not_independent_truth', summary_byte_limit=8192)
        while True:
            consumed = offset + len(result['items'])
            result['next_offset'] = consumed if consumed < len(rows) else None
            result['omitted_count'] = max(0, len(rows) - consumed)
            if len(json.dumps(result, ensure_ascii=False, sort_keys=True,
                              allow_nan=False).encode('utf-8')) <= 8192:
                return result
            if len(result['items']) <= 1:
                raise KnowledgeError('improvement-list entry exceeds byte budget')
            result['items'].pop()

    def record_outcome(self, args, *, source_ids=None):
        assert_writable(self.store)
        required = {'id', 'skill_ids', 'evidence_ids', 'observation_evidence_id', 'task_result',
                    'correction_performed', 'remaining_issues', 'summary'}
        _exact(args, required, {'knowledge_ids', 'knowledge_scope'})
        _identifier(args['id'])
        if (type(args['task_result']) is not str or args['task_result'] not in {'success', 'failure', 'uncertain'}
                or type(args['correction_performed']) is not bool):
            raise KnowledgeError('explicit task result and correction boolean required')
        skills = self._skill_references(args['skill_ids'])
        if not skills and not args.get('knowledge_ids'):
            raise KnowledgeError('outcome must reference a skill or knowledge record')
        evidence = self.store._references(args['evidence_ids'])
        if (type(args['observation_evidence_id']) is not str
                or args['observation_evidence_id'] not in args['evidence_ids']):
            raise KnowledgeError('outcome observation must be a registered referenced evidence ID')
        remaining = _list(args['remaining_issues'], 'remaining_issues')
        record = self._append('outcome', args['id'], dict(
            skill_ids=skills, evidence_ids=list(args['evidence_ids']), evidence=evidence,
            observation_evidence_id=args['observation_evidence_id'], task_result=args['task_result'],
            correction_performed=args['correction_performed'], remaining_issues=remaining,
            summary=_text(args['summary'], 'summary'), basis='model_report', independently_verified=False,
            **self._knowledge_metadata(args)), source_ids=source_ids)
        return self._summary(record)

    @read_operation
    def resolve_call(self, args):
        from astra_robotics_agent.physical_keyboard.skills.programs import validate_arguments
        _exact(args, {'package_id', 'arguments'})
        package = self._record(args['package_id'])
        if package['kind'] != 'package':
            raise KnowledgeError('call_skill requires an explicit skill package')
        arguments = validate_arguments(package['contract'], args['arguments'])
        candidate, _ = self._resolve_skill(args['package_id'])
        profile = dict(candidate['profile'], **arguments)
        profile = _decoded(_profile(profile))
        _validate_source(candidate['source'], demo_tools=self.store.demo_tools)
        return dict(source=candidate['source'], profile=profile,
                    label='package-' + _sha(package['record_id'].encode())[:32],
                    package_id=package['record_id'], candidate_id=candidate['candidate_id'],
                    dependency_pins=_copy(package['dependency_pins']))

    @read_operation
    def inspect_evidence(self, args):
        """Inspect registered history without granting a fresh observation.

        Legacy requests retain their original result shape and prefix budgets.
        Any paging/JSON option requires exactly one selected UTF-8 text file.
        Offsets, limits, and ``total`` count Unicode code points, never bytes.
        A JSON pointer selects before paging its compact, sorted-key JSON text.
        ``truncated`` means more text follows; ``next_offset`` is null at EOF.
        """
        page_fields = {'text_offset', 'text_limit', 'json_pointer'}
        _exact(args, {'evidence_id'}, {'file_indices', 'include_images'} | page_fields)
        evidence = self.store._evidence(args['evidence_id'])
        indices = args.get('file_indices', [])
        images = args.get('include_images', False)
        if (type(images) is not bool or type(indices) is not list or len(indices) > 4
                or any(type(i) is not int or not 0 <= i < len(evidence['files']) for i in indices)
                or len(set(indices)) != len(indices)):
            raise KnowledgeError('bounded unique file indices and explicit image boolean required')
        paged = bool(page_fields & args.keys())
        offset, limit = args.get('text_offset', 0), args.get('text_limit', 8000)
        if paged and (len(indices) != 1 or type(offset) is not int or offset < 0
                      or type(limit) is not int or not 1 <= limit <= 8000):
            raise KnowledgeError('text paging requires one file index, nonnegative text_offset and text_limit 1..8000')
        files, text_left, image_count = [], 16000, 0
        for index, item in enumerate(evidence['files']):
            entry = dict(index=index, name=Path(item['source_path']).name, sha256=item['sha256'], bytes=item['bytes'])
            if index in indices:
                raw = _read(self.store.namespace / 'objects' / item['sha256'])
                if _sha(raw) != item['sha256'] or len(raw) != item['bytes']:
                    raise KnowledgeError('selected evidence object hash mismatch')
                mime = ('image/jpeg' if raw.startswith(b'\xff\xd8\xff') else
                        'image/png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else None)
                if mime is not None:
                    if paged:
                        raise KnowledgeError('text paging requires UTF-8 text evidence, not an image')
                    entry['mime_type'] = mime
                    if images:
                        image_count += 1
                        if image_count > 2 or len(raw) > 4 * 1024 * 1024:
                            raise KnowledgeError('historical image request exceeds image budget')
                        entry['image_base64'] = base64.b64encode(raw).decode('ascii')
                else:
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeError as exc:
                        if paged:
                            raise KnowledgeError('text paging requires UTF-8 text evidence') from exc
                        entry['content_status'] = 'binary_not_returned'
                    else:
                        if paged:
                            if 'json_pointer' in args:
                                text = _evidence_json_text(raw, args['json_pointer'])
                                entry.update(json_pointer=args['json_pointer'], text_format='canonical_json')
                            if offset > len(text):
                                raise KnowledgeError('text_offset exceeds selected text total')
                            allowance = min(limit, text_left)
                            entry['text'] = text[offset:offset + allowance]
                            end = offset + len(entry['text'])
                            entry.update(text_offset=offset, text_limit=limit, offset_unit='unicode_code_points',
                                         total=len(text), next_offset=end if end < len(text) else None,
                                         truncated=end < len(text))
                        else:
                            allowance = min(8000, text_left)
                            entry['text'] = text[:allowance]
                            entry['truncated'] = len(text) > allowance
                        text_left -= len(entry['text'])
            files.append(entry)
        return {'evidence_id': args['evidence_id'], 'scope': self.scope, 'run_id': evidence['run_id'],
                'historical': True, 'files': files, 'images_requested': images,
                'meaning': 'Registered historical evidence; not a fresh current robot observation.'}

    @read_operation
    def context(self, limit=5):
        if type(limit) is not int or not 1 <= limit <= 16:
            raise KnowledgeError('development context limit must be 1..16')
        records = self._records()
        latest = {}
        for item in records:
            if item['kind'] == 'improvement':
                latest.pop(item['id'], None)
                latest[item['id']] = item
        ordered = list(latest.values())
        pending = [item for _, item in sorted(enumerate(ordered),
                   key=lambda pair: (pair[1].get('priority', 50), pair[0]), reverse=True)
                   if item['status'] in {'open', 'reopened', 'deferred'}]
        outcomes = [item for item in records if item['kind'] == 'outcome']
        result = {'protocol': PROTOCOL, 'scope': self.scope,
                'pending_improvements': [self._summary(item) for item in pending[:limit]],
                'recent_outcomes': [self._summary(item) for item in outcomes[-limit:]],
                'pending_count': len(pending), 'outcome_count': len(outcomes),
                'basis': 'Model-authored opportunities and task reports; not independent truth or automatic promotion.',
                'selection': 'model-declared priority, then latest logical registration; no host-assigned urgency',
                'summary_byte_limit': 8192, 'summary_truncated': False}
        while len(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')) > 8192:
            result['summary_truncated'] = True
            entries = result['recent_outcomes'] or result['pending_improvements']
            if not entries:
                raise KnowledgeError('development summary cannot fit byte budget')
            entries.pop(0 if result['recent_outcomes'] else -1)
        return result
