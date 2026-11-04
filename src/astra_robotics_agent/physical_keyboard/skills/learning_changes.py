"""Finite learning declarations with a final, append-only visibility boundary.

The existing candidate/package writers remain the source of executable formats.
Preflight uses an in-memory overlay and performs their validation without disk
writes. A failed write can leave inert candidates; only a complete commit makes
the newly created IDs eligible for normal reads or selection.
"""
from copy import copy, deepcopy
import os
import uuid

from ..records.learning_access import LearningAccess, LearningAccessDenied
from ..records.knowledge_domains import validate_knowledge_scope
from ..records.learning_contracts import (
    PROTOCOLS, LearningContractError, clone, digest, encoded, exact, ids,
    identifier, text,
)
from .development import PROTOCOL as DEVELOPMENT_PROTOCOL, SkillDevelopment, _check_directory
from .knowledge import (_decoded, _encoded, _identifier, _profile, _read, _sha,
                        _text, _texts, _validate_source)


RELATIONS = {'derived_from', 'uses_lesson', 'uses_resource', 'addresses_issue',
             'supersedes', 'contradicts', 'alias'}
_ACYCLIC_GROUPS = ({'alias', 'supersedes'}, {'derived_from'}, {'dependency'})
_GROUPS = ('resources', 'lessons', 'packages', 'workflows', 'improvements', 'relations', 'selections')


class _OverlayStore:
    """Lazy read-through store with only new candidate payloads held in memory."""
    def __init__(self, store, access):
        self.original = copy(store)
        self.original.learning_access = deepcopy(access)
        self.learning_access = self.original.learning_access
        self._scope, self.run_id, self.demo_tools = store._scope, store.run_id, store.demo_tools
        self.new = {}

    def _events(self):
        return self.original._events() + [dict(type='candidate_proposed', **{
            key: row[key] for key in ('candidate_id', 'id', 'kind', 'version', 'scope')})
            for row in self.new.values()]

    def _candidate(self, value):
        return self.new[value] if value in self.new else self.original._candidate(value)

    def _check_knowledge_access(self, value, record):
        return self.original._check_knowledge_access(value, record)

    def _evidence_source_ids(self, evidence):
        return self.original._evidence_source_ids(evidence)

    def _evidence(self, value):
        return self.original._evidence(value)

    def _references(self, value):
        return self.original._references(value)

    def propose(self, kind, args):
        return self._propose(kind, args)

    def _propose(self, kind, args, *, source_ids=None):
        required = {'id', 'applicability', 'limitations', 'evidence_ids'} | (
            {'claim'} if kind == 'lesson' else {'source_lines', 'profile'})
        exact(args, required, {'knowledge_scope'})
        logical_id = _identifier(args['id'])
        data = dict(id=logical_id, applicability=_texts(args['applicability'], 'applicability'),
                    limitations=_texts(args['limitations'], 'limitations'),
                    evidence_ids=ids(args['evidence_ids'], 'evidence_ids', 16))
        evidence = self._references(args['evidence_ids'])
        if 'knowledge_scope' in args:
            data['knowledge_scope'] = validate_knowledge_scope(args['knowledge_scope'])
        if kind == 'lesson':
            data['claim'] = _text(args['claim'], 'claim', 4096)
        else:
            lines = args['source_lines']
            if type(lines) is not list or not 1 <= len(lines) <= 1024 or any(type(line) is not str for line in lines):
                raise LearningContractError('bounded source_lines required')
            source = '\n'.join(lines) + '\n'
            _validate_source(source, demo_tools=self.demo_tools)
            profile = _profile(args['profile'])
            data.update(source=source, profile=_decoded(profile),
                        source_sha256=_sha(source.encode()), profile_sha256=_sha(profile))
        record = {'protocol': 'physical-keyboard-knowledge-candidate-v1', 'kind': kind,
                  'scope': self._scope, 'run_id': self.run_id, 'origin': 'astra_generated',
                  'status': 'candidate_not_promoted', 'source': 'astra_proposal', 'evidence': evidence, **data}
        if self.learning_access is not None:
            sources = set(self._evidence_source_ids(evidence))
            if source_ids is not None:
                sources.update(self.learning_access.check_sources(source_ids, require_provenance=True))
            record['source_ids'] = self.learning_access.check_sources(sorted(sources), require_provenance=True)
        candidate_id = kind + '-' + logical_id + '-' + _sha(_encoded(record))
        if self.learning_access is not None:
            self.learning_access.bind_knowledge(candidate_id, record['source_ids'])
        if candidate_id not in self.new:
            previous = [row for row in self._events() if row.get('candidate_id') == candidate_id]
            if previous:
                return self.original._summary(self.original._candidate(candidate_id),
                    self.original.namespace / 'candidates' / candidate_id)
            versions = [row['version'] for row in self._events()
                        if row.get('type') == 'candidate_proposed' and row.get('kind') == kind and row.get('id') == logical_id]
            record.update(candidate_id=candidate_id, version=max(versions, default=0) + 1)
            self.new[candidate_id] = record
        return {key: self.new[candidate_id][key] for key in ('candidate_id', 'kind', 'id', 'version')}


class _OverlayLibrary(SkillDevelopment):
    def __init__(self, library, access):
        self.original = copy(library)
        self.store = _OverlayStore(library.store, access)
        self.original.store = self.store.original
        self._scope = library.scope
        self.new = []

    def _records(self, include_denied=False):
        return self.original._records(include_denied=include_denied) + self.new

    def _append(self, kind, logical_id, fields, *, source_ids=None):
        _identifier(logical_id)
        previous = [row['version'] for row in self._records(include_denied=True) if row['kind'] == kind and row['id'] == logical_id]
        if self.store.learning_access is not None:
            fields = dict(fields, source_ids=self._field_source_ids(fields, extra=source_ids))
        record = dict(protocol=DEVELOPMENT_PROTOCOL, kind=kind, id=logical_id,
                      version=max(previous, default=0) + 1, scope=self.scope,
                      run_id=self.store.run_id, origin='astra_generated', stable=False, **fields)
        record['record_id'] = kind + '-' + logical_id + '-' + _sha(_encoded(record))
        if self.store.learning_access is not None:
            self.store.learning_access.bind_knowledge(record['record_id'], record['source_ids'])
        if len(_encoded(record)) > 512 * 1024:
            raise LearningContractError('development record exceeds byte budget')
        self.new.append(record)
        return record


class LearningChanges:
    def __init__(self, library, resources, *, access=None):
        self.library, self.resources = library, resources
        self.store = library.store
        self.scope = library.scope
        self.access = access or resources.access or LearningAccess(self.scope)
        if resources.scope != self.scope or self.access.scope != self.scope:
            raise LearningContractError('learning components must share exact scope')
        if resources.access is not self.access:
            # The caller may retain a broad resource service. All resource reads
            # through this boundary use its exact policy, without mutating that
            # caller-owned service or trusting its more permissive decisions.
            self.resources = copy(resources)
            self.resources.access = self.access
            self.resources.store = copy(resources.store)
            self.resources.store.learning_access = self.access
        if self.store.learning_access is None:
            self.store.learning_access = self.access
        self.root = self.store.namespace / 'learning' / 'changes'
        for directory in (self.root.parent, self.root, self.root / 'intents', self.root / 'commits', self.root / 'staging', self.root / 'publications'):
            _check_directory(directory)
            directory.mkdir(exist_ok=True)
        self.failure_hook = None
        self._restore_bindings()

    def _hook(self, stage):
        if self.failure_hook is not None:
            self.failure_hook(stage)

    def _read_record(self, path):
        raw = _read(path, maximum=2 * 1024 * 1024)
        record = _decoded(raw)
        if type(record) is not dict or record.get('scope') != self.scope:
            raise LearningContractError('learning change scope mismatch')
        if record.get('record_sha256') != digest({key: value for key, value in record.items() if key != 'record_sha256'}):
            raise LearningContractError('learning change hash mismatch')
        if path.stem != record.get('commit_id'):
            raise LearningContractError('learning change identity mismatch')
        return record

    def _write_record(self, directory, record):
        record = dict(record)
        record['record_sha256'] = digest(record)
        # Publish only fully flushed bytes. A crash during staging leaves no
        # complete registration and cannot replace an earlier selected view.
        stage = self.root / 'staging' / (uuid.uuid4().hex + '.json')
        with stage.open('xb') as stream:
            stream.write(encoded(record, 2 * 1024 * 1024))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, self.root / directory / (record['commit_id'] + '.json'), follow_symlinks=False)
        return record

    def commits(self):
        records = []
        for path in sorted((self.root / 'commits').glob('*.json')):
            record = self._read_record(path)
            intent_path = self.root / 'intents' / (record['commit_id'] + '.json')
            intent = self._read_record(intent_path)
            if (record.get('protocol') != PROTOCOLS['commit'] or record.get('status') != 'complete'
                    or record.get('intent_sha256') != intent['record_sha256']
                    or record.get('sequence') != intent.get('sequence')):
                raise LearningContractError('incomplete or mismatched learning commit')
            records.append(record)
        records.sort(key=lambda row: row['sequence'])
        if len({row['sequence'] for row in records}) != len(records):
            raise LearningContractError('duplicate learning commit sequence')
        return records

    def _intents(self):
        return [self._read_record(path) for path in sorted((self.root / 'intents').glob('*.json'))]

    def is_committed(self, value):
        identifier(value)
        intents = self._intents()
        owners = [row['commit_id'] for row in intents if value in row['new_ids']]
        publication = self._publication(value)
        if not owners and publication is None:
            return True  # Pre-existing exact IDs retain their original semantics.
        # A retry may revalidate an identical immutable lesson/resource left by
        # an earlier failed batch. Its later complete declaration activates it.
        return any(value in row['record_ids'] for row in self.commits())

    def _publication(self, value, *, intent=None):
        """Independent immutable anchor prevents a lost intent becoming legacy."""
        path = self.root / 'publications' / (digest(value) + '.json')
        if path.exists() or path.is_symlink():
            record = _decoded(_read(path))
            exact(record, {'protocol', 'identifier', 'scope', 'first_commit_id',
                           'intent_sha256', 'record_sha256'})
            expected = digest({key: item for key, item in record.items() if key != 'record_sha256'})
            if (record['protocol'] != PROTOCOLS['commit'] + ':publication'
                    or record['identifier'] != value or record['scope'] != self.scope
                    or record['record_sha256'] != expected):
                raise LearningContractError('publication anchor identity or hash mismatch')
            return record
        if intent is None:
            return None
        record = dict(protocol=PROTOCOLS['commit'] + ':publication', identifier=value,
            scope=self.scope, first_commit_id=intent['commit_id'], intent_sha256=intent['record_sha256'])
        record['record_sha256'] = digest(record)
        stage = self.root / 'staging' / (uuid.uuid4().hex + '.json')
        with stage.open('xb') as stream:
            stream.write(encoded(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, path, follow_symlinks=False)
        return record

    def _restore_bindings(self):
        for commit in self.commits():
            for value, sources in commit['source_bindings'].items():
                try:
                    self.access.check_sources(sources, require_provenance=True)
                except LearningAccessDenied:
                    # A filtered session can open the same namespace, but must
                    # never expose or adopt a forbidden historical binding.
                    continue
                self.access.bind_knowledge(value, sources)

    def source_ids(self, value):
        if not self.is_committed(value):
            raise LearningContractError('uncommitted learning object is not active')
        if value.startswith('resource-'):
            return self.resources.get(value)['source_ids']
        # Load persisted provenance through the guarded legacy reader on a
        # reopened session, even before its in-memory binding has been rebuilt.
        reader = copy(self.library)
        reader.store = copy(self.store)
        reader.store.learning_access = self.access
        reader.inspect(value)
        return self._bound_sources(value, self.access)

    def _bound_sources(self, value, access):
        found = None
        for commit in self.commits():
            if value in commit['source_bindings']:
                sources = commit['source_bindings'][value]
                if found is not None and found != sources:
                    raise LearningContractError('knowledge source closure changed')
                found = sources
        return access.check_sources(found, require_provenance=True) if found is not None else access.knowledge_sources(value)

    def _reference(self, value, bindings):
        if type(value) is dict:
            exact(value, {'local_ref'})
            local = identifier(value['local_ref'], 'local_ref')
            if local not in bindings:
                raise LearningContractError('unknown or forward local_ref')
            return bindings[local]
        return identifier(value)

    @staticmethod
    def _declarations(args):
        """Topologically order only finite ID references, never arbitrary JSON."""
        declarations, names = [], set()
        for group in _GROUPS:
            for item in args.get(group, []):
                if type(item) is not dict:
                    raise LearningContractError('finite declaration object required')
                local = item.get('local_ref')
                if 'local_ref' in item:
                    identifier(local, 'local_ref')
                    if local in names:
                        raise LearningContractError('duplicate local_ref')
                    names.add(local)
                references = []
                if group == 'packages':
                    references.extend(item[field] for field in ('parent_id', 'profile_resource_id') if field in item)
                    dependencies = item.get('dependencies', {})
                    if type(dependencies) is not dict:
                        raise LearningContractError('finite dependency alias object required')
                    references.extend(dependencies.values())
                    values = item.get('improvement_ids', [])
                    if type(values) is not list:
                        raise LearningContractError('improvement_ids must be a finite ID list')
                    references.extend(values)
                elif group == 'workflows':
                    if 'parent_id' in item:
                        references.append(item['parent_id'])
                    values = item.get('improvement_ids', [])
                    if type(values) is not list:
                        raise LearningContractError('improvement_ids must be a finite ID list')
                    references.extend(values)
                    steps = item.get('steps', [])
                    if type(steps) is not list:
                        raise LearningContractError('steps must be a finite step list')
                    for step in steps:
                        if type(step) is not dict or type(step.get('skill_ids')) is not list:
                            raise LearningContractError('workflow step requires finite skill_ids')
                        references.extend(step['skill_ids'])
                elif group == 'improvements':
                    values = item.get('skill_ids', [])
                    if type(values) is not list:
                        raise LearningContractError('skill_ids must be a finite ID list')
                    references.extend(values)
                    knowledge = item.get('knowledge_ids', [])
                    if type(knowledge) is not list:
                        raise LearningContractError('knowledge_ids must be a finite ID list')
                    references.extend(knowledge)
                    if 'merged_into' in item:
                        references.append(item['merged_into'])
                elif group == 'relations':
                    references.extend(item[field] for field in ('from_id', 'to_id') if field in item)
                elif group == 'selections' and 'identifier' in item:
                    references.append(item['identifier'])
                required = set()
                for reference in references:
                    if type(reference) is dict:
                        exact(reference, {'local_ref'})
                        required.add(identifier(reference['local_ref'], 'local_ref'))
                    else:
                        identifier(reference)
                declarations.append((group, item, local, required))
        if any(not required <= names for _, _, _, required in declarations):
            raise LearningContractError('unknown local_ref')
        ordered, ready = [], set()
        while declarations:
            progress = False
            remaining = []
            for group, item, local, required in declarations:
                if required <= ready:
                    ordered.append((group, item))
                    if local is not None:
                        ready.add(local)
                    progress = True
                else:
                    remaining.append((group, item, local, required))
            if not progress:
                raise LearningContractError('cyclic finite local_ref dependencies')
            declarations = remaining
        return ordered

    def _existing(self, value, overlay, resource_map, source_map):
        if value in resource_map:
            return resource_map[value], source_map[value]
        if value in source_map:
            if value.startswith('resource-'):
                return resource_map[value], source_map[value]
            return overlay.inspect(value), source_map[value]
        if not self.is_committed(value):
            raise LearningContractError('pending object cannot be referenced')
        if value.startswith('resource-'):
            record = overlay.resources.get(value)
            return record, record['source_ids']
        record = overlay.inspect(value)
        if record.get('scope') != self.scope:
            raise LearningContractError('reference scope mismatch')
        sources = self._bound_sources(value, overlay.store.learning_access)
        # Declared package dependencies and parent must retain their full closure.
        if record.get('kind') == 'package':
            for dependency in record.get('dependencies', {}).values():
                if not self.is_committed(dependency):
                    raise LearningContractError('package has pending dependency')
                overlay.inspect(dependency)
                dependency_sources = self._bound_sources(dependency, overlay.store.learning_access)
                sources = sorted(set(sources) | set(dependency_sources))
        return record, self.access.check_sources(sources, require_provenance=True)

    def _planned_sources(self, declarations, overlay):
        """Compute source closure before source-bearing content IDs are hashed."""
        locals_to_key = {item['local_ref']: id(item) for _, item in declarations if 'local_ref' in item}
        values, edges, resources = {}, [], set()
        access = overlay.store.learning_access

        def endpoint(value):
            if type(value) is dict:
                return locals_to_key[value['local_ref']], None
            _, sources = self._existing(value, overlay, {}, {})
            return None, set(sources)

        for group, item in declarations:
            if group not in {'resources', 'lessons', 'packages', 'workflows', 'improvements'}:
                continue
            key = id(item)
            sources = set(access.check_sources(item.get('source_ids', [])))
            for evidence_id in ids(item.get('evidence_ids', []), 'evidence_ids', 16):
                overlay.store._evidence(evidence_id)
                sources.update(access.evidence_sources(evidence_id))
            values[key] = sources
            references = []
            if group == 'resources':
                resources.add(key)
            elif group == 'packages':
                references.extend(item[field] for field in ('parent_id', 'profile_resource_id') if field in item)
                references.extend(item.get('dependencies', {}).values())
                references.extend(item.get('improvement_ids', []))
            elif group == 'workflows':
                if 'parent_id' in item:
                    references.append(item['parent_id'])
                references.extend(item.get('improvement_ids', []))
                for step in item.get('steps', []):
                    references.extend(step['skill_ids'])
            elif group == 'improvements':
                references.extend(item.get('skill_ids', []))
                references.extend(item.get('knowledge_ids', []))
                if 'merged_into' in item:
                    references.append(item['merged_into'])
            for reference in references:
                target, inherited = endpoint(reference)
                edges.append((key, target, inherited))
        for group, item in declarations:
            if (group != 'relations' or type(item.get('kind')) is not str
                    or item.get('kind') not in {'derived_from', 'uses_lesson', 'uses_resource', 'addresses_issue'}
                    or not {'from_id', 'to_id'} <= set(item)):
                continue
            if type(item['from_id']) is dict:
                origin = locals_to_key[item['from_id']['local_ref']]
                target, inherited = endpoint(item['to_id'])
                edges.append((origin, target, inherited))
        for _ in range(len(values) + 1):
            changed = False
            for origin, target, inherited in edges:
                if origin not in values or target is not None and target not in values:
                    raise LearningContractError('learning dependency must name an object declaration')
                merged = values[origin] | (values[target] if target is not None else inherited)
                if merged != values[origin]:
                    if origin in resources:
                        raise LearningContractError('resource declaration omits relationship source closure')
                    values[origin] = merged
                    changed = True
            if not changed:
                break
        return {key: access.check_sources(sorted(sources), require_provenance=True) for key, sources in values.items()}

    @staticmethod
    def _cycles(relations):
        # Provenance and use are distinct graphs: a lesson may derive from an
        # attempt whose later package uses that lesson without a logical cycle.
        for kinds in _ACYCLIC_GROUPS:
            graph = {}
            for relation in relations:
                if relation['kind'] in kinds:
                    graph.setdefault(relation['from_id'], set()).add(relation['to_id'])
            visiting, done = set(), set()
            def visit(node):
                if node in visiting:
                    raise LearningContractError('dependency, alias or replacement cycle')
                if node in done:
                    return
                visiting.add(node)
                for target in graph.get(node, ()):
                    visit(target)
                visiting.remove(node)
                done.add(node)
            for node in graph:
                visit(node)

    def preflight(self, args):
        """Return an inert exact plan; no candidate, resource or index is written."""
        exact(args, set(), set(_GROUPS) | {'summary'})
        clone(args, 1024 * 1024)
        if 'summary' in args:
            text(args['summary'], 'summary')
        count = 0
        for group in _GROUPS:
            values = args.get(group, [])
            if type(values) is not list or len(values) > 16:
                raise LearningContractError('each declaration group is bounded to 16 items')
            count += len(values)
        if not 1 <= count <= 32:
            raise LearningContractError('commit requires 1..32 finite declarations')
        overlay = _OverlayLibrary(self.library, self.access)
        overlay.resources = copy(self.resources)
        overlay.resources.store = overlay.store
        overlay.resources.access = overlay.store.learning_access
        declarations = self._declarations(args)
        planned_sources = self._planned_sources(declarations, overlay)
        bindings, resource_map, source_map, operations, relations, selections = {}, {}, {}, [], [], []
        existing_ids = {row['candidate_id'] for row in self.store._events() if row.get('type') == 'candidate_proposed'}
        existing_ids.update(row['record_id'] for row in overlay._records(include_denied=True))
        existing_ids.update(row['resource_id'] for row in overlay.resources.records())

        def bind(item, value):
            local = item.get('local_ref')
            if local is not None:
                identifier(local, 'local_ref')
                if local in bindings:
                    raise LearningContractError('duplicate local_ref')
                bindings[local] = value

        def sources_for(item, references=()):
            for value in references:
                self._existing(value, overlay, resource_map, source_map)
            result = planned_sources[id(item)]
            if not result:
                raise LearningContractError('learning declaration requires explicit provenance')
            return result

        for group, item in declarations:
            if type(item) is not dict:
                raise LearningContractError('finite declaration object required')
            if group == 'resources':
                exact(item, {'id', 'value', 'source_ids'}, {'local_ref', 'evidence_ids', 'kind', 'metadata'})
                record = overlay.resources.prepare(item['id'], item['value'], source_ids=planned_sources[id(item)],
                    evidence_ids=item.get('evidence_ids'), kind=item.get('kind', 'analysis_result'), metadata=item.get('metadata'))
                value = record['resource_id']
                resource_map[value], source_map[value] = record, record['source_ids']
                operations.append(dict(group=group, args=clone(item), identifier=value))
                bind(item, value)
                continue
            if group in {'lessons', 'packages', 'workflows', 'improvements'}:
                payload = {key: clone(value, 512 * 1024) for key, value in item.items() if key not in {'local_ref', 'source_ids'}}
                references = []
                if group == 'packages':
                    if ('profile' in payload) == ('profile_resource_id' in payload):
                        raise LearningContractError('choose profile or profile_resource_id exactly once')
                    if 'profile_resource_id' in payload:
                        resource_id = self._reference(payload.pop('profile_resource_id'), bindings)
                        resource, _ = self._existing(resource_id, overlay, resource_map, source_map)
                        if resource.get('kind') not in {'profile', 'analysis_result'} or type(resource.get('value')) is not dict:
                            raise LearningContractError('package requires a registered complete profile object')
                        payload['profile'] = clone(resource['value'])
                        _profile(payload['profile'])
                        references.append(resource_id)
                    for field in ('parent_id',):
                        if field in payload:
                            payload[field] = self._reference(payload[field], bindings)
                            references.append(payload[field])
                    if 'dependencies' in payload:
                        if type(payload['dependencies']) is not dict:
                            raise LearningContractError('dependency aliases must be an object')
                        payload['dependencies'] = {key: self._reference(value, bindings)
                                                   for key, value in payload['dependencies'].items()}
                        references.extend(payload['dependencies'].values())
                    if 'improvement_ids' in payload:
                        payload['improvement_ids'] = [self._reference(value, bindings) for value in payload['improvement_ids']]
                        references.extend(payload['improvement_ids'])
                elif group == 'workflows':
                    if 'parent_id' in payload:
                        payload['parent_id'] = self._reference(payload['parent_id'], bindings)
                        references.append(payload['parent_id'])
                    if 'improvement_ids' in payload:
                        payload['improvement_ids'] = [self._reference(value, bindings) for value in payload['improvement_ids']]
                        references.extend(payload['improvement_ids'])
                    for step in payload.get('steps', []):
                        step['skill_ids'] = [self._reference(value, bindings) for value in step['skill_ids']]
                        references.extend(step['skill_ids'])
                elif group == 'improvements':
                    if type(payload.get('skill_ids')) is not list:
                        raise LearningContractError('skill_ids must be a finite ID list')
                    payload['skill_ids'] = [self._reference(value, bindings) for value in payload['skill_ids']]
                    references.extend(payload['skill_ids'])
                    if 'knowledge_ids' in payload:
                        payload['knowledge_ids'] = [self._reference(value, bindings) for value in payload['knowledge_ids']]
                        references.extend(payload['knowledge_ids'])
                    if 'merged_into' in payload:
                        payload['merged_into'] = self._reference(payload['merged_into'], bindings)
                        references.append(payload['merged_into'])
                sources = sources_for(item, references)
                if group == 'lessons':
                    result = overlay.store._propose('lesson', payload, source_ids=sources)
                    value = result['candidate_id']
                elif group == 'packages':
                    result = overlay.save_package(payload, source_ids=sources)
                    value = result['package_id']
                    source_map[result['candidate_id']] = sources
                elif group == 'workflows':
                    result = overlay.save_application_skill(payload, source_ids=sources)
                    value = result['record_id']
                else:
                    result = overlay.record_improvement(payload, source_ids=sources)
                    value = result['record_id']
                if value in source_map and source_map[value] != sources:
                    raise LearningContractError('one immutable object cannot acquire different provenance')
                source_map[value] = sources
                operations.append(dict(group=group, args=payload, identifier=value, source_ids=sources))
                bind(item, value)
                continue
            if group == 'relations':
                exact(item, {'kind', 'from_id', 'to_id'}, {'local_ref', 'conditions', 'reason'})
                if type(item['kind']) is not str or item['kind'] not in RELATIONS:
                    raise LearningContractError('recognized finite relationship required')
                relation = dict(kind=item['kind'], from_id=self._reference(item['from_id'], bindings),
                                to_id=self._reference(item['to_id'], bindings), scope=self.scope,
                                conditions=clone(item.get('conditions', [])), reason=item.get('reason', 'Explicit model declaration'))
                for value in (relation['from_id'], relation['to_id']):
                    self._existing(value, overlay, resource_map, source_map)
                self._conditions(relation['conditions']); text(relation['reason'], 'reason')
                target, _ = self._existing(relation['to_id'], overlay, resource_map, source_map)
                required_kind = {'uses_lesson': 'lesson', 'uses_resource': 'resource', 'addresses_issue': 'improvement'}.get(relation['kind'])
                if required_kind and (not relation['to_id'].startswith('resource-') if required_kind == 'resource' else target.get('kind') != required_kind):
                    raise LearningContractError('relationship target has wrong knowledge kind')
                relation['relation_id'] = 'relation-' + digest(relation)
                relations.append(relation); bind(item, relation['relation_id'])
                continue
            exact(item, {'logical_id', 'identifier', 'conditions', 'reason'}, {'local_ref'})
            logical = identifier(item['logical_id'], 'logical_id')
            value = self._reference(item['identifier'], bindings)
            record, _ = self._existing(value, overlay, resource_map, source_map)
            if record.get('id') != logical or record.get('kind') not in {'package', 'skill', 'lesson', 'workflow'}:
                raise LearningContractError('selection must match the exact logical knowledge ID')
            self._conditions(item['conditions']); text(item['reason'], 'reason')
            selection = dict(logical_id=logical, identifier=value, kind=record['kind'],
                             conditions=clone(item['conditions']), reason=item['reason'], scope=self.scope)
            selection['selection_id'] = 'selection-' + digest(selection)
            selections.append(selection); bind(item, selection['selection_id'])
        # Relationships cannot launder a new summary's provenance. Propagate
        # actual learning dependencies to every new declaration and its frozen
        # executable candidate; contrast/replacement links remain associations.
        for _ in range(count + 1):
            changed = False
            for relation in relations:
                if relation['kind'] not in {'derived_from', 'uses_lesson', 'uses_resource', 'addresses_issue'}:
                    continue
                origin, target = relation['from_id'], relation['to_id']
                if origin not in source_map:
                    continue
                _, inherited = self._existing(target, overlay, resource_map, source_map)
                merged = self.access.check_sources(sorted(set(source_map[origin]) | set(inherited)), require_provenance=True)
                if origin.startswith('resource-') and merged != source_map[origin]:
                    raise LearningContractError('resource declaration omits relationship source closure')
                if merged != source_map[origin]:
                    source_map[origin] = merged
                    changed = True
            for record in overlay.new:
                if record['kind'] == 'package' and record['record_id'] in source_map:
                    candidate_id = record['candidate_id']
                    merged = sorted(set(source_map[candidate_id]) | set(source_map[record['record_id']]))
                    if source_map[candidate_id] != merged:
                        source_map[candidate_id] = merged
                        changed = True
            if not changed:
                break
        self._cycles(self.relations() + relations + [dict(kind='dependency', from_id=row['record_id'], to_id=dependency)
            for row in overlay.new if row['kind'] == 'package' for dependency in row.get('dependencies', {}).values()])
        all_ids = set(resource_map) | set(source_map)
        # Bindings are immutable even if a retry names a pre-existing candidate.
        known_sources = {}
        for commit in self.commits():
            known_sources.update(commit['source_bindings'])
        for value, sources in source_map.items():
            if value in known_sources and known_sources[value] != sources:
                raise LearningContractError('knowledge source binding cannot change')
            if value in self.access._knowledge and self.access._knowledge[value] != sources:
                raise LearningContractError('existing access binding cannot change')
        plan = dict(operations=operations, bindings=bindings, record_ids=sorted(all_ids),
                    new_ids=sorted(all_ids - existing_ids), source_bindings=source_map,
                    relations=relations, selections=selections, scope=self.scope,
                    summary=args.get('summary', 'Finite model-authored learning declarations'))
        return clone(plan, 1536 * 1024)

    @staticmethod
    def _conditions(value):
        if type(value) is not list or len(value) > 16:
            raise LearningContractError('bounded explicit conditions required')
        for condition in value:
            text(condition, 'condition', 2048)

    def relations(self):
        result = []
        for commit in self.commits():
            for relation in commit['relations']:
                if relation not in result:
                    result.append(relation)
        return clone(result)

    def commit(self, args):
        from .read_scope import assert_writable
        assert_writable(self.store)
        self._hook('before_preflight')
        plan = self.preflight(args)
        self._hook('after_preflight')
        sequence = max((row['sequence'] for row in self._intents()), default=0) + 1
        commit_id = 'commit-' + uuid.uuid4().hex
        intent = self._write_record('intents', dict(protocol=PROTOCOLS['commit'], status='pending',
            commit_id=commit_id, sequence=sequence, **plan))
        for value in plan['new_ids']:
            self._publication(value, intent=intent)
        self._hook('after_intent')
        for index, operation in enumerate(plan['operations']):
            self._hook('before_object_write:' + str(index))
            payload = operation['args']
            group = operation['group']
            if group == 'resources':
                payload = {key: value for key, value in payload.items() if key != 'local_ref'}
                payload['identifier'] = payload.pop('id')
                result = self.resources.put(**payload)
                value = result['resource_id']
            elif group == 'lessons':
                value = self.store._propose('lesson', payload, source_ids=operation['source_ids'])['candidate_id']
            elif group == 'packages':
                value = self.library.save_package(payload, source_ids=operation['source_ids'])['package_id']
            elif group == 'workflows':
                value = self.library.save_application_skill(payload, source_ids=operation['source_ids'])['record_id']
            else:
                value = self.library.record_improvement(payload, source_ids=operation['source_ids'])['record_id']
            if value != operation['identifier']:
                raise LearningContractError('single-owner preflight identity drift')
            self._hook('after_object_write:' + str(index))
        self._hook('before_final_commit')
        for value in plan['record_ids']:
            if value.startswith('resource-'):
                self.resources.get(value)
            else:
                self.library.inspect(value)
        result = self._write_record('commits', dict(protocol=PROTOCOLS['commit'], status='complete',
            commit_id=commit_id, sequence=sequence, intent_sha256=intent['record_sha256'],
            bindings=plan['bindings'], record_ids=plan['record_ids'], source_bindings=plan['source_bindings'],
            relations=plan['relations'], selections=plan['selections'], scope=self.scope, summary=plan['summary']))
        self._restore_bindings()
        self._hook('after_final_commit')
        return clone(result)
