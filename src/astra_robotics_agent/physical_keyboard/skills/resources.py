"""Scoped immutable JSON resources; analysis results never grant authority."""
import os
import uuid
from ..records.learning_access import LearningAccess, LearningAccessDenied
from ..records.learning_contracts import (
    PROTOCOLS, LearningContractError, clone, digest, encoded, exact, ids,
    identifier as valid_identifier, integer, sha256,
)
from .development import _check_directory
from .knowledge import _decoded, _profile, _read, _sha


class LearningResources:
    def __init__(self, store, *, access=None):
        self.store = store
        self.scope = clone(store._scope)
        self.access = access or LearningAccess(self.scope)
        if self.access.scope != self.scope:
            raise LearningContractError('resource access scope mismatch')
        self.root = store.namespace / 'learning' / 'resources'
        for directory in (self.root.parent, self.root, self.root / 'objects', self.root / 'registrations', self.root / 'staging'):
            _check_directory(directory)
            directory.mkdir(exist_ok=True)

    def prepare(self, identifier, value, *, source_ids, evidence_ids=None,
                kind='analysis_result', metadata=None):
        """Validate and compute exact identity without writing any object."""
        valid_identifier(identifier)
        if type(kind) is not str or kind not in {'analysis_result', 'profile', 'attachment', 'device_knowledge'}:
            raise LearningContractError('recognized inert resource kind required')
        value = clone(value, 512 * 1024)
        if kind == 'profile':
            _profile(value)  # Existing structural and authority checks, unchanged.
        sources = sorted(self.access.check_sources(source_ids, require_provenance=True))
        if not sources:
            raise LearningContractError('resource requires explicit source provenance')
        evidence = ids([] if evidence_ids is None else evidence_ids, 'evidence_ids', 16)
        for evidence_id in evidence:
            self.store._evidence(evidence_id)
            if not set(self.access.evidence_sources(evidence_id)) <= set(sources):
                raise LearningContractError('resource omits evidence source closure')
        metadata = {} if metadata is None else clone(metadata, 16384)
        if type(metadata) is not dict:
            raise LearningContractError('resource metadata must be a JSON object')
        record = dict(protocol=PROTOCOLS['resource'], id=identifier, kind=kind,
                      scope=clone(self.scope), source_ids=sources, evidence_ids=evidence,
                      value=value, sha256=_sha(encoded(value, 512 * 1024)), metadata=metadata,
                      authority='inert_model_proposal_not_execution_permission')
        record['resource_id'] = 'resource-' + digest(record)
        encoded(record, 640 * 1024)
        return record

    def _put_prepared(self, record):
        raw = encoded(record, 640 * 1024)
        object_sha = _sha(raw)
        target = self.root / 'objects' / (object_sha + '.json')
        if target.exists() or target.is_symlink():
            if _read(target, maximum=640 * 1024) != raw:
                raise LearningContractError('resource object hash collision or corruption')
        else:
            self._publish(target, raw)
        registration = dict(resource_id=record['resource_id'], scope=self.scope,
                            object_sha256=object_sha, bytes=len(raw))
        path = self.root / 'registrations' / (record['resource_id'] + '.json')
        registration_raw = encoded(registration)
        if path.exists() or path.is_symlink():
            if _read(path) != registration_raw:
                raise LearningContractError('resource registration is immutable')
        else:
            self._publish(path, registration_raw)
        return self.describe(record)

    def _publish(self, target, raw):
        stage = self.root / 'staging' / (uuid.uuid4().hex + '.json')
        with stage.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, target, follow_symlinks=False)

    def put(self, identifier, value, *, source_ids, evidence_ids=None,
            kind='analysis_result', metadata=None):
        from .read_scope import assert_writable
        assert_writable(self.store)
        return self._put_prepared(self.prepare(identifier, value, source_ids=source_ids,
            evidence_ids=evidence_ids, kind=kind, metadata=metadata))

    @staticmethod
    def describe(record):
        return clone({key: record[key] for key in ('resource_id', 'id', 'kind', 'scope',
            'sha256', 'source_ids', 'evidence_ids', 'metadata', 'authority')})

    def get(self, resource_id, *, expected_sha256=None):
        valid_identifier(resource_id)
        if not resource_id.startswith('resource-'):
            raise LearningContractError('registered resource ID required')
        path = self.root / 'registrations' / (resource_id + '.json')
        registration = _decoded(_read(path))
        exact(registration, {'resource_id', 'scope', 'object_sha256', 'bytes'})
        sha256(registration['object_sha256'])
        integer(registration['bytes'], 'bytes', 1, 640 * 1024)
        if registration['resource_id'] != resource_id or registration['scope'] != self.scope:
            raise LearningContractError('resource registration scope or identity mismatch')
        raw = _read(self.root / 'objects' / (registration['object_sha256'] + '.json'), maximum=640 * 1024)
        if _sha(raw) != registration['object_sha256'] or len(raw) != registration['bytes']:
            raise LearningContractError('resource object hash mismatch')
        record = _decoded(raw)
        exact(record, {'protocol', 'id', 'kind', 'scope', 'source_ids', 'evidence_ids',
                       'value', 'sha256', 'metadata', 'authority', 'resource_id'})
        if record['scope'] != self.scope:
            raise LearningContractError('resource scope mismatch')
        prepared = self.prepare(record['id'], record['value'], source_ids=record['source_ids'],
            evidence_ids=record['evidence_ids'], kind=record['kind'], metadata=record['metadata'])
        if record != prepared or record['resource_id'] != resource_id:
            raise LearningContractError('resource content identity mismatch')
        if expected_sha256 is not None and sha256(expected_sha256) != record['sha256']:
            raise LearningContractError('resource expected value hash mismatch')
        return clone(record, 640 * 1024)

    def records(self):
        result = []
        for path in sorted((self.root / 'registrations').glob('*.json')):
            try:
                result.append(self.get(path.stem))
            except LearningAccessDenied:
                continue
        return result
