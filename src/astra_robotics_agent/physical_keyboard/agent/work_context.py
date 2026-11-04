"""Auditable work checkpoints and explicit, host-authorized context operations.

The caller supplies authoritative facts separately from the model declaration.
Neither a checkpoint nor a new provider thread confers execution authority.
"""
import hashlib
import json
from pathlib import Path
import threading
import time

from ..records.learning_access import LearningAccess, LearningAccessDenied
from ..records.learning_contracts import (
    PROTOCOLS, LearningContractError, clone, digest, encoded, exact, identifier,
    integer, sha256, text, validate_checkpoint,
)
from ..records.journal import SegmentedJournal
from ..records.storage import parent_fd, read_bytes, write_new
from ..skills.development import SkillDevelopment, _check_directory
from ..skills.learning_changes import LearningChanges
from ..skills.resources import LearningResources


class WorkContextError(LearningContractError):
    pass


_FACTS = {'task_contract_sha256', 'observation_evidence_id', 'generation', 'last_execution',
          'motion_available', 'inflight_rpc', 'inflight_action', 'inflight_model',
          'stop_requested', 'assistance', 'fault', 'source_policy_sha256'}
_MAX_RECORD = 2 * 1024 * 1024


def _decoded(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WorkContextError('duplicate checkpoint JSON field')
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(WorkContextError('nonfinite checkpoint')))
        encoded(value, _MAX_RECORD)
        return value
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise WorkContextError('invalid checkpoint JSON') from exc


class WorkContext:
    def __init__(self, session, store, *, output, access=None):
        self.session, self.store = session, store
        self.scope = clone(store._scope)
        self.access = access or getattr(store, 'learning_access', None) or LearningAccess(self.scope)
        if self.access.scope != self.scope:
            raise WorkContextError('work context access scope mismatch')
        self.library = SkillDevelopment(store)
        self.resources = LearningResources(store, access=self.access)
        self.changes = LearningChanges(self.library, self.resources, access=self.access)
        self.output = Path(output)
        _check_directory(self.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.output / 'checkpoints'
        self.checkpoints.mkdir(exist_ok=True)
        # Validate all existing ancestors without following symlinks.
        import os
        descriptor, _ = parent_fd(self.checkpoints / 'probe')
        os.close(descriptor)
        self.journal = SegmentedJournal(self.output / 'context-events.jsonl', max_line_bytes=_MAX_RECORD)
        self._lock = threading.Lock()

    def _event(self, event, **fields):
        self.journal.append(encoded(dict(event=event, time=time.time(), **fields), _MAX_RECORD) + b'\n')

    def _facts(self, value):
        exact(value, _FACTS)
        result = clone(value, 65536)
        for key in ('task_contract_sha256', 'source_policy_sha256'):
            sha256(result[key])
        integer(result['generation'], 'generation', 0, 2 ** 63 - 1)
        for key in ('motion_available', 'inflight_rpc', 'inflight_action', 'inflight_model', 'stop_requested'):
            if type(result[key]) is not bool:
                raise WorkContextError('explicit authoritative boolean required: ' + key)
        for key in ('assistance', 'fault'):
            if result[key] is not None and type(result[key]) is not dict:
                raise WorkContextError('bounded authoritative object or null required: ' + key)
        if result['observation_evidence_id'] is None:
            if result['motion_available']:
                raise WorkContextError('motion boundary needs a current registered observation')
        else:
            identifier(result['observation_evidence_id'])
        execution = result['last_execution']
        if execution is not None:
            exact(execution, {'evidence_id', 'status', 'motion_outcome_uncertain', 'controller_terminal'})
            identifier(execution['evidence_id'])
            text(execution['status'], 'execution status', 128)
            if any(type(execution[key]) is not bool for key in ('motion_outcome_uncertain', 'controller_terminal')):
                raise WorkContextError('explicit execution uncertainty/terminal flags required')
        return result

    @staticmethod
    def _unknown(facts):
        execution = facts['last_execution']
        return execution is not None and (execution['status'] == 'unknown'
            or execution['motion_outcome_uncertain'] or execution['controller_terminal'])

    def _references(self, declaration, facts):
        knowledge, evidence = {}, {}
        def check_evidence(value):
            if value in evidence:
                return
            record = self.store._evidence(value)  # Includes immutable object hash/size checks.
            if record.get('source_ids'):
                self.access.bind_evidence(value, record['source_ids'])
            sources = self.access.evidence_sources(value)
            evidence[value] = dict(sha256=record['sha256'], source_ids=sorted(sources))
        def check_knowledge(value):
            if value in knowledge:
                return
            identifier(value)
            if not self.changes.is_committed(value):
                raise WorkContextError('uncommitted knowledge cannot enter a checkpoint')
            if value.startswith('resource-'):
                record = self.resources.get(value)
            else:
                record = self.library.inspect(value)
            if record.get('scope') != self.scope:
                raise WorkContextError('checkpoint reference scope mismatch')
            if record.get('source_ids'):
                self.access.bind_knowledge(value, record['source_ids'])
            sources = self.changes.source_ids(value)
            self.access.check_sources(sources)
            knowledge[value] = dict(sha256=digest(record), kind=record['kind'], source_ids=sorted(sources))
            for evidence_id in record.get('evidence_ids', []):
                check_evidence(evidence_id)
            dependencies = list(record.get('dependencies', {}).values())
            if record.get('parent_id'):
                dependencies.append(record['parent_id'])
            for key in ('improvement_ids', 'skill_ids', 'knowledge_ids', 'related_skill_ids', 'related_lesson_ids', 'resource_ids'):
                dependencies.extend(record.get(key, []))
            if record.get('merged_into'):
                dependencies.append(record['merged_into'])
            for dependency in dependencies:
                check_knowledge(dependency)
        for value in declaration['knowledge_ids'] + declaration['pending_issues']:
            check_knowledge(value)
        for issue in declaration['pending_issues']:
            if knowledge[issue]['kind'] != 'improvement':
                raise WorkContextError('pending_issues needs exact improvement record IDs')
        for value in declaration['evidence_ids']:
            check_evidence(value)
        if facts['observation_evidence_id'] is not None:
            check_evidence(facts['observation_evidence_id'])
        if facts['last_execution'] is not None:
            check_evidence(facts['last_execution']['evidence_id'])
        return dict(knowledge=knowledge, evidence=evidence)

    def _access_snapshot(self, saved):
        current = self.access.descriptor()
        if saved.get('scope') != self.scope:
            raise WorkContextError('checkpoint source scope mismatch')
        if not current['restricted']:
            return
        if not saved.get('restricted'):
            raise LearningAccessDenied('unrestricted checkpoint payload is outside restricted source policy')
        for key in ('allowed_source_ids', 'allowed_knowledge_ids', 'allowed_evidence_ids'):
            if not set(saved[key]) <= set(current[key]):
                raise LearningAccessDenied('checkpoint payload was created under broader source permissions')
        if saved['allow_own_experience'] and not current['allow_own_experience']:
            raise LearningAccessDenied('checkpoint own-experience permissions changed')

    def save(self, args, facts):
        try:
            declaration = validate_checkpoint(args)
            authoritative = self._facts(facts)
            references = self._references(declaration, authoritative)
            record = dict(protocol=PROTOCOLS['checkpoint'], scope=self.scope,
                          declaration=declaration, authoritative_facts=authoritative,
                          facts_sha256=digest(authoritative), references=references,
                          access_policy=self.access.descriptor(),
                          provider_identity=self._session_identity(),
                          thread_id=getattr(self.session, 'thread_id', None),
                          authority='auditable_task_state_not_execution_permission',
                          model_reasoning_saved=False)
            checkpoint_id = 'checkpoint-' + digest(record)
            raw = encoded(record, _MAX_RECORD)
            path = self.checkpoints / (checkpoint_id + '.json')
            if path.exists() or path.is_symlink():
                if read_bytes(path, _MAX_RECORD) != raw:
                    raise WorkContextError('checkpoint object is corrupted')
            else:
                write_new(path, raw)
            descriptor = dict(checkpoint_id=checkpoint_id, sha256=hashlib.sha256(raw).hexdigest(),
                              bytes=len(raw), facts_sha256=record['facts_sha256'], scope=clone(self.scope))
            self._event('checkpoint_saved', **descriptor)
            return descriptor
        except Exception as exc:
            self._event('checkpoint_save_rejected', error_type=type(exc).__name__)
            raise

    def read(self, checkpoint_id):
        try:
            identifier(checkpoint_id, 'checkpoint_id')
            if not checkpoint_id.startswith('checkpoint-'):
                raise WorkContextError('exact checkpoint ID required')
            sha256(checkpoint_id[len('checkpoint-'):])
            raw = read_bytes(self.checkpoints / (checkpoint_id + '.json'), _MAX_RECORD)
            record = _decoded(raw)
            exact(record, {'protocol', 'scope', 'declaration', 'authoritative_facts', 'facts_sha256',
                           'references', 'access_policy', 'provider_identity', 'thread_id',
                           'authority', 'model_reasoning_saved'})
            if (record['protocol'] != PROTOCOLS['checkpoint'] or record['scope'] != self.scope
                    or 'checkpoint-' + digest(record) != checkpoint_id
                    or record['model_reasoning_saved'] is not False
                    or record['authority'] != 'auditable_task_state_not_execution_permission'):
                raise WorkContextError('checkpoint hash, identity or scope mismatch')
            self._access_snapshot(record['access_policy'])
            declaration = validate_checkpoint(record['declaration'])
            facts = self._facts(record['authoritative_facts'])
            if digest(facts) != record['facts_sha256'] or self._references(declaration, facts) != record['references']:
                raise WorkContextError('checkpoint facts or pinned reference changed')
            return clone(dict(record, checkpoint_id=checkpoint_id), _MAX_RECORD)
        except Exception as exc:
            self._event('checkpoint_read_rejected', error_type=type(exc).__name__)
            raise

    def _session_identity(self):
        result = {}
        for key in ('model', 'effort', 'service_tier'):
            value = getattr(self.session, key, None)
            if value is not None:
                text(value, key, 128)
            result[key] = value
        return result

    def request(self, args, facts):
        if not self._lock.acquire(blocking=False):
            raise WorkContextError('context operation already in flight')
        old_thread = getattr(self.session, 'thread_id', None)
        provider_called = False
        try:
            exact(args, {'checkpoint_id', 'method', 'reason'})
            identifier(args['checkpoint_id']); text(args['reason'], 'reason', 1024)
            if args['method'] not in ('native_compact', 'rebuild'):
                raise WorkContextError('explicit native_compact or rebuild required')
            current = self._facts(facts)
            record = self.read(args['checkpoint_id'])
            if record['authoritative_facts'] != current:
                raise WorkContextError('checkpoint is stale for the current authoritative boundary')
            if current['stop_requested']:
                raise WorkContextError('STOP prevents context operations')
            if any(current[key] for key in ('inflight_rpc', 'inflight_action', 'inflight_model')):
                raise WorkContextError('context operations require no in-flight model, action or RPC')
            if current['motion_available'] and (self._unknown(current) or current['fault'] is not None):
                raise WorkContextError('unknown or fault permits only explicit read-only context')
            if type(old_thread) is not str or not old_thread:
                raise WorkContextError('started provider thread required')
            identity = self._session_identity()
            if record['provider_identity'] != identity or record['thread_id'] != old_thread:
                raise WorkContextError('checkpoint provider configuration or thread is stale')
            self._event('context_operation_requested', checkpoint_id=args['checkpoint_id'],
                        method=args['method'], reason=args['reason'], thread_id=old_thread,
                        facts_sha256=record['facts_sha256'])
            operation = getattr(self.session, 'compact_context' if args['method'] == 'native_compact'
                                else 'rotate_thread', None)
            if not callable(operation):
                provider = dict(status='unsupported', requires_explicit_rebuild_choice=True)
            else:
                provider_called = True
                provider = operation() if args['method'] == 'native_compact' else operation(args['reason'])
            provider = clone(provider, 128 * 1024)
            if type(provider) is not dict:
                raise WorkContextError('provider context operation requires an object receipt')
            if (args['method'] == 'native_compact' and provider.get('status') not in {'completed', 'unsupported'}
                    or 'status' in provider and provider['status'] not in {'completed', 'unsupported'}):
                raise WorkContextError('provider context operation did not explicitly complete')
            if self._session_identity() != identity:
                self.session.close()
                raise WorkContextError('provider changed pinned model configuration')
            new_thread = getattr(self.session, 'thread_id', None)
            unsupported = provider.get('status') == 'unsupported'
            if ((args['method'] == 'native_compact' or unsupported) and new_thread != old_thread
                    or args['method'] == 'rebuild' and not unsupported and
                    (type(new_thread) is not str or not new_thread or new_thread == old_thread)):
                self.session.close()
                raise WorkContextError('provider context lineage mismatch')
            working = dict(checkpoint_id=args['checkpoint_id'], scope=clone(self.scope),
                           task_state=clone(record['declaration']), authoritative_facts=clone(current),
                           facts_sha256=record['facts_sha256'], references=clone(record['references']),
                           current_state=('unknown' if self._unknown(current) or current['fault'] is not None
                                          or current['observation_evidence_id'] is None
                                          else 'observation_requires_refresh'),
                           current_observation_evidence_id=None, historical_observation_evidence_id=current['observation_evidence_id'],
                           motion_available=current['motion_available'], fresh_observation_required=True,
                           authority='checkpoint_does_not_authorize_actions_or_restore_hidden_reasoning')
            result = dict(status='unsupported' if unsupported else 'completed', method=args['method'],
                          previous_thread_id=old_thread, thread_id=new_thread, working_state=working,
                          fresh_observation_required=True, provider=provider,
                          total_usage=clone(getattr(self.session, 'total_usage', {})),
                          total_usage_complete=getattr(self.session, 'total_usage_complete', False),
                          generation=current['generation'], motion_available=current['motion_available'])
            self._event('context_operation_result', **result)
            return result
        except BaseException as exc:
            if provider_called:
                close = getattr(self.session, 'close', None)
                if callable(close):
                    close()
            self._event('context_operation_rejected', previous_thread_id=old_thread,
                        error_type=type(exc).__name__, total_usage=clone(getattr(self.session, 'total_usage', {})),
                        total_usage_complete=getattr(self.session, 'total_usage_complete', False))
            raise
        finally:
            self._lock.release()
