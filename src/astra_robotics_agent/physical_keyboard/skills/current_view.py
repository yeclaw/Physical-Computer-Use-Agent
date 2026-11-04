"""Mechanical logical directory and explicit condition-specific selections.

The view never evaluates natural-language conditions or chooses executable
versions. Its state is rebuilt from complete learning commits on each read.
"""
import re

from ..records.learning_access import LearningAccessDenied
from ..records.knowledge_domains import knowledge_scope, validate_domain_filter, matches_knowledge_scope
from ..records.learning_contracts import LearningContractError, clone, digest, identifier, integer


class CurrentView:
    def __init__(self, library, changes):
        self.library, self.changes = library, changes
        self.scope = library.scope
        if self.scope != changes.scope:
            raise LearningContractError('current view scope mismatch')

    def _visible(self, value):
        if not self.changes.is_committed(value):
            return False
        try:
            self.changes.source_ids(value)
        except LearningAccessDenied:
            return False
        return True

    def selections(self, logical_id):
        with self.library.store.read_snapshot():
            return self._selections(logical_id)

    def _selections(self, logical_id):
        """Return current explicit branches; equal conditions replace explicitly."""
        identifier(logical_id, 'logical_id')
        current = {}
        for commit in self.changes.commits():
            for selection in commit['selections']:
                if selection['logical_id'] != logical_id or not self._visible(selection['identifier']):
                    continue
                # Full object/hash validation is independent of the selection.
                record = self.library.inspect(selection['identifier'])
                if record['scope'] != self.scope or record['id'] != logical_id:
                    raise LearningContractError('selected object identity or scope changed')
                key = (selection['kind'], digest(selection['conditions']))
                current[key] = dict(selection, commit_id=commit['commit_id'])
        return clone(list(current.values()))

    def relations(self):
        with self.library.store.read_snapshot():
            return self._relations()

    def _relations(self):
        return [relation for relation in self.changes.relations()
                if self._visible(relation['from_id']) and self._visible(relation['to_id'])]

    def directory(self, query='', kind='all', limit=8, offset=0, *,
                  version_offset=0, version_limit=4, branch_offset=0, branch_limit=4,
                  domain=None, application_id=None):
        with self.library.store.read_snapshot():
            return self._directory(query, kind, limit, offset, version_offset=version_offset,
                version_limit=version_limit, branch_offset=branch_offset, branch_limit=branch_limit,
                domain=domain, application_id=application_id)

    def _directory(self, query='', kind='all', limit=8, offset=0, *,
                   version_offset=0, version_limit=4, branch_offset=0, branch_limit=4,
                  domain=None, application_id=None):
        validate_domain_filter(domain, application_id)
        if type(query) is not str or len(query) > 512:
            raise LearningContractError('bounded directory query required')
        try:
            query.encode('utf-8')
        except UnicodeError as exc:
            raise LearningContractError('valid Unicode directory query required') from exc
        if type(kind) is not str or kind not in {'all', 'skill', 'package', 'lesson', 'improvement', 'outcome', 'resource', 'workflow'}:
            raise LearningContractError('recognized directory kind required')
        integer(limit, 'limit', 1, 32); integer(offset, 'offset', 0, 1000000)
        integer(version_offset,'version_offset',0,1000000);integer(version_limit,'version_limit',1,32)
        integer(branch_offset,'branch_offset',0,1000000);integer(branch_limit,'branch_limit',1,32)
        records = []
        development = self.library._records()
        backing = {row['candidate_id'] for row in development if row['kind'] == 'package'
                   and self._visible(row['record_id'])}
        for event in self.library.store._events():
            value = event.get('candidate_id')
            if event.get('type') == 'candidate_proposed' and value not in backing:
                try:
                    record = self.library.inspect(value)
                    if self._visible(value):
                        records.append(record)
                except LearningAccessDenied:
                    continue
        for row in development:
            if self._visible(row['record_id']):
                records.append(self.library.inspect(row['record_id']))
        for row in self.changes.resources.records():
            if self._visible(row['resource_id']):
                records.append(dict(row, kind='resource', resource_kind=row['kind']))
        superseded = {relation['to_id'] for relation in self.relations()
                      if relation['kind'] == 'supersedes' and not relation['conditions']}
        superseded.update(value for event in self.library.store._events()
                          if event.get('type') == 'skill_superseded' for value in event['candidate_ids'])
        groups = {}
        for record in records:
            if not matches_knowledge_scope(record, domain, application_id):
                continue
            value = record.get('record_id', record.get('candidate_id', record.get('resource_id')))
            key = (record['kind'], record['id'])
            group = groups.setdefault(key, dict(logical_id=record['id'], kind=record['kind'],
                scope=self.scope, versions=[], branches=[], selected=False, automatic_latest=False))
            descriptor = dict(identifier=value, version=record.get('version'),
                              superseded=value in superseded, knowledge_scope=knowledge_scope(record))
            for name in ('source_sha256', 'profile_sha256', 'compiled_source_sha256',
                         'compiled_profile_sha256', 'sha256', 'resource_kind', 'status'):
                if name in record:
                    descriptor[name] = record[name]
            group['versions'].append(descriptor)
        tokens = re.findall(r'[a-z0-9]+|[\u3400-\u9fff]+', query.casefold())
        rows = []
        for (record_kind, logical_id), group in sorted(groups.items()):
            if kind != 'all' and record_kind != kind:
                continue
            active = [version for version in group['versions'] if not version['superseded']]
            if not active:
                continue
            group['version_count'] = len(group['versions'])
            group['active_version_count'] = len(active)
            eligible_ids = {version['identifier'] for version in group['versions']}
            branches = [selection for selection in self.selections(logical_id)
                        if selection['kind'] == record_kind and selection['identifier'] not in superseded
                        and selection['identifier'] in eligible_ids]
            # Ordering is only a bounded metadata view. Full explicit branches
            # determine selection, including when a requested branch page is
            # empty or points to a version outside the current version page.
            group['selected'] = bool(branches)
            group['selection_state'] = 'explicit_branches' if group['selected'] else 'unselected'
            versions = [entry for _,entry in sorted(enumerate(group['versions']),key=lambda pair:(
                pair[1]['version'] if type(pair[1]['version']) is int else -1,
                pair[0],pair[1]['identifier']),reverse=True)]
            group['versions'] = versions[version_offset:version_offset+version_limit]
            version_end = version_offset+len(group['versions'])
            group.update(version_offset=version_offset,version_limit=version_limit,
                version_next_offset=version_end if version_end<len(versions) else None)
            group['branches'] = branches[branch_offset:branch_offset+branch_limit]
            branch_end = branch_offset+len(group['branches'])
            group.update(branches_count=len(branches),branch_offset=branch_offset,branch_limit=branch_limit,
                branch_next_offset=branch_end if branch_end<len(branches) else None)
            if tokens and not all(token in (logical_id + ' ' + record_kind).casefold() for token in tokens):
                continue
            rows.append(group)
        selected = rows[offset:offset + limit]
        next_offset = offset + len(selected)
        return clone(dict(scope=self.scope, query=query, kind=kind, items=selected,
            total_count=len(rows), offset=offset, next_offset=next_offset if next_offset < len(rows) else None,
            version_offset=version_offset,version_limit=version_limit,branch_offset=branch_offset,branch_limit=branch_limit,
            ordering=dict(versions='version_descending_then_registration_descending',
                          branches='stable_explicit_condition_registration_order',authority='metadata_only_no_automatic_latest'),
            selection='explicit_model_conditions_only_no_latest_execution',
            grouping='logical_id_and_kind; package backing candidates counted once'))
