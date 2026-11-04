"""Mechanical, bounded learning links and evidence cursors; no semantic writer.

Association is not independent validation. All state changes remain explicit
RSI declarations; this module only reads registered, permitted snapshots.
"""
from __future__ import annotations

import hashlib

from ..records.learning_access import LearningAccessDenied
from ..records.learning_contracts import LearningContractError, clone, digest, encoded, exact, integer
from .knowledge import KnowledgeError
from .recall import SkillRecall


PROTOCOL = 'physical-keyboard-learning-attention-v1'
CURSOR_PROTOCOL = 'physical-keyboard-learning-evidence-cursor-v1'
_BYTE_LIMIT = 8192
_AUDIT_NAMES = ('score', 'typing-buffer', 'typing_buffer', 'editor-buffer', 'editor_buffer', 'hidden-case', 'hidden_case', 'ground-truth')


def _excerpt(value, size=180):
    text = ' '.join(str(value).split())
    return text if len(text)<=size else text[:size-3]+'...'


class LearningAttention:
    def __init__(self, library, changes, reader=None):
        self.library, self.store, self.changes, self.reader = library, library.store, changes, reader
        self.scope = library.scope
        if changes is not None and changes.scope != self.scope:
            raise LearningContractError('attention scope mismatch')
        self._policies()
        self._resolved_seen = {}

    def _policies(self):
        accesses=[]
        reader,changes=self.reader,self.changes
        for access in (getattr(reader,'access',None),getattr(changes,'access',None),self.store.learning_access):
            if access is not None and all(access is not prior for prior in accesses):
                if access.scope!=self.scope:raise LearningContractError('attention policy scope mismatch')
                accesses.append(access)
        return accesses

    def _active(self, identifier):
        if self.changes is not None:
            if not self.changes.is_committed(identifier):
                return False
            self.changes.source_ids(identifier)
        if self.reader is not None:
            self.reader._get(identifier)
        return True

    def _get(self, identifier):
        if not self._active(identifier):
            raise LearningContractError('pending learning object is not active')
        if self.reader is not None:
            return self.reader._get(identifier)
        if identifier.startswith('resource-') and self.changes is not None:
            return self.changes.resources.get(identifier)
        return self.library.inspect(identifier)

    def _snapshot(self):
        candidates, packages, records, order, _ = SkillRecall(self.library)._snapshot()
        def allowed(identifier):
            try:
                return self._active(identifier)
            except LearningAccessDenied:
                return False
        candidates = {key:value for key,value in candidates.items() if allowed(key)}
        packages = {key:value for key,value in packages.items() if allowed(key)}
        records = [row for row in records if allowed(row['record_id'])]
        lessons = {}
        for event in self.store._events():
            if event.get('type') == 'candidate_proposed' and event.get('kind') == 'lesson':
                identifier = event['candidate_id']
                try:
                    if allowed(identifier):
                        lessons[identifier] = self.store._candidate(identifier)
                except LearningAccessDenied:
                    continue
        return candidates, packages, records, lessons

    @staticmethod
    def _related(focus, candidates, packages):
        registered = set(candidates)|set(packages)
        if any(identifier not in registered for identifier in focus):
            raise KnowledgeError('attention skill/package is not registered or permitted')
        neighbours = {identifier:set() for identifier in registered}
        def connect(first, second):
            if first in registered and second in registered:
                neighbours[first].add(second);neighbours[second].add(first)
        families = {}
        for identifier, candidate in candidates.items():
            connect(families.setdefault(candidate['id'],identifier),identifier)
            parent=candidate.get('revision',{}).get('parent_candidate_id')
            if parent is not None:connect(identifier,parent)
        for identifier, package in packages.items():
            connect(identifier,package['candidate_id'])
            if package.get('parent_id') is not None:connect(identifier,package['parent_id'])
        related,pending=set(),list(focus)
        while pending:
            identifier=pending.pop()
            if identifier in related:continue
            if identifier not in registered:raise KnowledgeError('attention dependency missing')
            related.add(identifier);pending.extend(neighbours[identifier]-related)
            if identifier in packages:
                # Composition dependencies are one-way. A shared component
                # never adds all of its callers to the requested family.
                pending.extend(packages[identifier].get('dependencies',{}).values())
        return related

    def _evidence(self, identifier):
        record=self.store._evidence(identifier)
        names=[item['source_path'].replace('\\','/').rsplit('/',1)[-1].lower() for item in record['files']]
        if any(token in name for name in names for token in _AUDIT_NAMES):
            raise LearningAccessDenied('audit-only evidence is not a learning attention source')
        sources=set(record.get('source_ids',[]))
        policies=self._policies()
        for access in policies:
            if 'source_ids' in record:
                access.bind_evidence(identifier,access.check_sources(record['source_ids'],require_provenance=True))
            sources.update(access.evidence_sources(identifier))
        for access in policies:access.check_sources(sorted(sources))
        file_names=set(names)
        kind=('execution' if {'skill.py','profile.json','response.json'}<=file_names
              else 'observation' if any(name.endswith(('.jpg','.png')) for name in names)
              else 'registered_evidence')
        group=digest(sorted([item['sha256'],item['bytes']] for item in record['files']))
        return record,dict(evidence_id=identifier,sha256=record['sha256'],kind=kind,
            source_ids=sorted(sources),run_id=record['run_id'],object_count=len(record['files']),
            evidence_group_id=group,historical=True,independently_verified=False)

    def _evidence_events(self):
        events=self.store._events();visible=[]
        for position,event in enumerate(events):
            if event.get('type')!='evidence_registered':continue
            try:
                record,meta=self._evidence(event['evidence_id'])
            except LearningAccessDenied:
                continue
            if record['sha256']!=event['sha256']:
                raise LearningContractError('registered evidence event hash mismatch')
            visible.append((position,record,meta))
        return events,visible

    @staticmethod
    def _referenced(records):
        return {identifier for row in records for identifier in row.get('evidence_ids',[])}

    def _associations(self, related, candidates, packages, records, lessons):
        associations=[];issue_ids=set();lesson_ids=set()
        focus_evidence=set()
        for identifier in sorted(related):
            row=packages.get(identifier,candidates.get(identifier))
            focus_evidence.update(row.get('evidence_ids',[]))
            for issue in row.get('improvement_ids',[]):
                issue_ids.add(issue)
                associations.append(dict(from_id=identifier,to_id=issue,association_kind='package_improvement_ids'))
        for identifier,lesson in lessons.items():
            shared=sorted(focus_evidence & set(lesson.get('evidence_ids',[])))
            if shared:
                lesson_ids.add(identifier)
                associations.append(dict(to_id=identifier,association_kind='same_registered_evidence',evidence_ids=shared[:8]))
        origins=set(related)
        relations=[] if self.changes is None else self.changes.relations()
        checked=set()
        while origins-checked:
            origin=sorted(origins-checked)[0];checked.add(origin)
            if len(checked)>1024:raise LearningContractError('attention relationship traversal exceeds bounded graph')
            for relation in relations:
                if relation['from_id']!=origin or relation['kind'] not in {'uses_lesson','derived_from','addresses_issue'}:continue
                target=relation['to_id']
                try:
                    row=self._get(target)
                except LearningAccessDenied:
                    continue
                if row.get('kind')=='lesson':lesson_ids.add(target)
                if row.get('kind')=='improvement':issue_ids.add(target)
                associations.append(dict(from_id=origin,to_id=target,association_kind='explicit_'+relation['kind'],
                    relation_id=relation['relation_id'],conditions=[_excerpt(value,100) for value in relation.get('conditions',[])[:2]],
                    reason=_excerpt(relation.get('reason','explicit registered relationship'),160),
                    conditions_evaluated=False))
                if relation['kind']=='derived_from':origins.add(target)
        return lesson_ids,issue_ids,associations

    def for_skills(self, skill_ids, limit=4):
        with self.store.read_snapshot():
            return self._for_skills(skill_ids, limit)

    def _for_skills(self, skill_ids, limit=4):
        focus=SkillRecall._focus(skill_ids);integer(limit,'limit',1,8)
        if not focus:
            result=dict(protocol=PROTOCOL,scope=clone(self.scope),focus_skill_ids=[],lessons=[],linked_improvements=[],
                revisit_resolved=[],associations=[],attention_record_ids=[],independent_validation_count=0,
                basis='explicit relationships or exact registered evidence; no semantic inference',
                summary_byte_limit=_BYTE_LIMIT,omitted={},fingerprint='')
            self._finish(result,dict(lessons=0,linked_improvements=0,revisit_resolved=0,associations=0))
            return result
        for identifier in focus:self._get(identifier)
        candidates,packages,records,lessons=self._snapshot()
        related=self._related(focus,candidates,packages)
        lesson_ids,explicit_issues,associations=self._associations(related,candidates,packages,records,lessons)
        latest={};record_order={}
        for position,row in enumerate(records):
            record_order[row['record_id']]=position
            if row['kind']=='improvement':latest[row['id']]=row
        issue_logical={row['id'] for row in records if row['record_id'] in explicit_issues}
        linked=[row for row in latest.values() if row['id'] in issue_logical or set(row['skill_ids'])&related]
        events,evidence=self._evidence_events()
        evidence_positions={meta['evidence_id']:position for position,_,meta in evidence}
        all_rows=list(candidates.values())+list(packages.values())+records+list(lessons.values())
        associated=self._referenced(all_rows)
        outcomes=[row for row in records if row['kind']=='outcome' and set(row['skill_ids'])&related]
        revisit=[]
        def summary(row):
            return dict(record_id=row['record_id'],id=row['id'],version=row['version'],status=row['status'],
                        summary=_excerpt(row['summary']),basis='model_report',independently_verified=False)
        for issue in linked:
            if issue['status']!='resolved':continue
            related_issue_skills=set(issue['skill_ids'])&related
            if issue['id'] in issue_logical:related_issue_skills=related
            negative=[row for row in outcomes if record_order[row['record_id']]>record_order[issue['record_id']]
                and set(row['skill_ids'])&related_issue_skills
                and (row['task_result']!='success' or row['correction_performed'] or row['remaining_issues'])]
            reasons=[]
            for row in negative:
                groups=sorted({self._evidence(value)[1]['evidence_group_id'] for value in row['evidence_ids']})
                reasons.append(dict(record_id=row['record_id'],association_kind='later_negative_model_outcome',
                    task_result=row['task_result'],correction_performed=row['correction_performed'],
                    remaining_issues=[_excerpt(value,120) for value in row['remaining_issues'][:2]],
                    evidence_group_ids=groups,after_resolution=True,independently_verified=False))
            issue_boundary=max((evidence_positions.get(value,-1) for value in issue['evidence_ids']),default=-1)
            source_hashes={candidates[value]['source_sha256'] for value in related_issue_skills if value in candidates}
            source_hashes.update(packages[value]['compiled_source_sha256'] for value in related_issue_skills if value in packages)
            seen_boundary=self._resolved_seen.get(issue['record_id'])
            for position,record,meta in evidence:
                if meta['kind']!='execution' or meta['evidence_id'] in associated or position<=issue_boundary:continue
                hashes={item['sha256'] for item in record['files'] if item['source_path'].replace('\\','/').endswith('/skill.py')}
                if not source_hashes & hashes:continue
                reasons.append(dict(evidence_id=meta['evidence_id'],evidence_group_ids=[meta['evidence_group_id']],
                    association_kind='unassociated_execution_source_hash',
                    after_resolution=True if seen_boundary is not None and position>=seen_boundary else 'unknown',
                    temporal_basis='observed_after_resolved_view' if seen_boundary is not None and position>=seen_boundary
                        else 'registered_after_issue_referenced_evidence; cross_journal_resolution_time_unknown',
                    independently_verified=False))
            self._resolved_seen.setdefault(issue['record_id'],len(events))
            if reasons:
                revisit.append(dict(**summary(issue),state_unchanged=True,new_evidence=reasons[:limit],
                    omitted_reason_count=max(0,len(reasons)-limit),meaning='Possible recurrence for RSI review; issue remains resolved'))
        lesson_rows=[dict(candidate_id=identifier,id=lessons[identifier]['id'],version=lessons[identifier]['version'],
            claim=_excerpt(lessons[identifier]['claim']),applicability=[_excerpt(value,100) for value in lessons[identifier]['applicability'][:2]],
            limitations=[_excerpt(value,100) for value in lessons[identifier]['limitations'][:2]],
            evidence_ids=lessons[identifier]['evidence_ids'][:8],basis='model_hypothesis',independently_verified=False)
            for identifier in lesson_ids if identifier in lessons]
        lesson_rows.sort(key=lambda row:row['candidate_id'])
        linked.sort(key=lambda row:record_order[row['record_id']],reverse=True)
        result=dict(protocol=PROTOCOL,scope=clone(self.scope),focus_skill_ids=focus,
            lessons=lesson_rows[:limit],linked_improvements=[summary(row) for row in linked[:limit]],
            revisit_resolved=revisit[:limit],associations=associations[:32],attention_record_ids=[],
            independent_validation_count=0,basis='explicit relationships or exact registered evidence; no semantic inference',
            summary_byte_limit=_BYTE_LIMIT,omitted={},fingerprint='')
        totals=dict(lessons=len(lesson_rows),linked_improvements=len(linked),revisit_resolved=len(revisit),associations=len(associations))
        self._finish(result,totals)
        return result

    @staticmethod
    def _finish(result,totals):
        def update():
            result['attention_record_ids']=[row['candidate_id'] for row in result['lessons']]
            result['attention_record_ids'] += [row['record_id'] for row in result['linked_improvements']
                if row['status'] in {'open','reopened','deferred'}]
            result['attention_record_ids'] += [row['record_id'] for row in result['revisit_resolved']]
            result['attention_record_ids']=list(dict.fromkeys(result['attention_record_ids']))
            result['omitted']={key:totals[key]-len(result[key]) for key in totals}
            result['summary_truncated']=any(result['omitted'].values())
            result['fingerprint']=digest({key:value for key,value in result.items() if key!='fingerprint'})
        update()
        while len(encoded(result,1024*1024))>_BYTE_LIMIT:
            for key in ('associations','lessons','linked_improvements','revisit_resolved'):
                if result[key]:result[key].pop();break
            else:raise LearningContractError('attention metadata cannot fit byte budget')
            update()

    def _cursor(self, events, position):
        fingerprint=hashlib.sha256()
        for event in events[:position]:
            fingerprint.update(encoded(event,1024*1024));fingerprint.update(b'\n')
        cursor=dict(protocol=CURSOR_PROTOCOL,scope=clone(self.scope),namespace_key=digest(str(self.store.namespace)),
                    position=position,prefix_sha256=fingerprint.hexdigest())
        return dict(cursor,sha256=digest(cursor))

    def _position(self, cursor, events):
        exact(cursor,{'protocol','scope','namespace_key','position','prefix_sha256','sha256'})
        integer(cursor['position'],'cursor position',0,len(events))
        if cursor!=self._cursor(events,cursor['position']):
            raise LearningContractError('evidence cursor scope, prefix or hash mismatch')
        return cursor['position']

    def new_evidence(self, cursor=None, limit=8, *, offset=0):
        with self.store.read_snapshot():
            return self._new_evidence(cursor, limit, offset=offset)

    def _new_evidence(self, cursor=None, limit=8, *, offset=0):
        integer(limit,'limit',1,16);integer(offset,'offset',0,1000000)
        if cursor is not None and offset:
            raise LearningContractError('cursor and historical offset cannot be mixed')
        events=self.store._events()
        position=self._position(cursor,events) if cursor is not None else None
        if position == len(events):
            # A validated unchanged journal needs no repeated object scans or
            # semantic/model work. This reports no new evidence, not a fresh
            # verification of all historical bytes.
            return dict(protocol=PROTOCOL,scope=clone(self.scope),items=[],next_cursor=self._cursor(events,len(events)),
                omitted_count=0,summary_byte_limit=_BYTE_LIMIT,
                basis='registered evidence not yet referenced by visible active learning objects',
                independent_validation_count=0,next_offset=None,unique_evidence_group_count=0)
        events,evidence=self._evidence_events()
        candidates,packages,records,lessons=self._snapshot()
        associated=self._referenced(list(candidates.values())+list(packages.values())+records+list(lessons.values()))
        available=[(index,meta) for index,_,meta in evidence if meta['evidence_id'] not in associated
                   and (position is None or index>=position)]
        if position is None:
            end=max(0,len(available)-offset);start=max(0,end-limit);selected=available[start:end]
        else:selected=available[:limit]
        result=dict(protocol=PROTOCOL,scope=clone(self.scope),items=[],next_cursor=None,
            omitted_count=0,summary_byte_limit=_BYTE_LIMIT,basis='registered evidence not yet referenced by visible active learning objects',
            independent_validation_count=0,next_offset=None)
        def update():
            result['items']=[clone(meta) for _,meta in selected]
            consumed=(selected[-1][0]+1 if selected and position is not None else len(events))
            if position is not None and not selected:consumed=len(events)
            result['next_cursor']=self._cursor(events,consumed)
            result['omitted_count']=max(0,len(available)-len(selected))
            result['next_offset']=(offset+len(selected) if position is None and len(available)>offset+len(selected) else None)
            result['unique_evidence_group_count']=len({row['evidence_group_id'] for row in result['items']})
        update()
        while len(encoded(result,1024*1024))>_BYTE_LIMIT:
            if len(selected)<=1:raise LearningContractError('one evidence metadata entry cannot fit byte budget; narrow source closure')
            selected.pop(0 if position is None else -1);update()
        return result
