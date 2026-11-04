"""Read-only, scope-matched recall for model-declared skill relationships.

Version/parent links define capability families. Dependencies are followed only
downwards: sharing a component does not make two compositions the same skill.
This provides historical model reports, not a repair, ranking of physical
quality, authorization to execute, or an independently verified outcome.
"""
import json

from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeError, _encoded, _sha
from astra_robotics_agent.physical_keyboard.runtime.skill_language import Limits
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccessDenied


PROTOCOL = 'physical-keyboard-skill-recall-v1'
_BYTE_LIMIT = 6144


def _excerpt(value, maximum=160):
    value = ' '.join(str(value).split())
    return value if len(value) <= maximum else value[:maximum - 3] + '...'


def _size(value):
    # Match the Agent's canonical JSON, including its default separators.
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8'))


class SkillRecall:
    """A fresh validated snapshot on each call; no persistence or stale cache."""

    def __init__(self, library):
        self.library = library
        self.store = library.store

    def _snapshot(self):
        events = self.store._events()
        candidates, order = {}, {}
        for index, event in enumerate(events):
            if event.get('type') == 'candidate_proposed' and event.get('kind') == 'skill':
                identifier = event['candidate_id']
                try:
                    candidates[identifier] = self.store._candidate(identifier)
                except LearningAccessDenied:
                    continue
                order[identifier] = index
        records = self.library._records()
        packages = {item['record_id']: item for item in records if item['kind'] == 'package'}
        # Validate source/profile/dependency pins through the established reader.
        for identifier in packages:
            self.library._resolve_skill(identifier)
        superseded = {identifier for event in events if event.get('type') == 'skill_superseded'
                      for identifier in event['candidate_ids']}
        return candidates, packages, records, order, superseded

    @staticmethod
    def _focus(skill_ids):
        if (type(skill_ids) is not list or len(skill_ids) > 16
                or any(type(value) is not str for value in skill_ids)
                or len(set(skill_ids)) != len(skill_ids)):
            raise KnowledgeError('recall skill_ids must contain 0..16 unique registered skill/package IDs')
        return list(skill_ids)

    def matching_source(self, source):
        if type(source) is not str:
            raise KnowledgeError('matching source must be an exact Unicode string')
        try:
            raw = source.encode('utf-8')
        except UnicodeError as exc:
            raise KnowledgeError('matching source must be valid Unicode') from exc
        if len(raw) > Limits().max_source_bytes:
            raise KnowledgeError('matching source exceeds interpreter source budget')
        digest = _sha(raw)
        candidates, _, _, order, _ = self._snapshot()
        matches = [identifier for identifier, item in candidates.items()
                   if item['source_sha256'] == digest and item['source'] == source]
        return sorted(matches, key=lambda item: order[item], reverse=True)[:16]

    def for_skills(self, skill_ids, limit=4):
        focus = self._focus(skill_ids)
        if type(limit) is not int or not 1 <= limit <= 8:
            raise KnowledgeError('recall limit must be 1..8')
        candidates, packages, records, order, superseded = self._snapshot()
        registered = set(candidates) | set(packages)
        if any(identifier not in registered for identifier in focus):
            raise KnowledgeError('recall skill/package is not registered in this scope')

        neighbours = {identifier: set() for identifier in registered}
        def connect(first, second):
            if first not in registered or second not in registered:
                raise KnowledgeError('skill family reference is not registered in this scope')
            neighbours[first].add(second)
            neighbours[second].add(first)

        families = {}
        for identifier, candidate in candidates.items():
            first = families.setdefault(candidate['id'], identifier)
            connect(first, identifier)
            parent = candidate.get('revision', {}).get('parent_candidate_id')
            if parent is not None:
                connect(identifier, parent)
        for identifier, package in packages.items():
            connect(identifier, package['candidate_id'])
            if package.get('parent_id') is not None:
                connect(identifier, package['parent_id'])

        related, pending = set(), list(focus)
        while pending:
            identifier = pending.pop()
            if identifier in related:
                continue
            if identifier not in registered:
                raise KnowledgeError('skill dependency is not registered in this scope')
            related.add(identifier)
            pending.extend(neighbours[identifier] - related)
            if identifier in packages:
                # No reverse dependency edge: a shared approach does not recall
                # every unrelated composition which also uses that approach.
                pending.extend(packages[identifier]['dependencies'].values())

        latest_issues, latest_outcomes, record_order = {}, {}, {}
        issue_records = {}
        for index, item in enumerate(records):
            record_order[item['record_id']] = index
            if item['kind'] == 'improvement':
                latest_issues[item['id']] = item
                issue_records[item['record_id']] = item
            elif item['kind'] == 'outcome':
                latest_outcomes[item['id']] = item
        selected, related_issue_versions, merged_matches = {}, set(), {}
        for initial in latest_issues.values():
            if not set(initial['skill_ids']) & related:
                continue
            item, seen = initial, set()
            while True:
                if item['id'] in seen:
                    raise KnowledgeError('cyclic merged improvement family')
                seen.add(item['id'])
                related_issue_versions.add(item['record_id'])
                if item['status'] != 'merged':
                    break
                target = issue_records.get(item.get('merged_into'))
                if target is None:
                    raise KnowledgeError('merged improvement target is not registered in this scope')
                item = latest_issues[target['id']]
            if item['status'] in {'open', 'reopened', 'deferred'}:
                selected[item['id']] = item
                if initial['id'] != item['id']:
                    merged_matches.setdefault(item['record_id'], []).append(initial['record_id'])
        issues = sorted(selected.values(),
                        key=lambda item: (item.get('priority', 50), record_order[item['record_id']]), reverse=True)
        outcomes = [item for item in latest_outcomes.values() if set(item['skill_ids']) & related]
        def negative(item):
            return (item['task_result'] != 'success' or item['correction_performed']
                    or bool(item['remaining_issues']))
        # Keep the current result visible even after many old failures. Use the
        # remaining slots for negative/compensated history, without presenting
        # that history as proof that a now-resolved issue remains open.
        newest_outcome = max((record_order[item['record_id']] for item in outcomes), default=-1)
        outcomes.sort(key=lambda item: (record_order[item['record_id']] == newest_outcome,
                      negative(item), record_order[item['record_id']]), reverse=True)
        focus_candidates = {packages[value]['candidate_id'] if value in packages else value for value in focus}
        packaged_candidates = {item['candidate_id'] for item in packages.values()}
        revisions = [item for item in packages.values()
                     if item['record_id'] in related and item['candidate_id'] not in focus_candidates]
        revisions += [item for identifier, item in candidates.items()
                      if identifier in related and identifier not in focus_candidates
                      and identifier not in packaged_candidates]
        revisions.sort(key=lambda item: order[item['candidate_id']], reverse=True)

        def report_summary(item):
            result = {key: item[key] for key in ('record_id', 'kind', 'id', 'version')}
            result.update(summary=_excerpt(item['summary']), skill_ids=item['skill_ids'][:2],
                          evidence_ids=item['evidence_ids'][:2], basis='model_report')
            if item['kind'] == 'improvement':
                result.update(status=item['status'], rationale=_excerpt(item['rationale']),
                              priority=item.get('priority', 50),
                              revisit_when=[_excerpt(value, 100) for value in item.get('revisit_when', [])[:2]])
                if item['record_id'] in merged_matches:
                    result['matched_via_merged_ids'] = merged_matches[item['record_id']][:2]
            else:
                result.update(task_result=item['task_result'], correction_performed=item['correction_performed'],
                              remaining_issues=[_excerpt(value, 100) for value in item['remaining_issues'][:2]],
                              historical=True, independently_verified=False)
            return result

        def revision_summary(item):
            identifier = item.get('record_id', item['candidate_id'])
            result = dict(record_id=identifier, candidate_id=item['candidate_id'], kind=item['kind'],
                          id=item['id'], version=item['version'], origin=item['origin'],
                          superseded=item['candidate_id'] in superseded, status='candidate_not_promoted')
            if item['kind'] == 'package':
                result.update(parent_id=item.get('parent_id'), description=_excerpt(item['contract']['description']),
                              preconditions=[_excerpt(value, 100) for value in item['contract']['preconditions'][:1]])
            else:
                result.update(parent_id=item.get('revision', {}).get('parent_candidate_id'),
                              applicability=[_excerpt(value, 100) for value in item['applicability'][:1]])
            return result

        signature = dict(scope=self.library.scope, focus_skill_ids=sorted(focus),
                         related_skill_ids=sorted(related), issue_versions=sorted(related_issue_versions),
                         outcome_versions=sorted(item['record_id'] for item in outcomes),
                         revision_ids=sorted(item.get('record_id', item['candidate_id']) for item in revisions))
        result = dict(protocol=PROTOCOL, scope=self.library.scope, focus_skill_ids=focus,
                      pending_improvements=[report_summary(item) for item in issues[:limit]],
                      recent_outcomes=[report_summary(item) for item in outcomes[:limit]],
                      candidate_revisions=[revision_summary(item) for item in revisions[:limit]],
                      pending_count=len(issues), outcome_count=len(outcomes), revision_count=len(revisions),
                      attention_record_ids=[], fingerprint=_sha(_encoded(signature)),
                      summary_truncated=False, summary_byte_limit=_BYTE_LIMIT,
                      meaning='Model reports and related candidates; no automatic repair, selection or promotion.')

        def update_metadata():
            attention = [item['record_id'] for item in result['pending_improvements']]
            attention += [item['record_id'] for item in result['recent_outcomes'] if negative(item)]
            if attention:
                attention += [item['record_id'] for item in result['candidate_revisions']]
            result['attention_record_ids'] = list(dict.fromkeys(attention))
            result['omitted_counts'] = {
                'pending_improvements': len(issues) - len(result['pending_improvements']),
                'recent_outcomes': len(outcomes) - len(result['recent_outcomes']),
                'candidate_revisions': len(revisions) - len(result['candidate_revisions'])}
            result['summary_truncated'] = any(result['omitted_counts'].values()) or shrunk

        shrunk = False
        update_metadata()
        while _size(result) > _BYTE_LIMIT:
            shrunk = True
            if result['candidate_revisions']:
                result['candidate_revisions'].pop()
            elif len(result['recent_outcomes']) > (0 if result['pending_improvements'] else 1):
                result['recent_outcomes'].pop()
            elif len(result['pending_improvements']) > 1:
                result['pending_improvements'].pop()
            else:
                # Retain one minimal issue (or outcome) rather than silently
                # losing the reason for a model-facing contextual review.
                entries = result['pending_improvements'] or result['recent_outcomes']
                if not entries:
                    raise KnowledgeError('recall metadata exceeds byte budget')
                item = entries[0]
                minimal = {key: item[key] for key in ('record_id', 'kind', 'id', 'version', 'summary')}
                minimal['summary'] = _excerpt(item['summary'], 64)
                if item['kind'] == 'improvement':
                    minimal.update(status=item['status'], priority=item.get('priority', 50),
                                   revisit_when=[_excerpt(value, 40) for value in item.get('revisit_when', [])[:1]])
                else:
                    minimal.update(task_result=item['task_result'], correction_performed=item['correction_performed'],
                                   remaining_issues=[_excerpt(value, 40) for value in item['remaining_issues'][:1]])
                if minimal == item:
                    raise KnowledgeError('minimal recall summary exceeds byte budget')
                entries[0] = minimal
            update_metadata()
        return result
