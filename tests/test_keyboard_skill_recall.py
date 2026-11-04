"""Real temporary knowledge stores; no model, network, camera or robot IO."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeError, KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.recall import SkillRecall


SCOPE = {'task_id': 'recall_fixture', 'device_id': 'fake_device', 'layout_id': 'fake_layout'}
SOURCE = ['def main(robot, profile):', '    return {"text": profile["text"]}']


class SkillRecallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='skill-recall-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'first')
        evidence = self.root / 'evidence.json'
        evidence.write_text('{"fixture":"not a physical result"}')
        self.store.register_evidence('obs-fixture', [evidence])
        self.library = SkillDevelopment(self.store)
        self.recall = SkillRecall(self.library)

    def legacy(self, identifier='seed', lines=None):
        return self.store.propose('skill', dict(id=identifier, source_lines=lines or SOURCE,
            profile={'text': 'fixture'}, applicability=['Fixture only'],
            limitations=['No physical verification'], evidence_ids=['obs-fixture']))['candidate_id']

    def package(self, identifier='composable', parent=None, dependencies=None):
        args = dict(id=identifier, source_lines=SOURCE, profile={'text': 'fixture'},
            contract={'description': 'Model-authored fixture operation',
                'input_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}},
                                 'required': ['text'], 'additionalProperties': False},
                'preconditions': ['Inspect the current fixture state'],
                'postconditions': ['Assess fresh evidence']},
            applicability=['Fixture only'], limitations=['Not physically tested'], evidence_ids=['obs-fixture'])
        if parent:
            args['parent_id'] = parent
        if dependencies:
            args.update(dependencies=dependencies, source_lines=['def main(robot, profile):',
                '    return ' + next(iter(dependencies)) + '(robot, {"text": profile["text"]})'])
        return self.library.save_package(args)

    def issue(self, skill, identifier='extra_output', status='open', **extra):
        args = dict(id=identifier, summary='The fixture needed a compensating correction.',
            rationale='Keep a reusable skill defect after fixing the current task.',
            status=status, skill_ids=[skill], evidence_ids=['obs-fixture'])
        args.update(extra)
        return self.library.record_improvement(args)

    def outcome(self, skill, identifier='attempt', result='success', correction=True, remaining=None):
        return self.library.record_outcome(dict(id=identifier, skill_ids=[skill],
            evidence_ids=['obs-fixture'], observation_evidence_id='obs-fixture',
            task_result=result, correction_performed=correction,
            remaining_issues=['The underlying fixture defect remains.'] if remaining is None else remaining,
            summary='Historical fixture model report, not independent physical truth.'))

    def files(self):
        return {str(p.relative_to(self.store.namespace)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.store.namespace.rglob('*') if p.is_file()}

    def test_buried_related_issue_recalled_without_unrelated_recent_issues_or_writes(self):
        seed, other = self.legacy(), self.legacy('other')
        old = self.issue(seed)
        for number in range(10):
            self.issue(other, identifier='unrelated_' + str(number))
        self.assertNotIn(old['record_id'], json.dumps(self.library.context()))
        before = self.files()
        recall = self.recall.for_skills([seed])
        self.assertEqual(recall['pending_count'], 1)
        self.assertEqual(recall['pending_improvements'][0]['record_id'], old['record_id'])
        self.assertEqual(recall['attention_record_ids'], [old['record_id']])
        self.assertEqual(before, self.files())
        self.assertNotIn('source', recall['pending_improvements'][0])

    def test_parent_versions_and_package_alias_recall_the_original_issue(self):
        seed = self.legacy()
        issue = self.issue(seed)
        first = self.package('child', parent=seed)
        second = self.package('child', parent=first['package_id'])
        for identifier in (first['package_id'], first['candidate_id'], second['package_id']):
            recall = self.recall.for_skills([identifier])
            self.assertEqual(recall['pending_improvements'][0]['record_id'], issue['record_id'])
        alternatives = self.recall.for_skills([seed])['candidate_revisions']
        self.assertIn(second['package_id'], [item['record_id'] for item in alternatives])
        self.assertTrue(all(item['status'] == 'candidate_not_promoted' for item in alternatives))

    def test_logical_versions_work_without_an_explicit_parent(self):
        first = self.legacy('same_capability')
        issue = self.issue(first)
        second = self.legacy('same_capability', SOURCE[:-1] + ['    return {"revision": 2}'])
        self.assertEqual(self.recall.for_skills([second])['pending_improvements'][0]['record_id'], issue['record_id'])

    def test_downward_dependency_does_not_follow_shared_component_back_to_a_sibling(self):
        approach = self.legacy('shared_approach')
        left = self.package('left_composition', dependencies={'approach': approach})
        right = self.package('right_composition', dependencies={'approach': approach})
        left_issue = self.issue(left['package_id'], identifier='left_only')
        shared_issue = self.issue(approach, identifier='shared')
        right_recall = self.recall.for_skills([right['package_id']])
        self.assertEqual(right_recall['pending_count'], 1)
        self.assertEqual(right_recall['pending_improvements'][0]['record_id'], shared_issue['record_id'])
        self.assertNotIn(left_issue['record_id'], json.dumps(right_recall))
        self.assertNotIn(left['package_id'], [item['record_id'] for item in right_recall['candidate_revisions']])
        self.assertEqual(self.recall.for_skills([approach])['pending_count'], 1)

    def test_nested_dependencies_recall_a_leaf_issue(self):
        leaf = self.legacy('leaf')
        issue = self.issue(leaf)
        middle = self.package('middle', dependencies={'leaf': leaf})
        outer = self.package('outer', dependencies={'middle': middle['package_id']})
        self.assertEqual(self.recall.for_skills([outer['package_id']])['pending_improvements'][0]['record_id'], issue['record_id'])

    def test_compensated_success_is_history_and_does_not_close_a_pending_issue(self):
        seed = self.legacy()
        issue = self.issue(seed)
        compensated = self.outcome(seed)
        self.outcome(seed, identifier='later_clean', correction=False, remaining=[])
        recall = self.recall.for_skills([seed])
        self.assertEqual(recall['pending_improvements'][0]['record_id'], issue['record_id'])
        self.assertIn(compensated['record_id'], recall['attention_record_ids'])
        historical = next(item for item in recall['recent_outcomes'] if item['record_id'] == compensated['record_id'])
        self.assertEqual(historical['task_result'], 'success')
        self.assertTrue(historical['historical'])
        self.assertFalse(historical['independently_verified'])

    def test_latest_resolved_issue_does_not_remain_pending_and_reopen_is_visible(self):
        seed = self.legacy()
        self.issue(seed)
        open_fingerprint = self.recall.for_skills([seed])['fingerprint']
        self.issue(seed, status='resolved')
        closed = self.recall.for_skills([seed])
        self.assertEqual(closed['pending_improvements'], [])
        self.assertEqual(closed['attention_record_ids'], [])
        self.assertNotEqual(open_fingerprint, closed['fingerprint'])
        reopened = self.issue(seed, status='reopened')
        self.assertEqual(self.recall.for_skills([seed])['pending_improvements'][0]['record_id'], reopened['record_id'])

    def test_latest_improved_result_remains_visible_after_many_old_negative_reports(self):
        seed = self.legacy()
        self.issue(seed)
        for number in range(7):
            self.outcome(seed, identifier='old_attempt_' + str(number), result='failure')
        newest = self.outcome(seed, identifier='new_result', result='success', correction=False, remaining=[])
        self.issue(seed, status='resolved')
        recall = self.recall.for_skills([seed])
        self.assertEqual(recall['pending_count'], 0)
        self.assertEqual(recall['recent_outcomes'][0]['record_id'], newest['record_id'])
        self.assertTrue(any(item['task_result'] == 'failure' for item in recall['recent_outcomes'][1:]))
        self.assertGreater(recall['omitted_counts']['recent_outcomes'], 0)
        self.assertFalse(recall['recent_outcomes'][0]['independently_verified'])
        only_one = self.recall.for_skills([seed], limit=1)
        self.assertEqual(only_one['recent_outcomes'][0]['record_id'], newest['record_id'])

    def test_empty_focus_and_unrelated_append_have_no_spurious_attention(self):
        seed, other = self.legacy(), self.legacy('other')
        self.issue(seed)
        first = self.recall.for_skills([seed])
        self.issue(other, identifier='other_issue')
        self.outcome(other, identifier='other_result')
        self.package('unrelated_package', parent=other)
        self.assertEqual(first, self.recall.for_skills([seed]))
        empty = self.recall.for_skills([])
        self.assertEqual(empty['pending_count'], 0)
        self.assertEqual(empty['outcome_count'], 0)
        self.assertEqual(empty['revision_count'], 0)
        self.assertEqual(empty['attention_record_ids'], [])

    def test_unknown_foreign_scope_and_lesson_ids_fail_closed(self):
        seed = self.legacy()
        other = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='other'), 'other'))
        lesson = self.store.propose('lesson', dict(id='lesson', claim='Not executable',
            applicability=['fixture'], limitations=['unverified'], evidence_ids=['obs-fixture']))['candidate_id']
        before = self.files()
        for identifier in ('skill-missing-' + '0' * 64, lesson, '../file'):
            with self.subTest(identifier=identifier), self.assertRaises(KnowledgeError):
                self.recall.for_skills([identifier])
        with self.assertRaises(KnowledgeError):
            SkillRecall(other).for_skills([seed])
        self.assertEqual(before, self.files())

    def test_focus_and_limit_validation(self):
        seed = self.legacy()
        for value in (None, 'seed', [seed, seed], [1], [{}], [str(n) for n in range(17)]):
            with self.subTest(value=value), self.assertRaises(KnowledgeError):
                self.recall.for_skills(value)
        for limit in (0, 9, True, '4'):
            with self.subTest(limit=limit), self.assertRaises(KnowledgeError):
                self.recall.for_skills([seed], limit)

    def test_matching_source_is_exact_bounded_and_never_interprets_code(self):
        source = '\n'.join(SOURCE) + '\n'
        for number in range(18):
            self.legacy('matching_' + str(number))
        before = self.files()
        matches = self.recall.matching_source(source)
        self.assertEqual(len(matches), 16)
        self.assertEqual(len(set(matches)), 16)
        self.assertEqual(self.recall.matching_source(source + '\n'), [])
        self.assertEqual(self.recall.matching_source('raise RuntimeError("not executed")'), [])
        for invalid in (None, '\ud800', 'x' * 65537):
            with self.subTest(invalid_type=type(invalid)), self.assertRaises(KnowledgeError):
                self.recall.matching_source(invalid)
        self.assertEqual(before, self.files())

    def test_utf8_summary_is_bounded_and_keeps_one_pending_issue_with_omission_counts(self):
        seeds = [self.legacy(('long_name_' + str(index)).ljust(64, 'x')) for index in range(16)]
        for index in range(9):
            self.issue(seeds[0], identifier='pending_' + str(index),
                summary='可补偿的输入问题' * 160, rationale='后续相关调用应该考虑优化' * 140)
            self.outcome(seeds[0], identifier='historical_' + str(index),
                remaining=['反复出现的误触需要验证' * 15])
        value = self.recall.for_skills(seeds, limit=8)
        self.assertLessEqual(len(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()), 6144)
        self.assertEqual(value['pending_count'], 9)
        self.assertTrue(value['pending_improvements'])
        self.assertTrue(value['summary_truncated'])
        self.assertGreater(value['omitted_counts']['pending_improvements'], 0)
        displayed = {item['record_id'] for key in ('pending_improvements', 'recent_outcomes', 'candidate_revisions')
                     for item in value[key]}
        self.assertTrue(set(value['attention_record_ids']) <= displayed)

    def test_corrupt_registered_source_is_rejected(self):
        seed = self.legacy()
        self.issue(seed)
        path = self.store.namespace / 'candidates' / seed / 'source.py'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(KnowledgeError):
            self.recall.for_skills([seed])
        with self.assertRaises(KnowledgeError):
            self.recall.matching_source('\n'.join(SOURCE) + '\n')

    def test_model_priority_orders_pending_and_revisit_conditions_are_retained(self):
        seed = self.legacy()
        high = self.issue(seed, identifier='important', priority=90, revisit_when=['Before this capability is reused'])
        self.issue(seed, identifier='newer_low', priority=10)
        recalled = self.recall.for_skills([seed])['pending_improvements']
        self.assertEqual(recalled[0]['record_id'], high['record_id'])
        self.assertEqual(recalled[0]['priority'], 90)
        self.assertEqual(recalled[0]['revisit_when'], ['Before this capability is reused'])

    def test_merge_alias_follows_latest_canonical_issue_without_duplicate_pending(self):
        seed, target_skill = self.legacy(), self.legacy('target_capability')
        source = self.issue(seed, identifier='duplicate')
        target = self.issue(target_skill, identifier='canonical', priority=80)
        self.issue(seed, identifier='duplicate', status='merged', merged_into=target['record_id'])
        recalled = self.recall.for_skills([seed])
        self.assertEqual(recalled['pending_count'], 1)
        self.assertEqual(recalled['pending_improvements'][0]['record_id'], target['record_id'])
        self.assertNotIn(source['record_id'], recalled['attention_record_ids'])
        self.issue(target_skill, identifier='canonical', status='resolved', priority=80)
        self.assertEqual(self.recall.for_skills([seed])['pending_count'], 0)

    def test_merge_chain_and_latest_model_priority_are_visible_from_the_original_skill(self):
        seed, second_skill, third_skill = self.legacy(), self.legacy('second'), self.legacy('third')
        self.issue(seed, identifier='first')
        second = self.issue(second_skill, identifier='second_issue')
        final = self.issue(third_skill, identifier='final_issue', priority=20)
        self.issue(seed, identifier='first', status='merged', merged_into=second['record_id'])
        self.issue(second_skill, identifier='second_issue', status='merged', merged_into=final['record_id'])
        before = self.recall.for_skills([seed])
        updated = self.issue(third_skill, identifier='final_issue', priority=95,
                             revisit_when=['At the next relevant application use'])
        after = self.recall.for_skills([seed])
        self.assertEqual(after['pending_count'], 1)
        self.assertEqual(after['pending_improvements'][0]['record_id'], updated['record_id'])
        self.assertEqual(after['pending_improvements'][0]['priority'], 95)
        self.assertNotEqual(before['fingerprint'], after['fingerprint'])

    def test_latest_outcome_revision_is_used_without_implicitly_closing_the_issue(self):
        seed = self.legacy()
        issue = self.issue(seed)
        previous = self.outcome(seed)
        revised = self.outcome(seed, correction=False, remaining=[])
        recall = self.recall.for_skills([seed])
        self.assertEqual(recall['outcome_count'], 1)
        self.assertEqual(recall['recent_outcomes'][0]['record_id'], revised['record_id'])
        self.assertNotIn(previous['record_id'], recall['attention_record_ids'])
        self.assertEqual(recall['pending_improvements'][0]['record_id'], issue['record_id'])

    def test_cyclic_or_unknown_merged_target_cannot_silently_hide_an_issue(self):
        seed = self.legacy()
        first = self.issue(seed, identifier='first')
        second = self.issue(seed, identifier='second')
        records = self.library._records()
        # Fault-injected reader output exercises the defensive graph check; the
        # real content-addressed reader independently rejects corrupt bytes.
        for target in ('missing-record', second['record_id']):
            altered = json.loads(json.dumps(records))
            altered[0].update(status='merged', merged_into=target)
            altered[1].update(status='merged', merged_into=first['record_id'])
            with patch.object(self.library, '_records', return_value=altered), self.assertRaises(KnowledgeError):
                self.recall.for_skills([seed])

    def test_fresh_instance_observes_append_without_reusing_a_stale_snapshot(self):
        seed = self.legacy()
        self.assertEqual(self.recall.for_skills([seed])['pending_count'], 0)
        issue = self.issue(seed)
        later = SkillRecall(SkillDevelopment(KnowledgeStore(self.root / 'knowledge', SCOPE, 'later')))
        self.assertEqual(later.for_skills([seed])['pending_improvements'][0]['record_id'], issue['record_id'])


if __name__ == '__main__':
    unittest.main()
