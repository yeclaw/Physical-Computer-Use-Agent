"""Model-owned backlog semantics on real temporary stores, without live IO."""
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore


SCOPE = {'task_id': 'backlog_fixture', 'device_id': 'fake', 'layout_id': 'fixed'}


class ImprovementBacklogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'first')
        evidence = self.root / 'observation.json'
        evidence.write_text('{"fixture":true}')
        self.store.register_evidence('obs-fixture', [evidence])
        self.library = SkillDevelopment(self.store)

    def args(self, identifier, **fields):
        return dict(id=identifier, summary='Observed reusable problem, reported by the model.',
                    rationale='Worth improving in subsequent relevant use.',
                    status='open', skill_ids=[], evidence_ids=['obs-fixture'], **fields)

    def test_model_priority_and_conditions_persist_when_only_status_is_updated(self):
        first = self.library.record_improvement(self.args('costly', priority=90,
                    revisit_when=['At the next relevant skill use.']))
        self.library.record_improvement(self.args('other', priority=20))
        changed = self.args('costly')
        changed['status'] = 'deferred'
        revision = self.library.record_improvement(changed)
        current = self.library.list_improvements()
        self.assertEqual([item['id'] for item in current['items']], ['costly', 'other'])
        self.assertEqual(revision['priority'], 90)
        self.assertEqual(revision['revisit_when'], first['revisit_when'])
        self.assertEqual(self.library.context()['pending_improvements'][0]['record_id'], revision['record_id'])
        changed.update(priority=5, revisit_when=['After another prerequisite is available.'])
        self.library.record_improvement(changed)
        self.assertEqual(self.library.list_improvements()['items'][0]['id'], 'other')
        self.assertEqual(self.library.inspect(first['record_id'])['status'], 'open')

    def test_explicit_model_merge_keeps_original_history_and_canonical_pending(self):
        canonical = self.library.record_improvement(self.args('canonical', priority=80))
        duplicate = self.library.record_improvement(self.args('duplicate'))
        before = self.library.index.read_bytes()
        args = self.args('duplicate')
        args.update(status='merged', merged_into=canonical['record_id'])
        merged = self.library.record_improvement(args)
        self.assertTrue(self.library.index.read_bytes().startswith(before))
        self.assertEqual([item['id'] for item in self.library.list_improvements()['items']], ['canonical'])
        self.assertEqual(self.library.list_improvements(status='merged')['items'][0]['record_id'], merged['record_id'])
        self.assertEqual(self.library.inspect(duplicate['record_id'])['status'], 'open')
        self.assertEqual(self.library.inspect(canonical['record_id'])['status'], 'open')
        later = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', SCOPE, 'later'))
        self.assertEqual(later.list_improvements()['pending_count'], 1)
        reopened = self.args('duplicate')
        reopened['status'] = 'reopened'
        self.assertNotIn('merged_into', later.record_improvement(reopened))
        self.assertEqual(later.list_improvements()['pending_count'], 2)

    def test_bad_merge_and_cross_scope_references_cannot_append(self):
        target = self.library.record_improvement(self.args('target'))
        source = self.library.record_improvement(self.args('source'))
        merged_args = self.args('source')
        merged_args.update(status='merged', merged_into=target['record_id'])
        self.library.record_improvement(merged_args)
        foreign = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='elsewhere'), 'foreign'))
        foreign.store.register_evidence('obs-fixture', [self.root / 'observation.json'])
        foreign_target = foreign.record_improvement(self.args('foreign'))
        invalid = [dict(self.args('self'), status='merged'),
                   dict(self.args('target'), status='merged', merged_into=target['record_id']),
                   dict(self.args('new'), status='merged', merged_into=source['record_id']),
                   dict(self.args('new'), status='merged', merged_into=foreign_target['record_id']),
                   dict(self.args('new'), merged_into=target['record_id'])]
        before = self.library.index.read_bytes()
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.library.record_improvement(args)
        self.assertEqual(self.library.index.read_bytes(), before)

    def test_invalid_priority_and_revisit_values_do_not_write(self):
        self.library.record_improvement(self.args('seed'))
        before = self.library.index.read_bytes()
        for field, value in [('priority', True), ('priority', 1.5), ('priority', -1),
                             ('priority', 101), ('revisit_when', 'soon'),
                             ('revisit_when', ['duplicate', 'duplicate']),
                             ('revisit_when', ['x'] * 9)]:
            args = self.args('bad')
            args[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.library.record_improvement(args)
        self.assertEqual(self.library.index.read_bytes(), before)

    def test_pagination_contains_only_latest_versions_and_does_not_lose_low_priority_items(self):
        for index in range(11):
            self.library.record_improvement(self.args('item_' + str(index), priority=index))
        updated = self.args('item_0')
        updated.update(priority=100, status='resolved')
        self.library.record_improvement(updated)
        offset, seen = 0, []
        while offset is not None:
            result = self.library.list_improvements(limit=3, offset=offset)
            seen += [item['id'] for item in result['items']]
            offset = result['next_offset']
        self.assertEqual(seen, ['item_' + str(index) for index in range(10, 0, -1)])
        all_items = self.library.list_improvements(status='all', limit=16)
        self.assertEqual(all_items['total_count'], 11)
        self.assertEqual(all_items['items'][0]['id'], 'item_0')
        self.assertEqual(all_items['items'][0]['version'], 2)

    def test_unicode_backlog_pages_are_bounded_and_make_forward_progress(self):
        for index in range(12):
            args = self.args('item_' + str(index), priority=index,
                             revisit_when=['下一次再次使用这个能力时检查。' * 12])
            args.update(summary='仍然有待解决的现象。' * 70, rationale='保留证据并比较复用结果。' * 60)
            self.library.record_improvement(args)
        first = self.library.list_improvements(limit=16)
        self.assertLessEqual(len(json.dumps(first, ensure_ascii=False).encode()), 8192)
        self.assertTrue(first['items'])
        self.assertIsNotNone(first['next_offset'])
        self.assertGreater(first['next_offset'], first['offset'])
        before = self.library.index.read_bytes()
        for args in ({'limit': True}, {'offset': -1}, {'offset': True}, {'status': []}):
            with self.assertRaises(ValueError):
                self.library.list_improvements(**args)
        self.assertEqual(before, self.library.index.read_bytes())

    def test_actual_json_context_and_pages_fit_utf8_budget_at_variable_boundaries(self):
        from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
        source = ['def main(robot, profile):', '    return robot.observe()']
        def skill(name):
            return self.store.propose('skill', dict(id=name, source_lines=source, profile={},
                applicability=['Fixture'], limitations=['Unverified'], evidence_ids=['obs-fixture']))['candidate_id']
        target, other = skill('target'), skill('other')
        def issue(name, reference, length, priority):
            args = self.args(name, priority=priority, revisit_when=['次' * length])
            args.update(summary='示' * length, rationale='证' * length, skill_ids=[reference])
            self.library.record_improvement(args)
        issue('target_issue', target, 100, 10)
        for index in range(5):
            issue('other_' + str(index), other, 200, 60 + index)
        development = DevelopmentActions(self.store)
        development.focus_skill_ids = [target]
        for length in range(30, 241, 5):
            issue('variable', other, length, 100)
            context = development.context()
            page = self.library.list_improvements(limit=16)
            with self.subTest(length=length):
                for value in (context, page):
                    self.assertLessEqual(len(json.dumps(value, ensure_ascii=False,
                        sort_keys=True, allow_nan=False).encode('utf-8')), 8192)
                self.assertEqual(context['online_recall']['pending_count'], 1)
                self.assertTrue(context['online_recall']['pending_improvements'])

    def test_an_issue_id_inside_model_profile_is_not_a_presented_issue_description(self):
        from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
        source = ['def main(robot, profile):', '    return robot.observe()']
        def skill(profile):
            return self.store.propose('skill', dict(id='target', source_lines=source, profile=profile,
                applicability=['Fixture'], limitations=['Unverified'], evidence_ids=['obs-fixture']))['candidate_id']
        old = skill({})
        args = self.args('old_defect', priority=1)
        args['skill_ids'] = [old]
        issue = self.library.record_improvement(args)
        for index in range(7):
            self.library.record_improvement(self.args('other_' + str(index), priority=90))
        new = skill({'candidate_id': issue['record_id'], 'record_id': old})
        development = DevelopmentActions(self.store)
        context = {'skill_development': development.context(),
                   'feedback': {'action': 'inspect_skill', 'skill': self.store.get_skill(new)}}
        self.assertNotIn(issue['record_id'], json.dumps(context['skill_development']))
        shown = development.presented(context)
        self.assertNotIn(issue['record_id'], shown)
        recall, unseen = development.before_use([new], '\n'.join(source) + '\n')
        self.assertEqual(recall['pending_count'], 1)
        self.assertIn(issue['record_id'], unseen)


if __name__ == '__main__':
    unittest.main()
