"""Related links and append-only evidence cursors without semantic writes."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS
from astra_robotics_agent.physical_keyboard.skills.learning_attention import LearningAttention
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from test_keyboard_learning_resources import Fixture, SCOPE


class LearningAttentionTests(Fixture,unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.access=LearningAccess(SCOPE)
        self.access.bind_evidence('obs',['source-a'])
        self.store.learning_access=self.access
        self.resources=LearningResources(self.store,access=self.access)
        self.changes=LearningChanges(self.library,self.resources,access=self.access)
        self.attention=LearningAttention(self.library,self.changes)

    def snapshot(self):
        return {str(path.relative_to(self.store.root)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.store.root.rglob('*') if path.is_file()}

    def saved(self,name='press',**kw):
        return self.library.save_package(self.package(name,**kw))

    def add_evidence(self,name,payload=None,source='source-a'):
        path=self.root/(name+'.json');path.write_text(json.dumps({'event':name} if payload is None else payload))
        return self.store.register_evidence(name,[path],source_ids=[source])

    def issue(self,skill=None,status='open',name='issue'):
        return self.library.record_improvement(dict(id=name,summary='model observation',rationale='follow up',
            status=status,skill_ids=[] if skill is None else [skill],evidence_ids=['obs']))

    def outcome(self,skill,name='attempt',result='uncertain',correction=False):
        return self.library.record_outcome(dict(id=name,skill_ids=[skill],evidence_ids=['obs'],
            observation_evidence_id='obs',task_result=result,correction_performed=correction,
            remaining_issues=[],summary='Model report; no independent evaluation'))

    def test_lesson_same_registered_evidence_is_recalled_without_issue(self):
        saved=self.saved()
        args=self.lesson();args.pop('source_ids')
        lesson=self.store.propose('lesson',args)
        before=self.snapshot()
        got=self.attention.for_skills([saved['package_id']])
        self.assertEqual(got['lessons'][0]['candidate_id'],lesson['candidate_id'])
        self.assertEqual(got['associations'][0]['association_kind'],'same_registered_evidence')
        self.assertEqual(got['linked_improvements'],[])
        self.assertEqual(before,self.snapshot())
        self.assertEqual(got['fingerprint'],self.attention.for_skills([saved['package_id']])['fingerprint'])

    def test_explicit_lesson_relation_works_without_shared_evidence(self):
        saved=self.saved()
        self.add_evidence('lesson-obs')
        args=self.lesson(evidence_ids=['lesson-obs']);args.pop('source_ids')
        lesson=self.store.propose('lesson',args)
        self.changes.commit({'relations':[dict(kind=kind,from_id=saved['package_id'],to_id=lesson['candidate_id'])
            for kind in ('uses_lesson','derived_from')]})
        got=self.attention.for_skills([saved['package_id']])
        self.assertEqual([row['candidate_id'] for row in got['lessons']],[lesson['candidate_id']])
        self.assertEqual(got['associations'][0]['association_kind'],'explicit_uses_lesson')
        self.assertEqual({row['association_kind'] for row in got['associations']},{'explicit_uses_lesson','explicit_derived_from'})

    def test_package_improvement_ids_work_with_issue_empty_skill_ids(self):
        issue=self.issue()
        saved=self.saved(improvement_ids=[issue['record_id']])
        got=self.attention.for_skills([saved['package_id']])
        self.assertEqual(got['linked_improvements'][0]['record_id'],issue['record_id'])
        self.assertIn(issue['record_id'],got['attention_record_ids'])
        self.assertIn('package_improvement_ids',[row['association_kind'] for row in got['associations']])

    def test_shared_component_does_not_recall_sibling_or_unrelated_issue(self):
        shared=self.saved('shared')
        source=['def main(robot, profile):','    return part(robot, {})']
        left=self.saved('left',dependencies={'part':shared['package_id']},source_lines=source)
        right=self.saved('right',dependencies={'part':shared['package_id']},source_lines=source)
        hidden=self.issue(left['package_id'],name='left-only')
        common=self.issue(shared['package_id'],name='shared-issue')
        got=self.attention.for_skills([right['package_id']])
        self.assertEqual([row['record_id'] for row in got['linked_improvements']],[common['record_id']])
        self.assertNotIn(hidden['record_id'],json.dumps(got))

    def test_resolved_after_negative_outcomes_remains_resolved_and_same_evidence_not_validation(self):
        saved=self.saved();self.outcome(saved['package_id'],'earlier','failure')
        issue=self.issue(saved['package_id'],status='resolved')
        quiet=self.attention.for_skills([saved['package_id']])
        self.assertEqual(quiet['revisit_resolved'],[])
        self.assertNotIn(issue['record_id'],quiet['attention_record_ids'])
        self.outcome(saved['package_id'],'later1','uncertain')
        self.outcome(saved['package_id'],'later2','success',True)
        before=self.snapshot();got=self.attention.for_skills([saved['package_id']])
        review=got['revisit_resolved'][0]
        self.assertEqual(review['status'],'resolved');self.assertTrue(review['state_unchanged'])
        self.assertEqual(len(review['new_evidence']),2)
        self.assertEqual(review['new_evidence'][0]['evidence_group_ids'],review['new_evidence'][1]['evidence_group_ids'])
        self.assertEqual(got['independent_validation_count'],0)
        self.assertEqual(self.library.inspect(issue['record_id'])['status'],'resolved')
        self.assertEqual(before,self.snapshot())

    def execution(self,saved,name):
        folder=self.root/name;folder.mkdir()
        candidate=self.store.get_skill(saved['candidate_id'])
        (folder/'skill.py').write_text(candidate['source'])
        (folder/'profile.json').write_text(json.dumps(candidate['profile']))
        (folder/'response.json').write_text('{"status":"unknown"}')
        return self.store.register_evidence(name,[folder/'skill.py',folder/'profile.json',folder/'response.json'],source_ids=['source-a'])

    def test_unassociated_execution_matches_exact_source_and_does_not_clear_fault(self):
        saved=self.saved();issue=self.issue(saved['package_id'],status='resolved')
        self.attention.for_skills([saved['package_id']])
        execution=self.execution(saved,'new-exec')
        before=self.snapshot();got=self.attention.for_skills([saved['package_id']])
        reason=got['revisit_resolved'][0]['new_evidence'][0]
        self.assertEqual(reason['evidence_id'],'new-exec')
        self.assertEqual(reason['association_kind'],'unassociated_execution_source_hash')
        self.assertIs(reason['after_resolution'],True)
        fresh=LearningAttention(self.library,self.changes).for_skills([saved['package_id']])
        self.assertEqual(fresh['revisit_resolved'][0]['new_evidence'][0]['after_resolution'],'unknown')
        self.assertEqual(self.library.inspect(issue['record_id'])['status'],'resolved')
        self.assertEqual(before,self.snapshot())

    def test_cursor_initial_recent_paging_repeated_empty_and_new_increment(self):
        self.saved() # initial obs is now associated
        for index in range(12):self.add_evidence('new-%02d'%index)
        first=self.attention.new_evidence(limit=8)
        self.assertEqual([row['evidence_id'] for row in first['items']],['new-%02d'%i for i in range(4,12)])
        self.assertEqual(first['omitted_count'],4)
        self.assertEqual(self.attention.new_evidence(first['next_cursor'])['items'],[])
        with patch.object(self.attention,'_snapshot',side_effect=AssertionError('no repeated full scan')):
            self.assertEqual(self.attention.new_evidence(first['next_cursor'])['items'],[])
        older=self.attention.new_evidence(limit=8,offset=8)
        self.assertEqual(len(older['items']),4)
        for index in range(12,17):self.add_evidence('new-%02d'%index)
        second=self.attention.new_evidence(first['next_cursor'],limit=2)
        self.assertEqual([row['evidence_id'] for row in second['items']],['new-12','new-13'])
        third=self.attention.new_evidence(second['next_cursor'],limit=8)
        self.assertEqual([row['evidence_id'] for row in third['items']],['new-14','new-15','new-16'])
        self.assertEqual(self.attention.new_evidence(third['next_cursor'])['items'],[])
        self.assertLessEqual(len(json.dumps(first,sort_keys=True,separators=(',',':')).encode()),8192)
        self.assertNotIn('text',first['items'][0]);self.assertNotIn('files',first['items'][0])

    def test_duplicate_raw_source_contents_have_one_evidence_group_and_no_independent_count(self):
        self.saved();self.add_evidence('alias-one',{'same':'event'});self.add_evidence('alias-two',{'same':'event'})
        got=self.attention.new_evidence()
        self.assertEqual(len(got['items']),2)
        self.assertEqual(got['unique_evidence_group_count'],1)
        self.assertEqual(got['independent_validation_count'],0)

    def test_cursor_wrong_scope_namespace_prefix_hash_and_corrupted_evidence_rejected(self):
        first=self.attention.new_evidence()
        for key,value in [('scope',dict(SCOPE,layout_id='different')),('position',999999),('prefix_sha256','0'*64),('sha256','0'*64)]:
            cursor=copy.deepcopy(first['next_cursor']);cursor[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.attention.new_evidence(cursor)
        other_store=KnowledgeStore(self.root/'other',SCOPE,'different')
        other=LearningAttention(SkillDevelopment(other_store),None)
        with self.assertRaises(ValueError):other.new_evidence(first['next_cursor'])
        self.add_evidence('bad')
        record=self.store._evidence('bad')
        (self.store.namespace/'objects'/record['files'][0]['sha256']).write_bytes(b'corrupted')
        with self.assertRaises(ValueError):self.attention.new_evidence()

    def test_restricted_sources_and_pending_or_audit_evidence_do_not_leak(self):
        saved=self.saved()
        args={'lessons':[self.lesson('pending')]}
        plan=self.changes.preflight(args)
        def fail(stage):
            if stage=='before_final_commit':raise RuntimeError('fixture pending')
        self.changes.failure_hook=fail
        with self.assertRaises(RuntimeError):self.changes.commit(args)
        self.changes.failure_hook=None
        pending_id=plan['operations'][0]['identifier']
        self.assertFalse(self.changes.is_committed(pending_id))
        self.assertNotIn(pending_id,json.dumps(self.attention.for_skills([saved['package_id']])))
        self.add_evidence('allowed-new')
        self.add_evidence('hidden-new',source='source-b')
        hidden_args=self.lesson('hidden',evidence_ids=['hidden-new']);hidden_args.pop('source_ids')
        hidden=self.store.propose('lesson',hidden_args)
        access=self.access_for_test()
        restricted_store=KnowledgeStore(self.store.root,SCOPE,'filtered',learning_access=access)
        library=SkillDevelopment(restricted_store)
        resources=LearningResources(restricted_store,access=access)
        changes=LearningChanges(library,resources,access=access)
        attention=LearningAttention(library,changes)
        got=attention.new_evidence()
        self.assertEqual([row['evidence_id'] for row in got['items']],['allowed-new'])
        self.assertNotIn('hidden',json.dumps(attention.for_skills([saved['package_id']])))
        score=self.root/'typing-buffer-score.json';score.write_text('{"secret":"must never be shown"}')
        self.store.register_evidence('score-record',[score],source_ids=['source-a'])
        self.assertNotIn('score-record',json.dumps(attention.new_evidence()))

    def test_large_related_set_reports_omissions_within_eight_kib_without_writes(self):
        saved=self.saved()
        for index in range(20):
            args=self.lesson('lesson-'+str(index),claim='hypothesis '*350);args.pop('source_ids')
            self.store.propose('lesson',args)
            self.issue(saved['package_id'],name='issue-'+str(index))
        before=self.snapshot();got=self.attention.for_skills([saved['package_id']],limit=8)
        self.assertLessEqual(len(json.dumps(got,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),8192)
        self.assertTrue(got['summary_truncated'])
        self.assertGreater(got['omitted']['lessons'],0)
        self.assertEqual(got['independent_validation_count'],0)
        self.assertEqual(before,self.snapshot())

    def test_empty_focus_does_not_scan_global_library_or_create_maintenance(self):
        with patch.object(self.attention,'_snapshot',side_effect=AssertionError('no full scan without focus')):
            got=self.attention.for_skills([])
        self.assertEqual(got['attention_record_ids'],[])
        self.assertFalse(got['summary_truncated'])

    def test_attention_uses_narrow_policy_even_when_supplied_store_is_broad(self):
        self.saved()
        self.add_evidence('hidden-derived',source='source-b')
        strict=self.access_for_test()
        strict.register_own_evidence('hidden-derived')
        changes=LearningChanges(self.library,self.resources,access=strict)
        attention=LearningAttention(self.library,changes)
        self.assertNotIn('hidden-derived',json.dumps(attention.new_evidence()))
        self.assertEqual(self.store._evidence('hidden-derived')['source_ids'],['source-b'])

    def access_for_test(self):
        access=LearningAccess(SCOPE,dict(protocol=PROTOCOLS['access'],scope=SCOPE,allowed_source_ids=['source-a'],
            allowed_knowledge_ids=[],allowed_evidence_ids=['obs'],allow_own_experience=True))
        access.bind_evidence('obs',['source-a'])
        return access


if __name__=='__main__':unittest.main()
