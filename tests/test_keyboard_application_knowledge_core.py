"""Application knowledge is inert, scope-bound and uses the existing evidence store."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

from astra_robotics_agent.physical_keyboard.records.knowledge_domains import validate_knowledge_scope
from astra_robotics_agent.physical_keyboard.skills.current_view import CurrentView
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges
from astra_robotics_agent.physical_keyboard.skills.reader import KnowledgeReader
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources
from test_keyboard_learning_resources import Fixture, SCOPE


class ApplicationKnowledgeTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.changes = LearningChanges(self.library, self.resources)
        self.view = CurrentView(self.library, self.changes)
        self.reader = KnowledgeReader(self.library, resources=self.resources, changes=self.changes, view=self.view)

    def snapshot(self):
        return {str(p.relative_to(self.store.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.store.root.rglob('*') if p.is_file()}

    def workflow(self, skills=(), **changes):
        result = dict(id='draft_and_confirm', application_id='wechat', description='Inert fixture workflow',
            applicability=['Authorized matching application and physical scope'],
            limitations=['Semantic success remains a model assessment'], evidence_ids=['obs'],
            steps=[dict(id='compose', instruction='Use a suitable typing capability for the current draft',
                skill_ids=list(skills), requires_observation=True),
                dict(id='verify', instruction='Read actual external evidence before deciding the next action',
                    skill_ids=[], requires_observation=True)])
        result.update(changes)
        return result

    def bare_lesson(self, name='old', **updates):
        args=self.lesson(name, **updates);args.pop('source_ids', None)
        return args

    def test_legacy_records_project_unspecified_without_rewrite(self):
        old=self.store.propose('lesson', self.bare_lesson())
        before=self.snapshot()
        brief=self.reader.read(dict(identifier=old['candidate_id']))['data']
        self.assertEqual(brief['knowledge_scope'], {'domain':'unspecified'})
        self.assertNotIn('knowledge_scope', self.library.inspect(old['candidate_id']))
        self.assertEqual(self.library.search(kind='lesson',domain='unspecified')['matched_count'],1)
        self.assertEqual(self.snapshot(),before)

    def test_metadata_strict_application_validation(self):
        for value in ({'domain':'application'}, {'domain':'robot','application_id':'wechat'},
                      {'domain':'unknown'}, {'domain':'application','application_id':''},
                      {'domain':'application','application_id':'wechat','permissions':True},
                      {'domain':True}, []):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_knowledge_scope(value)
        self.assertEqual(validate_knowledge_scope({'domain':'robot'}),{'domain':'robot'})

    def test_atomic_metadata_hashes_match_writers_and_directory_filter(self):
        app=dict(domain='application',application_id='wechat')
        args=dict(lessons=[self.lesson('focus',knowledge_scope=app),
                           self.lesson('motor',knowledge_scope={'domain':'robot'})],
                  packages=[self.package('press',knowledge_scope={'domain':'robot'},source_ids=['source-a'])])
        before=self.snapshot();plan=self.changes.preflight(args)
        self.assertEqual(self.snapshot(),before)
        commit=self.changes.commit(args)
        self.assertEqual(plan['record_ids'],commit['record_ids'])
        items=self.view.directory(domain='application',application_id='wechat')['items']
        self.assertEqual([(i['kind'],i['logical_id']) for i in items],[('lesson','focus')])
        self.assertEqual(items[0]['versions'][0]['knowledge_scope'],app)
        self.assertEqual(self.view.directory(domain='robot')['total_count'],2)
        for kwargs in ({'domain':'bad'},{'domain':'robot','application_id':'wechat'},{'application_id':False}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.view.directory(**kwargs)

    def test_workflow_save_inspect_reader_and_robot_dispatch_separation(self):
        package=self.library.save_package(self.package())
        saved=self.library.save_application_skill(self.workflow([package['package_id']]))
        self.assertEqual(saved['kind'],'workflow')
        full=self.library.inspect(saved['record_id'])
        self.assertNotIn('source',full)
        self.assertEqual(full['skill_pins'][0]['reference_id'],package['package_id'])
        self.assertEqual(full['skill_pins'][0]['candidate_id'],package['candidate_id'])
        self.assertEqual(full['knowledge_scope'],dict(domain='application',application_id='wechat'))
        brief=self.reader.read(dict(identifier=saved['record_id']))['data']
        self.assertEqual((brief['step_count'],brief['skill_pin_count']),(2,1))
        self.assertEqual(self.reader.read(dict(identifier=saved['record_id'],view='full'))['data']['steps'],full['steps'])
        with self.assertRaises(ValueError):self.library.resolve_call(dict(package_id=saved['record_id'],arguments={}))
        with self.assertRaises(ValueError):self.library._resolve_skill(saved['record_id'])
        with self.assertRaises(ValueError):self.reader.read(dict(identifier=saved['record_id'],view='source'))

    def test_workflow_revision_keeps_original_steps_and_frozen_pins(self):
        package=self.library.save_package(self.package())
        original=self.library.save_application_skill(self.workflow([package['package_id']]))
        before=self.library.inspect(original['record_id'])
        newer=self.library.save_package(self.package(parent_id=package['package_id'],profile={'text':'changed'}))
        second=self.library.save_application_skill(self.workflow([newer['package_id']],parent_id=original['record_id']))
        self.assertEqual(second['version'],2)
        self.assertEqual(self.library.inspect(original['record_id']),before)
        self.assertNotEqual(before['skill_pins'],self.library.inspect(second['record_id'])['skill_pins'])
        with self.assertRaises(ValueError):
            self.library.save_application_skill(self.workflow(application_id='other',parent_id=original['record_id']))

    def test_workflow_steps_require_valid_finite_references(self):
        mutations=[{'steps':[]},{'steps':[dict(id='a',instruction='x',skill_ids=[],requires_observation='yes')]},
            {'steps':[dict(id='a',instruction='x',skill_ids=['package-missing'],requires_observation=True)]},
            {'steps':[dict(id='a',instruction='x',skill_ids=[],requires_observation=True)]*2},
            {'source_lines':['def main(robot, profile):','    return 1']}]
        for updates in mutations:
            before=self.snapshot()
            with self.subTest(updates=updates),self.assertRaises(ValueError):
                self.library.save_application_skill(self.workflow(**updates))
            self.assertEqual(self.snapshot(),before)

    def test_atomic_workflow_local_references_issue_links_and_selection(self):
        args=dict(packages=[self.package(local_ref='motor',source_ids=['source-a'])],
            lessons=[self.lesson('focus',local_ref='focus',knowledge_scope=dict(domain='application',application_id='wechat'))],
            workflows=[self.workflow([{'local_ref':'motor'}],local_ref='workflow')],
            improvements=[dict(id='delivery',summary='Model reported application concern',rationale='Physical and application results differ',
                status='open',skill_ids=[{'local_ref':'motor'}],knowledge_ids=[{'local_ref':'workflow'},{'local_ref':'focus'}],
                evidence_ids=['obs'],knowledge_scope=dict(domain='application',application_id='wechat'),local_ref='issue')],
            relations=[dict(kind='uses_lesson',from_id={'local_ref':'workflow'},to_id={'local_ref':'focus'})],
            selections=[dict(logical_id='draft_and_confirm',identifier={'local_ref':'workflow'},conditions=['Matching app'],reason='Explicit model choice')])
        before=self.snapshot();plan=self.changes.preflight(args);self.assertEqual(self.snapshot(),before)
        result=self.changes.commit(args);self.assertEqual(plan['bindings'],result['bindings'])
        workflow_id=result['bindings']['workflow'];issue_id=result['bindings']['issue']
        self.assertEqual(self.changes.source_ids(workflow_id),['source-a'])
        self.assertEqual(self.library.inspect(issue_id)['knowledge_ids'],[workflow_id,result['bindings']['focus']])
        summary=self.library.list_improvements()['items'][0]
        self.assertEqual(summary['knowledge_ids'],[workflow_id,result['bindings']['focus']])
        directory=self.view.directory(kind='workflow',application_id='wechat')['items'][0]
        self.assertTrue(directory['selected']);self.assertFalse(directory['automatic_latest'])

    def test_atomic_failure_keeps_workflow_pending_not_readable(self):
        args=dict(workflows=[self.workflow(local_ref='app',source_ids=['source-a'])])
        plan=self.changes.preflight(args)
        def fail(stage):
            if stage=='before_final_commit':raise RuntimeError('Injected interruption')
        self.changes.failure_hook=fail
        with self.assertRaises(RuntimeError):self.changes.commit(args)
        self.assertFalse(self.changes.is_committed(plan['bindings']['app']))
        with self.assertRaises(ValueError):self.reader.read(dict(identifier=plan['bindings']['app']))

    def test_application_only_outcome_can_link_workflow_without_robot_skill(self):
        workflow=self.library.save_application_skill(self.workflow())
        result=self.library.record_outcome(dict(id='application_result',skill_ids=[],knowledge_ids=[workflow['record_id']],
            evidence_ids=['obs'],observation_evidence_id='obs',task_result='uncertain',correction_performed=False,
            remaining_issues=['Need fresh visual confirmation'],summary='Application-level model report',
            knowledge_scope=dict(domain='application',application_id='wechat')))
        full=self.library.inspect(result['record_id'])
        self.assertEqual(full['knowledge_ids'],[workflow['record_id']]);self.assertFalse(full['independently_verified'])
        self.assertEqual(full['task_result'],'uncertain')

    def test_cross_physical_scope_is_not_made_executable_by_application_metadata(self):
        package=self.library.save_package(self.package())
        scope=dict(SCOPE,layout_id='another-layout')
        other=KnowledgeStore(self.root/'knowledge',scope,'new-device')
        other.register_evidence('obs',[self.evidence])
        library=SkillDevelopment(other)
        with self.assertRaises(ValueError):library.save_application_skill(self.workflow([package['package_id']]))

    def test_source_policy_covers_workflow_pins_and_shared_issue_knowledge(self):
        hidden=self.changes.commit(dict(packages=[self.package('hidden',source_ids=['source-b'])]))
        robot=hidden['record_ids'][0]
        robot=next(value for value in hidden['record_ids'] if value.startswith('package-'))
        workflow=self.changes.commit(dict(workflows=[self.workflow([robot],source_ids=['source-a'])]))['record_ids'][0]
        self.assertEqual(self.changes.source_ids(workflow),['source-a','source-b'])
        access=self.access();resources=LearningResources(self.store,access=access)
        changes=LearningChanges(self.library,resources,access=access)
        with self.assertRaises(ValueError):changes.source_ids(workflow)
        with self.assertRaises(ValueError):changes.commit(dict(improvements=[dict(id='launder',summary='x',rationale='x',status='open',
            skill_ids=[],knowledge_ids=[workflow],evidence_ids=['obs'],source_ids=['source-a'])]))

    def test_filtered_version_group_does_not_leak_another_application_selection(self):
        first=self.changes.commit(dict(lessons=[self.lesson('shared',local_ref='first',
            knowledge_scope=dict(domain='application',application_id='wechat'))],
            selections=[dict(logical_id='shared',identifier={'local_ref':'first'},conditions=['wechat'],reason='one app')]))
        other=self.changes.commit(dict(lessons=[self.lesson('shared',local_ref='other',
            knowledge_scope=dict(domain='application',application_id='notepad'))],
            selections=[dict(logical_id='shared',identifier={'local_ref':'other'},conditions=['notepad'],reason='other app')]))
        item=self.view.directory(application_id='wechat')['items'][0]
        self.assertEqual([version['identifier'] for version in item['versions']],[first['bindings']['first']])
        self.assertEqual([branch['identifier'] for branch in item['branches']],[first['bindings']['first']])
        self.assertNotIn(other['bindings']['other'],str(item))

    def test_invalid_late_workflow_reference_rejects_whole_commit_without_writes(self):
        args=dict(packages=[self.package(local_ref='motor',source_ids=['source-a'])],
            workflows=[self.workflow([{'local_ref':'missing'}],source_ids=['source-a'])])
        before=self.snapshot()
        with self.assertRaises(ValueError):self.changes.commit(args)
        self.assertEqual(self.snapshot(),before)

    def test_forged_workflow_pin_is_detected_by_existing_content_hash(self):
        saved=self.library.save_application_skill(self.workflow())
        row=self.library._record(saved['record_id'])
        objects=list(self.library.objects.glob('*.json'))
        target=next(path for path in objects if saved['record_id'] in path.read_text())
        target.write_text(target.read_text().replace('Inert fixture workflow','Forged fixture workflow'))
        with self.assertRaises(ValueError):self.library.inspect(saved['record_id'])


if __name__=='__main__':unittest.main()
