"""Fault-only learning/context routes: literal model/session and cache fixtures."""
import json
from pathlib import Path
import unittest

from astra_robotics_agent.physical_keyboard.agent.fault_review import run_fault_review
from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop
from astra_robotics_agent.physical_keyboard.adapters.skill_sources.learning_sources import LearningSources
from astra_robotics_agent.physical_keyboard.records.learning_contracts import digest
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources
import test_keyboard_fault_review as fault_fixtures
from test_keyboard_fault_review import Model, finish, SCOPE


class ContextModel(Model):
    thread_id='fault-thread'
    model,effort,service_tier='fixture-model','fixture-effort','default'
    total_usage,total_usage_complete={},False

    def __init__(self,choose,native=False):
        super().__init__(choose,native=native)
        self.rotations=[];self.compactions=[];self.closed=False

    def rotate_thread(self,reason):
        self.rotations.append(reason)
        old=self.thread_id;self.thread_id='fault-thread-rebuilt'
        return dict(status='completed',previous_thread_id=old,thread_id=self.thread_id)

    def compact_context(self):
        self.compactions.append('checked')
        return dict(status='unsupported',requires_explicit_rebuild_choice=True)

    def close(self):self.closed=True


class ContinualFaultTests(unittest.TestCase):
    setUp=fault_fixtures.FaultReviewTests.setUp

    def run_review(self,model,**kwargs):
        kwargs.setdefault('continual_learning',True)
        kwargs.setdefault('max_decisions',12)
        return run_fault_review(session=model,store=self.store,output=self.output,fault=self.fault,**kwargs)

    def boundary(self,**updates):
        value=dict(task_contract_sha256='a'*64,observation_evidence_id='before-execution',generation=7,
            last_execution=dict(evidence_id='failed-execution',status='unknown',motion_outcome_uncertain=True,controller_terminal=True),
            motion_available=False,inflight_rpc=False,inflight_action=False,inflight_model=False,
            stop_requested=False,assistance=None,fault=dict(stopped_reason='unknown_motion_no_retry',physical_outcome='unknown'),
            source_policy_sha256='b'*64)
        value.update(updates);return value

    @staticmethod
    def declaration(context,**updates):
        value=dict(summary='Work remains incomplete; physical outcome unknown',knowledge_ids=[],
            evidence_ids=[context['fault_evidence_id']],pending_issues=[],remaining=['Future authorized verification'])
        value.update(updates);return value

    def test_analyze_exact_resource_save_context_rebuild_stays_motion_disabled(self):
        sources=LearningSources(SCOPE)
        sources.register('numeric',kind='robot_attempt',data={'samples':[{'offset':.12345678912345678}]})
        ids={}
        def choose(index,context):
            self.assertFalse(context['motion_available']);self.assertIsNone(context['current_observation_evidence_id'])
            self.assertFalse(context['learning_boundary']['motion_available'])
            self.assertIsNone(context['learning_boundary']['observation_evidence_id'])
            if index==0:return dict(action='analyze_learning_sources',arguments=dict(id='exact',
                source_lines=['def analyze(data, parameters):','    return {"offset": data["sample"]["samples"][0]["offset"]}'],
                inputs={'sample':{'source_id':'numeric'}},parameters={}))
            if index==1:
                ids['resource']=context['feedback']['result']['resource']['resource_id']
                args=fault_fixtures.FaultReviewTests.package_arguments(self,context['fault_evidence_id'])
                del args['profile'];args['profile_resource_id']=ids['resource']
                return dict(action='save_skill_package',arguments=args)
            if index==2:
                ids['package']=context['feedback']['result']['package_id']
                ids['candidate']=context['feedback']['result']['candidate_id']
                return dict(action='save_working_context',arguments=self.declaration(context,knowledge_ids=[ids['package']]))
            if index==3:
                ids['checkpoint']=context['feedback']['result']['checkpoint_id']
                return dict(action='manage_working_context',arguments=dict(checkpoint_id=ids['checkpoint'],method='rebuild',reason='Explicit readonly review'))
            self.assertEqual(context['feedback']['passive_observation']['status'],'passive_recorder_unavailable')
            self.assertTrue(context['feedback']['fresh_observation_required'])
            self.assertEqual(context['current_state'],'unknown')
            return finish(skills=[ids['candidate']])
        model=ContextModel(choose,native=True)
        report=self.run_review(model,learning_sources=sources)
        self.assertEqual(len(model.contexts),5)
        self.assertEqual(model.rotations,['Explicit readonly review'])
        self.assertTrue(all(not reply['success'] for reply in model.native_replies))
        self.assertEqual(self.library.resolve_call({'package_id':ids['package'],'arguments':{}})['profile']['offset'],.12345678912345678)
        self.assertEqual(report['verifications'],[])
        self.assertEqual(report['stopped_reason'],'unknown_motion_no_retry')
        self.assertFalse(report['task_completed'])
        self.assertEqual(report['learning_boundary_source'],'fault_review_fallback_not_original_task')

    def test_frozen_boundary_unknown_stop_and_generation_survive_checkpoint_and_deny_manage(self):
        self.fault['learning_boundary']=self.boundary(stop_requested=True)
        def choose(index,context):
            facts=context['learning_boundary']
            self.assertEqual(facts['generation'],7)
            self.assertTrue(facts['stop_requested'])
            self.assertEqual(facts['last_execution']['status'],'unknown')
            self.assertTrue(facts['last_execution']['controller_terminal'])
            self.assertIsNone(facts['observation_evidence_id'])
            if index==0:return dict(action='save_working_context',arguments=self.declaration(context))
            if index==1:return dict(action='manage_working_context',arguments=dict(
                checkpoint_id=context['feedback']['result']['checkpoint_id'],method='rebuild',reason='Should remain stopped'))
            self.assertEqual(context['feedback']['phase'],'request_rejected')
            self.assertIn('STOP',context['feedback']['error'])
            return finish()
        model=ContextModel(choose)
        report=self.run_review(model)
        self.assertEqual(model.rotations,[])
        self.assertEqual(report['learning_boundary_source'],'frozen_run_boundary')
        record=json.loads(next((self.output/'continual-learning/work-context/checkpoints').glob('*.json')).read_text())
        facts=record['authoritative_facts']
        self.assertTrue(facts['stop_requested']);self.assertFalse(facts['motion_available'])
        self.assertEqual(facts['generation'],7)
        self.assertEqual(facts['task_contract_sha256'],'a'*64)
        self.assertEqual(facts['fault']['physical_outcome'],'unknown')

    def test_context_refresh_uses_only_passive_cache_and_refreshes_policy_not_current_state(self):
        self.fault['learning_boundary']=self.boundary()
        calls=[];checkpoint_policy={}
        def passive(folder):
            calls.append(folder);folder.mkdir()
            path=folder/'response.json';path.write_text('{"state":"unknown","snapshot":"cache"}')
            self.store.register_evidence('passive-after-rebuild',[path])
            return dict(success=True,evidence_id='passive-after-rebuild',motion_available=True,
                contentItems=[dict(type='inputImage',imageUrl='data:image/jpeg;base64,passive-fixture')])
        def choose(index,context):
            if index==0:
                checkpoint_policy['before']=context['learning_boundary']['source_policy_sha256']
                return dict(action='save_working_context',arguments=self.declaration(context))
            if index==1:return dict(action='manage_working_context',arguments=dict(
                checkpoint_id=context['feedback']['result']['checkpoint_id'],method='rebuild',reason='Explicit readonly rebuild'))
            self.assertIsNone(context['current_observation_evidence_id'])
            self.assertIsNone(context['learning_boundary']['observation_evidence_id'])
            self.assertFalse(context['feedback']['passive_observation']['motion_available'])
            self.assertTrue(context['feedback']['passive_observation']['historical'])
            self.assertEqual(context['feedback']['receipt']['historical_image_count'],1)
            self.assertGreater(context['feedback']['receipt']['extra_content_json_bytes'],0)
            self.assertNotEqual(context['learning_boundary']['source_policy_sha256'],checkpoint_policy['before'])
            return finish()
        model=ContextModel(choose)
        report=self.run_review(model,passive_observe=passive)
        self.assertEqual(len(calls),1);self.assertEqual(len(model.contexts),3)
        self.assertEqual(model.images[2],['data:image/jpeg;base64,passive-fixture'])
        self.assertIn('HISTORICAL passive',model.prompts[2])
        self.assertEqual(report['passive_camera_reads'][0]['current_observation_evidence_id'],None)
        self.assertIn('passive-after-rebuild',report['evidence_ids'])
        self.assertEqual(report['stopped_reason'],'unknown_motion_no_retry')

    def test_commit_learning_in_fault_registers_only_model_declared_lesson_for_reflection(self):
        ids={}
        def choose(index,context):
            if index==0:return dict(action='commit_learning',arguments=dict(lessons=[dict(id='fault-hypothesis',
                claim='Outcome remains unknown; this is a model hypothesis',applicability=['Current fixture'],
                limitations=['No current controller or outcome verification'],evidence_ids=[context['fault_evidence_id']],local_ref='lesson')]))
            ids['lesson']=context['feedback']['result']['bindings']['lesson']
            return finish(lessons=[ids['lesson']])
        report=self.run_review(ContextModel(choose))
        self.assertEqual(report['saved_candidates'][0]['candidate_id'],ids['lesson'])
        self.assertEqual(len(report['learning_commits']),1)
        self.assertFalse(report['stable_promoted']);self.assertEqual(report['verifications'],[])

    def test_pending_legacy_inspect_is_rejected_and_disabled_mode_has_no_new_capability(self):
        resources=LearningResources(self.store)
        changes=LearningChanges(self.library,resources)
        args={'lessons':[dict(id='pending',claim='MUST_NOT_BE_EXPOSED',applicability=['fixture'],limitations=['unknown'],
                             evidence_ids=['before-execution'],source_ids=['fixture-source'])]}
        plan=changes.preflight(args)
        def fail(stage):
            if stage=='before_final_commit':raise RuntimeError('fixture interruption')
        changes.failure_hook=fail
        with self.assertRaises(RuntimeError):changes.commit(args)
        identifier=plan['operations'][0]['identifier']
        def choose(index,context):
            if index==0:return dict(action='inspect_knowledge',arguments={'identifier':identifier})
            self.assertEqual(context['feedback']['phase'],'request_rejected')
            self.assertNotIn('MUST_NOT_BE_EXPOSED',json.dumps(context))
            return finish()
        report=self.run_review(ContextModel(choose))
        self.assertEqual(report['saved_candidates'],[])
        self.output=self.root/'disabled-review'
        def disabled(index,context):
            self.assertNotIn('continual_learning',context)
            if index==0:return dict(action='knowledge_directory',arguments={})
            self.assertEqual(context['feedback']['phase'],'request_rejected');return finish()
        disabled_report=self.run_review(ContextModel(disabled),continual_learning=False)
        self.assertNotIn('continual_learning',disabled_report)
        self.assertFalse((self.output/'continual-learning').exists())

    def test_current_stop_callback_aborts_before_new_model_or_provider_call(self):
        model=ContextModel(lambda *_:finish())
        with self.assertRaises(OperatorStop):self.run_review(model,stop_requested=lambda:True)
        self.assertEqual(model.contexts,[]);self.assertEqual(model.rotations,[])
        report=json.loads((self.output/'report.json').read_text())
        self.assertEqual(report['stopped_reason'],'unknown_motion_no_retry')
        self.assertFalse(report['task_completed'])

    def test_inflight_rpc_is_not_cleared_to_allow_context_rebuild(self):
        self.fault['learning_boundary']=self.boundary(inflight_rpc=True)
        def choose(index,context):
            self.assertTrue(context['learning_boundary']['inflight_rpc'])
            if index==0:return dict(action='save_working_context',arguments=self.declaration(context))
            if index==1:return dict(action='manage_working_context',arguments=dict(checkpoint_id=context['feedback']['result']['checkpoint_id'],
                method='rebuild',reason='Must remain blocked while request is unresolved'))
            self.assertEqual(context['feedback']['phase'],'request_rejected')
            self.assertIn('in-flight',context['feedback']['error']);return finish()
        model=ContextModel(choose)
        self.run_review(model);self.assertEqual(model.rotations,[])


if __name__=='__main__':unittest.main()
