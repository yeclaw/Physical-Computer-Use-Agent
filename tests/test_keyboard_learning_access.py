"""Actual knowledge and generic material readers enforce source closure, not only lists."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess, LearningAccessDenied
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS, LearningContractError
from astra_robotics_agent.physical_keyboard.adapters.skill_sources.learning_sources import LearningSources
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.recall import SkillRecall
from test_keyboard_skill_development import contract, SOURCE


SCOPE={'task_id':'access-fixture','device_id':'robot-fixture','layout_id':'layout-fixture'}


class LearningAccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='learning-access-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def access(self,sources=(),knowledge=(),evidence=(),own=False):
        return LearningAccess(SCOPE,dict(protocol=PROTOCOLS['access'],scope=SCOPE,
            allowed_source_ids=list(sources),allowed_knowledge_ids=list(knowledge),
            allowed_evidence_ids=list(evidence),allow_own_experience=own))

    def observation(self,store,name='obs-first',**kwargs):
        path=self.root/(name+'.json');path.write_text(json.dumps({'state':'historical','name':name}))
        return store.register_evidence(name,[path],**kwargs)

    def package(self,name='skill',evidence='obs-first',**kwargs):
        return dict(id=name,source_lines=list(SOURCE),profile={'text':'private-exact-profile'},
            contract=contract(),applicability=['scoped fixture'],limitations=['not verified'],
            evidence_ids=[evidence],**kwargs)

    def test_default_policy_none_keeps_old_bytes_and_exact_candidate_identity(self):
        store=KnowledgeStore(self.root/'store',SCOPE,'run')
        self.observation(store)
        library=SkillDevelopment(store)
        saved=library.save_package(self.package())
        candidate=store._candidate(saved['candidate_id'])
        self.assertNotIn('source_ids',candidate)
        self.assertNotIn('source_ids',library._record(saved['package_id']))
        self.assertNotIn('source_ids',store._evidence('obs-first'))
        before={p:p.read_bytes() for p in store.namespace.rglob('*') if p.is_file()}
        store.learning_access=None
        self.assertEqual(library.resolve_call({'package_id':saved['package_id'],'arguments':{'text':'new'}})['profile']['text'],'new')
        self.assertEqual({p:p.read_bytes() for p in before},before)

    def test_ordinary_no_demo_new_own_experience_package_issue_outcome_and_restart(self):
        access=self.access(own=True)
        store=KnowledgeStore(self.root/'store',SCOPE,'first',learning_access=access)
        observed=self.observation(store)
        library=SkillDevelopment(store)
        saved=library.save_package(self.package())
        issue=library.record_improvement(dict(id='fix',summary='negative event',rationale='evidence',status='open',
            skill_ids=[saved['package_id']],evidence_ids=['obs-first']))
        outcome=library.record_outcome(dict(id='attempt',skill_ids=[saved['package_id']],evidence_ids=['obs-first'],
            observation_evidence_id='obs-first',task_result='uncertain',correction_performed=True,
            remaining_issues=['unknown'],summary='model report'))
        self.assertEqual(store._candidate(saved['candidate_id'])['source_ids'],observed['source_ids'])
        fresh=self.access(sources=observed['source_ids'])
        reopened=KnowledgeStore(self.root/'store',SCOPE,'later',learning_access=fresh)
        later=SkillDevelopment(reopened)
        self.assertEqual(later.resolve_call({'package_id':saved['package_id'],'arguments':{'text':'later'}})['profile']['text'],'later')
        self.assertIn(saved['candidate_id'],reopened.context_text())
        self.assertEqual(later.list_improvements()['items'][0]['record_id'],issue['record_id'])
        self.assertEqual(later.inspect(outcome['record_id'])['source_ids'],observed['source_ids'])
        self.assertEqual(SkillRecall(later).for_skills([saved['package_id']])['pending_count'],1)

    def test_direct_profile_evidence_and_saved_compilation_do_not_bypass_policy(self):
        old=KnowledgeStore(self.root/'store',SCOPE,'legacy')
        self.observation(old)
        library=SkillDevelopment(old)
        saved=library.save_package(self.package())
        original={p:p.read_bytes() for p in old.namespace.rglob('*') if p.is_file()}
        old.learning_access=self.access()
        for action in [lambda:old.get_skill(saved['candidate_id']),lambda:library.inspect(saved['package_id']),
                       lambda:library.resolve_call({'package_id':saved['package_id'],'arguments':{'text':'bypass'}}),
                       lambda:library.inspect_evidence({'evidence_id':'obs-first','file_indices':[0]}),
                       lambda:old.record_verification(saved['candidate_id'],['obs-first'],'ok')]:
            with self.assertRaises(LearningAccessDenied):action()
        self.assertEqual(library.search()['results'],[])
        self.assertNotIn('private-exact-profile',old.context_text())
        self.assertEqual(SkillRecall(library).matching_source('\n'.join(SOURCE)+'\n'),[])
        self.assertEqual({p:p.read_bytes() for p in original},original)

    def test_allowed_outer_package_cannot_wash_hidden_dependency_or_parent(self):
        store=KnowledgeStore(self.root/'store',SCOPE,'legacy')
        self.observation(store)
        library=SkillDevelopment(store)
        dependency=library.save_package(self.package('hidden'))
        composite=self.package('allowed',dependencies={'step':dependency['package_id']},parent_id=dependency['package_id'])
        composite['source_lines']=['def main(robot, profile):','    return step(robot, {"text": profile["text"]})']
        saved=library.save_package(composite)
        allowed=[saved['package_id'],saved['candidate_id'],dependency['candidate_id']]
        store.learning_access=self.access(knowledge=allowed,evidence=['obs-first'])
        with self.assertRaises(LearningAccessDenied):library.inspect(saved['package_id'])
        with self.assertRaises(LearningAccessDenied):library.resolve_call({'package_id':saved['package_id'],'arguments':{'text':'bypass'}})
        self.assertEqual(library.search(kind='package')['results'],[])
        # Even when every knowledge ID is granted, its evidence is independent.
        store.learning_access=self.access(knowledge=allowed+[dependency['package_id']])
        with self.assertRaises(LearningAccessDenied):library.inspect(saved['package_id'])

    def test_fully_allowed_legacy_ids_are_readable_but_unbound_origins_cannot_be_derived(self):
        store=KnowledgeStore(self.root/'store',SCOPE,'old')
        self.observation(store)
        library=SkillDevelopment(store)
        saved=library.save_package(self.package())
        access=self.access(knowledge=[saved['package_id'],saved['candidate_id']],evidence=['obs-first'],own=True)
        store.learning_access=access
        self.assertEqual(library.inspect(saved['package_id'])['declared_profile']['text'],'private-exact-profile')
        with self.assertRaises(LearningAccessDenied):library.save_package(self.package('unbound'))
        own=self.observation(store,'new-obs')
        # Mixing a known new observation does not wash an old unbound dependency.
        with self.assertRaises(LearningAccessDenied):library.save_package(self.package('mixed','new-obs',dependencies={'step':saved['package_id']}))
        access.allowed_sources.add('declared-legacy-source')
        access.bind_evidence('obs-first',['declared-legacy-source'])
        created=library.save_package(self.package('explicitly-bound'))
        self.assertEqual(store._candidate(created['candidate_id'])['source_ids'],['declared-legacy-source'])

    def test_derived_evidence_and_host_sources_checked_before_visibility(self):
        access=self.access(sources=['allowed'],own=False)
        store=KnowledgeStore(self.root/'store',SCOPE,'run',learning_access=access)
        with self.assertRaises(LearningAccessDenied):self.observation(store,source_ids=['hidden-demo'])
        self.assertFalse(list((store.namespace/'evidence').iterdir()))
        self.observation(store,source_ids=['allowed'])
        library=SkillDevelopment(store)
        with self.assertRaises(LearningAccessDenied):library.save_package(self.package(),source_ids=['hidden-demo'])
        saved=library.save_package(self.package(),source_ids=['allowed'])
        access.allowed_sources.clear()
        self.assertEqual(library.search()['results'],[])
        self.assertEqual(library.context()['pending_count'],0)
        with self.assertRaises(LearningAccessDenied):store._candidate(saved['candidate_id'])

    def test_revoked_verification_evidence_does_not_leak_status_into_allowed_context(self):
        access=self.access(sources=['a','b'])
        store=KnowledgeStore(self.root/'store',SCOPE,'run',learning_access=access)
        self.observation(store,source_ids=['a'])
        self.observation(store,'later-exec',source_ids=['b'])
        library=SkillDevelopment(store)
        saved=library.save_package(self.package())
        store.record_verification(saved['candidate_id'],['later-exec'],'ok')
        self.assertIn('verified_execution_only',store.context_text())
        access.allowed_sources.remove('b')
        self.assertIn(saved['candidate_id'],store.context_text())
        self.assertNotIn('verified_execution_only',store.context_text())
        with self.assertRaises(LearningAccessDenied):store.record_verification(saved['candidate_id'],['later-exec'],'ok')




    def test_generic_material_inspect_direct_id_and_analysis_use_one_filter(self):
        access = self.access(sources=['human_a', 'human_b'])
        reader = LearningSources(SCOPE, access=access)
        measured = dict(calibration=dict(frame='fixture_tool', tip_offset_m=[.001, .002, -.03]))
        first = reader.register('human_a', kind='human_demo', data=measured)
        reader.register('human_b', kind='human_demo', data=dict(calibration=dict(private_hint='held_out')))
        self.assertEqual(reader.list_sources()['total'], 2)
        measured['calibration']['tip_offset_m'][2] = -99
        access.allowed_sources.remove('human_b')
        visible = reader.list_sources()
        self.assertEqual([item['source_id'] for item in visible['sources']], ['human_a'])
        self.assertNotIn('human_b', json.dumps(visible))
        self.assertEqual(reader.describe('human_a')['sha256'], first['sha256'])
        self.assertEqual(reader.analysis_data('human_a', fields=['calibration'])['calibration']['tip_offset_m'][2], -.03)
        for read in (lambda: reader.describe('human_b'), lambda: reader.inspect('human_b'),
                     lambda: reader.analysis_data('human_b', fields=['calibration']),
                     lambda: reader.register('allowed_copy', kind='device_reference',
                         data=dict(claim='copied from held-out source'), source_ids=['human_b'])):
            with self.assertRaises(LearningAccessDenied):
                read()
        access.allowed_sources.clear()
        self.assertEqual(reader.list_sources()['sources'], [])
        with self.assertRaises(LearningAccessDenied):
            reader.analysis_data('human_a', fields=['calibration'])
        with self.assertRaises(LearningContractError):
            reader.describe('unknown')


if __name__=='__main__':unittest.main()
