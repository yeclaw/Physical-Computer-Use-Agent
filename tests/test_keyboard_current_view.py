"""No automatic latest execution; branches, counterexamples and recovery."""
import hashlib
import json
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.skills.current_view import CurrentView
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.resources import LearningResources
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS
from test_keyboard_learning_resources import Fixture


class CurrentViewTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.changes = LearningChanges(self.library, self.resources)
        self.view = CurrentView(self.library, self.changes)

    def save(self, text, selection=None):
        args = self.package(profile={'text': text}, source_ids=['source-a'], local_ref='p')
        batch = {'packages': [args]}
        if selection is not None:
            batch['selections'] = [dict(logical_id='press', identifier={'local_ref': 'p'},
                conditions=[selection], reason='Model explicitly chose the applicable start')]
        return self.changes.commit(batch)['bindings']['p']

    def test_unselected_versions_group_without_latest_and_backing_duplicates(self):
        first, second = self.save('one'), self.save('two')
        result = self.view.directory()
        self.assertEqual(result['total_count'], 1)
        group = result['items'][0]
        self.assertEqual(group['version_count'], 2)
        self.assertEqual(group['kind'], 'package')
        self.assertFalse(group['selected'])
        self.assertFalse(group['automatic_latest'])
        self.assertEqual({row['identifier'] for row in group['versions']}, {first, second})
        self.assertEqual(group['selection_state'], 'unselected')
        self.assertNotIn('source_lines', str(result))
        self.assertNotIn('declared_profile', str(result))

    def test_hover_neutral_branches_persist_and_replace_only_explicit_conditions(self):
        neutral = self.save('neutral-v1', 'neutral start')
        hover = self.save('hover-v1', 'hover start')
        self.assertEqual({row['identifier'] for row in self.view.selections('press')}, {neutral, hover})
        replacement = self.save('neutral-v2', 'neutral start')
        branches = self.view.selections('press')
        self.assertEqual({row['identifier'] for row in branches}, {replacement, hover})
        self.assertTrue(all(row['scope'] == self.library.scope and row['commit_id'] for row in branches))
        restored = CurrentView(self.library, LearningChanges(self.library, self.resources))
        self.assertEqual(restored.selections('press'), branches)
        self.assertEqual(self.library.resolve_call({'package_id': neutral, 'arguments': {}})['profile']['text'], 'neutral-v1')

    def test_half_commit_cannot_replace_existing_selection(self):
        selected = self.save('selected', 'neutral')
        before = self.view.selections('press')
        def fail(stage):
            if stage == 'before_final_commit':
                raise RuntimeError('Interrupted before visibility')
        self.changes.failure_hook = fail
        with self.assertRaises(RuntimeError):
            self.save('pending replacement', 'neutral')
        self.assertEqual(self.view.selections('press'), before)
        self.assertEqual(self.view.directory()['items'][0]['version_count'], 1)
        self.assertEqual(self.view.selections('press')[0]['identifier'], selected)
        group=self.view.directory(version_limit=1,branch_limit=1)['items'][0]
        self.assertEqual(group['version_count'],1)
        self.assertEqual(group['versions'][0]['identifier'],selected)
        self.assertEqual(group['branches_count'],1)

    def test_alias_and_replacement_cycles_reject_but_contradictions_can_be_mutual(self):
        one, two = self.save('one'), self.save('two')
        for relation_kind in ('alias', 'supersedes', 'derived_from'):
            with self.subTest(kind=relation_kind), self.assertRaises(ValueError):
                self.changes.commit({'relations': [dict(kind=relation_kind, from_id=one, to_id=two),
                    dict(kind=relation_kind, from_id=two, to_id=one)]})
        result = self.changes.commit({'relations': [dict(kind='contradicts', from_id=one, to_id=two),
            dict(kind='contradicts', from_id=two, to_id=one)]})
        self.assertEqual(len(result['relations']), 2)

    def test_scoped_supersession_hides_default_only_and_preserves_exact_old(self):
        one = self.save('one')
        two = self.save('two')
        self.changes.commit({'relations': [dict(kind='supersedes', from_id=two, to_id=one,
            conditions=['only hover branch'], reason='Limited explicit scope')]})
        self.assertEqual(self.view.directory()['items'][0]['active_version_count'], 2)
        self.changes.commit({'relations': [dict(kind='supersedes', from_id=two, to_id=one)]})
        self.assertEqual(self.view.directory()['items'][0]['active_version_count'], 1)
        self.assertEqual(self.library.inspect(one)['declared_profile']['text'], 'one')

    def test_same_attempt_multiple_relations_does_not_create_new_evidence_or_resolve_issue(self):
        package = self.save('compensated')
        issue = dict(id='repeat', summary='Current text corrected after repeated six',
            rationale='Underlying primitive remains uncertain', status='open', skill_ids=[package],
            evidence_ids=['obs'], local_ref='issue')
        lesson = self.lesson('compensation', local_ref='lesson')
        result = self.changes.commit(dict(lessons=[lesson], improvements=[issue], relations=[
            dict(kind='derived_from', from_id={'local_ref': 'lesson'}, to_id=package),
            dict(kind='addresses_issue', from_id=package, to_id={'local_ref': 'issue'}),
            dict(kind='uses_lesson', from_id=package, to_id={'local_ref': 'lesson'})]))
        self.assertEqual(len([row for row in self.store._events() if row.get('type') == 'evidence_registered']), 1)
        self.assertEqual(self.library.inspect(result['bindings']['issue'])['status'], 'open')
        self.assertFalse(self.library.inspect(package)['stable'])

    def test_directory_is_bounded_filterable_and_validates_corruption(self):
        self.save('one')
        self.changes.commit({'lessons': [self.lesson('explanation')]})
        page = self.view.directory(limit=1)
        self.assertEqual(page['total_count'], 2)
        self.assertEqual(page['next_offset'], 1)
        self.assertEqual(self.view.directory(kind='package')['total_count'], 1)
        self.assertEqual(self.view.directory(query='explanation')['items'][0]['kind'], 'lesson')
        for kwargs in ({'limit': True}, {'offset': -1}, {'kind': 'execute'}, {'query': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.view.directory(**kwargs)
        path = next((self.changes.root / 'commits').glob('*.json'))
        path.write_text('{}')
        with self.assertRaises(ValueError):
            self.view.directory()

    def snapshot(self):
        return {str(path.relative_to(self.store.root)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.store.root.rglob('*') if path.is_file()}

    def test_fifty_versions_default_four_and_exact_ids_are_fully_pageable_without_writes(self):
        self.changes.access.bind_evidence('obs',['source-a'])
        saved=[]
        for index in range(50):
            saved.append(self.store.propose('skill',dict(id='long-history',
                source_lines=['def main(robot, profile):','    return profile["version"]'],
                profile={'version':index},applicability=['Explicit fixture start'],limitations=['Not verified'],
                evidence_ids=['obs']))['candidate_id'])
        before=self.snapshot()
        brief=self.view.directory(kind='skill',query='long-history')
        group=brief['items'][0]
        self.assertEqual([row['identifier'] for row in group['versions']],saved[-1:-5:-1])
        self.assertEqual(group['version_count'],50)
        self.assertEqual(group['active_version_count'],50)
        self.assertEqual(group['version_next_offset'],4)
        self.assertFalse(group['selected']);self.assertFalse(group['automatic_latest'])
        self.assertEqual(brief['ordering']['authority'],'metadata_only_no_automatic_latest')
        self.assertLess(len(json.dumps(brief,sort_keys=True).encode()),8192)
        found=[];offset=0
        while True:
            page=self.view.directory(kind='skill',query='long-history',version_offset=offset,version_limit=7)['items'][0]
            found.extend(row['identifier'] for row in page['versions'])
            offset=page['version_next_offset']
            if offset is None:break
        self.assertEqual(found,list(reversed(saved)))
        self.assertEqual(self.view.directory(kind='skill',version_offset=49)['items'][0]['versions'][0]['identifier'],saved[0])
        self.assertEqual(self.snapshot(),before)

    def test_branch_pages_preserve_full_conditions_and_selected_even_when_page_empty(self):
        first=self.save('neutral');second=self.save('hover')
        selections=[]
        for index in range(11):
            selections.append(dict(logical_id='press',identifier=first if index%2 else second,
                conditions=['branch-'+str(index)+' '+('complete condition '*12),'second exact condition '+str(index)],
                reason='Explicit Astra choice with complete scope '+str(index)))
        self.changes.commit({'selections':selections})
        expected=self.view.selections('press');before=self.snapshot()
        first_page=self.view.directory()['items'][0]
        self.assertEqual(first_page['branches_count'],11)
        self.assertEqual(first_page['branch_next_offset'],4)
        self.assertEqual(first_page['branches'],expected[:4])
        found=[];offset=0
        while True:
            group=self.view.directory(branch_offset=offset,branch_limit=3)['items'][0]
            found.extend(group['branches']);offset=group['branch_next_offset']
            if offset is None:break
        self.assertEqual(found,expected)
        empty=self.view.directory(branch_offset=100,version_offset=100)['items'][0]
        self.assertEqual(empty['branches'],[]);self.assertEqual(empty['versions'],[])
        self.assertTrue(empty['selected']);self.assertEqual(empty['selection_state'],'explicit_branches')
        self.assertEqual(empty['branches_count'],11);self.assertEqual(empty['version_count'],2)
        self.assertEqual(self.library.resolve_call({'package_id':first,'arguments':{}})['profile']['text'],'neutral')
        self.assertEqual(self.snapshot(),before)

    def test_page_filters_scope_sources_and_query_before_counts(self):
        self.changes.access.bind_evidence('obs',['source-a'])
        visible=self.save('visible','full visible condition')
        path=self.root/'hidden.json';path.write_text('{"historical":"hidden-source"}')
        self.store.register_evidence('hidden-observation',[path],source_ids=['source-b'])
        args=self.package(profile={'text':'hidden'},source_ids=['source-b'],evidence_ids=['hidden-observation'])
        hidden=self.changes.commit({'packages':[args]})['record_ids'][0]
        policy=LearningAccess(self.library.scope,dict(protocol=PROTOCOLS['access'],scope=self.library.scope,
            allowed_source_ids=['source-a'],allowed_knowledge_ids=[],allowed_evidence_ids=['obs'],allow_own_experience=True))
        policy.bind_evidence('obs',['source-a'])
        filtered_changes=LearningChanges(self.library,LearningResources(self.store,access=policy),access=policy)
        filtered=CurrentView(self.library,filtered_changes)
        result=filtered.directory(query='press',kind='package',version_limit=1,branch_limit=1)
        self.assertEqual(result['total_count'],1)
        self.assertEqual(result['items'][0]['version_count'],1)
        self.assertEqual(result['items'][0]['versions'][0]['identifier'],visible)
        self.assertNotIn(hidden,json.dumps(result))
        self.assertEqual(filtered.directory(query='absent')['items'],[])
        self.assertEqual(result['scope'],self.library.scope)
        foreign=KnowledgeStore(self.store.root,dict(self.library.scope,layout_id='foreign'),'foreign')
        foreign.register_evidence('foreign-observation',[path])
        self.assertEqual(filtered.directory(query='press',kind='package')['items'][0]['version_count'],1)
        for kwargs in ({'version_limit':True},{'branch_offset':True},{'version_limit':0},{'branch_limit':33},
                       {'version_offset':-1},{'branch_offset':1000001},{'version_offset':1.0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):filtered.directory(**kwargs)

    def test_selection_b_does_not_change_frozen_a_source_profile_or_dependency_pins(self):
        def batch(text):
            component=self.package('component',profile={'text':'component-'+text},source_ids=['source-a'],local_ref='component')
            if text=='B':component['source_lines']=['def main(robot, profile):','    return {"changed": profile["text"]}']
            package=self.package(profile={'text':text},source_ids=['source-a'],local_ref='press',
                source_lines=['def main(robot, profile):','    return leaf(robot, {})'],
                dependencies={'leaf':{'local_ref':'component'}})
            return dict(packages=[component,package],selections=[dict(logical_id='press',identifier={'local_ref':'press'},
                conditions=['neutral start'],reason='Explicit model choice '+text)])
        with patch('astra_robotics_agent.physical_keyboard.runtime.skill_language.run_skill') as executor:
            first=self.changes.commit(batch('A'))['bindings']
            frozen=self.library.resolve_call({'package_id':first['press'],'arguments':{}})
            original=json.dumps(frozen,sort_keys=True,separators=(',',':')).encode()
            second=self.changes.commit(batch('B'))['bindings']
            self.assertEqual([row['identifier'] for row in self.view.selections('press')],[second['press']])
            self.assertEqual(json.dumps(frozen,sort_keys=True,separators=(',',':')).encode(),original)
            later=self.library.resolve_call({'package_id':first['press'],'arguments':{}})
            self.assertEqual(json.dumps(later,sort_keys=True,separators=(',',':')).encode(),original)
            new=self.library.resolve_call({'package_id':second['press'],'arguments':{}})
            self.assertNotEqual(new['source'],frozen['source'])
            self.assertNotEqual(new['profile'],frozen['profile'])
            self.assertNotEqual(new['dependency_pins'],frozen['dependency_pins'])
            self.assertEqual(self.library.inspect(first['press'])['source'],frozen['source'])
            self.assertEqual(self.library.inspect(first['press'])['profile'],frozen['profile'])
            executor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
