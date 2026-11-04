"""Online improvement integration with real stores and fixture-only execution.

The scripted model owns every decision below. This verifies plumbing, not a
frontier model's initiative, physical typing quality, or stable promotion.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.agent.learning_loop import parse_decision, run_episode
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from test_keyboard_skill_development_loop import (
    CORRECT, EMIT, GLYPHS, NOISY, SCOPE, Environment, Model, declaration, finish,
)


class OnlineRsiLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='online-rsi-loop-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'fixture')
        self.library = SkillDevelopment(self.store)
        seed = self.root / 'initial.json'
        seed.write_text('{"fixture":true}')
        self.store.register_evidence('seed-observation', [seed])

    def package_args(self, name, source=EMIT):
        return dict(id=name, source_lines=list(source), profile={'glyphs': GLYPHS, 'payload': 'a'},
                    contract=declaration(), applicability=['Only the fake layout.'],
                    limitations=['No physical or model-efficacy claim.'], evidence_ids=['seed-observation'])

    def package(self, name, source=EMIT):
        return self.library.save_package(self.package_args(name, source))

    def issue_args(self, name, skill, *, priority=5, status='open', evidence='seed-observation'):
        return dict(id=name, summary='A reusable fixture operation produces an extra symbol.',
                    rationale='A compensating edit does not repair the reusable operation.',
                    status=status, priority=priority, revisit_when=['When that operation is next reused.'],
                    skill_ids=[skill], evidence_ids=[evidence])

    def bury(self):
        unrelated = self.package('unrelated_operation')
        for index in range(8):
            self.library.record_improvement(self.issue_args('unrelated_' + str(index),
                unrelated['package_id'], priority=90))

    def output(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def call(self, package, payload='a'):
        return {'action': 'call_skill', 'arguments': {
            'package_id': package['package_id'], 'arguments': {'payload': payload}},
            'hypothesis': 'Fixture input should have the requested visible result.'}

    def inline(self, source=EMIT, *, refs=None, payload='a'):
        args = {'label': 'fixture_inline', 'source_lines': list(source),
                'profile': {'glyphs': GLYPHS, 'payload': payload}}
        if refs is not None:
            args['related_skill_ids'] = refs
        return {'action': 'run_skill', 'arguments': args,
                'hypothesis': 'Use the recalled evidence for this fixture input.'}

    def run_model(self, choose, env, name, limit=12, execute=None, **kwargs):
        model = Model(choose)
        report = run_episode(session=model, observe=env.observe, execute=execute or env.execute,
            store=env.store, output=self.output(name), max_decisions=limit,
            skill_development=True, **kwargs)
        return model, report

    def test_compensated_defect_is_recalled_for_later_task_and_model_authors_reusable_revision(self):
        noisy = self.package('initial_noisy_operation', NOISY)
        env = Environment(self.root, self.store, 'first')
        captured = {}
        original = self.store.namespace / 'candidates' / noisy['candidate_id']
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir()}

        def first(index, context, raw):
            current = context['current_observation_evidence_id']
            if index == 0:
                return self.call(noisy)
            if index == 1:
                self.assertEqual(env.text, 'as')
                return {'action': 'record_improvement', 'arguments': self.issue_args(
                    'extra_symbol', noisy['package_id'], evidence=current)}
            if index == 2:
                captured['issue'] = context['feedback']['result']['record_id']
                return {'action': 'run_skill', 'arguments': {
                    'label': 'compensate_current_input', 'source_lines': CORRECT, 'profile': {}},
                    'hypothesis': 'Remove the extra symbol from the current draft.'}
            if index == 3:
                self.assertEqual(env.text, 'a')
                return {'action': 'record_skill_outcome', 'arguments': {
                    'id': 'compensated', 'skill_ids': [noisy['package_id']], 'evidence_ids': [current],
                    'observation_evidence_id': current, 'task_result': 'success',
                    'correction_performed': True, 'remaining_issues': ['The original operation is still defective.'],
                    'summary': 'This input is correct after compensation; reusable defect remains.'}}
            return finish()

        _, first_report = self.run_model(first, env, 'first-output')
        self.assertEqual(len(env.executions), 2)
        self.assertFalse(first_report['stable_promoted'])
        self.assertEqual(self.library.inspect(captured['issue'])['status'], 'open')
        self.bury()
        self.assertNotIn(captured['issue'], json.dumps(self.library.context()))

        later_store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'later-task')
        later = Environment(self.root, later_store, 'later')
        def second(index, context, raw):
            current = context['current_observation_evidence_id']
            if index == 0:
                self.assertNotIn(captured['issue'], raw)
                return self.call(noisy)
            if index == 1:
                self.assertEqual(context['feedback']['phase'], 'related_skill_memory')
                self.assertFalse(context['feedback']['dispatched'])
                self.assertFalse(context['feedback']['auto_resume'])
                self.assertEqual(later.executions, [])
                self.assertEqual(later.observations, 1)
                self.assertEqual(context['current_observation_evidence_id'], 'later-obs-1')
                memory = context['skill_development']['online_recall']
                self.assertIn(captured['issue'], [x['record_id'] for x in memory['pending_improvements']])
                self.assertTrue(memory['recent_outcomes'][0]['correction_performed'])
                # The mock RSI chooses a different program, rather than allowing
                # the previously requested noisy call to be resumed by the host.
                return self.inline(refs=[noisy['package_id']])
            if index == 2:
                self.assertEqual(later.text, 'a')
                return {'action': 'record_skill_outcome', 'arguments': {
                    'id': 'first_repair', 'skill_ids': [noisy['package_id']], 'evidence_ids': [current],
                    'observation_evidence_id': current, 'task_result': 'success',
                    'correction_performed': False, 'remaining_issues': ['Repeatability is not established.'],
                    'summary': 'The fixture revision works once; keep the item open for subsequent reuse.'}}
            if index == 3:
                args = self.package_args('model_revised_operation')
                args.update(parent_id=noisy['package_id'], improvement_ids=[captured['issue']],
                            evidence_ids=[current])
                return {'action': 'save_skill_package', 'arguments': args}
            captured['revision'] = context['feedback']['result']
            return finish(skills=[captured['revision']['candidate_id']])

        _, later_report = self.run_model(second, later, 'later-output')
        self.assertEqual(len(later.executions), 1)
        self.assertEqual(len(later_report['skill_memory_reviews']), 1)
        self.assertTrue(any(captured['issue'] in entry['record_ids']
                            for entry in later_report['skill_memory_contexts']))
        self.assertTrue(all(entry['meaning'] == 'context_presented_not_proof_of_use_or_improvement'
                            for entry in later_report['skill_memory_contexts']))
        self.assertEqual(self.library.inspect(captured['issue'])['status'], 'open')
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir()})

        reuse_store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'reuse-task')
        reuse = Environment(self.root, reuse_store, 'reuse')
        def third(index, context, raw):
            if index == 0:
                return {'action': 'recall_skills', 'arguments': {
                    'skill_ids': [captured['revision']['package_id']]}}
            if index == 1:
                return self.call(captured['revision'], 'orbit')
            if index == 2:
                self.assertEqual(reuse.text, 'orbit')
                return self.call(captured['revision'], 'delta')
            self.assertEqual(reuse.text, 'orbitdelta')
            return finish()
        _, reused = self.run_model(third, reuse, 'reuse-output')
        self.assertEqual(len(reuse.executions), 2)
        self.assertEqual(reused['skill_memory_reviews'], [])
        self.assertFalse(reused['stable_promoted'])

    def test_revised_backlog_version_is_presented_but_known_issue_does_not_repeatedly_pause(self):
        package = self.package('reused_operation')
        first = self.library.record_improvement(self.issue_args('known_defect', package['package_id']))
        env = Environment(self.root, self.store, 'versions')
        captured = {}
        def choose(index, context, raw):
            if index in (0, 1):
                return self.call(package)
            if index == 2:
                args = self.issue_args('known_defect', package['package_id'], priority=97,
                                       status='deferred', evidence=context['current_observation_evidence_id'])
                args['revisit_when'] = ['When a verified starting condition can be restored.']
                return {'action': 'record_improvement', 'arguments': args}
            if index == 3:
                captured['new'] = context['feedback']['result']['record_id']
                self.assertNotEqual(captured['new'], first['record_id'])
                pending = context['skill_development']['online_recall']['pending_improvements']
                self.assertEqual(pending[0]['record_id'], captured['new'])
                self.assertEqual(pending[0]['priority'], 97)
                return self.call(package)
            return finish()
        _, report = self.run_model(choose, env, 'versions-output')
        self.assertEqual(len(env.executions), 3)
        self.assertEqual(report['skill_memory_reviews'], [])
        shown = {identifier for entry in report['skill_memory_contexts'] for identifier in entry['record_ids']}
        self.assertTrue({first['record_id'], captured['new']} <= shown)
        self.assertEqual(self.library.inspect(first['record_id'])['status'], 'open')

    def test_inline_reuse_metadata_is_consumed_before_runtime_and_known_memory_needs_only_one_briefing(self):
        package = self.package('related_operation', NOISY)
        issue = self.library.record_improvement(self.issue_args('buried_inline_issue', package['package_id']))
        self.bury()
        env = Environment(self.root, self.store, 'inline')
        source = EMIT + ['# A novel model-authored inline revision.']
        request = self.inline(source, refs=[package['package_id']])
        def choose(index, context, raw):
            if index == 0:
                return request
            if index == 1:
                self.assertIn(issue['record_id'], raw)
                self.assertEqual(env.executions, [])
                self.assertEqual(context['feedback']['phase'], 'related_skill_memory')
                return request
            if index == 2:
                self.assertEqual(env.text, 'a')
                return request
            return finish()
        _, report = self.run_model(choose, env, 'inline-output')
        self.assertEqual(len(env.executions), 2)
        self.assertEqual(len(report['skill_memory_reviews']), 1)
        for execution in env.executions:
            self.assertEqual(set(execution['arguments']), {'source', 'profile', 'label'})
            self.assertNotIn('related_skill_ids', execution['arguments'])

    def test_exact_registered_source_recalls_prior_issue_without_model_metadata(self):
        skill = self.store.propose('skill', {'id': 'legacy_noisy', 'source_lines': NOISY,
            'profile': {'glyphs': GLYPHS, 'payload': 'a'}, 'applicability': ['Fixture only.'],
            'limitations': ['No physical result.'], 'evidence_ids': ['seed-observation']})
        self.library.record_improvement(self.issue_args('exact_source_issue', skill['candidate_id']))
        self.bury()
        env = Environment(self.root, self.store, 'exact')
        def choose(index, context, raw):
            if index < 2:
                if index == 1:
                    self.assertEqual(context['feedback']['phase'], 'related_skill_memory')
                    self.assertEqual(env.executions, [])
                return self.inline(NOISY)
            return finish()
        _, report = self.run_model(choose, env, 'exact-output')
        self.assertEqual(len(env.executions), 1)
        self.assertEqual(env.text, 'as')  # The model chose to repeat, not the framework.
        self.assertEqual(len(report['skill_memory_reviews']), 1)
        self.assertIn(skill['candidate_id'], report['skill_use_contexts'][0]['focus_skill_ids'])

    def test_returning_after_unrelated_work_rebriefs_if_issue_is_absent_from_current_memory(self):
        package = self.package('occasionally_reused', NOISY)
        issue = self.library.record_improvement(self.issue_args('older_local_issue', package['package_id']))
        self.bury()
        unrelated = self.package('different_task_operation')
        env = Environment(self.root, self.store, 'switch')
        def choose(index, context, raw):
            if index == 0:
                return self.call(package)
            if index == 1:
                self.assertEqual(context['feedback']['phase'], 'related_skill_memory')
                return self.call(package)
            if index == 2:
                return self.call(unrelated, 'z')
            if index == 3:
                # The old issue was seen earlier in this very session, but is
                # no longer in the supplied bounded memory for this decision.
                self.assertNotIn(issue['record_id'], json.dumps(context['skill_development']))
                return self.call(package)
            if index == 4:
                self.assertEqual(context['feedback']['phase'], 'related_skill_memory')
                self.assertIn(issue['record_id'], raw)
                self.assertEqual(len(env.executions), 2)
                return self.call(package)
            return finish()
        _, report = self.run_model(choose, env, 'switch-output')
        self.assertEqual(len(env.executions), 3)
        self.assertEqual(len(report['skill_memory_reviews']), 2)
        self.assertEqual(env.text, 'aszas')

    def test_foreign_references_and_disabled_metadata_never_reach_execution(self):
        foreign_store = KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='foreign'), 'foreign')
        foreign_store.register_evidence('seed-observation', [self.root / 'initial.json'])
        foreign = SkillDevelopment(foreign_store).save_package(self.package_args('foreign_operation'))
        env = Environment(self.root, self.store, 'invalid')
        request = self.inline(refs=[foreign['package_id']])
        model, report = self.run_model(lambda index, *unused: request if index == 0 else finish(),
            env, 'invalid-output')
        self.assertEqual(env.executions, [])
        self.assertEqual(model.contexts[1]['feedback']['phase'], 'request_rejected')
        self.assertEqual(report['skill_memory_reviews'], [])
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(request), skill_development=False)
        with self.assertRaises(ValueError):
            parse_decision(json.dumps({'action': 'recall_skills', 'arguments': {'skill_ids': []}}),
                           skill_development=False)

    def test_unknown_or_terminal_after_briefing_stops_after_one_real_dispatch(self):
        package = self.package('uncertain_operation', NOISY)
        self.library.record_improvement(self.issue_args('uncertain_issue', package['package_id']))
        self.bury()
        for outcome in ('unknown', 'terminal'):
            with self.subTest(outcome=outcome):
                env = Environment(self.root, self.store, outcome)
                sent = []
                def execute(arguments):
                    sent.append(arguments)
                    response = ({'status': 'unknown', 'result': {'motion_outcome_uncertain': True}}
                        if outcome == 'unknown' else {'status': 'ok', 'result': {
                            'receipts': [], 'motion_outcome_uncertain': False, 'controller_terminal': True}})
                    return {'response': response, 'evidence_id': env.evidence('execution', response)}
                model, report = self.run_model(lambda *unused: self.call(package), env,
                    outcome + '-output', execute=execute, limit=7)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(len(sent), 1)
                self.assertEqual(env.observations, 1)
                self.assertEqual(len(report['skill_memory_reviews']), 1)
                self.assertEqual(report['stopped_reason'], 'unknown_motion_no_retry' if outcome == 'unknown'
                                 else 'native_controller_terminal_no_retry')

    def test_model_can_abandon_a_briefed_request_without_queued_motion(self):
        package = self.package('abandoned_operation', NOISY)
        self.library.record_improvement(self.issue_args('abandoned_issue', package['package_id']))
        self.bury()
        env = Environment(self.root, self.store, 'abandoned')
        _, report = self.run_model(lambda index, *unused: self.call(package) if index == 0 else finish(),
            env, 'abandoned-output')
        self.assertEqual(env.executions, [])
        self.assertEqual(env.observations, 1)
        self.assertEqual(report['verifications'], [])
        self.assertEqual(report['skill_calls'], [])
        self.assertEqual(len(report['skill_memory_reviews']), 1)


if __name__ == '__main__':
    unittest.main()
