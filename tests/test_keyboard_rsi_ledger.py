"""Adversarial software checks; no model, simulation, camera or robot imports."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from astra_robotics_agent.physical_keyboard import (
    EvaluationPlan, IntegrityError, Ledger, LedgerError, Scope, StaleParentError, Usage)
from astra_robotics_agent.physical_keyboard.contracts import aggregate_usage, digest


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'ledger'
        self.ledger = Ledger(self.root)
        self.addCleanup(lambda: self.ledger.close())
        self.scope = Scope('synthetic', 'fixture-device', 'layout-1',
                           *[digest(x.encode()) for x in ['config', 'protocol', 'evaluator', 'goal', 'safety']])
        self.seed = self.bundle('seed', kind='seed')
        self.ledger.bootstrap(self.seed, reason='Synthetic fixture seed, not a learned skill')
        self.candidate = self.bundle('candidate', parent=self.seed)
        self.plan = EvaluationPlan({'a': 'cat', 'b': 'dog'}, 2, 1, 1000)
        self.open_round()

    def bundle(self, label, *, parent=None, kind='skill_program', scope=None):
        return self.ledger.freeze_bundle(label=label, kind=kind, scope=scope or self.scope,
                                        files={'skill.py': ('# synthetic ' + label).encode(),
                                               'profile.json': json.dumps({'label': label}).encode()},
                                        parent=parent, provenance={'origin': 'original', 'fixture': True})

    def open_round(self, name='software-round', candidate=None, plan=None, usage=None, mode='synthetic'):
        self.ledger.open_round(name, candidate=candidate or self.candidate, plan=plan or self.plan,
                               generation_usage=usage or Usage.zero(),
                               metadata={'evidence_mode': mode, 'experiment_id': 'software-only', 'model': None})

    def trial(self, variant, case='a', repetition=0, *, round_id='software-round',
              status='observed', observed=None, usage=None, condition=None, execution=None):
        trial_id = f'{variant}-{case}-{repetition}'
        before = self.ledger.put_evidence(json.dumps({'synthetic_before': trial_id, 'round': round_id}).encode())
        after = self.ledger.put_evidence(json.dumps({'synthetic_after': trial_id, 'round': round_id}).encode())
        self.ledger.start_trial(round_id, trial_id, variant=variant, case_id=case,
                                repetition=repetition, condition_id=condition or f'{case}-{repetition}',
                                before_evidence=[before])
        if observed is None and status == 'observed':
            observed = self.plan.cases[case] if variant == 'candidate' else '?'
        kwargs = dict(status=status, observed_text=observed, after_evidence=[after],
                      execution_id=execution or f'{round_id}-{trial_id}', usage=usage or Usage.zero())
        self.ledger.finish_trial(round_id, trial_id, **kwargs)
        return trial_id, kwargs

    def complete(self, *, round_id='software-round', result_override=None, usage=None):
        for case in self.plan.cases:
            for repetition in range(self.plan.repetitions):
                for variant in ['parent', 'candidate']:
                    override = (result_override or {}).get((variant, case, repetition), {})
                    self.trial(variant, case, repetition, round_id=round_id, usage=usage, **override)

    def test_frozen_files_are_bytes_not_executed(self):
        sentinel = Path(self.temp.name) / 'must-not-exist'
        self.ledger.freeze_bundle(label='not-executed', kind='skill_program', scope=self.scope,
                                  files={'skill.py': f'open({str(sentinel)!r},"w").write("bad")'.encode()},
                                  parent=self.seed, provenance={'origin': 'original'})
        self.assertFalse(sentinel.exists())

    def test_complete_comparison_promotes_and_replays(self):
        self.complete()
        result = self.ledger.decide('software-round')
        self.assertEqual(result['verdict'], 'accepted')
        self.assertEqual(result['correct'], {'parent': 0, 'candidate': 4})
        with Ledger(self.root) as reopened:
            self.assertEqual(reopened.verify()['stable'][self.scope.key], self.candidate)

    def test_bootstrap_explicit_and_idempotent(self):
        before = self.ledger.verify()['event_count']
        self.ledger.bootstrap(self.seed, reason='Synthetic fixture seed, not a learned skill')
        self.assertEqual(self.ledger.verify()['event_count'], before)
        with self.assertRaises(LedgerError):
            self.ledger.bootstrap(self.candidate, reason='Cannot fake validation')

    def test_duplicate_finish_is_idempotent_and_conflict_rejected(self):
        trial_id, kwargs = self.trial('candidate', usage=Usage(10, 6, 2, 1))
        before = self.ledger.verify()
        self.ledger.finish_trial('software-round', trial_id, **kwargs)
        after = self.ledger.verify()
        self.assertEqual(before, after)
        with self.assertRaises(LedgerError):
            self.ledger.finish_trial('software-round', trial_id, **{**kwargs, 'observed_text': 'different'})

    def test_duplicate_start_is_idempotent_and_slot_cannot_hide_failure(self):
        ref = self.ledger.put_evidence(b'synthetic initial')
        kwargs = dict(variant='parent', case_id='a', repetition=0, condition_id='a-0', before_evidence=[ref])
        self.ledger.start_trial('software-round', 'one', **kwargs)
        count = self.ledger.verify()['event_count']
        self.ledger.start_trial('software-round', 'one', **kwargs)
        self.assertEqual(self.ledger.verify()['event_count'], count)
        with self.assertRaises(LedgerError):
            self.ledger.start_trial('software-round', 'hidden-retry', **kwargs)

    def test_missing_trial_is_inconclusive_not_best_subset(self):
        self.trial('candidate')
        result = self.ledger.decide('software-round')
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertIn('incomplete_schedule', result['reasons'])
        self.assertEqual(self.ledger.verify()['stable'][self.scope.key], self.seed)

    def test_unfinished_trial_survives_reopen_unknown_cost_and_late_receipt(self):
        ref = self.ledger.put_evidence(b'initial pending fixture')
        self.ledger.start_trial('software-round', 'pending', variant='candidate', case_id='a',
                                repetition=0, condition_id='a-0', before_evidence=[ref])
        with Ledger(self.root) as reopened:
            report = reopened.verify()['rounds']['software-round']['comparison']
            self.assertFalse(report['usage']['complete'])
            self.assertIsNone(report['usage']['total_input_plus_output'])
            self.assertFalse(reopened.verify()['overall_usage']['complete'])
            self.assertIsNone(reopened.verify()['overall_usage']['total_input_plus_output'])
            with self.assertRaises(LedgerError):
                reopened.decide('software-round')
            reopened.finish_trial('software-round', 'pending', status='aborted', observed_text=None,
                                   after_evidence=[ref], execution_id='crash-receipt',
                                   usage=Usage(None, None, None, None))
            result = reopened.decide('software-round')
            self.assertEqual(result['verdict'], 'inconclusive')
            self.assertIn('unknown_usage', result['reasons'])

    def test_uncertain_blocked_aborted_never_promote(self):
        for status in ['uncertain', 'blocked', 'aborted']:
            with self.subTest(status=status):
                # Separate candidates/rounds ensure all evidence is retained, not overwritten.
                candidate = self.bundle('candidate-' + status, parent=self.seed)
                name = 'round-' + status
                self.open_round(name, candidate)
                self.complete(round_id=name, result_override={('candidate', 'a', 0): {'status': status}})
                self.assertEqual(self.ledger.decide(name)['verdict'], 'inconclusive')

    def test_observed_text_drives_score_no_caller_success_flag(self):
        self.complete(result_override={('candidate', 'a', 0): {'observed': 'wrong'}})
        self.assertEqual(self.ledger.decide('software-round')['correct']['candidate'], 3)

    def test_per_case_regression_rejected_even_with_total_gain(self):
        self.complete(result_override={('parent', 'a', 0): {'observed': 'cat'},
                                       ('candidate', 'a', 0): {'observed': '?'},
                                       ('candidate', 'a', 1): {'observed': '?'}})
        result = self.ledger.decide('software-round')
        self.assertEqual(result['correct'], {'parent': 1, 'candidate': 2})
        self.assertEqual(result['verdict'], 'rejected')

    def test_condition_mismatch_is_inconclusive(self):
        self.complete(result_override={('candidate', 'b', 0): {'condition': 'different-initial-state'}})
        self.assertIn('initial_condition_mismatch', self.ledger.decide('software-round')['reasons'])

    def test_unknown_cost_is_not_zero(self):
        self.complete(usage=Usage(None, None, None, None))
        result = self.ledger.decide('software-round')
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertIsNone(result['usage']['total_input_plus_output'])

    def test_token_budget_rejects_promotion(self):
        self.complete(usage=Usage(200, 100, 20, 5))
        self.assertIn('token_budget_exceeded', self.ledger.decide('software-round')['reasons'])

    def test_token_subsets_not_double_counted(self):
        self.complete(usage=Usage(10, 6, 2, 1))
        usage = self.ledger.decide('software-round')['usage']
        self.assertEqual(usage['total_input_plus_output'], 96)
        self.assertEqual(usage['cached_input_tokens'], 48)
        self.assertEqual(usage['reasoning_output_tokens'], 8)

    def test_overall_cost_includes_rejected_round_and_generation(self):
        other = self.bundle('rejected-cost', parent=self.seed)
        self.open_round('rejected-cost-round', other, usage=Usage(5, 2, 3, 1))
        overrides = {('candidate', c, r): {'observed': '?'}
                     for c in self.plan.cases for r in range(self.plan.repetitions)}
        self.complete(round_id='rejected-cost-round', usage=Usage(10, 6, 2, 1), result_override=overrides)
        self.assertEqual(self.ledger.decide('rejected-cost-round')['verdict'], 'rejected')
        self.complete(usage=Usage(1, 0, 1, 1))
        self.assertEqual(self.ledger.decide('software-round')['verdict'], 'accepted')
        usage = self.ledger.verify()['overall_usage']
        self.assertTrue(usage['complete'])
        self.assertEqual(usage['total_input_plus_output'], 120)
        self.assertEqual(usage['cached_input_tokens'], 50)
        self.assertEqual(usage['reasoning_output_tokens'], 17)

    def test_bad_usage_values_rejected(self):
        for values in [(-1, 0, 0, 0), (1, 2, 0, 0), (0, 0, 1, 2),
                       (float('nan'), 0, 0, 0), (True, 0, 0, 0), (None, 1, 0, 0)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Usage(*values)

    def test_unknown_generation_cost_also_blocks(self):
        c = self.bundle('generation-unknown', parent=self.seed)
        self.open_round('unknown-gen', c, usage=Usage(None, None, None, None))
        self.complete(round_id='unknown-gen')
        self.assertIn('unknown_usage', self.ledger.decide('unknown-gen')['reasons'])

    def test_evidence_generator_validated_once_and_empty_rejected(self):
        for value in [[], iter([])]:
            with self.assertRaises(LedgerError):
                self.ledger.start_trial('software-round', 'empty', variant='parent', case_id='a',
                                        repetition=0, condition_id='a-0', before_evidence=value)
        ref = self.ledger.put_evidence(b'valid evidence')
        self.ledger.start_trial('software-round', 'valid', variant='parent', case_id='a',
                                repetition=0, condition_id='a-0', before_evidence=iter([ref]))
        with self.assertRaises(LedgerError):
            self.ledger.finish_trial('software-round', 'valid', status='observed', observed_text='cat',
                                     after_evidence=iter([]), execution_id='empty-after', usage=Usage.zero())
        self.ledger.finish_trial('software-round', 'valid', status='observed', observed_text='cat',
                                 after_evidence=iter([ref]), execution_id='valid-after', usage=Usage.zero())

    def test_missing_or_corrupted_evidence_fails_closed(self):
        ref = self.ledger.put_evidence(b'will be corrupted')
        self.ledger.start_trial('software-round', 'one', variant='parent', case_id='a',
                                repetition=0, condition_id='a-0', before_evidence=[ref])
        (self.root / 'objects' / ref).write_bytes(b'corrupted')
        with self.assertRaises(IntegrityError):
            self.ledger.verify()
        with self.assertRaises(IntegrityError):
            self.ledger.decide('software-round')

    def test_corrupted_frozen_source_rejected(self):
        m = json.loads(self.ledger.read_object(self.candidate))
        (self.root / 'objects' / m['files']['skill.py']['sha256']).write_bytes(b'changed after freeze')
        with self.assertRaises(IntegrityError):
            self.ledger.verify()

    def test_event_update_delete_protected_and_chain_detects_forced_tamper(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.db.execute('UPDATE events SET payload=? WHERE seq=1', ('{}',))
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.db.execute('DELETE FROM events WHERE seq=1')
        # Trusted-host corruption simulation, not proof against malicious equal privilege.
        self.ledger.db.execute('DROP TRIGGER events_no_update')
        self.ledger.db.execute('UPDATE events SET payload=? WHERE seq=1', ('{}',))
        with self.assertRaises(IntegrityError):
            self.ledger.verify()

    def test_failed_transaction_does_not_leave_half_event(self):
        count = self.ledger.verify()['event_count']
        original = self.ledger._append
        def fail_after_insert(*args):
            original(*args)
            raise RuntimeError('simulated failure before COMMIT')
        self.ledger._append = fail_after_insert
        ref = self.ledger.put_evidence(b'transaction evidence')
        with self.assertRaises(RuntimeError):
            self.ledger.start_trial('software-round', 'not-committed', variant='parent', case_id='a',
                                    repetition=0, condition_id='a-0', before_evidence=[ref])
        self.ledger._append = original
        self.assertEqual(self.ledger.verify()['event_count'], count)

    def test_process_death_inside_sqlite_transaction_rolls_back(self):
        count = self.ledger.verify()['event_count']
        code = """import os,sqlite3,sys
c=sqlite3.connect(sys.argv[1],isolation_level=None)
c.execute('BEGIN IMMEDIATE')
c.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',(999,'uncommitted','bad','{}','now','x','x'))
os._exit(17)
"""
        result = subprocess.run([sys.executable, '-c', code, str(self.root / 'events.sqlite3')], timeout=10)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.ledger.verify()['event_count'], count)

    def test_scope_change_cannot_inherit_comparison(self):
        for field in ['layout_id', 'configuration_sha256', 'protocol_sha256', 'evaluator_sha256', 'goal_sha256', 'safety_sha256']:
            value = 'different-layout' if field == 'layout_id' else digest(field.encode())
            with self.subTest(field=field), self.assertRaises(LedgerError):
                self.bundle('changed-' + field, parent=self.seed, scope=replace(self.scope, **{field: value}))

    def test_synthetic_stable_cannot_become_physical(self):
        with self.assertRaises(LedgerError):
            self.open_round('physical-mismatch', mode='physical')
        with self.assertRaises(LedgerError):
            self.bundle('physical-bundle', parent=self.seed, scope=replace(self.scope, evidence_mode='physical'))

    def test_learned_profile_can_change_without_changing_physical_configuration(self):
        m = json.loads(self.ledger.read_object(self.candidate))
        seed = json.loads(self.ledger.read_object(self.seed))
        self.assertEqual(m['scope'], seed['scope'])
        self.assertNotEqual(m['files']['profile.json'], seed['files']['profile.json'])

    def test_mutated_plan_revalidated_when_freezing(self):
        plan = EvaluationPlan({'x': 'valid'}, 1, 1, 0)
        plan.cases['x'] = ''
        with self.assertRaises(ValueError):
            self.open_round('mutated-plan', plan=plan)

    def test_same_candidate_cannot_be_repeated_until_lucky(self):
        with self.assertRaises(LedgerError):
            self.open_round('same-candidate-again')

    def test_reused_execution_receipt_rejected(self):
        self.trial('parent', execution='same-execution')
        with self.assertRaises(LedgerError):
            self.trial('candidate', execution='same-execution')

    def test_rollback_is_append_only_and_target_must_have_been_stable(self):
        self.complete()
        self.ledger.decide('software-round')
        count = self.ledger.verify()['event_count']
        self.ledger.rollback(scope=self.scope, target=self.seed, expected_current=self.candidate,
                              reason='Synthetic rollback test', event_id='rollback-1')
        self.assertEqual(self.ledger.verify()['event_count'], count + 1)
        self.assertEqual(self.ledger.verify()['stable'][self.scope.key], self.seed)
        self.assertEqual(self.ledger.verify()['rounds']['software-round']['decision']['verdict'], 'accepted')
        with self.assertRaises(LedgerError):
            self.ledger.rollback(scope=self.scope, target=self.bundle('never-stable', parent=self.seed),
                                  expected_current=self.seed, reason='invalid', event_id='invalid-rollback')

    def test_two_connections_concurrently_promote_only_one_parent(self):
        other = self.bundle('other', parent=self.seed)
        self.open_round('other-round', other)
        self.complete()
        self.complete(round_id='other-round')
        barrier = threading.Barrier(2)
        results = []
        def worker(name):
            try:
                with Ledger(self.root) as ledger:
                    barrier.wait(timeout=5)
                    results.append(ledger.decide(name)['verdict'])
            except StaleParentError:
                results.append('stale_parent')
            except Exception as exc:
                results.append(repr(exc))
        threads = [threading.Thread(target=worker, args=(n,)) for n in ['software-round', 'other-round']]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertCountEqual(results, ['accepted', 'stale_parent'])

    def test_unsafe_bundle_paths_and_symlinks_rejected(self):
        for name in ['../escape', '/absolute', 'a/../b', 'a//b', '.', 'a\\b']:
            with self.subTest(name=name), self.assertRaises(LedgerError):
                self.ledger.freeze_bundle(label='bad-path', kind='seed', scope=self.scope,
                                          files={name: b'bad'}, provenance={'origin': 'original'})
        ref = self.ledger.put_evidence(b'symlink target')
        path = self.root / 'objects' / ref
        path.unlink()
        path.symlink_to(Path(self.temp.name) / 'outside')
        with self.assertRaises(IntegrityError):
            self.ledger.read_object(ref)

    def test_export_preserves_pending_and_explicit_synthetic_label(self):
        export = Path(self.temp.name) / 'export'
        report = self.ledger.export(export)
        self.assertIn('synthetic', (export / 'SUMMARY.md').read_text())
        self.assertIsNone(report['rounds']['software-round']['decision'])
        self.assertEqual(report['event_count'], len((export / 'events.jsonl').read_text().splitlines()))


if __name__ == '__main__':
    unittest.main()
