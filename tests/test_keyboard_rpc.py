"""Synthetic RPC fault-injection checks, never physical robot tests."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from astra_robotics_agent.physical_keyboard import Ledger, Scope, EvaluationPlan, Usage
from astra_robotics_agent.physical_keyboard.contracts import canonical, digest
from astra_robotics_agent.physical_keyboard.rpc import SyntheticServer, RpcClient, default_safety


class RpcTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / 'ledger')
        self.addCleanup(self.ledger.close)
        self.safety = default_safety()
        self.scope = Scope('synthetic', 'fake-device', 'layout-a',
                           *[digest(x.encode()) for x in ('config','rpc','evaluator','goal')],
                           digest(canonical(self.safety)))
        self.profile = canonical({'synthetic': True, 'learned': False})
        self.program = canonical({'protocol': 'physical-keyboard-action-plan-v1', 'actions': []})
        self.profile_id = digest(self.profile)
        files = {'profile.json': self.profile, 'actions.json': self.program}
        self.seed = self.ledger.freeze_bundle(label='seed', kind='seed', scope=self.scope,
                                              files=files, provenance={'origin':'original'})
        self.ledger.bootstrap(self.seed, reason='synthetic fixture')
        self.candidate = self.ledger.freeze_bundle(label='candidate', kind='skill_program',
            scope=self.scope, files=files, provenance={'origin':'original'}, parent=self.seed)
        self.ledger.open_round('round', candidate=self.candidate,
            plan=EvaluationPlan({'case':'cat'},2,1,0), generation_usage=Usage.zero(),
            metadata={'evidence_mode':'synthetic'})
        self.before = self.ledger.put_evidence(b'synthetic-before')
        self.start()
        self.config = dict(protocol='physical-keyboard-rpc-config-v1', backend='synthetic-fake',
            scope=asdict(self.scope), safety=self.safety, allowed_profiles=[self.profile_id],
            ledger_root=str(self.ledger.root),
            initial_pose={'xyz_m':[0,0,.2], 'quaternion_xyzw':[0,0,0,1]})
        self.server = None

    def start(self, trial='trial', variant='parent', repetition=0):
        self.ledger.start_trial('round',trial,variant=variant,case_id='case',repetition=repetition,
                                condition_id='condition',before_evidence=[self.before])

    def grant_args(self, trial='trial', variant='parent'):
        state = self.ledger.trial_state('round',trial)
        context = self.ledger.execution_context('round',variant)
        return dict(trial_id=trial,round_id='round',variant=variant,
                    bundle_id=context['bundle_id'],profile_id=self.profile_id,
                    scope_key=self.scope.key,start_event_sha256=state['start_event_sha256'],
                    program_sha256=digest(self.program))

    def local(self):
        self.server = SyntheticServer(self.config,self.root/'journal')
        self.addCleanup(self.server.close)
        return self.server

    def envelope(self, method, arguments, rpc_id=None):
        self.counter = getattr(self,'counter',0)+1
        return dict(rpc_id=rpc_id or 'rpc'+str(self.counter),method=method,arguments=arguments)

    def call(self, method, arguments, rpc_id=None):
        return self.server.rpc(self.envelope(method,arguments,rpc_id))['result']

    def grant(self):
        self.permit = self.call('authorize_trial',self.grant_args())['permit_id']

    def move(self, request_id='move', xyz=None, obs=None, **overrides):
        if obs is None:
            obs = self.call('observe',{})['observation']['observation_id']
        return dict(request_id=request_id,permit_id=self.permit,scope_key=self.scope.key,
                    profile_id=self.profile_id,observation_id=obs,xyz_m=xyz or [.01,0,.2],
                    quaternion_xyzw=[0,0,0,1],**overrides)

    def test_read_queries_verify_start_anchor_and_detached_bundle(self):
        context = self.ledger.execution_context('round','parent')
        self.assertEqual(context['bundle_id'],self.seed)
        self.assertEqual(len(context['inflight_trials']),1)
        state = self.ledger.trial_state('round','trial')
        self.assertEqual(len(state['start_event_sha256']),64)
        context['bundle']['files'].clear()
        self.assertIn('actions.json',self.ledger.execution_context('round','parent')['bundle']['files'])
        self.assertIsNone(self.ledger.trial_state('round','missing'))

    def test_real_start_is_required_not_client_claim(self):
        self.local()
        args=self.grant_args()
        for changes in ({'trial_id':'invented'}, {'start_event_sha256':'0'*64},
                        {'variant':'candidate'}, {'bundle_id':self.candidate},
                        {'profile_id':'0'*64}, {'program_sha256':'0'*64}, {'scope_key':'0'*64}):
            result=self.call('authorize_trial',{**args,**changes})
            self.assertEqual(result['status'],'rejected',changes)
        self.assertEqual(self.server.move_calls,0)
        self.grant()

    def test_motion_does_not_claim_text_success(self):
        self.local(); self.grant()
        result=self.call('move_tcp',self.move())
        self.assertEqual(result['status'],'ok')
        self.assertFalse(result['text_verified'])
        self.assertEqual(result['start_event_sha256'],self.grant_args()['start_event_sha256'])
        self.assertNotIn('observed_text',result)
        self.assertEqual(self.server.move_calls,1)

    def test_same_action_returns_original_before_freshness_or_lease_checks(self):
        self.local(); self.grant()
        args=self.move()
        first=self.call('move_tcp',args)
        self.server.permit['expires_at']=0
        self.assertEqual(self.call('move_tcp',args),first)
        self.assertEqual(self.server.move_calls,1)
        self.assertEqual(self.call('move_tcp',{**args,'xyz_m':[.02,0,.2]})['reason'],'conflicting_request')
        self.assertEqual(self.server.move_calls,1)

    def test_rpc_replay_preserves_observation_not_new_timestamp(self):
        self.local(); self.grant()
        old=self.call('observe',{},'same-observe')
        self.call('move_tcp',self.move(obs=old['observation']['observation_id']))
        again=self.call('observe',{},'same-observe')
        self.assertEqual(again,old)
        self.assertEqual(self.call('observe',{'ignored':True},'same-observe')['reason'],'conflicting_rpc_id')

    def test_observation_consumed_by_motion_and_new_observation_required(self):
        self.local(); self.grant()
        obs=self.call('observe',{})['observation']['observation_id']
        self.assertEqual(self.call('move_tcp',self.move(obs=obs))['status'],'ok')
        self.assertEqual(self.call('move_tcp',self.move('second',obs=obs))['reason'],'stale_observation')
        self.assertEqual(self.server.move_calls,1)

    def test_observation_source_monotonic_age_not_client_time(self):
        self.local(); self.grant()
        args=self.move()
        self.server.observation['source_monotonic_s']-=10
        self.assertEqual(self.call('move_tcp',args)['reason'],'expired_observation')
        forged=self.move('forged'); forged['source_monotonic_s']=time.monotonic()
        self.assertEqual(self.call('move_tcp',forged)['reason'],'action_schema')
        self.assertEqual(self.server.move_calls,0)

    def test_lease_expiry_stop_still_works(self):
        self.local(); self.grant()
        args=self.move()
        self.server.permit['expires_at']=0
        self.assertEqual(self.call('move_tcp',args)['reason'],'lease_expired')
        self.assertEqual(self.call('stop',{'request_id':'stop'})['status'],'ok')
        self.assertEqual(self.call('authorize_trial',self.grant_args())['status'],'rejected')

    def test_wait_must_fit_remaining_lease_and_limit(self):
        self.local(); self.grant()
        args=dict(request_id='wait',permit_id=self.permit,scope_key=self.scope.key,
                  profile_id=self.profile_id,seconds=.1)
        self.server.permit['expires_at']=time.monotonic()+.03
        self.assertEqual(self.call('wait_bounded',args)['reason'],'wait_exceeds_lease')
        self.server.permit['expires_at']=time.monotonic()+5
        result=self.call('wait_bounded',{**args,'request_id':'wait2','seconds':.3})
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(self.call('wait_bounded',{**args,'request_id':'wait3','seconds':0})['status'],'ok')

    def test_finite_bounds_unit_quaternion_scope_and_profile(self):
        self.local(); self.grant()
        base=self.move()
        cases=[('bool',{'xyz_m':[True,0,.2]}),('inf',{'xyz_m':[math.inf,0,.2]}),
            ('nan',{'xyz_m':[math.nan,0,.2]}),('workspace',{'xyz_m':[.4,0,.2]}),
            ('translation',{'xyz_m':[.1,0,.2]}),('unit',{'quaternion_xyzw':[0,0,0,2]}),
            ('rotation',{'quaternion_xyzw':[0,0,1,0]}),('scope',{'scope_key':'0'*64}),
            ('profile',{'profile_id':'0'*64})]
        for key,overrides in cases:
            args={**base,'request_id':key,**overrides}
            if key in ('inf','nan'):
                # strict transport rejects these before journal serialization
                with self.assertRaises(ValueError): self.call('move_tcp',args)
            else:
                self.assertEqual(self.call('move_tcp',args)['status'],'rejected',key)
        self.assertEqual(self.server.move_calls,0)
        self.assertEqual(self.call('move_tcp',{**base,'request_id':'negative-q',
                        'quaternion_xyzw':[0,0,0,-1]})['status'],'ok')

    def test_finished_trial_cannot_use_old_permit(self):
        self.local(); self.grant()
        self.ledger.finish_trial('round','trial',status='uncertain',observed_text=None,
            after_evidence=[self.before],execution_id='finished',usage=Usage.zero())
        self.assertEqual(self.call('move_tcp',self.move())['reason'],'unfinished_durable_start_required')
        self.assertEqual(self.server.move_calls,0)

    def test_permit_action_count_cannot_be_increased_by_worker(self):
        self.local(); self.grant()
        self.server.permit['actions']=self.safety['max_actions']
        self.assertEqual(self.call('move_tcp',self.move())['reason'],'action_budget')

    def test_only_one_journal_owner_and_config_cannot_change(self):
        server=self.local()
        with self.assertRaises(ValueError): SyntheticServer(self.config,self.root/'journal')
        # A different profile/config cannot silently attach to old durable history.
        with self.assertRaises(ValueError):
            SyntheticServer({**self.config,'backend':'real-sdk'},self.root/'another')
        with self.assertRaises(ValueError):
            SyntheticServer({**self.config,'scope':asdict(replace(self.scope,evidence_mode='physical'))},self.root/'another')

    def test_sqlite_journal_is_append_only_and_full(self):
        self.local(); self.grant(); self.call('move_tcp',self.move())
        self.assertEqual(self.server.journal.db.execute('PRAGMA synchronous').fetchone()[0],2)
        with self.assertRaises(sqlite3.IntegrityError):
            self.server.journal.db.execute('DELETE FROM events')
        kinds=[e['kind'] for e in self.server.journal.events]
        self.assertLess(kinds.index('action_reserved'),kinds.index('action_dispatching'))
        self.assertLess(kinds.index('action_dispatching'),kinds.index('action_receipt'))

    def client(self):
        config=self.root/'config.json';config.write_bytes(canonical(self.config))
        client=RpcClient(config,self.root/'remote-journal',self.root/'rpc-logs')
        self.addCleanup(client.close)
        return client

    def test_actual_stdio_process_and_reaped_exit(self):
        client=self.client()
        obs=client.call('observe',{})
        self.assertEqual(obs['observation']['evidence_mode'],'synthetic')
        grant=client.call('authorize_trial',self.grant_args())
        self.permit=grant['permit_id']
        args=dict(request_id='process-move',permit_id=self.permit,scope_key=self.scope.key,
            profile_id=self.profile_id,observation_id=obs['observation']['observation_id'],
            xyz_m=[.01,0,.2],quaternion_xyzw=[0,0,0,1])
        first=client.call('move_tcp',args)
        self.assertEqual(first['status'],'ok')
        self.assertEqual(client.call('move_tcp',args),first)
        exit_record=client.close()
        self.assertEqual(exit_record['pid'],client.pid)
        self.assertTrue(exit_record['waited'])
        self.assertIsNotNone(exit_record['returncode'])

    def test_dispatch_crash_survives_restart_and_cannot_be_bypassed(self):
        for crash_at in ('after_dispatch','after_adapter'):
            with self.subTest(crash_at=crash_at):
                # Independent journals keep both fault attempts, no hidden retries.
                self.config['fault']={'crash_at':crash_at,'request_id':'crash'}
                config=self.root/(crash_at+'.json');config.write_bytes(canonical(self.config))
                journal=self.root/('journal-'+crash_at)
                client=RpcClient(config,journal,self.root/('log-'+crash_at))
                try:
                    obs=client.call('observe',{})['observation']['observation_id']
                    permit=client.call('authorize_trial',self.grant_args())['permit_id']
                    args=dict(request_id='crash',permit_id=permit,scope_key=self.scope.key,
                        profile_id=self.profile_id,observation_id=obs,xyz_m=[.01,0,.2],
                        quaternion_xyzw=[0,0,0,1])
                    with self.assertRaises((EOFError,RuntimeError,BrokenPipeError)):
                        client.call('move_tcp',args)
                finally: exit_record=client.close()
                self.assertIn(exit_record['returncode'],(81,82))
                reopened=RpcClient(config,journal,self.root/('log-reopen-'+crash_at))
                try:
                    self.assertEqual(reopened.call('move_tcp',args)['status'],'unknown')
                    self.assertEqual(reopened.call('move_tcp',{**args,'request_id':'new-id'})['status'],'rejected')
                    self.assertEqual(reopened.call('stop',{'request_id':'stop'})['status'],'ok')
                    self.assertEqual(reopened.call('authorize_trial',self.grant_args())['status'],'rejected')
                    obs=reopened.call('observe',{})['observation']
                    self.assertEqual(obs['pending_requests'],['crash'])
                    self.assertEqual(obs['move_calls'],0)  # new fake adapter is not continuity evidence
                finally: reopened.close()
                db=sqlite3.connect(journal/'actions.sqlite3')
                self.assertEqual(db.execute("SELECT count(*) FROM events WHERE kind='action_dispatching' AND key='crash'").fetchone()[0],1)
                db.close()

    def test_rpc_duplicate_observation_after_process_restart_is_historical(self):
        client=self.client()
        old=client.call('observe',{},rpc_id='historic')
        client.close()
        client=self.client()
        self.assertEqual(client.call('observe',{},rpc_id='historic'),old)
        new=client.call('observe',{})
        self.assertNotEqual(old['observation']['session'],new['observation']['session'])

    def test_huge_integer_is_rejected_with_receipt_without_server_exit(self):
        client=self.client()
        obs=client.call('observe',{})['observation']['observation_id']
        permit=client.call('authorize_trial',self.grant_args())['permit_id']
        args=dict(request_id='huge',permit_id=permit,scope_key=self.scope.key,
            profile_id=self.profile_id,observation_id=obs,xyz_m=[10**1000,0,.2],
            quaternion_xyzw=[0,0,0,1])
        result=client.call('move_tcp',args)
        self.assertEqual(result['status'],'rejected')
        self.assertIn('finite float range',result['reason'])
        self.assertEqual(client.call('observe',{})['observation']['move_calls'],0)

    def test_stop_persists_across_restart_and_retains_trial_anchor(self):
        client=self.client()
        client.call('authorize_trial',self.grant_args())
        stopped=client.call('stop',{'request_id':'stop'})
        self.assertEqual(stopped['start_event_sha256'],self.grant_args()['start_event_sha256'])
        client.close()
        reopened=self.client()
        self.assertTrue(reopened.call('observe',{})['observation']['stopped'])
        self.assertEqual(reopened.call('authorize_trial',self.grant_args())['status'],'rejected')

    def test_journal_rejects_config_change_after_prior_session(self):
        client=self.client();client.call('observe',{});client.close()
        self.config['initial_pose']['xyz_m'][0]=.01
        config=self.root/'config.json';config.write_bytes(canonical(self.config))
        reopened=RpcClient(config,self.root/'remote-journal',self.root/'changed-log')
        try:
            with self.assertRaises((EOFError,RuntimeError)):
                reopened.call('observe',{})
        finally: self.assertNotEqual(reopened.close()['returncode'],0)


if __name__ == '__main__': unittest.main()
