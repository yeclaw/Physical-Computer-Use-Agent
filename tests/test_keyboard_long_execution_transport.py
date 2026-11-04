"""Real local subprocess protocol regressions; no robot, model, or network."""
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import unittest

from astra_robotics_agent.physical_keyboard.runtime.remote_host import RemoteHost

SERVER = r'''
import json,sys,time
from pathlib import Path
from astra_robotics_agent.physical_keyboard.runtime.request_lifecycle import descriptor,PROTOCOL

def emit(value):
 print('@@PCU@@'+json.dumps(value),flush=True)
emit({'type':'ready','request_lifecycle':descriptor()})
for line in sys.stdin:
 request=json.loads(line)
 with Path(sys.argv[1]).open('a') as f:f.write(json.dumps(request)+'\n')
 mode=sys.argv[2]
 if request['method']=='run_skill':
  if mode=='late':
   time.sleep(.19)
  else:
   emit(dict(type='request_update',protocol=PROTOCOL,id=request['id'],sequence=1,
             phase='accepted',progress_seq=0,progress_kind=None,progress_anchor=None))
   for i in range(24):
    time.sleep(.015)
    emit(dict(type='request_update',protocol=PROTOCOL,
       id='wrong-id' if mode=='wrong-id' else request['id'],
       sequence=1 if mode=='duplicate' else i+2,phase='running',
       progress_seq=0 if mode=='stalled' else i+1,
       progress_kind='trajectory_point_dispatched',
       progress_anchor={'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns()}))
  emit({'id':request['id'],'status':'ok','result':{'done':True,'blob':'x'*100000}})
 else:
  emit({'id':request['id'],'status':'ok','result':{}})
 if request['method']=='close':break
'''


class LongExecutionTransportTests(unittest.TestCase):
    def host(self, root, mode, **kwargs):
        return RemoteHost([sys.executable, '-u', '-c', SERVER, str(root/'requests.jsonl'), mode],
                          root, timeout=2, **kwargs)

    @staticmethod
    def requests(root):
        return [json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]

    def test_total_duration_exceeds_wait_window_with_real_progress_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);host=self.host(root,'healthy');host.timeout=.12
            start=time.monotonic()
            try:
                response=host.request('run_skill',{'label':'once'})
                self.assertTrue(response['result']['done'])
                self.assertEqual(len(response['result']['blob']),100000)
                self.assertGreater(time.monotonic()-start,host.timeout*2)
            finally:closed=host.close()
            self.assertEqual(closed['returncode'],0)
            self.assertEqual([x['method'] for x in self.requests(root)],['run_skill','close'])
            for line in (root/'transport.jsonl').read_text().splitlines():json.loads(line)

    def test_heartbeat_without_real_progress_does_not_renew_progress_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);host=self.host(root,'stalled');host.timeout=.10
            start=time.monotonic()
            try:
                with self.assertRaises(queue.Empty):host.request('run_skill',{})
                self.assertLess(time.monotonic()-start,.28)
            finally:closed=host.close()
            self.assertIn('unresolved_request_id',closed)
            self.assertEqual([x['method'] for x in self.requests(root)],['run_skill'])

    def test_late_terminal_is_reconciled_before_separate_close_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);host=self.host(root,'late');host.timeout=.13
            with self.assertRaises(queue.Empty):host.request('run_skill',{})
            closed=host.close()
            self.assertEqual(closed['late_response']['id'],'actor-000001')
            self.assertEqual(closed['response']['id'],'actor-000002')
            self.assertNotIn('error',closed)
            self.assertEqual(host.close(),closed)
            self.assertEqual([x['method'] for x in self.requests(root)],['run_skill','close'])

    def test_unresolved_request_forbids_new_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);host=self.host(root,'late');host.timeout=.13
            try:
                with self.assertRaises(queue.Empty):host.request('run_skill',{})
                with self.assertRaisesRegex(RuntimeError,'unresolved'):host.request('run_skill',{})
            finally:host.close()
            self.assertEqual(sum(x['method']=='run_skill' for x in self.requests(root)),1)

    def test_wrong_id_and_duplicate_sequence_are_not_accepted_as_progress(self):
        for mode in ('wrong-id','duplicate'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);host=self.host(root,mode)
                try:
                    with self.assertRaises(RuntimeError):host.request('run_skill',{})
                finally:host.close()
                self.assertEqual(sum(x['method']=='run_skill' for x in self.requests(root)),1)

    def test_stop_callback_interrupts_wait_without_new_request(self):
        class Stopped(Exception):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);start=[None]
            def check():
                if start[0] is not None and time.monotonic()-start[0]>.06:raise Stopped()
            host=self.host(root,'healthy',wait_check=check);start[0]=time.monotonic()
            try:
                with self.assertRaises(Stopped):host.request('run_skill',{})
                self.assertLess(time.monotonic()-start[0],.25)
            finally:closed=host.close()
            self.assertIn('unresolved_request_id',closed)
            self.assertEqual([x['method'] for x in self.requests(root)],['run_skill'])

if __name__=='__main__':unittest.main()

# This fixture runs the REAL serve/Audit/input-cancellation path in a child,
# with an inert fake execution owner instead of any SDK or robot adapter.
REAL_SERVE = r'''
import json,sys,os,signal,time,threading
from pathlib import Path
from astra_robotics_agent.physical_keyboard.runtime import request_lifecycle as lifecycle
from astra_robotics_agent.physical_keyboard.runtime.startouch_control import serve,Audit
original=lifecycle.RequestLifecycle
lifecycle.RequestLifecycle=lambda key,emit:original(key,emit,interval_s=.02)
lock=threading.RLock()
def emit(value):
 with lock:print('@@PCU@@'+json.dumps(value),flush=True)
class Cancelled(BaseException):pass
def cancelled(*_):raise Cancelled()
signal.signal(signal.SIGTERM,cancelled)
class Host:
 closed=False
 audit=Audit(Path(sys.argv[1]).parent/'audit.jsonl','inert')
 def handle(self,request):
  with Path(sys.argv[1]).open('a') as f:f.write(json.dumps(request)+'\n')
  if request['method']=='run_skill':
   for i in range(35):
    self.audit('trajectory_point_dispatched',index=i)
    time.sleep(.012)
  else:self.close()
  return {'id':request['id'],'status':'ok','result':{'done':True}}
 def close(self):
  if not self.closed:
   self.closed=True
   Path(sys.argv[1]+'.closed').write_text('stopped by sole execution thread')
   self.audit('host_closed')
  return {}
host=Host();emit({'type':'ready','request_lifecycle':lifecycle.descriptor()})
try:serve(host,sys.stdin.buffer,emit,cancel=lambda:os.kill(os.getpid(),signal.SIGTERM))
except Cancelled:pass
finally:host.close();host.audit.close()
'''



