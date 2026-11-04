"""Public assembly seam must close resources and preserve primary failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.agent.runner import AgentRunner, run_config
from astra_robotics_agent.physical_keyboard.runtime.device_adapter import CallReceipt, Observation
from astra_robotics_agent.physical_keyboard.runtime.mock_device import MockDevice
from astra_robotics_agent.physical_keyboard.runtime.program_runtime import ProgramRuntime


SCOPE = dict(task_id='lifecycle', device_id='mock_keyboard', layout_id='generic')


class Session:
    def __init__(self):
        self.closed = False

    def run_turn(self, prompt, callback, images):
        raise RuntimeError('provider turn failed')

    def close(self):
        self.closed = True


class RunnerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def runner(self, session=None, device=None):
        session = session or Session()
        device = device or MockDevice()
        value = AgentRunner(session=session, device=device, knowledge=self.root/'knowledge',
                            scope=SCOPE, output=self.root/'run', task='synthetic lifecycle fixture')
        return value, session, device

    def test_invalid_loop_options_still_close_the_owned_device_and_provider(self):
        runner, session, device = self.runner()
        with self.assertRaises(ValueError):
            runner.run(execute=lambda: None)
        self.assertTrue(device.closed)
        self.assertTrue(session.closed)
        self.assertTrue((self.root/'run'/'cleanup.json').is_file())

    def test_provider_turn_failure_closes_both_and_preserves_original_error(self):
        runner, session, device = self.runner()
        with self.assertRaisesRegex(RuntimeError, 'provider turn failed'):
            runner.run()
        self.assertTrue(device.closed)
        self.assertTrue(session.closed)
        self.assertIn('provider turn failed', json.loads((self.root/'run'/'cleanup.json').read_text())['original_error'])

    def test_provider_startup_error_is_not_replaced_by_device_cleanup_error(self):
        class ProviderInitError(Exception):
            pass
        class CleanupFails(MockDevice):
            def close(self):
                super().close()
                raise RuntimeError('cleanup failed too')
        device = CleanupFails()
        def provider_factory(*args, **kwargs):
            raise ProviderInitError('primary provider initialization error')
        config = dict(task='fixture', scope=SCOPE, knowledge=str(self.root/'knowledge'),
                      device=dict(factory='configured:device', settings={}),
                      model=dict(factory='configured:provider', settings={}))
        target = 'astra_robotics_agent.physical_keyboard.agent.runner'
        with patch(target+'.load_adapter', return_value=device), patch(target+'.load_factory', return_value=provider_factory):
            with self.assertRaisesRegex(ProviderInitError, 'primary provider initialization error') as failure:
                run_config(config, output=self.root/'run')
        self.assertTrue(device.closed)
        self.assertTrue(any('cleanup' in note.lower() for note in getattr(failure.exception, '__notes__', [])))

    def test_driver_status_field_cannot_cause_a_false_unknown_execution(self):
        class HasDriverStatus(MockDevice):
            def observe(self):
                state = super().observe().state
                state['status'] = 'driver_metric'
                return Observation(state, [])
            def dispatch(self, prepared):
                receipt = super().dispatch(prepared)
                receipt.result['status'] = 'driver_metric'
                return CallReceipt(receipt.status, receipt.dispatch_state, receipt.result)
        adapter = HasDriverStatus()
        host = ProgramRuntime(adapter, self.root/'runtime', run_id='audit')
        self.addCleanup(host.close)
        executed = host.execute(dict(label='status', profile={}, source='''def main(robot, profile):
    state = robot.observe()
    robot.move_tcp([0, 0, 0.16], [0, 0, 0, 1])
    return state
'''))
        self.assertEqual(executed['response']['status'], 'ok')
        self.assertFalse(host.uncertain)
        self.assertEqual(adapter.dispatch_count, 1)

    def test_provider_factory_can_initialize_its_log_directory(self):
        class FinishSession(Session):
            def run_turn(self, prompt, callback, images):
                decision = dict(action='finish', result=dict(status='partial', reflection=dict(
                    summary='Synthetic factory fixture only', skills_saved=[], lessons_saved=[],
                    limitations=['No model or hardware execution'])))
                return dict(turn_id='fixture', final_text=json.dumps(decision), usage=None)
        created, workspaces = [], []
        def provider_factory(settings, *, instructions, cwd, output):
            cwd = Path(cwd)
            workspaces.append((cwd.resolve(), cwd.is_dir(), list(cwd.iterdir())))
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output/'provider-created.json').write_text('{}')
            value = FinishSession()
            created.append(value)
            return value
        device = MockDevice()
        config = dict(task='fixture', scope=SCOPE, knowledge=str(self.root/'knowledge'),
                      device=dict(factory='configured:device', settings={}),
                      model=dict(factory='configured:provider', settings={}))
        target = 'astra_robotics_agent.physical_keyboard.agent.runner'
        with patch(target+'.load_adapter', return_value=device), patch(target+'.load_factory', return_value=provider_factory):
            report = run_config(config, output=self.root/'run')
        self.assertTrue(report['finished_by_model'])
        self.assertTrue(device.closed)
        self.assertTrue(created[0].closed)
        self.assertNotEqual(workspaces[0][0], Path.cwd().resolve())
        self.assertTrue(workspaces[0][1])
        self.assertEqual(workspaces[0][2], [])


if __name__ == '__main__':
    unittest.main()
