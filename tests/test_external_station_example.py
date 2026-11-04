"""External device/model factories on new synthetic geometry and view counts."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.agent.runner import run_config
from astra_robotics_agent.physical_keyboard.runtime.device_adapter import (
    AdmissionRejected, CallReceipt, load_adapter,
)
from astra_robotics_agent.physical_keyboard.runtime.program_runtime import ProgramRuntime


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class ExternalStationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(EXAMPLES))
        cls.example = importlib.import_module('external_station')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(EXAMPLES))

    def settings(self, views=('inspection',)):
        config = json.loads((EXAMPLES / 'external_station.json').read_text())
        settings = config['device']['settings']
        settings['views'] = list(views)
        return settings

    def test_actual_factories_run_different_geometry_and_zero_one_two_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for index, views in enumerate(((), ('inspection',), ('near', 'wide'))):
                config = json.loads((EXAMPLES / 'external_station.json').read_text())
                config['knowledge'] = str(root / 'shared-knowledge')
                config['scope']['layout_id'] = 'layout-' + str(index)
                config['device']['settings']['views'] = list(views)
                target = [0.24 + index * .03, -.18 + index * .02, .37 + index * .01]
                config['device']['settings']['target_xyz_m'] = target
                output = root / ('run-' + str(index))
                with self.subTest(views=views):
                    report = run_config(config, output=output)
                    self.assertTrue(report['finished_by_model'])
                    self.assertEqual(len(report['turns']), 2)
                    observations = sorted((output / 'runtime').glob('*-observation-*/record.json'))
                    actual = json.loads(observations[-1].read_text())
                    self.assertEqual(actual['state']['observation']['pose']['xyz_m'], target)
                    self.assertTrue(actual['state']['synthetic_target_reached'])
                    self.assertEqual([x['view_id'] for x in actual['images']], list(views))
                    self.assertEqual(len(actual['images']), len(views))
                    self.assertEqual(actual['adapter']['coordinate_frame'], 'synthetic_station_base')
                    self.assertFalse(json.loads((output / 'run.json').read_text())['task_success_verified'])
                    self.assertTrue(json.loads((output / 'cleanup.json').read_text())['runtime']['adapter_result']['closed'])
            self.assertEqual(len(list((root / 'shared-knowledge/scopes').iterdir())), 3)

    def test_factory_exposes_declared_frame_without_reference_robot_cameras(self):
        settings = self.settings(('custom_view',))
        settings['coordinate_frame'] = 'another_declared_command_frame'
        device = load_adapter('external_station:device_factory', settings)
        self.addCleanup(device.close)
        self.assertEqual(device.describe().coordinate_frame, 'another_declared_command_frame')
        self.assertEqual(device.describe().views, ('custom_view',))
        self.assertEqual(device.describe().evidence_mode, 'synthetic')

    def test_invalid_pose_rejects_before_dispatch_and_has_no_effects(self):
        device = self.example.SyntheticStation(self.settings())
        self.addCleanup(device.close)
        before = device.observe().state
        for xyz, quaternion in [([9, 9, 9], [0, 0, 0, 1]),
                                 ([.24, -.18, .37], [0, 0, 0, 2])]:
            with self.subTest(xyz=xyz), self.assertRaises(AdmissionRejected):
                device.prepare('move_tcp', dict(xyz_m=xyz, quaternion_xyzw=quaternion))
        self.assertEqual(device.observe().state, before)
        self.assertEqual(device.dispatches, 0)

    def test_unknown_dispatch_is_stopped_latched_and_never_replayed(self):
        class UnknownStation(self.example.SyntheticStation):
            def dispatch(self, prepared):
                super().dispatch(prepared)
                return CallReceipt('unknown', 'unknown', dict(reason='synthetic_missing_completion_receipt'))
        device = UnknownStation(self.settings())
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ProgramRuntime(device, Path(temporary) / 'runtime', run_id='external-unknown')
            self.addCleanup(runtime.close)
            request = dict(label='target', profile={}, source='''def main(robot, profile):
    robot.move_tcp([0.24, -0.18, 0.37], [0, 0, 0, 1])
    return robot.observe()
''')
            result = runtime.execute(request)
            self.assertEqual(result['response']['status'], 'error')
            self.assertTrue(runtime.uncertain)
            self.assertTrue(device.stopped)
            with self.assertRaises(RuntimeError):
                runtime.execute(request)
            self.assertEqual(device.dispatches, 1)

    def test_invalid_configuration_and_view_ids_are_not_silently_repaired(self):
        for change in [{'views': ['bad/view']}, {'views': ['same', 'same']}, {'views': 'abc'},
                       {'initial_xyz_m': [1, 1, 1]}, {'extra': True}]:
            settings = self.settings()
            settings.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.example.SyntheticStation(settings)


if __name__ == '__main__':
    unittest.main()
