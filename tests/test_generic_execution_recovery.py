"""Fault recovery descriptions belong to the external device integration."""
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.agent.execution_service import (
    PersistentExecutionSupervisor, make_resume_event,
)


class GenericExecutionRecoveryTests(unittest.TestCase):
    def run_recovery(self, *, instructions=None, observation_home=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        states, calls = [], []
        def call(name, result):
            def callback(*args):
                calls.append(name)
                return result
            return callback
        def idle(seconds):
            request = next(value['recovery'] for value in reversed(states) if 'recovery' in value)
            event = make_resume_event(request['fault_id'], observation_home=observation_home)
            service.resume_file.write_text(json.dumps(event))
            service.resume_file.chmod(0o600)
        service = PersistentExecutionSupervisor(output=Path(temp.name) / 'service',
            close_host=call('close', dict(closed=True)), review_fault=call('review', {}),
            restore_host=call('restore', dict(generation=2)),
            observe=call('observe', dict(evidence_id='fresh', contentItems=[])),
            status=states.append, sleep_fn=idle, recovery_instructions=instructions,
            observation_home=observation_home)
        result = service.handle_fault(dict(stopped_reason='unknown', last_observation=dict(evidence_id='old')))
        request = next(value['recovery'] for value in states if value.get('phase') == 'execution_suspended')
        return request, result, calls

    def test_default_recovery_description_contains_no_reference_machine_initialization(self):
        request, result, calls = self.run_recovery()
        description = request['initialization_side_effect']
        self.assertIn('configured recovery', description)
        self.assertNotIn('.08', description)
        self.assertNotIn('right-only', description)
        self.assertNotIn('gripper', description)
        self.assertFalse(result['recovery']['original_request_replayed'])
        self.assertEqual(calls, ['close', 'review', 'restore', 'observe'])

    def test_external_recovery_instructions_are_published_without_replacing_the_protocol(self):
        description = 'Owner-supplied fixture recovery: verify the tool is clear.'
        request, result, calls = self.run_recovery(instructions=description)
        self.assertEqual(request['initialization_side_effect'], description)
        self.assertFalse(result['recovery']['implicit_home'])
        self.assertEqual(calls, ['close', 'review', 'restore', 'observe'])

    def test_optional_reference_restore_has_no_fixed_arm_or_pose(self):
        request, result, calls = self.run_recovery(observation_home=True)
        self.assertIn('Reference pose restoration', request['initialization_side_effect'])
        self.assertNotIn('right-only', request['initialization_side_effect'])
        self.assertIn('allow_observation_home', request['required_confirmations'])
        self.assertFalse(result['recovery']['original_request_replayed'])


if __name__ == '__main__':
    unittest.main()
