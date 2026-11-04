"""Independent adversarial host checks, executed only against the clean core."""
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.runtime.device_adapter import AdmissionRejected, CallReceipt
from astra_robotics_agent.physical_keyboard.runtime.mock_device import MockDevice
from astra_robotics_agent.physical_keyboard.runtime.program_runtime import ProgramRuntime
from astra_robotics_agent.physical_keyboard.runtime.skill_language import ToolError


Q = "[0, 0, 0, 1]"


def program(*lines):
    return dict(label="counterexample", profile={}, source="def main(robot, profile):\n" +
                "\n".join("    " + line for line in lines) + "\n")


class ProgramRuntimeCounterexamples(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def runtime(self, device):
        value = ProgramRuntime(device, Path(self.folder.name) / "runtime", run_id="audit")
        self.addCleanup(value.close)
        return value

    def test_dispatch_toolerror_cannot_be_caught_as_a_proved_rejection(self):
        class FailsAfterSend(MockDevice):
            def dispatch(self, prepared):
                super().dispatch(prepared)
                raise ToolError("driver exception after dispatch")
        adapter = FailsAfterSend()
        host = self.runtime(adapter)
        result = host.execute(program(
            "try:",
            "    robot.move_tcp([0, 0, 0.12], " + Q + ")",
            "except ToolError:",
            "    robot.move_tcp([0, 0, 0.2], " + Q + ")",
            "return {'model_success': True}",
        ))
        self.assertTrue(host.uncertain)
        self.assertTrue(result["response"]["result"]["motion_outcome_uncertain"])
        self.assertEqual(adapter.dispatch_count, 1)

    def test_prepare_rejection_can_be_revised_without_replaying_completed_tools(self):
        class RejectsOnePreparedTarget(MockDevice):
            def prepare(self, method, arguments):
                if method == 'move_tcp' and arguments.get('xyz_m') == [0, 0, .12]:
                    raise AdmissionRejected('mock preflight rejects this otherwise well-formed target')
                return super().prepare(method, arguments)
        adapter = RejectsOnePreparedTarget()
        host = self.runtime(adapter)
        result = host.execute(program(
            "robot.move_tcp([0, 0, 0.16], " + Q + ")",
            "try:",
            "    robot.move_tcp([0, 0, 0.12], " + Q + ")",
            "except ToolError:",
            "    robot.move_tcp([0, 0, 0.2], " + Q + ")",
            "return {'attempt_complete': True}",
        ))
        self.assertFalse(host.uncertain)
        self.assertEqual(adapter.dispatch_count, 2)
        self.assertEqual(result["response"]["status"], "ok")
        self.assertFalse(result["response"]["result"]["task_success_verified"])

    def test_admissionrejected_from_dispatch_is_unknown_even_if_program_catches_toolerror(self):
        class FailsAfterSend(MockDevice):
            def dispatch(self, prepared):
                super().dispatch(prepared)
                raise AdmissionRejected("wrong exception phase")
        adapter = FailsAfterSend()
        host = self.runtime(adapter)
        result = host.execute(program(
            "try:",
            "    robot.move_tcp([0, 0, 0.12], " + Q + ")",
            "except ToolError:",
            "    robot.move_tcp([0, 0, 0.2], " + Q + ")",
        ))
        self.assertTrue(host.uncertain)
        self.assertEqual(adapter.dispatch_count, 1)
        self.assertTrue(result["response"]["result"]["controller_terminal"])

    def test_unknown_receipt_latches_and_further_execute_is_refused(self):
        class UnknownAfterSend(MockDevice):
            def dispatch(self, prepared):
                super().dispatch(prepared)
                return CallReceipt("unknown", "unknown", {"reason": "lost acknowledgement"})
        adapter = UnknownAfterSend()
        host = self.runtime(adapter)
        host.execute(program("robot.move_tcp([0, 0, 0.12], " + Q + ")"))
        with self.assertRaises(RuntimeError):
            host.execute(program("robot.move_tcp([0, 0, 0.2], " + Q + ")"))
        self.assertEqual(adapter.dispatch_count, 1)

    def test_nonfinite_driver_result_after_send_latches_unknown(self):
        class NonfiniteAfterSend(MockDevice):
            def dispatch(self, prepared):
                receipt = super().dispatch(prepared)
                # The callback mutates nested data after dataclass validation.
                receipt.result["driver_value"] = float("nan")
                return receipt
        adapter = NonfiniteAfterSend()
        host = self.runtime(adapter)
        result = host.execute(program("robot.move_tcp([0, 0, 0.12], " + Q + ")"))
        self.assertTrue(host.uncertain)
        self.assertTrue(result["response"]["result"]["motion_outcome_uncertain"])
        self.assertEqual(adapter.dispatch_count, 1)

    def test_stop_failure_does_not_erase_unknown_or_repeat_move(self):
        class UnknownAndStopFails(MockDevice):
            def dispatch(self, prepared):
                super().dispatch(prepared)
                raise RuntimeError("acknowledgement lost")
            def stop(self):
                raise RuntimeError("stop channel unavailable")
        adapter = UnknownAndStopFails()
        host = self.runtime(adapter)
        result = host.execute(program("robot.move_tcp([0, 0, 0.12], " + Q + ")"))
        self.assertTrue(host.uncertain)
        stops = result["response"]["result"]["stop_attempts"]
        self.assertTrue(any("stop channel unavailable" in str(value) for value in stops))
        self.assertEqual(adapter.dispatch_count, 1)

    def test_interpreter_validation_rejects_before_first_move(self):
        adapter = MockDevice()
        host = self.runtime(adapter)
        result = host.execute(program("import os", "return 'success'"))
        self.assertFalse(host.uncertain)
        self.assertEqual(adapter.dispatch_count, 0)
        self.assertEqual(result["response"]["status"], "blocked")

    def test_return_value_never_becomes_independent_task_success(self):
        adapter = MockDevice()
        host = self.runtime(adapter)
        result = host.execute(program("return {'task_success': True, 'observed_text': 'a'}"))
        self.assertFalse(result["response"]["result"]["task_success_verified"])
        self.assertEqual(adapter.observe().state["observed_text"], "")

    def test_completed_move_then_interpreter_error_can_be_revised_by_new_program(self):
        adapter = MockDevice()
        host = self.runtime(adapter)
        first = host.execute(program(
            "robot.move_tcp([0, 0, 0.16], " + Q + ")",
            "value = 1 / 0",
            "robot.move_tcp([0, 0, 0.12], " + Q + ")",
        ))
        self.assertEqual(adapter.dispatch_count, 1)
        self.assertFalse(host.uncertain)
        self.assertFalse(adapter.stopped)
        self.assertEqual(first['response']['status'], 'error')
        self.assertFalse(first['response']['result']['motion_outcome_uncertain'])
        self.assertTrue(first['response']['result']['recoverable_program_error'])
        second = host.execute(program("robot.move_tcp([0, 0, 0.2], " + Q + ")"))
        self.assertEqual(second['response']['status'], 'ok')
        self.assertEqual(adapter.dispatch_count, 2)
        self.assertEqual(adapter.observe().state['observed_text'], '')


if __name__ == "__main__":
    unittest.main()
