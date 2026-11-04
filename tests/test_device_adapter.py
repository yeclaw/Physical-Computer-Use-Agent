"""Portable device-seam counterexamples; no private package is required."""
import math
import sys
import types
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.runtime.device_adapter import (
    AdapterDescription, AdmissionRejected, CallReceipt, Observation, PreparedCall,
    json_copy, load_adapter, load_factory,
)
from astra_robotics_agent.physical_keyboard.runtime.mock_device import MockDevice


FACTORY = "astra_robotics_agent.physical_keyboard.runtime.mock_device:factory"


class DeviceAdapterTests(unittest.TestCase):
    def move(self, device, z, x=0):
        return device.prepare("move_tcp", dict(xyz_m=[x, 0, z], quaternion_xyzw=[0, 0, 0, 1]))

    def test_loaded_default_is_explicitly_synthetic_and_has_no_reference_cameras(self):
        device = load_adapter(FACTORY, {"evidence_mode": "synthetic"})
        self.assertEqual(device.describe(), AdapterDescription(
            "mock_keyboard", "synthetic_world", (), "synthetic"))
        self.assertEqual(device.observe().state["observed_text"], "")

    def test_prepare_has_no_effects_and_copies_input(self):
        device = MockDevice()
        before = device.observe().state
        args = dict(xyz_m=[0, 0, .12], quaternion_xyzw=[0, 0, 0, 1])
        prepared = device.prepare("move_tcp", args)
        args["xyz_m"][2] = .04
        self.assertEqual(prepared.arguments["xyz_m"][2], .12)
        self.assertEqual(before, device.observe().state)

    def test_shallow_stroke_does_not_register_and_deep_stroke_requires_release(self):
        device = MockDevice()
        for z in (.16, .2):
            device.dispatch(self.move(device, z))
        self.assertEqual(device.observe().state["observed_text"], "")
        device.dispatch(self.move(device, .12))
        self.assertEqual(device.observe().state["observed_text"], "")
        device.dispatch(self.move(device, .2))
        self.assertEqual(device.observe().state["observed_text"], "a")
        device.dispatch(self.move(device, .2))
        self.assertEqual(device.observe().state["observed_text"], "a")

    def test_unaligned_stroke_never_registers(self):
        device = MockDevice()
        device.dispatch(self.move(device, .12, x=.02))
        device.dispatch(self.move(device, .2, x=.02))
        self.assertEqual(device.observe().state["observed_text"], "")

    def test_wait_is_fake_and_does_not_sleep_or_type(self):
        device = MockDevice()
        receipt = device.dispatch(device.prepare("wait_bounded", {"seconds": 1_000_000}))
        self.assertEqual(receipt.result["clock"], "synthetic_no_wall_clock_sleep")
        self.assertEqual(receipt.result["observed_text"], "")

    def test_invalid_pose_is_zero_dispatch(self):
        device = MockDevice()
        cases = [
            {"xyz_m": [0, 0, math.nan], "quaternion_xyzw": [0, 0, 0, 1]},
            {"xyz_m": [0, 0, math.inf], "quaternion_xyzw": [0, 0, 0, 1]},
            {"xyz_m": [0, 0, True], "quaternion_xyzw": [0, 0, 0, 1]},
            {"xyz_m": [0, 0, .2], "quaternion_xyzw": [0, 0, 0, 2]},
            {"xyz_m": [0, .2], "quaternion_xyzw": [0, 0, 0, 1]},
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(AdmissionRejected):
                device.prepare("move_tcp", args)
        self.assertEqual(device.dispatch_count, 0)

    def test_prepared_token_is_not_cross_adapter_authority(self):
        first, second = MockDevice(), MockDevice()
        with self.assertRaises(RuntimeError):
            second.dispatch(self.move(first, .12))
        self.assertEqual(second.dispatch_count, 0)

    def test_closed_or_stopped_mock_cannot_send_more_targets(self):
        for operation in ("stop", "close"):
            device = MockDevice()
            prepared = self.move(device, .12)
            getattr(device, operation)()
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                device.dispatch(prepared)
            self.assertEqual(device.dispatch_count, 0)

    def test_observation_returns_independent_state(self):
        device = MockDevice()
        observed = device.observe()
        observed.state["observation"]["pose"]["xyz_m"][2] = 0
        self.assertEqual(device.observe().state["observation"]["pose"]["xyz_m"][2], .2)

    def test_call_receipt_cannot_hide_rejection_as_completed_or_unknown_as_completed(self):
        for status, state in (("rejected", "completed"), ("unknown", "completed"),
                              ("ok", "unknown"), ("ok", "not_started")):
            with self.subTest(status=status, state=state), self.assertRaises(ValueError):
                CallReceipt(status, state, {})
        self.assertEqual(CallReceipt("unknown", "unknown", {}).status, "unknown")

    def test_nonfinite_receipt_and_observation_are_rejected(self):
        with self.assertRaises(ValueError):
            CallReceipt("ok", "completed", {"x": float("nan")})
        with self.assertRaises(ValueError):
            Observation({"x": float("inf")}, [])
        with self.assertRaises(ValueError):
            PreparedCall("move_tcp", {"x": float("nan")})

    def test_json_copy_does_not_coerce_types_or_keys(self):
        for value in ({False: "bad"}, {"tuple": (1, 2)}, {"object": object()}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                json_copy(value)
        source = {"nested": [1, 2]}
        copied = json_copy(source)
        source["nested"].append(3)
        self.assertEqual(copied, {"nested": [1, 2]})

    def test_description_allows_single_arbitrary_view_and_rejects_duplicate_or_path_names(self):
        value = AdapterDescription("another_robot", "flange/calibrated_base", ("screen",))
        self.assertEqual(value.views, ("screen",))
        for views in (("screen", "screen"), ("../../screen",), ("bad/name",)):
            with self.subTest(views=views), self.assertRaises(ValueError):
                AdapterDescription("robot", "base", views)

    def test_loader_does_not_accept_arbitrary_expressions_or_missing_interface(self):
        for spec in ("module", "module:factory:other", "module:factory()", "module:_hidden"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                load_factory(spec)
        module = types.ModuleType("adapter_loader_fixture")
        instance = types.SimpleNamespace(closed=False)
        instance.close = lambda: setattr(instance, "closed", True)
        module.factory = lambda settings: instance
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaises(TypeError):
                load_adapter("adapter_loader_fixture:factory", {})
        self.assertTrue(instance.closed)

    def test_loader_validates_settings_before_factory_and_copies_them(self):
        module = types.ModuleType("adapter_settings_fixture")
        values = []
        def factory(settings):
            values.append(settings)
            settings["nested"][0] = "changed"
            return MockDevice()
        module.factory = factory
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaises(ValueError):
                load_adapter("adapter_settings_fixture:factory", {"value": object()})
            self.assertEqual(values, [])
            original = {"nested": ["original"]}
            load_adapter("adapter_settings_fixture:factory", original)
            self.assertEqual(original, {"nested": ["original"]})


if __name__ == "__main__":
    unittest.main()
