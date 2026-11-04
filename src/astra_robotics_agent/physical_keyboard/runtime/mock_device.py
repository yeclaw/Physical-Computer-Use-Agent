"""Deterministic single-key fake for the public integration example.

This is authored synthetic geometry, not a physics simulator, OS keyboard,
camera, learned behavior or reference robot implementation. Preparation never
changes the fake. A centered stroke must reach z <= 0.13 m and then release to
z >= 0.19 m to append one ``a``. A shallow stroke to z=0.16 m does not register.
"""
from __future__ import annotations

import math

from .device_adapter import (
    AdapterDescription, AdmissionRejected, CallReceipt, Observation,
    PreparedCall, json_copy,
)


def _number(value, name):
    if type(value) not in (int, float):
        raise AdmissionRejected(name + " must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise AdmissionRejected(name + " must be a finite number") from error
    if not math.isfinite(result):
        raise AdmissionRejected(name + " must be a finite number")
    return result


def _vector(value, count, name):
    if type(value) is not list or len(value) != count:
        raise AdmissionRejected(name + " requires " + str(count) + " numbers")
    return [_number(item, name) for item in value]


class MockDevice:
    def __init__(self, settings=None):
        settings = {} if settings is None else json_copy(settings)
        if type(settings) is not dict or set(settings) - {"evidence_mode"}:
            raise ValueError("mock settings only accept evidence_mode=synthetic")
        if settings.get("evidence_mode", "synthetic") != "synthetic":
            raise ValueError("mock evidence must be explicitly synthetic")
        self.xyz_m = [0.0, 0.0, 0.2]
        self.quaternion_xyzw = [0.0, 0.0, 0.0, 1.0]
        self.text = ""
        self.closed = False
        self.stopped = False
        self.dispatch_count = 0
        self._contact = False
        self._owner_token = object()

    def describe(self):
        return AdapterDescription(id="mock_keyboard", coordinate_frame="synthetic_world",
                                  views=(), evidence_mode="synthetic")

    def _state(self):
        return json_copy(dict(
            observation=dict(pose=dict(xyz_m=self.xyz_m, quaternion_xyzw=self.quaternion_xyzw)),
            observed_text=self.text, evidence_mode="synthetic", stopped=self.stopped,
            dispatch_count=self.dispatch_count,
        ))

    def observe(self):
        if self.closed:
            raise RuntimeError("mock adapter is closed")
        return Observation(self._state(), [{"type": "inputText", "text":
            "Synthetic keyboard observation: " + repr(self.text)}])

    def prepare(self, method, arguments):
        if self.closed:
            raise AdmissionRejected("mock adapter is closed")
        if self.stopped:
            raise AdmissionRejected("mock adapter is stopped")
        if type(arguments) is not dict:
            raise AdmissionRejected("tool arguments must be an object")
        if method == "move_tcp":
            if set(arguments) != {"xyz_m", "quaternion_xyzw"}:
                raise AdmissionRejected("move_tcp requires xyz_m and quaternion_xyzw")
            xyz = _vector(arguments["xyz_m"], 3, "xyz_m")
            quaternion = _vector(arguments["quaternion_xyzw"], 4, "quaternion_xyzw")
            if not math.isclose(math.hypot(*quaternion), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise AdmissionRejected("quaternion_xyzw must be unit length")
            copied = dict(xyz_m=xyz, quaternion_xyzw=quaternion)
        elif method == "wait_bounded":
            if set(arguments) != {"seconds"}:
                raise AdmissionRejected("wait_bounded requires seconds")
            seconds = _number(arguments["seconds"], "seconds")
            if seconds < 0:
                raise AdmissionRejected("seconds must be nonnegative")
            copied = dict(seconds=seconds)
        else:
            raise AdmissionRejected("mock does not implement tool: " + str(method))
        return PreparedCall(method, copied, token=self._owner_token)

    def dispatch(self, prepared):
        if self.closed or self.stopped:
            raise RuntimeError("mock adapter cannot dispatch after stop/close")
        if not isinstance(prepared, PreparedCall) or prepared.token is not self._owner_token:
            raise RuntimeError("prepared call belongs to a different mock adapter")
        # Revalidation protects the example adapter from a caller modifying the
        # nested argument dictionary between preparation and dispatch. It is
        # not a hardware protection mechanism or a reason for the host to retry.
        current = self.prepare(prepared.method, prepared.arguments)
        before = list(self.xyz_m)
        self.dispatch_count += 1
        if current.method == "move_tcp":
            xyz = current.arguments["xyz_m"]
            centered = (abs(before[0]) <= .005 and abs(before[1]) <= .005
                        and abs(xyz[0]) <= .005 and abs(xyz[1]) <= .005)
            if centered and before[2] > .13 >= xyz[2]:
                self._contact = True
            elif not centered:
                self._contact = False
            if xyz[2] >= .19:
                if self._contact:
                    self.text += "a"
                self._contact = False
            self.xyz_m = list(xyz)
            self.quaternion_xyzw = list(current.arguments["quaternion_xyzw"])
        result = self._state()
        if current.method == "wait_bounded":
            result["waited_seconds"] = current.arguments["seconds"]
            result["clock"] = "synthetic_no_wall_clock_sleep"
        return CallReceipt("ok", "completed", result)

    def stop(self):
        self.stopped = True
        return CallReceipt("ok", "completed", dict(stopped=True, evidence_mode="synthetic",
                                                   physical_release_verified=False))

    def close(self):
        self.closed = True
        return dict(closed=True, evidence_mode="synthetic")


def factory(settings):
    """Adapter-factory entry point used by an explicit JSON configuration."""
    return MockDevice(settings)
