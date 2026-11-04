"""Runnable external factories for an authored, synthetic waypoint task.

This module is example integration code, not a hardware driver, camera, physics
simulator or learned policy. Its geometry comes entirely from configuration.
Real adapters must read measured state and actual images from their own SDK.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import struct
import zlib

from astra_robotics_agent.physical_keyboard.runtime.device_adapter import (
    AdapterDescription, AdmissionRejected, CallReceipt, Observation,
    PreparedCall, json_copy,
)


def _numbers(value, count, name):
    if type(value) is not list or len(value) != count:
        raise ValueError(name + ' requires a list of ' + str(count) + ' numbers')
    if any(type(item) not in (int, float) for item in value):
        raise ValueError(name + ' requires finite numbers')
    try:
        result = [float(item) for item in value]
    except OverflowError as error:
        raise ValueError(name + ' requires finite numbers') from error
    if not all(math.isfinite(item) for item in result):
        raise ValueError(name + ' requires finite numbers')
    return result


def _pixel(index):
    """One generated pixel, explicitly synthetic; never a camera observation."""
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack(
            '!I', zlib.crc32(kind + payload) & 0xffffffff)
    rgb = bytes((40 + index % 100, 100, 180))
    raw = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(b'\0' + rgb)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


class SyntheticStation:
    """Example adapter with operator-declared frame, geometry and view names."""
    def __init__(self, settings):
        expected = {'id', 'coordinate_frame', 'views', 'workspace_m', 'initial_xyz_m', 'target_xyz_m'}
        if type(settings) is not dict or set(settings) != expected:
            raise ValueError('Synthetic station requires exactly the documented settings')
        settings = json_copy(settings)
        if type(settings['views']) is not list:
            raise ValueError('views requires a list of declared view identifiers')
        self.description = AdapterDescription(settings['id'], settings['coordinate_frame'],
                                              tuple(settings['views']), 'synthetic')
        bounds = settings['workspace_m']
        if type(bounds) is not list or len(bounds) != 2:
            raise ValueError('workspace_m requires lower and upper xyz bounds')
        self.lower = _numbers(bounds[0], 3, 'workspace lower')
        self.upper = _numbers(bounds[1], 3, 'workspace upper')
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError('Each lower workspace bound must be below its upper bound')
        self.xyz_m = _numbers(settings['initial_xyz_m'], 3, 'initial_xyz_m')
        self.target = _numbers(settings['target_xyz_m'], 3, 'target_xyz_m')
        if not self._inside(self.xyz_m) or not self._inside(self.target):
            raise ValueError('Initial and target positions must lie in the configured synthetic workspace')
        self.quaternion = [0.0, 0.0, 0.0, 1.0]
        self.closed = self.stopped = False
        self.dispatches = 0
        self.token = object()

    def _inside(self, xyz):
        return all(low <= value <= high for low, value, high in zip(self.lower, xyz, self.upper))

    def describe(self):
        return self.description

    def observe(self):
        if self.closed:
            raise RuntimeError('Synthetic station is closed')
        state = dict(
            observation=dict(pose=dict(xyz_m=list(self.xyz_m), quaternion_xyzw=list(self.quaternion))),
            target_xyz_m=list(self.target), synthetic_target_reached=math.dist(self.xyz_m, self.target) <= 1e-9,
            evidence_mode='synthetic', position_source='synthetic_state_not_measured_hardware',
            stopped=self.stopped, dispatches=self.dispatches,
            image_order=[dict(image_index=i, view_id=view) for i, view in enumerate(self.description.views)],
        )
        items = [dict(type='inputText', text='Authored waypoint fixture; generated pixels are not camera evidence.')]
        items.extend(dict(type='inputImage', imageUrl=_pixel(i), view_id=view)
                     for i, view in enumerate(self.description.views))
        return Observation(json_copy(state), items)

    def prepare(self, method, arguments):
        if self.closed or self.stopped:
            raise AdmissionRejected('Synthetic station is closed or stopped')
        try:
            if type(arguments) is not dict:
                raise ValueError('Arguments must be a dictionary')
            if method == 'move_tcp':
                if set(arguments) != {'xyz_m', 'quaternion_xyzw'}:
                    raise ValueError('move_tcp requires xyz_m and quaternion_xyzw')
                xyz = _numbers(arguments['xyz_m'], 3, 'xyz_m')
                quaternion = _numbers(arguments['quaternion_xyzw'], 4, 'quaternion_xyzw')
                if not self._inside(xyz):
                    raise ValueError('Target lies outside the configured synthetic workspace')
                if not math.isclose(math.hypot(*quaternion), 1.0, abs_tol=1e-6, rel_tol=0):
                    raise ValueError('quaternion_xyzw must have unit length')
                args = dict(xyz_m=xyz, quaternion_xyzw=quaternion)
            elif method == 'wait_bounded':
                if set(arguments) != {'seconds'}:
                    raise ValueError('wait_bounded requires seconds')
                seconds = _numbers([arguments['seconds']], 1, 'seconds')[0]
                if seconds < 0:
                    raise ValueError('seconds must be nonnegative')
                args = dict(seconds=seconds)
            else:
                raise ValueError('Unsupported method: ' + str(method))
        except ValueError as error:
            raise AdmissionRejected(str(error)) from error
        return PreparedCall(method, args, token=self.token)

    def dispatch(self, prepared):
        if not isinstance(prepared, PreparedCall) or prepared.token is not self.token:
            raise RuntimeError('Prepared call belongs to another adapter')
        current = self.prepare(prepared.method, prepared.arguments)
        self.dispatches += 1
        if current.method == 'move_tcp':
            self.xyz_m = list(current.arguments['xyz_m'])
            self.quaternion = list(current.arguments['quaternion_xyzw'])
        # In this mathematical fake only, the command updates synthetic state
        # synchronously. Real SDKs must wait for actual completion and query
        # measured state; acceptance/queueing is not a completed receipt.
        return CallReceipt('ok', 'completed', self.observe().state)

    def stop(self):
        self.stopped = True
        return CallReceipt('ok', 'completed', dict(stopped=True, evidence_mode='synthetic',
                                                 physical_release_verified=False))

    def close(self):
        self.closed = True
        return dict(closed=True, evidence_mode='synthetic')


class AuthoredWaypointSession:
    """Two scripted decisions to test integration, not an autonomous model."""
    def __init__(self):
        self.index = 0
        self.closed = False

    def run_turn(self, prompt, callback, images):
        records = []
        for line in prompt.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        current = next(row for row in records if type(row) is dict
                       and 'adapter' in row and 'current_state' in row)
        state = current['current_state']
        assert current['adapter']['evidence_mode'] == 'synthetic'
        assert len(images) == len(state['image_order'])
        if self.index == 0:
            decision = dict(action='run_skill', arguments=dict(label='configured_waypoint',
                source_lines=['def main(robot, profile):',
                              '    robot.move_tcp(profile["target"], [0, 0, 0, 1])',
                              '    return robot.observe()'],
                profile=dict(target=state['target_xyz_m'])),
                hypothesis='The configured synthetic target should be reached in the next observation.')
        elif self.index == 1:
            assert state['synthetic_target_reached'] is True
            decision = dict(action='finish', result=dict(status='partial', reflection=dict(
                summary='External factories executed one configured synthetic waypoint.',
                skills_saved=[], lessons_saved=[],
                limitations=['Authored fixture only; no model learning, physical device or transferable calibration.'])))
        else:
            raise RuntimeError('Unexpected extra authored fixture turn')
        self.index += 1
        return dict(turn_id='waypoint-' + str(self.index), final_text=json.dumps(decision), usage=None)

    def close(self):
        self.closed = True


def device_factory(settings):
    return SyntheticStation(settings)


def session_factory(settings, *, instructions, cwd, output):
    if settings != {}:
        raise ValueError('Authored session accepts no settings or credentials')
    if not Path(cwd).is_dir():
        raise ValueError('Provider workspace must exist')
    # A real provider supplies its own authentication externally. This fixture
    # does not read credentials, invoke a CLI, call a network or move hardware.
    return AuthoredWaypointSession()
