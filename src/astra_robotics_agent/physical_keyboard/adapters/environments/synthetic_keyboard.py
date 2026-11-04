"""Host-owned virtual keyboard evaluator driven by acknowledged fake TCP paths.

This deliberately simplified geometry is synthetic evidence, never physical
contact, an operating-system keyboard event, or OCR. Candidate return values and
commanded target positions cannot provide evidence that a key was registered.
"""

import json
import math

from astra_robotics_agent.physical_keyboard.records.contracts import canonical, digest, identifier


CONFIG_PROTOCOL = 'synthetic-keyboard-config-v1'
EVALUATION_PROTOCOL = 'synthetic-keyboard-evaluation-v1'


class IncompleteProgram(ValueError):
    """Literal text exists, but it cannot validate a completed skill attempt."""

    def __init__(self, reasons, evidence):
        super().__init__('Incomplete synthetic program: ' + ', '.join(reasons))
        self.evidence = evidence


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Synthetic geometry requires finite numeric values')
    return float(value)


def _vector(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError('Invalid synthetic coordinate dimension')
    return [_number(v) for v in value]


class SyntheticKeyboard:
    """Evaluate complete press/release strokes against one frozen host config.

    Configure ``keys`` as a mapping of individual lowercase letters or space to
    XY meters. A stroke begins on a downward crossing of contact_z_m while both
    segment endpoints lie inside the same keycap. It must remain aligned, reach
    registration_z_m without violating minimum_safe_z_m, and rise to release_z_m
    before contributing one character. Horizontal sweeps below contact do not
    initiate strokes. The approximation treats successful fake moves as straight
    segments; it does not model force, key travel, inertia or continuous sensors.
    """

    def __init__(self, config):
        fields = {'protocol', 'keys', 'initial_xyz_m', 'contact_z_m', 'registration_z_m',
                  'release_z_m', 'minimum_safe_z_m', 'key_radius_m'}
        if not isinstance(config, dict) or set(config) != fields or config['protocol'] != CONFIG_PROTOCOL:
            raise ValueError('Exact synthetic keyboard configuration fields required')
        keys = config['keys']
        if not isinstance(keys, dict) or not keys or len(keys) > 27:
            raise ValueError('Synthetic keyboard needs 1..27 keys')
        normalized = dict(protocol=CONFIG_PROTOCOL, keys={}, initial_xyz_m=_vector(config['initial_xyz_m'], 3))
        for key, xy in keys.items():
            if not isinstance(key, str) or len(key) != 1 or key not in 'abcdefghijklmnopqrstuvwxyz ':
                raise ValueError('Keys must be individual lowercase English letters or space')
            normalized['keys'][key] = _vector(xy, 2)
        for name in fields - {'protocol', 'keys', 'initial_xyz_m'}:
            normalized[name] = _number(config[name])
        low, register, contact, release = [normalized[name] for name in (
            'minimum_safe_z_m', 'registration_z_m', 'contact_z_m', 'release_z_m')]
        if not low < register < contact < release <= normalized['initial_xyz_m'][2]:
            raise ValueError('Synthetic height thresholds must be strictly ordered below initial clearance')
        if normalized['key_radius_m'] <= 0:
            raise ValueError('Key radius must be positive')
        positions = list(normalized['keys'].values())
        if any(math.dist(a, b) <= 2 * normalized['key_radius_m']
               for i, a in enumerate(positions) for b in positions[i + 1:]):
            raise ValueError('Synthetic keycaps cannot overlap or touch')
        # Copy every nested value; mutating the caller's profile/config later
        # cannot change this host evaluator's geometry or acceptance conditions.
        self._config = json.loads(canonical(normalized))
        self._config_sha256 = digest(canonical(self._config))

    @property
    def config_sha256(self):
        return self._config_sha256

    def public_description(self):
        """Known setup and safety information; omits the registration threshold."""
        c = self._config
        return json.loads(canonical(dict(protocol='synthetic-keyboard-public-v1', evidence_mode='synthetic',
                    keys=c['keys'], initial_xyz_m=c['initial_xyz_m'], contact_z_m=c['contact_z_m'],
                    release_z_m=c['release_z_m'], minimum_safe_z_m=c['minimum_safe_z_m'],
                    key_radius_m=c['key_radius_m'],
                    registration_rule='Unknown depth; use actual trajectory and registration feedback.',
                    stroke_rule='Descend while aligned; keep the same key alignment until release. '
                                'Move between keys at clearance. Unsafe depth invalidates the trial.',
                    evaluator_config_sha256=self.config_sha256)))

    def evaluate(self, report):
        if not isinstance(report, dict) or report.get('evidence_mode') != 'synthetic':
            raise ValueError('Synthetic keyboard accepts only explicit synthetic reports')
        receipts = report.get('receipts')
        if not isinstance(receipts, list) or len(receipts) > 1024:
            raise ValueError('Synthetic evaluation needs a bounded raw receipt sequence')
        c = self._config
        initial = list(c['initial_xyz_m'])
        initial_source = 'frozen_host_config'
        for entry in receipts:
            if not isinstance(entry, dict):
                raise ValueError('Malformed raw receipt entry')
            result = entry.get('result')
            if entry.get('method') == 'move_tcp':
                break
            if entry.get('method') == 'observe' and isinstance(result, dict) and result.get('status') == 'ok':
                observation = result.get('observation')
                if (not isinstance(observation, dict) or observation.get('evidence_mode') != 'synthetic'
                        or not isinstance(observation.get('pose'), dict)):
                    raise ValueError('Initial source observation must contain a synthetic pose')
                initial = _vector(observation['pose'].get('xyz_m'), 3)
                initial_source = 'source_observation'
                break

        previous = initial
        minimum_z = initial[2]
        unsafe = initial[2] < c['minimum_safe_z_m']
        unknown = 0
        ignored = 0
        duplicates = 0
        accepted = []
        seen = {}
        strokes = []
        skipped_contacts = []
        active = None
        text = []

        def aligned(xyz, key):
            return math.dist(xyz[:2], c['keys'][key]) <= c['key_radius_m']

        for index, entry in enumerate(receipts):
            if not isinstance(entry, dict):
                raise ValueError('Malformed raw receipt entry')
            if entry.get('method') != 'move_tcp':
                continue
            result = entry.get('result')
            if not isinstance(result, dict) or result.get('status') != 'ok' or result.get('executed') is not True:
                ignored += 1
                if isinstance(result, dict) and result.get('status') == 'unknown':
                    unknown += 1
                elif not isinstance(result, dict) or 'error' in entry:
                    unknown += 1
                continue
            if result.get('evidence_mode') != 'synthetic' or result.get('method', 'move_tcp') != 'move_tcp':
                raise ValueError('Acknowledged move must be explicitly synthetic and method-consistent')
            request_id = identifier(result.get('request_id'))
            fingerprint = digest(canonical(dict(arguments=entry.get('arguments'), result=result)))
            if request_id in seen:
                if seen[request_id] != fingerprint:
                    raise ValueError('Conflicting receipts for one executed request')
                duplicates += 1
                continue
            seen[request_id] = fingerprint
            if not isinstance(result.get('pose'), dict):
                raise ValueError('Acknowledged move is missing the executed TCP pose')
            xyz = _vector(result['pose'].get('xyz_m'), 3)
            accepted.append(dict(receipt_index=index, request_id=request_id, xyz_m=xyz))
            minimum_z = min(minimum_z, xyz[2])
            unsafe = unsafe or xyz[2] < c['minimum_safe_z_m']
            if active is None and previous[2] > c['contact_z_m'] >= xyz[2]:
                key = next((key for key in c['keys'] if aligned(previous, key) and aligned(xyz, key)), None)
                if key is not None:
                    active = dict(key=key, start_receipt_index=index, minimum_z_m=xyz[2],
                                  registered=False, released=False, counted=False, invalid_reasons=[])
                else:
                    skipped_contacts.append(dict(receipt_index=index, reason='unaligned_or_lateral_contact_crossing'))
            if active is not None:
                active['minimum_z_m'] = min(active['minimum_z_m'], xyz[2])
                if not aligned(previous, active['key']) or not aligned(xyz, active['key']):
                    if 'lateral_escape' not in active['invalid_reasons']:
                        active['invalid_reasons'].append('lateral_escape')
                if xyz[2] < c['minimum_safe_z_m'] and 'unsafe_depth' not in active['invalid_reasons']:
                    active['invalid_reasons'].append('unsafe_depth')
                if xyz[2] <= c['registration_z_m']:
                    active['registered'] = True
                if xyz[2] >= c['release_z_m']:
                    active.update(released=True, end_receipt_index=index)
                    active['counted'] = active['registered'] and not active['invalid_reasons']
                    if active['counted']:
                        text.append(active['key'])
                    strokes.append(active)
                    active = None
            previous = xyz

        if active is not None:
            strokes.append(active)
        valid_text = ''.join(text)
        observed = valid_text if not unsafe and not unknown else ''
        evidence = dict(protocol=EVALUATION_PROTOCOL, evidence_mode='synthetic',
                        physical_or_ocr=False, evaluator_config_sha256=self.config_sha256,
                        observed_text=observed, valid_stroke_text=valid_text,
                        actual_minimum_z_m=minimum_z, initial_xyz_m=initial,
                        initial_pose_source=initial_source, executed_move_count=len(accepted),
                        ignored_move_count=ignored, duplicate_receipt_count=duplicates,
                        unknown_move_count=unknown, trajectory_complete=unknown == 0,
                        safety_violation=unsafe, strokes=strokes, skipped_contacts=skipped_contacts,
                        registered_strokes=sum(s['registered'] for s in strokes),
                        released_strokes=sum(s['released'] for s in strokes),
                        source_path_sha256=digest(canonical(dict(initial_xyz_m=initial, moves=accepted))),
                        ignored_candidate_claims=True)
        return {'observed_text': observed, 'evidence': canonical(evidence)}

    def evaluate_completed_trial(self, report):
        """Additional fixed software gate: all strokes released and back home.

        ``evaluate`` measures literal virtual text only. An otherwise correct
        string followed by a held key must not promote a complete typing skill.
        Keep the literal text in evidence and raise rather than falsifying it.
        No real-world release or HOME tolerance follows from this fixture.
        """
        result = self.evaluate(report)
        evidence = json.loads(result['evidence'])
        final = evidence['initial_xyz_m']
        for entry in report['receipts']:
            receipt = entry.get('result', {})
            if (entry.get('method') == 'move_tcp' and receipt.get('status') == 'ok'
                    and receipt.get('executed') is True):
                final = receipt['pose']['xyz_m']
        reasons = []
        if math.dist(evidence['initial_xyz_m'], self._config['initial_xyz_m']) > 1e-5:
            reasons.append('initial_pose_mismatch')
        if math.dist(final, self._config['initial_xyz_m']) > 1e-5:
            reasons.append('final_pose_not_home')
        if any(not stroke['released'] for stroke in evidence['strokes']):
            reasons.append('unreleased_stroke')
        if evidence['safety_violation'] or not evidence['trajectory_complete']:
            reasons.append('unsafe_or_unknown_path')
        evidence.update(final_xyz_m=final, completion_valid=not reasons, completion_reasons=reasons,
                        completion_gate='all strokes released; synthetic HOME within 1e-5m; known safe path')
        result['evidence'] = canonical(evidence)
        if reasons:
            raise IncompleteProgram(reasons, result['evidence'])
        return result
