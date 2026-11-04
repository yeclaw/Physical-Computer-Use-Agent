"""Pinned occurrence-prefix boundaries; no I/O, policy, or robot dependency.

The original generated manifest remains the task and evaluation source. This
gate holds only identity metadata after validating that manifest. Reaching a
boundary ends a selected schedule; it says nothing about task correctness or
completion of a continuous training objective.
"""

from .typing_cases import manifest_sha256, validate_manifest
from astra_robotics_agent.physical_keyboard.records.typing_metrics import validate_case_identity


PROTOCOL = 'physical-keyboard-typing-schedule-prefix-v1'


class TypingSchedulePrefix:
    """Classify one strict identity against a pinned original schedule prefix.

    Checks are stateless: a boundary identity does not assert that every prior
    occurrence was observed or correct. Missing, foreign, unknown, or skipped
    boundary identities fail closed with ValueError, never implicit completion.
    """

    def __init__(self, manifest, *, occurrence_limit, expected_run_id,
                 expected_batch_id, expected_manifest_sha256):
        validated = validate_manifest(manifest)
        occurrences = validated['occurrences']
        if type(occurrence_limit) is not int or not 1 <= occurrence_limit <= len(occurrences):
            raise ValueError('occurrence_limit must be an integer in 1..original occurrence count')
        # Reuse the strict public identity schema to validate every external pin.
        first = occurrences[0]
        validate_case_identity(dict(protocol='physical-keyboard-typing-case-identity-v1',
            run_id=expected_run_id, batch_id=expected_batch_id,
            manifest_sha256=expected_manifest_sha256, **first, phase='typing'))
        if (validated['batch_id'] != expected_batch_id
                or manifest_sha256(validated) != expected_manifest_sha256):
            raise ValueError('schedule manifest batch or SHA-256 differs from expected pin')
        self._run_id = expected_run_id
        self._batch_id = expected_batch_id
        self._manifest_sha256 = expected_manifest_sha256
        self._limit = occurrence_limit
        self._occurrences = tuple(dict(item) for item in occurrences)
        self._positions = {item['occurrence_id']: index for index, item in enumerate(self._occurrences)}

    def describe(self):
        """Return detached, text-free metadata suitable for a frozen run input."""
        return dict(protocol=PROTOCOL, run_id=self._run_id, batch_id=self._batch_id,
            manifest_sha256=self._manifest_sha256, occurrence_limit=self._limit,
            original_occurrence_count=len(self._occurrences),
            selected_occurrence_ids=[item['occurrence_id'] for item in self._occurrences[:self._limit]],
            next_outside_occurrence_id=(self._occurrences[self._limit]['occurrence_id']
                                        if self._limit < len(self._occurrences) else None))

    def check_identity(self, identity):
        """Return an ID-only continue/stop decision or reject uncertain identity.

        A shortened prefix stops only at its exact first excluded occurrence,
        while that occurrence is still typing. A full-manifest prefix stops at
        the last original occurrence's normal batch_complete release boundary.
        No target, editor buffer, evaluation score, or coverage enters a result.
        """
        value = validate_case_identity(identity, expected_run_id=self._run_id)
        if value['batch_id'] != self._batch_id or value['manifest_sha256'] != self._manifest_sha256:
            raise ValueError('case identity belongs to another pinned batch or manifest')
        position = self._positions.get(value['occurrence_id'])
        if position is None:
            raise ValueError('case identity occurrence is not in the original manifest')
        expected = self._occurrences[position]
        if any(value[key] != expected[key] for key in expected):
            raise ValueError('case identity metadata differs from its original occurrence')
        if value['phase'] == 'closed':
            raise ValueError('closed case identity is not an active schedule boundary')
        if position > self._limit:
            raise ValueError('case identity has advanced beyond the exact selected boundary')
        if position == self._limit:
            if value['phase'] != 'typing':
                raise ValueError('first excluded occurrence is past its unsubmitted typing boundary')
            status, stop = 'next_outside_boundary', True
        elif value['phase'] == 'batch_complete':
            if self._limit != len(self._occurrences) or position != self._limit - 1:
                raise ValueError('batch_complete identity is not the selected full-manifest boundary')
            status, stop = 'full_manifest_complete', True
        else:
            status, stop = 'inside_prefix', False
        return dict(status=status, should_stop=stop, original_index=position,
            occurrence_limit=self._limit, original_occurrence_count=len(self._occurrences))
