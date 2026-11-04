"""Versioned, hardware-independent contracts for auditable keyboard adaptation."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

PROTOCOL = 'physical-keyboard-rsi-ledger-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', value):
        raise ValueError('Invalid identifier')
    return value


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Expected lowercase SHA-256')
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


@dataclass(frozen=True)
class Scope:
    evidence_mode: str
    device_id: str
    layout_id: str
    configuration_sha256: str
    protocol_sha256: str
    evaluator_sha256: str
    goal_sha256: str
    safety_sha256: str

    def __post_init__(self):
        if self.evidence_mode not in {'synthetic', 'physical'}:
            raise ValueError('Scope must separate synthetic and physical evidence')
        identifier(self.device_id)
        identifier(self.layout_id)
        for name, value in asdict(self).items():
            if name.endswith('_sha256'):
                sha256(value)

    @property
    def key(self):
        return digest(canonical(asdict(self)))


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_output_tokens: int | None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None:
                integer(value, name)
        for child, parent in [(self.cached_input_tokens, self.input_tokens),
                              (self.reasoning_output_tokens, self.output_tokens)]:
            if child is not None and (parent is None or child > parent):
                raise ValueError('Token subset exceeds or has unknown parent')

    @classmethod
    def zero(cls):
        """Explicitly measured no-model invocation, not a substitute for unknown."""
        return cls(0, 0, 0, 0)

    @property
    def complete(self):
        return all(v is not None for v in asdict(self).values())


@dataclass(frozen=True)
class EvaluationPlan:
    cases: dict[str, str]
    repetitions: int
    min_success_gain: int
    max_total_tokens: int | None

    def __post_init__(self):
        if not isinstance(self.cases, dict) or not self.cases:
            raise ValueError('At least one frozen comparison case is required')
        for case_id, text in self.cases.items():
            identifier(case_id)
            if not isinstance(text, str) or not text or any(c not in 'abcdefghijklmnopqrstuvwxyz ' for c in text):
                raise ValueError('v1 comparison targets are nonempty lowercase English and spaces')
        integer(self.repetitions, 'repetitions', 1)
        integer(self.min_success_gain, 'min_success_gain', 1)
        # None is explicit unlimited authorization, never an unknown cost.
        # Existing finite limits retain exactly their previous semantics.
        if self.max_total_tokens is not None:
            integer(self.max_total_tokens, 'max_total_tokens')
        if self.min_success_gain > len(self.cases) * self.repetitions:
            raise ValueError('Requested gain exceeds scheduled pairs')


def aggregate_usage(usages):
    usages = [Usage(**u) if isinstance(u, dict) else u for u in usages]
    fields = {}
    for name in Usage.__dataclass_fields__:
        values = [getattr(u, name) for u in usages]
        fields[name] = None if any(v is None for v in values) else sum(values)
    complete = all(u.complete for u in usages)
    total = (fields['input_tokens'] + fields['output_tokens']
             if fields['input_tokens'] is not None and fields['output_tokens'] is not None else None)
    return {**fields, 'complete': complete, 'total_input_plus_output': total}
