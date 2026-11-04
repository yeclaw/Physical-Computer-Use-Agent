"""Actual Codex candidate generation; execution/feedback remain host-owned.

This client is not an arbitrary-code sandbox for Codex itself. The separate
bounded Python worker validates and executes the returned source. Only final
candidate text and auditable metadata are saved, never model reasoning items.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from astra_robotics_agent.codex import CodexSession
from astra_robotics_agent.physical_keyboard.records.contracts import Usage, aggregate_usage, canonical, digest

MODEL = 'gpt-6-astra'
EFFORT = 'xhigh'
SERVICE_TIER = 'default'
PROTOCOL = 'physical-keyboard-astra-loop-v1'

INSTRUCTIONS = '''You generate a bounded Python skill for a host-owned synthetic keyboard experiment.
No real robot, computer input, camera, network or hardware is available or authorized.
Reply only with one JSON object with exactly two string keys:
{"summary": "brief observable change and intended effect", "skill_source": "complete skill.py source"}.
Do not use shell, file, browser, app, search or other built-in tools. No tool use is needed.
You cannot change the evaluator, authority, safety limits, other files or trial outcomes.
The host freezes your source, runs a fresh bounded worker against the fake RPC server,
then returns raw actual pose receipts and independently computed synthetic text feedback.
Treat your own returned success labels as unverified. Learn from actual feedback.
Do not claim real-world success or provide internal reasoning. Keep the summary concise.
The same conversation is used for repair; previous unsuccessful candidates remain recorded.
'''


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def native_usage(value):
    """Unknown native fields stay unknown; subsets are not added to totals."""
    if not isinstance(value, dict):
        return Usage(None, None, None, None)
    fields = {}
    for field, key in [('input_tokens', 'inputTokens'),
                       ('cached_input_tokens', 'cachedInputTokens'),
                       ('output_tokens', 'outputTokens'),
                       ('reasoning_output_tokens', 'reasoningOutputTokens')]:
        item = value.get(key)
        if item is not None and (type(item) is not int or item < 0):
            raise ValueError('Invalid native usage field: ' + key)
        fields[field] = item
    return Usage(**fields)


def decode_candidate(text):
    """Strict JSON, exact source artifact and bounded size, no fence guessing."""
    if not isinstance(text, str) or len(text.encode('utf-8')) > 100_000:
        raise ValueError('Candidate reply is missing or too large')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate candidate JSON key')
            value[key] = item
        return value

    value = json.loads(text, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value, dict) or set(value) != {'summary', 'skill_source'}:
        raise ValueError('Candidate must contain exactly summary and skill_source')
    if not isinstance(value['summary'], str) or not 1 <= len(value['summary']) <= 2000:
        raise ValueError('Candidate summary must be a short nonempty string')
    source = value['skill_source']
    if not isinstance(source, str) or not source.strip() or len(source.encode('utf-8')) > 60_000:
        raise ValueError('Candidate skill source is missing or too large')
    return value


class AstraSkillBridge:
    """Single-owner persistent conversation, with every attempted call recorded.

    No token cap is imposed here. The caller observes usage_updated in Codex's
    log plus per-call/cumulative usage here and stops when acceptance is done or
    progress/service anomalies require diagnosis. No automatic transport retry.
    """

    def __init__(self, workspace, artifact_root, *, timeout_s=600,
                 session_factory=CodexSession):
        self.root = Path(artifact_root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.session = session_factory(cwd=Path(workspace), tools=[], instructions=INSTRUCTIONS,
                                      model=MODEL, effort=EFFORT, service_tier=SERVICE_TIER,
                                      disable_execution_tools=True,
                                      log_dir=self.root / 'runtime', timeout=timeout_s)
        self.calls = []
        self.started = False

    def start(self):
        metadata = self.session.start()
        if metadata.get('model') != MODEL or metadata.get('effort') != EFFORT:
            self.session.close()
            raise ValueError('Requested Astra/xhigh was not preserved')
        # The CLI echo confirms its configuration, not an unreported billing tier.
        _new_json(self.root / 'session.json', {
            'protocol': PROTOCOL, 'started_at': _now(), 'metadata': metadata,
            'requested_service_tier': SERVICE_TIER, 'token_limit': None,
            'server_billing_tier': 'unknown_unless_explicitly_reported',
            'codex_builtin_isolation': 'workspace sandbox; feature disables requested; unexpected builtin event abort; not complete isolation',
            'model_reasoning_saved': False,
        })
        self.started = True
        return metadata

    def _append(self, event):
        with (self.root / 'calls.jsonl').open('ab') as stream:
            stream.write(canonical(event) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    @property
    def usage(self):
        return aggregate_usage([c['usage'] for c in self.calls])

    @staticmethod
    def _unexpected_tool(name, arguments):
        raise ValueError('Candidate generation cannot invoke dynamic tools')

    def propose(self, context):
        if not self.started:
            raise RuntimeError('Start the session before requesting a candidate')
        number = len(self.calls) + 1
        directory = self.root / ('call-%03d' % number)
        directory.mkdir(exist_ok=False)
        prompt = canonical(context).decode('utf-8')
        request = {'protocol': PROTOCOL, 'number': number, 'started_at': _now(),
                   'context': context, 'model': MODEL, 'effort': EFFORT,
                   'requested_service_tier': SERVICE_TIER, 'token_limit': None,
                   'thread_id': self.session.thread_id}
        _new_json(directory / 'request.json', request)
        self._append({'event': 'call_started', 'number': number,
                      'request_sha256': digest(canonical(request)), 'time': _now()})
        reply = None
        event = {'number': number, 'status': 'failed', 'started_at': request['started_at'],
                 'usage': asdict(Usage(None, None, None, None))}
        try:
            reply = self.session.run_turn(prompt, self._unexpected_tool)
            # run_turn exposes only final text/metadata, not hidden reasoning.
            _new_json(directory / 'reply.json', reply)
            event['usage'] = asdict(native_usage(reply.get('usage')))
            event['turn_id'] = reply.get('turn_id')
            if reply.get('model') != MODEL or reply.get('effort') != EFFORT:
                raise ValueError('Reply changed the requested model or effort')
            if reply.get('tool_calls') != 0:
                raise ValueError('Candidate generation unexpectedly used dynamic tools')
            candidate = decode_candidate(reply.get('final_text'))
            _new_json(directory / 'candidate.json', candidate)
            with (directory / 'skill.py').open('xb') as stream:
                stream.write(candidate['skill_source'].encode('utf-8'))
            event.update(status='candidate', source_sha256=digest(candidate['skill_source'].encode('utf-8')))
            return {**candidate, 'source_sha256': event['source_sha256'],
                    'call_number': number, 'usage': Usage(**event['usage'])}
        except BaseException as exc:
            event['error_type'] = type(exc).__name__
            # A transport-failed turn may have unreported usage beyond these
            # counters. Preserve the lower bound without asserting completeness.
            event['last_observed_session_native_usage'] = dict(self.session.total_usage)
            _new_json(directory / 'error.json', event)
            raise
        finally:
            event['finished_at'] = _now()
            self.calls.append(event)
            self._append({'event': 'call_finished', **event, 'cumulative_usage': self.usage})

    def close(self):
        self.session.close()
