"""Device-independent bounded programs and auditable, non-replayed receipts.

The adapter is trusted Python. This interpreter is not an OS sandbox; real
hardware limits, coordinate transforms and physical stop semantics live in the
adapter. An acknowledged command is never promoted to task success here.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path
import threading

from .control_protocol import Audit, anchor, digest, encoded, write_new
from .device_adapter import AdmissionRejected, CallReceipt, Observation, PreparedCall, json_copy
from .skill_language import SkillError, ToolError, run_skill
from ..records.redaction import redact_text


def clone(value):
    return json_copy(value)


class _ZeroDispatchRejection(Exception):
    pass


class ProgramRuntime:
    """Single owner. No reconnect, automatic retry, implicit HOME or reset.

    observe/execute return evidence files; the Agent runner registers them in
    its knowledge store. The Runtime never imports that store or the Agent.
    """

    def __init__(self, adapter, output, *, run_id):
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.adapter = adapter
        self.evidence_prefix = 'ev-' + digest(run_id.encode())[:12]
        self.description = clone(adapter.describe().as_dict())
        self._lock = threading.RLock()
        self.audit = Audit(self.root / 'events.jsonl', run_id)
        self.counter = 0
        self.uncertain = self.closed = False
        self.cleanup = None
        self.audit('runtime_started', adapter=self.description)

    def _record(self, kind, value):
        self.counter += 1
        ident = '%s-%s-%06d' % (self.evidence_prefix, kind, self.counter)
        folder = self.root / ident
        folder.mkdir()
        path = folder / 'record.json'
        write_new(path, encoded(value), readonly=True)
        return ident, folder, [path]

    def observe(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('Runtime is closed')
            observation = self.adapter.observe()
            if not isinstance(observation, Observation):
                raise TypeError('Adapter.observe must return Observation')
            state, items = clone(observation.state), clone(observation.content_items)
            if not isinstance(state, dict) or not isinstance(items, list):
                raise ValueError('Observation needs object state and list content_items')
            self.counter += 1
            ident = '%s-observation-%06d' % (self.evidence_prefix, self.counter)
            folder = self.root / ident
            folder.mkdir()
            files, images = [], []
            # Persist supplied pixels without assuming a head/left/right camera.
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError('Invalid observation content item')
                if item.get('type') == 'inputText' and isinstance(item.get('text'), str):
                    continue
                if item.get('type') != 'inputImage':
                    raise ValueError('Only inputText/inputImage observations are supported')
                url = item.get('imageUrl')
                if not isinstance(url, str) or ',' not in url:
                    raise ValueError('Observation image must be a data URL')
                header, payload = url.split(',', 1)
                extensions = {'data:image/jpeg;base64': 'jpg', 'data:image/png;base64': 'png'}
                if header not in extensions or len(payload) > 6 * 1024 * 1024:
                    raise ValueError('Unsupported/oversized observation image')
                raw = base64.b64decode(payload, validate=True)
                path = folder / ('image-%02d.' % index + extensions[header])
                write_new(path, raw, readonly=True)
                files.append(path)
                view_id = item.get('view_id', 'image_%02d' % index)
                if view_id not in self.description['views']:
                    raise ValueError('Each inputImage requires a declared view_id')
                images.append(dict(path=path.name, sha256=digest(raw), view_id=view_id))
            path = folder / 'record.json'
            write_new(path, encoded(dict(protocol='physical-agent-observation-v1', anchor=anchor(),
                adapter=self.description, state=state, content_items=items, images=images)), readonly=True)
            files.insert(0, path)
            items.insert(0, dict(type='inputText', text=encoded(dict(
                adapter=self.description, current_state=state,
                observation_evidence_mode=self.description['evidence_mode'])).decode()))
            self.audit('observation_recorded', evidence_id=ident, state_sha256=digest(encoded(state)))
            return dict(evidence_id=ident, contentItems=items, evidence_files=files, state=state)

    def _stop_after_unknown(self):
        self.uncertain = True
        try:
            receipt = self.adapter.stop()
            if not isinstance(receipt, CallReceipt):
                raise TypeError('Adapter.stop must return CallReceipt')
            result = receipt.as_dict()
        except BaseException as exc:
            result = dict(status='unknown', dispatch_state='unknown', error=redact_text(repr(exc)))
        self.audit('stop_attempt', receipt=result, physical_release_verified=False)
        return result

    def execute(self, arguments):
        with self._lock:
            if self.closed or self.uncertain:
                raise RuntimeError('Runtime closed/unknown; no action is replayed')
            if (type(arguments) is not dict or set(arguments) != {'source', 'profile', 'label'}
                    or type(arguments['source']) is not str or type(arguments['profile']) is not dict):
                raise ValueError('execute requires source, profile and label')
            source, profile = arguments['source'], clone(arguments['profile'])
            ident, folder, files = self._record('execution', dict(
                protocol='physical-agent-execution-v1', anchor=anchor(), label=arguments['label'], source_sha256=digest(source.encode()),
                profile_sha256=digest(encoded(profile)), adapter=self.description))
            for name, raw in [('skill.py', source.encode()), ('profile.json', encoded(profile))]:
                path = folder / name
                write_new(path, raw, readonly=True)
                files.append(path)
            receipts = []
            stop_attempts = []

            def tool(method, values):
                entry = dict(method=method, arguments=clone(values), started_anchor=anchor(),
                             status='not_started', motion_dispatch_attempted=False)
                receipts.append(entry)
                self.audit('tool_requested', execution_id=ident, receipt=entry)
                try:
                    if method == 'observe':
                        actual = self.observe()
                        result = {**actual['state'], 'status': 'ok'}
                        entry.update(status='ok', dispatch_state='completed', result=result,
                                     observation_evidence_id=actual['evidence_id'])
                        return result
                    if method == 'stop':
                        entry['motion_dispatch_attempted'] = True
                        receipt = self.adapter.stop()
                    else:
                        # Only this explicit pure-preflight rejection can become
                        # a catchable ToolError. Arbitrary adapter errors cannot.
                        prepared = self.adapter.prepare(method, clone(values))
                        if not isinstance(prepared, PreparedCall):
                            raise TypeError('Adapter.prepare must return PreparedCall')
                        if prepared.method != method or clone(prepared.arguments) != values:
                            raise ValueError('Prepared call changed the requested method/arguments')
                        entry['motion_dispatch_attempted'] = True
                        self.audit('dispatch_started', execution_id=ident, receipt=entry)
                        frozen = PreparedCall(prepared.method, clone(prepared.arguments), prepared.token)
                        receipt = self.adapter.dispatch(frozen)
                    if not isinstance(receipt, CallReceipt):
                        raise TypeError('Adapter.dispatch/stop must return CallReceipt')
                    data = clone(receipt.as_dict())
                    entry.update(data)
                    if data['status'] == 'unknown':
                        raise RuntimeError('Adapter reported unknown execution outcome')
                    if data['status'] == 'rejected':
                        entry['motion_dispatch_attempted'] = False
                        raise _ZeroDispatchRejection(str(data['result']))
                    return {**data['result'], 'status': 'ok'}
                except AdmissionRejected as exc:
                    if entry['motion_dispatch_attempted']:
                        entry.update(status='unknown', dispatch_state='unknown', error=redact_text(repr(exc)))
                        stop_attempts.append(self._stop_after_unknown())
                        raise SkillError('AdmissionRejected raised after dispatch; outcome unknown') from exc
                    entry.update(status='rejected', dispatch_state='not_started', error=redact_text(str(exc)))
                    raise ToolError(redact_text(str(exc))) from exc
                except _ZeroDispatchRejection as exc:
                    raise ToolError(redact_text(str(exc))) from exc
                except BaseException as exc:
                    entry.update(status='unknown' if entry['motion_dispatch_attempted'] else 'error', error=redact_text(repr(exc)))
                    # Even a broken read/preflight makes this adapter instance
                    # unhealthy. A successful stop cannot erase uncertainty.
                    stop_attempts.append(self._stop_after_unknown())
                    raise SkillError('Adapter failure; runtime latched closed to further motion') from exc
                finally:
                    entry['ended_anchor'] = anchor()
                    self.audit('tool_finished', execution_id=ident, receipt=entry)

            try:
                names = ('observe', 'move_tcp', 'wait_bounded', 'stop')
                tools = {name: (lambda name=name, **kw: tool(name, kw)) for name in names}
                result = run_skill([('skill.py', source)], profile=profile, tools=tools)
                response = dict(status='ok', result=dict(model_report=result, receipts=receipts,
                    motion_outcome_uncertain=False, task_success_verified=False))
            except BaseException as exc:
                interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
                if interrupted and not self.uncertain:
                    stop_attempts.append(self._stop_after_unknown())
                # A synchronous completed call remains completed when the
                # interpreter later rejects an expression. The next model turn
                # receives fresh observations and chooses a NEW remaining plan.
                partial_error = bool(receipts) and not self.uncertain and not isinstance(exc, ToolError)
                response = dict(status='error' if self.uncertain or partial_error else 'blocked', error=redact_text(repr(exc)), result=dict(
                    receipts=receipts, stop_attempts=stop_attempts,
                    motion_outcome_uncertain=self.uncertain, controller_terminal=self.uncertain,
                    recoverable_program_error=partial_error,
                    task_success_verified=False))
                if interrupted:
                    self.audit('execution_interrupted', execution_id=ident, response=response)
                    write_new(folder / 'response.json', encoded(response), readonly=True)
                    raise
            path = folder / 'response.json'
            write_new(path, encoded(response), readonly=True)
            files.append(path)
            self.audit('execution_finished', execution_id=ident, response=response)
            return dict(evidence_id=ident, response=response, evidence_files=files)

    def close(self):
        with self._lock:
            if self.closed:
                return self.cleanup
            self.closed = True
            try:
                result = self.adapter.close()
                self.cleanup = dict(status='closed', adapter_result=clone(result),
                                    motion_outcome_uncertain=self.uncertain,
                                    physical_release_verified=False)
            except BaseException as exc:
                self.cleanup = dict(status='error', error=redact_text(repr(exc)), motion_outcome_uncertain=True,
                                    physical_release_verified=False)
            self.audit('runtime_closed', cleanup=self.cleanup)
            self.audit.close()
            return self.cleanup
