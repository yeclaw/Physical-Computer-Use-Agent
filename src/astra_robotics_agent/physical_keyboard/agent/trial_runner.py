"""Trusted synthetic-only coordinator: durable start before any worker authority.

Workers execute frozen declarations or a resource-bounded Python subset. A
completed motion sequence never implies a keypress or an observed screen string.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import sys
import time
import uuid

from astra_robotics_agent.physical_keyboard.records.contracts import Scope, Usage, canonical, digest, identifier, integer
from astra_robotics_agent.physical_keyboard.records.ledger import LedgerError
from astra_robotics_agent.physical_keyboard.runtime.executable import resolve_execution
from astra_robotics_agent.physical_keyboard.runtime.transport import JsonProcess


class RecoveryRequired(LedgerError):
    """A prior start has no terminal outcome; never silently execute it again."""


class CoordinatorBusy(LedgerError):
    pass


class WorkerProtocolError(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class TrialRunner:
    """Host-owned bridge between ledger, one RPC session and fresh trial workers."""

    def __init__(self, ledger, rpc, artifact_root):
        self.ledger = ledger
        self.rpc = rpc
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _exclusive(self):
        # The lock belongs to the ledger, so a different artifact directory does
        # not permit another trusted coordinator to dispatch the same trial.
        with (self.ledger.root / 'execution.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CoordinatorBusy('Another coordinator owns this ledger') from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _match_schedule(state, requested):
        if any(state.get(name) != value for name, value in requested.items()):
            raise LedgerError('Existing trial has different scheduling arguments')

    def run(self, round_id, trial_id, variant, case_id, repetition, condition_id,
            timeout_s=5, evaluator=None):
        """Run one frozen synthetic program; optional evaluator is trusted host code.

        ``evaluator(report)`` must return literal ``observed_text`` and nonempty
        evidence bytes. It is called only after a fully completed worker sequence.
        The default result is uncertain even when every movement completed.
        """
        for value in (round_id, trial_id, case_id, condition_id):
            identifier(value)
        integer(repetition, 'repetition')
        if variant not in {'parent', 'candidate'}:
            raise ValueError('Variant must be parent or candidate')
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60):
            raise ValueError('Worker deadline must be finite and in (0, 60] seconds')
        if evaluator is not None and not callable(evaluator):
            raise TypeError('Evaluator must be callable or None')
        requested = dict(variant=variant, case_id=case_id, repetition=repetition,
                         condition_id=condition_id)
        with self._exclusive():
            existing = self.ledger.trial_state(round_id, trial_id)
            if existing is not None:
                self._match_schedule(existing, requested)
                if existing['result'] is not None:
                    return existing['result']
                raise RecoveryRequired('Trial is already in flight; collect receipts or explicitly abort')
            context = self.ledger.execution_context(round_id, variant)
            if context.get('inflight_trials'):
                raise RecoveryRequired('Another trial in this scope is in flight')
            if context['round_decided']:
                raise LedgerError('Cannot execute a decided comparison round')
            bundle = context['bundle']
            scope = Scope(**bundle['scope'])
            if scope.evidence_mode != 'synthetic':
                raise LedgerError('This coordinator only accepts synthetic fixtures')
            try:
                executable = resolve_execution(bundle, self.ledger.read_object)
                program_ref = executable['program_sha256']
                profile_ref = bundle['files']['profile.json']['sha256']
            except KeyError as exc:
                raise LedgerError('Frozen bundle is missing executable inputs') from exc

            execution_id = 'execution-' + uuid.uuid4().hex
            directory = self.artifact_root / (round_id + '-' + trial_id + '-' + execution_id)
            directory.mkdir(exist_ok=False)
            report = dict(protocol='physical-keyboard-trial-receipt-v1', evidence_mode='synthetic',
                          round_id=round_id, trial_id=trial_id, **requested,
                          execution_id=execution_id, bundle_id=context['bundle_id'],
                          scope_key=context['scope_key'], profile_id=profile_ref,
                          program_sha256=program_ref, execution_kind=executable['kind'],
                          program_file=executable['file_name'],
                          frozen_runtime_sha256={name: digest(content) for name, content in executable['sources'].items()},
                          started_at=_now(),
                          coordinator_pid=os.getpid(), server_pid=getattr(self.rpc, 'pid', None),
                          artifact_directory=str(directory.resolve()), messages=[], receipts=[], errors=[],
                          worker=None, motion_completed=False, text_verified=False)
            _write_new(directory / 'audit.jsonl', b'')
            audit_count = 0

            def record(kind, payload):
                nonlocal audit_count
                event = dict(sequence=audit_count + 1, timestamp=_now(), kind=kind, payload=payload)
                with (directory / 'audit.jsonl').open('ab') as stream:
                    stream.write(canonical(event) + b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                audit_count += 1

            record('coordinator_started', {k: report[k] for k in (
                'round_id', 'trial_id', 'execution_id', 'bundle_id', 'scope_key', 'program_sha256',
                'profile_id', 'coordinator_pid', 'server_pid')})
            deadline = time.monotonic() + timeout_s

            def remaining():
                value = deadline - time.monotonic()
                if value <= 0:
                    raise TimeoutError('Trial worker deadline expired')
                return value

            def call(method, arguments, *, cleanup=False):
                entry = dict(method=method, arguments=arguments, sent_at=_now())
                report['receipts'].append(entry)
                # Each outgoing request is durably recorded before dispatch; a
                # coordinator crash cannot erase the request or its owned PID.
                record('rpc_sent', entry)
                try:
                    receipt = self.rpc.call(method, arguments,
                                            timeout_s=0.5 if cleanup else remaining())
                    if not isinstance(receipt, dict) or receipt.get('status') not in {'ok', 'rejected', 'unknown'}:
                        raise WorkerProtocolError('RPC returned a malformed receipt')
                    entry['result'] = receipt
                    entry['received_at'] = _now()
                    record('rpc_received', entry)
                    return receipt
                except Exception as exc:
                    entry['error'] = dict(type=type(exc).__name__, message=str(exc))
                    entry['received_at'] = _now()
                    try:
                        record('rpc_error', entry)
                    except OSError as audit_error:
                        report['errors'].append(dict(stage='audit_write', type=type(audit_error).__name__,
                                                     message=str(audit_error)))
                    raise
                finally:
                    entry['received_at'] = _now()

            initial = None
            try:
                initial = call('observe', {})
            except Exception as exc:
                report['errors'].append(dict(stage='initial_observation', type=type(exc).__name__, message=str(exc)))
            before = self.ledger.put_evidence(canonical(dict(evidence_mode='synthetic',
                    observation_receipt=initial, errors=list(report['errors']), bundle_id=context['bundle_id'])))
            # No authorization or worker is created before this transaction commits.
            try:
                self.ledger.start_trial(round_id, trial_id, **requested, before_evidence=[before])
            except Exception as exc:
                report['errors'].append(dict(stage='ledger_start', type=type(exc).__name__, message=str(exc)))
                report['finished_at'] = _now()
                _write_new(directory / 'start-error.json', canonical(report))
                raise
            started = self.ledger.trial_state(round_id, trial_id)
            if started is None or started['result'] is not None:
                raise RecoveryRequired('Durable unfinished trial start is missing')
            report['start_event_sha256'] = started['start_event_sha256']
            start_anchor = dict(round_id=round_id, trial_id=trial_id,
                                start_event_sha256=started['start_event_sha256'],
                                bundle_id=context['bundle_id'], scope_key=context['scope_key'], **requested)
            _write_new(directory / 'trial-start.json', canonical(start_anchor))
            record('trial_started', start_anchor)
            permit = None
            worker = None
            status, observed_text = 'uncertain', None
            evaluator_evidence = None
            need_stop = False
            interruption = None
            try:
                if initial is None or initial['status'] != 'ok':
                    if initial is not None and initial['status'] == 'rejected':
                        status = 'blocked'
                    raise WorkerProtocolError('Initial observation is unavailable')
                observation = initial['observation']
                grant = call('authorize_trial', dict(trial_id=trial_id, round_id=round_id, variant=variant,
                    bundle_id=context['bundle_id'], profile_id=profile_ref, scope_key=context['scope_key'],
                    start_event_sha256=started['start_event_sha256'], program_sha256=program_ref))
                if grant['status'] != 'ok':
                    status = 'blocked' if grant['status'] == 'rejected' else 'uncertain'
                    raise WorkerProtocolError('Trial authorization was not granted')
                permit = grant['permit_id']
                for name, content in executable['sources'].items():
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_new(path, content)
                job = dict(protocol='physical-keyboard-worker-job-v1', scope_key=context['scope_key'],
                           profile_id=profile_ref, permit_id=permit, trial_id=trial_id,
                           observation_id=observation['observation_id'])
                if executable['kind'] == 'bounded-python-v1':
                    job.update(protocol='physical-keyboard-python-worker-job-v1', bundle_id=context['bundle_id'])
                    _write_new(directory / 'bundle.json', self.ledger.read_object(context['bundle_id']))
                    command = [sys.executable, '-m', 'astra_robotics_agent.physical_keyboard.python_worker',
                               '--manifest', str((directory / 'bundle.json').resolve()),
                               '--sha256', context['bundle_id'], '--job', str((directory / 'job.json').resolve())]
                else:
                    command = [sys.executable, '-m', 'astra_robotics_agent.physical_keyboard.worker',
                               '--program', str((directory / 'actions.json').resolve()),
                               '--sha256', program_ref, '--job', str((directory / 'job.json').resolve())]
                _write_new(directory / 'job.json', canonical(job))
                worker = JsonProcess(command, directory / 'worker.stderr')
                report['worker'] = dict(pid=worker.pid, command=command)
                _write_new(directory / 'worker-start.json', canonical(report['worker']))
                record('worker_started', report['worker'])
                ready = False
                expected_index = 0
                for _ in range(132):
                    message = worker.receive(remaining())
                    report['messages'].append(dict(direction='worker_to_host', message=message, received_at=_now()))
                    record('worker_message', report['messages'][-1])
                    if not isinstance(message, dict):
                        raise WorkerProtocolError('Worker message must be an object')
                    kind = message.get('type')
                    if kind == 'ready':
                        if ready or expected_index or message.get('program_sha256') != program_ref:
                            raise WorkerProtocolError('Worker ready/source mismatch')
                        if executable['kind'] == 'bounded-python-v1' and message.get('bundle_id') != context['bundle_id']:
                            raise WorkerProtocolError('Python worker bundle mismatch')
                        ready = True
                    elif kind == 'tool_call':
                        if not ready or message.get('index') != expected_index:
                            raise WorkerProtocolError('Worker call order mismatch')
                        method, arguments = message.get('method'), message.get('arguments')
                        allowed = {'observe': set(), 'move_tcp': {'xyz_m', 'quaternion_xyzw', 'observation_id'},
                                   'wait_bounded': {'seconds'}, 'stop': set()}
                        if (method not in allowed or not isinstance(arguments, dict)
                                or set(arguments) != allowed[method]):
                            raise WorkerProtocolError('Worker tool or authority fields are forbidden')
                        arguments = dict(arguments)
                        if method in {'move_tcp', 'wait_bounded', 'stop'}:
                            arguments['request_id'] = execution_id + '-action-' + str(expected_index)
                        if method in {'move_tcp', 'wait_bounded'}:
                            arguments.update(permit_id=permit, scope_key=context['scope_key'], profile_id=profile_ref)
                        receipt = call(method, arguments)
                        response = dict(type='tool_result', index=expected_index, result=receipt)
                        report['messages'].append(dict(direction='host_to_worker', message=response, sent_at=_now()))
                        record('coordinator_message', report['messages'][-1])
                        worker.send(response, timeout_s=remaining())
                        expected_index += 1
                        if receipt['status'] == 'unknown':
                            need_stop = True
                    elif kind == 'done':
                        result_status = message.get('status')
                        if result_status not in {'completed', 'blocked', 'error'}:
                            raise WorkerProtocolError('Unknown worker completion status')
                        if result_status == 'completed' and (not ready or any(
                                r.get('result', {}).get('status') != 'ok' for r in report['receipts'])):
                            raise WorkerProtocolError('Worker claimed completion after failed actions')
                        report['worker']['done'] = message
                        if 'skill_result' in message:
                            report['skill_result'] = message['skill_result']
                        report['motion_completed'] = result_status == 'completed'
                        if result_status == 'blocked':
                            status = 'uncertain' if need_stop else 'blocked'
                        elif result_status == 'error':
                            # Legacy declaration syntax is fully checked before
                            # its first action. Keep that zero-action rejection
                            # from latching a healthy service stopped. Python can
                            # fail during computation after motion, so every
                            # Python error still receives the host stop path.
                            need_stop = permit is not None and (
                                executable['kind'] == 'bounded-python-v1' or any(
                                    entry['method'] in {'move_tcp', 'wait_bounded'}
                                    for entry in report['receipts']))
                        break
                    else:
                        raise WorkerProtocolError('Unrecognized worker message')
                else:
                    raise WorkerProtocolError('Worker exceeded bounded message count')
                # done is a protocol message, not proof the process exited. Reap
                # its normal exit before cleanup can terminate a lingering child.
                worker.process.wait(timeout=min(remaining(), 0.5))
            except BaseException as exc:
                report['errors'].append(dict(stage='execution', type=type(exc).__name__, message=str(exc)))
                need_stop = permit is not None
                if not isinstance(exc, Exception):
                    interruption = exc
                try:
                    record('execution_error', report['errors'][-1])
                except OSError as audit_error:
                    report['errors'].append(dict(stage='audit_write', type=type(audit_error).__name__,
                                                 message=str(audit_error)))
            finally:
                if worker is not None:
                    try:
                        report['worker']['exit'] = worker.close()
                        record('worker_exit', report['worker']['exit'])
                        if (report['worker']['exit'].get('returncode') != 0
                                or report['worker']['exit'].get('terminate_sent')
                                or report['worker']['exit'].get('kill_sent')):
                            report['motion_completed'] = False
                    except Exception as exc:
                        report['errors'].append(dict(stage='worker_cleanup', type=type(exc).__name__, message=str(exc)))
                        report['motion_completed'] = False
                        need_stop = True
                if need_stop:
                    try:
                        call('stop', {'request_id': execution_id + '-stop'}, cleanup=True)
                    except Exception as exc:
                        report['errors'].append(dict(stage='stop', type=type(exc).__name__, message=str(exc)))
                if permit is not None:
                    try:
                        close = call('close_trial', {'permit_id': permit}, cleanup=True)
                        if close['status'] != 'ok':
                            report['motion_completed'] = False
                    except Exception as exc:
                        report['errors'].append(dict(stage='permit_cleanup', type=type(exc).__name__, message=str(exc)))
                        report['motion_completed'] = False

            if interruption is not None:
                # Preserve the unfinished start for explicit recovery rather than
                # converting a host interruption into an apparently normal trial.
                raise interruption
            if report['motion_completed'] and evaluator is not None:
                try:
                    evaluation = evaluator(report)
                    if (not isinstance(evaluation, dict) or set(evaluation) != {'observed_text', 'evidence'}
                            or not isinstance(evaluation['observed_text'], str)
                            or not isinstance(evaluation['evidence'], bytes) or not evaluation['evidence']):
                        raise ValueError('Evaluator must provide literal text and nonempty evidence bytes')
                    observed_text, evaluator_evidence = evaluation['observed_text'], evaluation['evidence']
                    status = 'observed'
                    report['text_verified'] = True
                except Exception as exc:
                    report['errors'].append(dict(stage='evaluator', type=type(exc).__name__, message=str(exc)))
            report.update(finished_at=_now(), outcome_status=status, observed_text=observed_text,
                          usage=asdict(Usage.zero()))
            record('trial_outcome', dict(status=status, observed_text=observed_text,
                                        motion_completed=report['motion_completed'],
                                        text_verified=report['text_verified'], errors=report['errors']))
            report['audit_events'] = audit_count
            report['artifact_sha256'] = {str(path.relative_to(directory)): digest(path.read_bytes())
                                        for path in directory.rglob('*') if path.is_file()}
            _write_new(directory / 'run-report.json', canonical(report))
            evidence = [self.ledger.put_evidence(canonical(report))]
            for name in sorted(set(executable['sources']) | {
                    'job.json', 'bundle.json', 'worker.stderr', 'audit.jsonl',
                    'trial-start.json', 'worker-start.json'}):
                path = directory / name
                if path.exists():
                    evidence.append(self.ledger.put_evidence(path.read_bytes()))
            if evaluator_evidence is not None:
                evidence.append(self.ledger.put_evidence(evaluator_evidence))
            # A write failure deliberately leaves the durable start unfinished.
            self.ledger.finish_trial(round_id, trial_id, status=status, observed_text=observed_text,
                    after_evidence=evidence, execution_id=execution_id, usage=Usage.zero())
            return self.ledger.trial_state(round_id, trial_id)['result']

    def abort_pending(self, round_id, trial_id, *, reason, evidence, usage=None):
        """Explicitly account an old unfinished attempt, without worker/RPC execution.

        Unknown historical usage remains unknown unless trusted recovery supplies
        its measurement. This does not reconcile an unknown device action journal.
        """
        if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, bytes) or not evidence:
            raise ValueError('Explicit abort reason and nonempty recovery evidence required')
        if usage is None:
            usage = Usage(None, None, None, None)
        if not isinstance(usage, Usage):
            raise TypeError('Recovery usage must be an explicit Usage measurement')
        with self._exclusive():
            state = self.ledger.trial_state(round_id, trial_id)
            if state is None or state['result'] is not None:
                raise RecoveryRequired('Only an unfinished existing trial can be explicitly aborted')
            context = self.ledger.execution_context(round_id, state['variant'])
            if Scope(**context['bundle']['scope']).evidence_mode != 'synthetic':
                raise LedgerError('This recovery helper only accepts synthetic fixtures')
            recovery = canonical(dict(protocol='physical-keyboard-explicit-abort-v1', reason=reason,
                                      round_id=round_id, trial_id=trial_id, timestamp=_now(),
                                      start_event_sha256=state['start_event_sha256'],
                                      reconciles_unknown_device_action=False))
            refs = [self.ledger.put_evidence(recovery), self.ledger.put_evidence(evidence)]
            self.ledger.finish_trial(round_id, trial_id, status='aborted', observed_text=None,
                                    after_evidence=refs, execution_id='abort-' + uuid.uuid4().hex, usage=usage)
            return self.ledger.trial_state(round_id, trial_id)['result']
