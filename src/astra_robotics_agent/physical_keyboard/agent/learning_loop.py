"""Model-owned physical learning decisions; host-owned evidence and permissions.

No SDK, SSH, model client or promotion is implemented here. Explicit training
reset and execution use callbacks through the existing guarded host.
"""
from __future__ import annotations

import json
from itertools import count
import math
from pathlib import Path
import re
import time
import hashlib

from astra_robotics_agent.physical_keyboard.adapters.environments.training_reset import resolve_recovery, validate_reset_arguments
from astra_robotics_agent.physical_keyboard.agent.campaign import CampaignLifecycle
from astra_robotics_agent.physical_keyboard.agent.application import application_for
from astra_robotics_agent.physical_keyboard.agent.execution_feedback import for_model as execution_for_model
from astra_robotics_agent.physical_keyboard.adapters.skill_sources.demo_support import DEMO_TRAJECTORY_HINTS, normalize_demo_capabilities
from astra_robotics_agent.physical_keyboard.runtime.visual_update import validate_roi


from astra_robotics_agent.physical_keyboard.agent.instructions import (
    COMMON_LEARNING_INSTRUCTIONS, LEGACY_FINISH_INSTRUCTIONS, CONTINUOUS_TASK_INSTRUCTIONS,
    CHAT_TASK_INSTRUCTIONS, CONTINUOUS_CHAT_INSTRUCTIONS, EXECUTION_CAUTION, LEARNING_INSTRUCTIONS, learning_instructions,
)
from astra_robotics_agent.physical_keyboard.records.run_recording import RunRecorder
from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def parse_decision(text, *, skill_development=False, fault_read_only=False, continual_learning=False):
    if type(skill_development) is not bool:
        raise ValueError('skill_development must be an explicit boolean')
    if type(continual_learning) is not bool or continual_learning and not skill_development:
        raise ValueError('continual learning requires explicit skill development')
    if type(fault_read_only) is not bool:
        raise ValueError('fault_read_only must be an explicit boolean')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate model decision key')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('Nonfinite model decision')

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite model decision')
        return result

    decision = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite,
                          parse_float=number)
    if type(decision) is not dict:
        raise ValueError('Model decision must be JSON object')
    action = decision.get('action')
    if type(action) is not str:
        raise ValueError('Model action must be a string')
    if action in ('finish', 'checkpoint') and set(decision) == {'action', 'result'}:
        result = decision['result']
        if (type(result) is not dict
                or result.get('status') not in ('completed', 'partial', 'blocked', 'view_ready', 'calibrated_partial')
                or any(k in result and type(result[k]) is not list for k in
                       ('device_hypotheses', 'completed_probes', 'next_needed'))
                or result.get('observed_text') is not None and type(result['observed_text']) is not str):
            raise ValueError('finish.result requires status=completed|partial|blocked|view_ready|calibrated_partial; '
                             'observed_text must be absent/null/string; legacy device_hypotheses, '
                             'completed_probes, next_needed must be lists only if included')
        return decision
    from astra_robotics_agent.physical_keyboard.agent.development import DEVELOPMENT_ACTIONS
    if action in DEVELOPMENT_ACTIONS and not skill_development:
        raise ValueError('Skill-development actions require explicit capability enablement')
    expected = {'action', 'arguments', 'hypothesis'} if action in ('run_skill', 'verify_skill', 'call_skill') else {'action', 'arguments'}
    if set(decision) != expected or type(decision.get('arguments')) is not dict:
        raise ValueError('Invalid action fields for ' + str(action) +
                         '; required top-level keys=' + ','.join(sorted(expected)) +
                         '; missing=' + ','.join(sorted(expected - set(decision))) +
                         '; extra=' + ','.join(sorted(set(decision) - expected)))
    args = decision['arguments']
    from astra_robotics_agent.physical_keyboard.agent.learning import LEARNING_ACTIONS
    if action in LEARNING_ACTIONS:
        if not continual_learning:
            raise ValueError('Continual-learning actions require explicit capability enablement')
        return decision
    if fault_read_only and action in ('inspect_recorder', 'refresh_fault_diagnostics') and args == {}:
        return decision
    if action in ('confirm_completion', 'request_assistance'):
        # The active task lifecycle validates their precise schema and current
        # evidence before accepting completion or requesting outside help.
        return decision
    if 'hypothesis' in expected and (type(decision['hypothesis']) is not str or not decision['hypothesis'].strip()):
        raise ValueError('Nonempty measurable hypothesis required')
    if action in DEVELOPMENT_ACTIONS and skill_development:
        return decision  # The development library validates before write/dispatch.
    if action == 'observe' and args == {}:
        return decision
    if action == 'wait_for_update' and not set(args) - {'roi', 'assessment'}:
        if 'roi' in args:
            validate_roi(args['roi'])
        if 'assessment' in args:
            from astra_robotics_agent.physical_keyboard.adapters.applications.chat_policy import validate_wait_assessment
            args['assessment'] = validate_wait_assessment(args['assessment'])
        return decision
    if action == 'reset':
        validate_reset_arguments(args)
        return decision
    if action in ('save_skill', 'save_lesson'):
        # The store validates the full kind-specific schema before any write.
        return decision
    if action in ('inspect_skill', 'verify_skill') and set(args) == {'candidate_id'} and type(args['candidate_id']) is str:
        return decision
    run_fields = {'source_lines', 'profile', 'label'}
    if (action == 'run_skill' and run_fields <= set(args)
            and not set(args) - run_fields - ({'related_skill_ids'} if skill_development else set())
            and type(args['source_lines']) is list and args['source_lines']
            and all(type(line) is str and '\n' not in line and '\r' not in line for line in args['source_lines'])
            and type(args['profile']) is dict and type(args['label']) is str):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args['label']):
            raise ValueError('run_skill.arguments.label must be 1..64 ASCII letters/digits/_/-; '
                             'use hypothesis for descriptive prose. No request was dispatched.')
        if 'related_skill_ids' in args:
            refs = args['related_skill_ids']
            if (type(refs) is not list or len(refs) > 16
                    or any(type(ref) is not str for ref in refs) or len(set(refs)) != len(refs)):
                raise ValueError('related_skill_ids must be at most16 unique registered skill/package IDs')
        return decision
    raise ValueError('Unknown learning action/schema')


def skill_arguments(args):
    return dict(label=args['label'], source='\n'.join(args['source_lines']) + '\n', profile=args['profile'])


def has_unknown_motion(response):
    return (response.get('status') not in ('ok', 'blocked', 'error')
            or (response.get('status') == 'error'
                and (type(response.get('result')) is not dict
                     or type(response['result'].get('receipts')) is not list
                     or response['result'].get('motion_outcome_uncertain') is not False))
            or response.get('result', {}).get('motion_outcome_uncertain') is True
            or any(r.get('status') == 'unknown' for r in response.get('result', {}).get('receipts', [])))


def controller_is_terminal(response):
    return type(response.get('result')) is dict and response['result'].get('controller_terminal') is True


def validate_finish(result, saved, *, legacy_view_setup=False):
    reflection = result.get('reflection')
    if (type(reflection) is not dict
            or set(reflection) != {'summary', 'lessons_saved', 'skills_saved', 'limitations'}
            or type(reflection['summary']) is not str or not reflection['summary'].strip()
            or type(reflection['limitations']) is not list or not reflection['limitations']
            or any(type(x) is not str or not x.strip() for x in reflection['limitations'])):
        raise ValueError('Finish requires explicit evidence-based reflection and limitations')
    for kind, field in (('lesson', 'lessons_saved'), ('skill', 'skills_saved')):
        ids = reflection[field]
        if type(ids) is not list or any(type(x) is not str or saved.get(x) != kind for x in ids):
            raise ValueError('Reflection must name only candidates saved in this episode')
    if legacy_view_setup and result.get('observed_text') is not None:
        raise ValueError('Legacy non-contact viewing experiment cannot report entered text')
    if legacy_view_setup and result['status'] in ('view_ready', 'completed'):
        assessment = result.get('view_assessment')
        if (type(assessment) is not dict
                or set(assessment) != {'visible_key_labels', 'petal_tip_visible', 'nearby_surfaces_described'}
                or type(assessment['visible_key_labels']) is not list
                or len(assessment['visible_key_labels']) != 10
                or any(type(x) is not str for x in assessment['visible_key_labels'])
                or set(assessment['visible_key_labels']) != set('QWERTYUIOP')
                or assessment['petal_tip_visible'] is not True
                or type(assessment['nearby_surfaces_described']) is not str
                or not assessment['nearby_surfaces_described'].strip()):
            raise ValueError('view_ready requires complete, explicit current visual assessment')


def run_episode(*, session, observe, execute, store, output: Path,
                max_decisions=None, report=None, legacy_view_setup=False, reset=None,
                task_contract=None, demo_capabilities=None, wait_for_update=None, status=None,
                operator_updates=None, skill_development=False, typing_case_context=None,
                capture_fault=False, fault_handler=None, assistance_wait=None,
                typing_schedule=None, continual_learning=False, learning_sources=None,
                learning_access=None, stop_requested=None, application_observer=None):
    """Every task/knowledge choice comes from session.run_turn, never a host plan.

    observe returns {contentItems, evidence_id}; execute returns
    {response, evidence_id}. Both callbacks freeze/register the actual evidence.
    Exceptions/unknown outcomes stop dispatch; an explicit fault handler may
    suspend and restore execution using a new registered observation. It never
    repeats an action. Without it, the original stop/exception behavior remains.
    """
    if max_decisions is not None and (type(max_decisions) is not int or max_decisions < 1):
        raise ValueError('Explicit max_decisions must be a positive integer, or None')
    if type(skill_development) is not bool:
        raise ValueError('skill_development must be an explicit boolean')
    if type(capture_fault) is not bool or capture_fault and not skill_development:
        raise ValueError('fault capture requires explicit skill development')
    if fault_handler is not None and (not callable(fault_handler) or not skill_development or not capture_fault):
        raise ValueError('fault handler requires explicit skill development and fault capture')
    if assistance_wait is not None and (not callable(assistance_wait) or fault_handler is None):
        raise ValueError('assistance waiting requires an explicit persistent fault handler')
    if typing_case_context is not None:
        if (not callable(typing_case_context) or not skill_development or task_contract is None
                or not getattr(task_contract, 'is_skill_development_training', False)):
            raise ValueError('typing case recording requires explicit continuous skill training')
    if typing_schedule is not None and typing_case_context is None:
        raise ValueError('typing schedule requires the recording-only case identity source')
    if legacy_view_setup and skill_development:
        raise ValueError('Legacy viewing cannot enable skill development')
    if (task_contract is not None and getattr(task_contract, 'is_skill_development_training', False)
            and not skill_development):
        raise ValueError('The skill-development training contract requires explicit enablement')
    if legacy_view_setup and task_contract is not None:
        raise ValueError('Legacy viewing and a continuous task contract cannot be combined')
    if type(continual_learning) is not bool or continual_learning and not skill_development:
        raise ValueError("continual learning requires explicit skill development")
    if (learning_sources is not None or learning_access is not None) and not continual_learning:
        raise ValueError("learning source/access configuration requires continual learning")
    application = application_for(task_contract)
    application.validate_callbacks(reset, wait_for_update)
    if application_observer is not None and (not application.continuous
            or not callable(getattr(application_observer, 'checkpoint', None))
            or not callable(getattr(application_observer, 'note_observation', None))):
        raise ValueError('Application observer requires continuous chat and read-only checkpoint/note callbacks')
    chat_service = application.continuous
    development = None
    if skill_development:
        from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
        development = DevelopmentActions(store)
    application_skill = None
    continual = boundary = None
    if continual_learning:
        from astra_robotics_agent.physical_keyboard.agent.learning import ContinualLearning
        from astra_robotics_agent.physical_keyboard.agent.learning_boundary import LearningBoundary
        continual = ContinualLearning(development, session, output=output / "continual-learning",
                                     sources=learning_sources, access=learning_access)
        boundary = LearningBoundary(store, task_contract, stop_requested)
    if development is not None:
        from astra_robotics_agent.physical_keyboard.agent.application_skill import ApplicationSkillSession
        application_skill = ApplicationSkillSession(development.library,
            getattr(application, 'application_id', None),
            read=continual.reader._get if continual is not None else None)
    if demo_capabilities is not None:
        demo_capabilities = normalize_demo_capabilities(demo_capabilities)
    lifecycle = CampaignLifecycle(task_contract) if task_contract is not None else None
    report = {} if report is None else report
    report.update(turns=[], saved_candidates=[], verifications=[], decision_owner='astra',
                  diagnostic_only=True, stable_promoted=False, max_decisions=max_decisions,
                  reset_attempts=[], training_reset_available=reset is not None and not legacy_view_setup)
    application.initialize_report(report)
    if development is not None:
        report.update(skill_development_enabled=True, skill_packages=[],
                      skill_calls=[], skill_feedback_basis='model_report_not_independent_truth',
                      skill_memory_reviews=[], skill_memory_contexts=[], skill_use_contexts=[])
        report.update(application_skills=[], application_skill_events=[])
    if demo_capabilities is not None:
        report['demo_trajectories'] = demo_capabilities
    if lifecycle is not None:
        report.update(task_completed=False, completed=False, finished_by_model=False,
                      phase_checkpoints=[], completion_reviews=[],
                      lifecycle='continuous_final_goal', task_contract=task_contract.metadata,
                      task_status='exploring', independently_verified=False)
    if continual is not None:
        report["continual_learning"] = continual.descriptor()
    saved = {}
    observed_ids = set()
    observation = None
    send_observation = True
    pending_content = []
    feedback = None
    event = RunRecorder(output, serializer=canonical).event
    if fault_handler is not None:
        report.update(persistent_execution_enabled=True, fault_history=[])

    class ObservationRecovered(Exception):
        """Unwind a model action after recovery without repeating its effects."""

    def accept_observation(current):
        if lifecycle is not None or fault_handler is not None or continual is not None:
            evidence_id = current.get('evidence_id')
            if type(evidence_id) is not str or not evidence_id or evidence_id in observed_ids:
                raise RuntimeError('Continuous task requires a new trusted observation evidence ID')
            if fault_handler is not None or continual is not None:
                store._evidence(evidence_id)
                if type(current.get('contentItems')) is not list:
                    raise RuntimeError('Recovered observations require registered contentItems')
            observed_ids.add(evidence_id)
            if lifecycle is not None:
                lifecycle.record_observation(evidence_id)
        if application_observer is not None:
            application_observer.note_observation(current)
        return current

    def trusted_callback(callback, *arguments):
        if boundary is None:
            return callback(*arguments)
        boundary.check_stop()
        boundary.inflight_rpc = boundary.inflight_action = True
        try:
            return callback(*arguments)
        finally:
            boundary.inflight_rpc = boundary.inflight_action = False

    def refresh_observation(*, attempted_program=None, failure_feedback=None, initial=False,
                            callback=None):
        try:
            return accept_observation(trusted_callback(observe if callback is None else callback))
        except OperatorStop:
            raise
        except Exception as exc:
            if fault_handler is None:
                raise
            remember_fault('observation_exception_no_retry', attempted_program,
                           failure_feedback=failure_feedback, error=exc, case_identity=case_for_records())
            resume_fault()
            if initial:
                return observation
            raise ObservationRecovered() from exc

    def publish(phase, **fields):
        if status is not None:
            status(dict(phase=phase, **fields))

    def remember_fault(reason, attempted_program, *, failure_feedback=None, error=None, case_identity=None):
        if not capture_fault:
            return
        snapshot = dict(stopped_reason=reason,
            last_observation=dict(evidence_id=None if observation is None else observation['evidence_id'],
                timing='unavailable' if observation is None else 'pre_execution_not_current'),
            attempted_program=(None if attempted_program is None else
                               {key: attempted_program[key] for key in ('source', 'profile', 'label')}),
            feedback=failure_feedback, case_identity=case_identity,
            physical_outcome='unknown_not_automatically_retried', motion_dispatch_disabled=True)
        if error is not None:
            snapshot.update(error_type=type(error).__name__, error=str(error)[:2048],
                            execution_status='exception_unknown')
        if boundary is not None:
            snapshot['learning_boundary'] = boundary.facts(None, motion_available=False,
                fault=dict(stopped_reason=reason, physical_outcome='unknown_not_automatically_retried'))
        report['fault_snapshot'] = snapshot
        event('fault_snapshot', snapshot)

    def resume_fault():
        nonlocal observation, send_observation, pending_content, feedback
        snapshot = report['fault_snapshot']
        # Freeze facts before handing them to an external lifecycle callback.
        entry = dict(number=len(report['fault_history']) + 1,
                     snapshot=json.loads(canonical(snapshot)), status='suspended')
        report['fault_history'].append(entry)
        report.update(task_completed=False, completed=False, task_status='execution_suspended')
        if lifecycle is not None:
            lifecycle.invalidate_completion_review()
        event('fault_suspended', entry)
        publish('execution_suspended', fault_number=entry['number'], stopped_reason=snapshot['stopped_reason'])
        resumed = fault_handler(json.loads(canonical(snapshot)))
        if type(resumed) is not dict or set(resumed) != {'observation', 'recovery'} or type(resumed['recovery']) is not dict:
            raise RuntimeError('Fault handler must return a new observation and recovery metadata')
        recovery = json.loads(canonical(resumed['recovery']))
        if boundary is not None:
            generation = recovery.get('generation', recovery.get('restore_receipt', {}).get('generation'))
            if type(generation) is not int or generation <= boundary.generation:
                raise RuntimeError('Recovery must supply a new authoritative generation')
            boundary.generation = generation
            # Unknown history remains in the recovery record; only supervised fresh generation grants motion.
            boundary.last_execution = None
        observation = accept_observation(resumed['observation'])
        # The same session's knowledge-only review may have saved candidates
        # outside the normal decision branch. Preserve their episode ownership
        # so a later phase reflection can cite them without claiming promotion.
        recovered_candidates = recovery.get('saved_candidates', [])
        if type(recovered_candidates) is not list:
            raise RuntimeError('Recovery saved_candidates must be a list')
        for candidate in recovered_candidates:
            if type(candidate) is not dict:
                raise RuntimeError('Recovery candidates require registered summaries')
            actual = store._candidate(candidate.get('candidate_id'))
            if actual['run_id'] != store.run_id:
                raise RuntimeError('Recovery candidate was not saved in this episode')
            identifier, kind = actual['candidate_id'], actual['kind']
            if identifier not in saved:
                saved[identifier] = kind
                report['saved_candidates'].append(candidate)
                if candidate.get('package_id') is not None:
                    report['skill_packages'].append(candidate)
        send_observation, pending_content = True, []
        recovery_feedback = dict(recovery)
        if recovered_candidates:
            recovery_feedback['saved_candidates'] = [dict(candidate_id=c['candidate_id'],
                package_id=c.get('package_id'), status='candidate_not_promoted') for c in recovered_candidates]
        feedback = dict(phase='execution_resumed', stopped_reason=snapshot['stopped_reason'],
            error_type=snapshot.get('error_type'), error=snapshot.get('error'), recovery=recovery_feedback,
            observation_evidence_id=observation['evidence_id'], replayed=False,
            instruction='The original continuous task resumes in this SAME session. '
                'Reassess the fresh observation and recorded partial/unknown outcome. '
                'No failed program, action prefix, reset or input was automatically replayed or cleared; '
                'choose the remaining task and any candidate revision yourself.')
        entry.update(status='resumed', observation_evidence_id=observation['evidence_id'], recovery=recovery)
        report['last_fault_snapshot'] = report.pop('fault_snapshot')
        report.pop('stopped_reason', None)
        report.update(task_status='exploring', fault_review_handled=True)
        event('fault_resumed', dict(number=entry['number'], **feedback))
        publish('execution_resumed', fault_number=entry['number'],
                observation_evidence_id=observation['evidence_id'])

    def case_for_records():
        if typing_case_context is None:
            return None
        from astra_robotics_agent.physical_keyboard.records.typing_metrics import validate_case_identity
        try:
            return validate_case_identity(typing_case_context())
        except Exception as exc:
            # Recording metadata cannot become text observations or trigger
            # motion retries. Never serialize the rejected (possibly textual) payload.
            report['typing_lab_context_errors'] = report.get('typing_lab_context_errors', 0) + 1
            event('typing_lab_context_unavailable', dict(error_type=type(exc).__name__))
            return None

    def record_typing_execution(action, arguments, before_case, started, ended,
                                response, candidate_id=None, package_id=None, related_skill_ids=()):
        if typing_case_context is None:
            return
        after_case = (None if has_unknown_motion(response) or controller_is_terminal(response)
                      else case_for_records())
        event('typing_lab_execution', dict(case=before_case, case_after=after_case,
            action=action, candidate_id=candidate_id, package_id=package_id,
            program_sha256=hashlib.sha256(arguments['source'].encode()).hexdigest(),
            # Match Runtime's frozen profile.json byte encoding. This pins the
            # requested version even when execution itself remains unknown.
            profile_sha256=hashlib.sha256(json.dumps(arguments['profile'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest(),
            related_skill_ids=list(related_skill_ids), elapsed_seconds=round(ended - started, 6),
            started_monotonic_ns=int(started * 1000000000),
            ended_monotonic_ns=int(ended * 1000000000),
            execution_status=response.get('status', 'unknown')))

    def no_native_tools(name, arguments):
        raise RuntimeError('Use the declared JSON learning actions only')

    observation = refresh_observation(initial=True)
    if feedback is None:
        feedback = {'phase': 'begin', 'observation_evidence_id': observation['evidence_id']}

    for index in count():
        if boundary is not None:
            boundary.check_stop()
        if max_decisions is not None and index >= max_decisions:
            break
        # Only between completed actions. The observer may inspect preview
        # changes, never interrupt a program or choose a conversational reply.
        if application_observer is not None and not send_observation:
            try:
                update = trusted_callback(application_observer.checkpoint)
                if update is not None:
                    observation = accept_observation(update)
                    send_observation = True
                    feedback = dict(phase='application_observation_update',
                        observation_evidence_id=observation['evidence_id'],
                        prior_action_feedback=feedback,
                        wake=observation.get('visual_update', observation.get('observation_recheck')),
                        instruction='Reassess fresh application evidence before continuing deferred knowledge work; no reply or motion was chosen or replayed.')
                    event('application_observation_update', feedback)
            except OperatorStop:
                raise
            except Exception as exc:
                if fault_handler is None:
                    raise
                remember_fault('application_observation_exception_no_retry', None,
                    failure_feedback=feedback, error=exc, case_identity=case_for_records())
                resume_fault()
                continue
        folder = output / ('turn-%02d' % index)
        folder.mkdir()
        items = list(observation['contentItems']) if send_observation else []
        items += pending_content
        pending_content = []
        images = [item['imageUrl'] for item in items if item['type'] == 'inputImage']
        context = dict(feedback=feedback, remaining_decisions=None if max_decisions is None else max_decisions-index,
                                training_reset_available=reset is not None and not legacy_view_setup,
                                current_observation_evidence_id=observation['evidence_id'],
                                observation_unchanged=not send_observation)
        if demo_capabilities is not None:
            context['demo_trajectories'] = demo_capabilities
        context.update(application.context_fields(report))
        if application_skill is not None:
            context['application_skill'] = application_skill.context()
        if continual is not None:
            context['continual_learning'] = continual.context(observation['evidence_id'])
            context['skill_development'] = dict(enabled=True, online_recall=(
                development.recall.for_skills(development.focus_skill_ids) if development.focus_skill_ids else {}))
        elif development is not None:
            context['skill_development'] = development.context()
        if lifecycle is not None:
            pending = lifecycle.pending_completion
            context['task_state'] = dict(contract=task_contract.document, task_completed=False,
                checkpoints_completed=lifecycle.checkpoint_count,
                pending_completion_observation_id=(pending['review_observation_evidence_id']
                                                  if pending is not None else None))
        prompt = canonical(context)
        if operator_updates is not None:
            updates = operator_updates()
            if updates:
                context['operator_feedback'] = updates
                event('operator_feedback', updates)
                prompt = canonical(context)
        prompt += '\n' + '\n'.join(item['text'] for item in items if item['type'] == 'inputText')
        (folder / 'request.txt').write_text(prompt)
        if continual is not None:
            event('learning_prompt_accounting', dict(turn_index=index, prompt_utf8_bytes=len(prompt.encode('utf-8')),
                image_count=len(images), accounting='actual text bytes, not token estimate'))
        recorded_case = case_for_records()
        if typing_schedule is not None:
            try:
                selection = typing_schedule.check_identity(recorded_case)
            except ValueError as exc:
                report.update(stopped_reason='typing_schedule_identity_unavailable',
                              task_completed=False, completed=False, task_status='stopped_incomplete',
                              typing_schedule_error=str(exc))
                event('typing_schedule_blocked', dict(error=str(exc), dispatched=False))
                return report
            if selection['should_stop']:
                report.update(stopped_reason='selected_typing_schedule_completed',
                              task_completed=False, completed=False, task_status='selected_schedule_completed',
                              typing_schedule_completion=selection)
                event('typing_schedule_completed', dict(selection, dispatched=False))
                publish('selected_schedule_completed', stopped_reason=report['stopped_reason'])
                return report
        publish('model_inference', model_turns=len(report['turns']),
                observation_evidence_id=observation['evidence_id'])
        model_started = time.monotonic()
        try:
            if boundary is not None:
                boundary.inflight_model = True
            try:
                reply = session.run_turn(prompt, no_native_tools, images=images)
            finally:
                if boundary is not None:
                    boundary.inflight_model = False
        except BaseException:
            if typing_case_context is not None:
                event('typing_lab_model_turn', dict(case=recorded_case, turn_index=index, usage=None,
                    elapsed_seconds=round(time.monotonic() - model_started, 6)))
            raise
        if typing_case_context is not None:
            event('typing_lab_model_turn', dict(case=recorded_case, turn_index=index, usage=reply.get('usage'),
                elapsed_seconds=round(time.monotonic() - model_started, 6)))
        (folder / 'reply.json').write_text(canonical(reply))
        report['turns'].append(dict(index=index, usage=reply.get('usage'), turn_id=reply['turn_id']))
        if typing_case_context is not None:
            report['turns'][-1]['typing_case'] = recorded_case
        if development is not None:
            presented = development.presented(context)
            if presented:
                entry = dict(turn_index=index, record_ids=presented,
                             meaning='context_presented_not_proof_of_use_or_improvement')
                report['skill_memory_contexts'].append(entry)
                event('skill_memory_context_presented', entry)
        send_observation = False
        external_callback_started = False
        try:
            decision = parse_decision(reply['final_text'], skill_development=skill_development,
                                      continual_learning=continual_learning)
            event('model_decision', decision)
            action = decision['action']
            publish('processing_action', action=action, model_turns=len(report['turns']))
            if continual is not None:
                from astra_robotics_agent.physical_keyboard.agent.learning import LEARNING_ACTIONS
                if action in LEARNING_ACTIONS:
                    facts = boundary.facts(observation['evidence_id'])
                    maintenance = action == 'manage_working_context'
                    before_usage = json.loads(canonical(getattr(session, 'total_usage', None)))
                    before_complete = getattr(session, 'total_usage_complete', False)
                    maintenance_started = time.monotonic()
                    operation_status = 'failed'
                    try:
                        feedback, pending_content = continual.handle(action, decision['arguments'], facts=facts)
                        operation_status = feedback['result'].get('status', 'completed')
                    finally:
                        if maintenance:
                            from astra_robotics_agent.physical_keyboard.records.usage import usage_delta
                            cost = dict(case=recorded_case, turn_index=index,
                                method=(decision['arguments'].get('method') if type(decision['arguments'].get('method')) is str
                                    and decision['arguments']['method'] in {'native_compact', 'rebuild'} else None), status=operation_status,
                                elapsed_seconds=round(time.monotonic() - maintenance_started, 6),
                                usage=usage_delta(before_usage, getattr(session, 'total_usage', None)),
                                usage_complete=bool(before_complete and getattr(session, 'total_usage_complete', False)))
                            event('typing_lab_context_operation' if typing_case_context is not None else
                                  'learning_context_operation_cost', cost)
                    for candidate in feedback['result'].get('saved_candidates', []):
                        if candidate['candidate_id'] not in saved:
                            saved[candidate['candidate_id']] = candidate['kind']
                            report['saved_candidates'].append(candidate)
                            if candidate.get('package_id'):
                                report['skill_packages'].append(candidate)
                    report.get('application_skills', []).extend(
                        feedback['result'].get('saved_application_skills', []))
                    event('continual_learning_action', feedback)
                    if feedback['result'].get('fresh_observation_required') is True:
                        boundary.require_unchanged(facts, observation['evidence_id'])
                        observation = refresh_observation()
                        send_observation = True
                        feedback.update(observation_unchanged=False, current_observation_evidence_id=observation['evidence_id'])
                    event('framework_feedback', feedback)
                    continue
                continual.validate_legacy(decision.get('arguments', {}))
            if action == 'application_skill':
                feedback = dict(action=action,
                    result=application_skill.handle(decision['arguments'], observation['evidence_id']),
                    observation_unchanged=True)
                report['application_skill_events'].append(feedback['result'])
                event('application_skill_checkpoint', feedback)
                continue
            if development is not None:
                from astra_robotics_agent.physical_keyboard.agent.development import KNOWLEDGE_ACTIONS
                if action in KNOWLEDGE_ACTIONS:
                    if continual is not None and action == 'save_skill_package':
                        feedback, pending_content = dict(action=action, result=continual.save_package(decision['arguments']),
                            observation_unchanged=True), []
                    else:
                        feedback, pending_content = development.knowledge_action(
                            action, decision['arguments'], observation['evidence_id'])
                    if continual is not None:
                        feedback = continual.published_feedback(feedback)
                    if action == 'save_skill_package':
                        package = feedback['result']
                        saved[package['candidate_id']] = 'skill'
                        report['saved_candidates'].append(package)
                        report['skill_packages'].append(package)
                    elif action == 'save_application_skill':
                        report['application_skills'].append(feedback['result'])
                    event('skill_development_action', feedback)
                    event('framework_feedback', feedback)
                    continue
            if action in ('finish', 'checkpoint'):
                validate_finish(decision['result'], saved, legacy_view_setup=legacy_view_setup)
                if (getattr(application, 'persistent_completion', chat_service)
                        and decision['result']['status'] == 'completed'):
                    entry, feedback = application.record_phase(
                        decision['result'], observation['evidence_id'], lifecycle, report)
                    event('chat_turn_recorded' if chat_service else 'development_phase_recorded', entry)
                    continue
                if lifecycle is not None:
                    result = decision['result']
                    if action == 'checkpoint' and result['status'] == 'completed':
                        raise ValueError('A checkpoint is nonterminal; use finish to request completion review')
                    lifecycle.invalidate_completion_review()
                    before = observation['evidence_id']
                    external_callback_started = True
                    observation = refresh_observation()
                    send_observation = True
                    if result['status'] == 'completed':
                        feedback = lifecycle.begin_completion_review(result, observation['evidence_id'])
                        report['task_status'] = 'reviewing_completion'
                        entry = dict(result=result, before_observation_evidence_id=before,
                                     review_observation_evidence_id=observation['evidence_id'])
                        report['completion_reviews'].append(entry)
                        event('completion_review_requested', entry)
                    else:
                        feedback = lifecycle.checkpoint(result, requested_action=action)
                        report['task_status'] = 'exploring'
                        entry = dict(number=lifecycle.checkpoint_count, requested_action=action,
                                     result=result, before_observation_evidence_id=before,
                                     observation_evidence_id=observation['evidence_id'])
                        report['phase_checkpoints'].append(entry)
                        event('phase_checkpoint', entry)
                    event('framework_feedback', feedback)
                    continue
                if action == 'checkpoint':
                    raise ValueError('checkpoint requires a continuous task contract')
                report.update(reply=decision['result'], finished_by_model=True)
                event('model_finished', decision['result'])
                return report
            args = decision['arguments']
            if action == 'confirm_completion':
                if chat_service:
                    raise ValueError('Continuous chat does not end per message; record a phase and use wait_for_update')
                if getattr(application, 'persistent_completion', False):
                    raise ValueError('Continuous skill training records nonterminal phases until operator stop')
                if lifecycle is None:
                    raise ValueError('confirm_completion requires a continuous task contract')
                result = lifecycle.confirm_completion(args, observation['evidence_id'], store.get_skill)
                report.update(reply=result, finished_by_model=True, task_completed=True,
                              completed=True, stopped_reason='final_goal_confirmed',
                              task_status='completed_by_visual_claim',
                              completion_basis='model_visual_confirmation')
                event('final_goal_confirmed', result)
                return report
            elif action == 'request_assistance':
                if lifecycle is None:
                    raise ValueError('request_assistance requires a continuous task contract')
                request = lifecycle.validate_assistance(args, observation['evidence_id'])
                ordinary_assistance = request['kind'] in ('operator_action', 'missing_external_capability')
                if ordinary_assistance and (application.recoverable_assistance(request) or assistance_wait is not None):
                    request['terminal'] = False
                    report.setdefault('assistance_history', []).append(request)
                    report['task_status'] = 'awaiting_assistance'
                    event('assistance_requested', request)
                    publish('awaiting_assistance', assistance=request)
                    lifecycle.invalidate_completion_review()
                    external_callback_started = True
                    observation = refresh_observation(callback=(
                        (lambda: assistance_wait(request)) if assistance_wait is not None
                        else (lambda: wait_for_update({}))),
                        failure_feedback=dict(action=action, assistance=request))
                    send_observation = True
                    report['task_status'] = 'chatting' if chat_service else 'exploring'
                    feedback = dict(phase='assistance_update', assistance=request,
                        observation_evidence_id=observation['evidence_id'],
                        operator_feedback=observation.get('operator_feedback', []),
                        instruction='Reassess this fresh evidence/operator feedback and continue the same session; no action was retried and no skill or usage was reset.')
                    event('assistance_update', feedback)
                    continue
                report.update(reply=dict(status='blocked', assistance=request),
                              assistance_requested=request, finished_by_model=True,
                              task_completed=False, completed=False,
                              task_status='awaiting_assistance',
                              stopped_reason='external_assistance_required')
                event('assistance_requested', request)
                return report
            elif action in ('save_skill', 'save_lesson'):
                kind = 'skill' if action == 'save_skill' else 'lesson'
                candidate = store.propose(kind, args)
                saved[candidate['candidate_id']] = kind
                report['saved_candidates'].append(candidate)
                feedback = dict(action=action, candidate=candidate, observation_unchanged=True)
            elif action == 'inspect_skill':
                feedback = dict(action=action, skill=store.get_skill(args['candidate_id']))
                if development is not None:
                    development.focus_skill_ids = [args['candidate_id']]
            elif action == 'observe':
                if lifecycle is not None:
                    lifecycle.invalidate_completion_review()
                    report['task_status'] = 'exploring'
                external_callback_started = True
                observation = refresh_observation()
                send_observation = True
                feedback = dict(action=action, evidence_id=observation['evidence_id'])
            elif action == 'wait_for_update':
                if not chat_service:
                    raise ValueError('wait_for_update requires a continuous chat session')
                assessment = application.validate_wait(args, observation['evidence_id'])
                if assessment is not None:
                    event('chat_wait_assessment', assessment)
                lifecycle.invalidate_completion_review()
                report['task_status'] = 'waiting_for_update'
                publish('waiting_for_update', **({'wait_assessment': assessment} if assessment is not None else {}))
                external_callback_started = True
                observation = refresh_observation(callback=lambda: wait_for_update(
                    {k:v for k,v in args.items() if k == 'roi'}),
                    failure_feedback=dict(action=action, wait_assessment=assessment))
                send_observation = True
                report['task_status'] = 'chatting'
                feedback = dict(action=action, evidence_id=observation['evidence_id'],
                                operator_feedback=observation.get('operator_feedback', []))
            elif action == 'reset':
                if reset is None or legacy_view_setup:
                    raise ValueError('Training reset unavailable; no motion or editor clear attempted')
                recovery = resolve_recovery(args, store)
                before = observation['evidence_id']
                if lifecycle is not None:
                    lifecycle.invalidate_completion_review()
                    report['task_status'] = 'exploring'
                external_callback_started = True
                reset_case = case_for_records()
                reset_started = time.monotonic()
                try:
                    applied = trusted_callback(reset, args, recovery)
                except BaseException as exc:
                    report['reset_exception_no_retry'] = True
                    if capture_fault:
                        report['stopped_reason'] = 'reset_exception_no_retry'
                    remember_fault('reset_exception_no_retry', recovery, error=exc, case_identity=reset_case)
                    try:
                        record_typing_execution(action, recovery, reset_case, reset_started, time.monotonic(),
                            {'status': 'exception_unknown'}, candidate_id=args['recovery'].get('candidate_id'))
                    except Exception as trace_error:
                        report['typing_lab_failed_trace_error'] = type(trace_error).__name__
                    if fault_handler is not None and isinstance(exc, Exception) and not isinstance(exc, OperatorStop):
                        resume_fault()
                        continue
                    raise
                response = applied['response']
                if boundary is not None:
                    boundary.execution(applied['evidence_id'], response, unknown=(has_unknown_motion(response) or
                        response.get('result', {}).get('reset_outcome_uncertain') is not False),
                        terminal=controller_is_terminal(response))
                record_typing_execution(action, recovery, reset_case, reset_started, time.monotonic(), response,
                                        candidate_id=args['recovery'].get('candidate_id'))
                evidence = [before, applied['evidence_id']]
                unknown = (has_unknown_motion(response)
                           or response.get('result', {}).get('reset_outcome_uncertain') is not False)
                entry = dict(status=response.get('status'), response=response, evidence_ids=evidence,
                             verification='training_reset_not_physical_correction')
                # Record the reset's actual callback result before requesting a
                # post-reset camera frame, which can independently fail.
                report['reset_attempts'].append(entry)
                if not unknown:
                    observation = refresh_observation(attempted_program=recovery,
                        failure_feedback=dict(action=action, response=response, evidence_ids=evidence))
                    send_observation = True
                    evidence.append(observation['evidence_id'])
                feedback = dict(action=action, **entry)
                if unknown:
                    report.update(stopped_reason='unknown_training_reset_no_retry', completed=False)
                    if lifecycle is not None:
                        report['task_status'] = 'stopped_incomplete'
                    event('unknown_training_reset_stopped', feedback)
                    remember_fault('unknown_training_reset_no_retry', recovery,
                                   failure_feedback=feedback, case_identity=reset_case)
                    if fault_handler is not None:
                        resume_fault()
                        continue
                    return report
            else:
                resolved = None
                use_memory = None
                if action == 'call_skill':
                    resolved = development.resolve_call(args)
                    candidate_id = resolved['candidate_id']
                    arguments = resolved
                    event('skill_call_resolved', dict(package_id=resolved['package_id'],
                        candidate_id=candidate_id, dependency_pins=resolved['dependency_pins'],
                        program_sha256=hashlib.sha256(resolved['source'].encode()).hexdigest()))
                else:
                    candidate_id = args['candidate_id'] if action == 'verify_skill' else None
                    arguments = store.get_skill(candidate_id) if candidate_id else skill_arguments(args)
                arguments = {key: arguments[key] for key in ('source', 'profile', 'label')}
                if development is not None:
                    references = ([resolved['package_id']] if resolved is not None else
                                  [candidate_id] if candidate_id is not None else
                                  args.get('related_skill_ids', []))
                    use_memory, unseen = development.before_use(references, arguments['source'])
                    if unseen:
                        # No executor/observation callback has started. This is
                        # a model-owned new decision, never an automatic retry.
                        entry = dict(turn_index=index, requested_action=action,
                                     focus_skill_ids=use_memory['focus_skill_ids'],
                                     recall_fingerprint=use_memory['fingerprint'],
                                     newly_relevant_record_ids=unseen, dispatched=False,
                                     observation_evidence_id=observation['evidence_id'])
                        report['skill_memory_reviews'].append(entry)
                        feedback = dict(phase='related_skill_memory', **entry,
                            observation_unchanged=True, auto_resume=False,
                            instruction='Related skill memory is supplied in skill_development.online_recall. '
                            'Consider the outstanding issue, choose an improvement/verification or explain '
                            'deferral yourself. No motion was dispatched or queued; choose your next action.')
                        event('skill_memory_review', dict(**entry, memory=use_memory))
                        event('framework_feedback', feedback)
                        continue
                before = observation['evidence_id']
                if lifecycle is not None:
                    lifecycle.invalidate_completion_review()
                    report['task_status'] = 'exploring'
                # Transport/hardware exceptions must escape, never become a
                # recoverable model-validation error or repeat an uncertain move.
                publish('executing_skill', label=arguments['label'],
                        model_turns=len(report['turns']))
                external_callback_started = True
                execution_case = case_for_records()
                execution_started = time.monotonic()
                try:
                    executed = trusted_callback(execute, arguments)
                    execution_ended = time.monotonic()
                except BaseException as exc:
                    report['execution_exception_no_retry'] = True
                    if capture_fault:
                        report['stopped_reason'] = 'execution_exception_no_retry'
                    remember_fault('execution_exception_no_retry', arguments, error=exc,
                                   case_identity=execution_case)
                    try:
                        record_typing_execution(action, arguments, execution_case, execution_started,
                            time.monotonic(), {'status': 'exception_unknown'}, candidate_id=candidate_id,
                            package_id=resolved['package_id'] if resolved is not None else None,
                            related_skill_ids=use_memory['focus_skill_ids'] if use_memory is not None else ())
                    except Exception as trace_error:
                        report['typing_lab_failed_trace_error'] = type(trace_error).__name__
                    if fault_handler is not None and isinstance(exc, Exception) and not isinstance(exc, OperatorStop):
                        resume_fault()
                        continue
                    raise
                response = executed['response']
                record_typing_execution(action, arguments, execution_case, execution_started, execution_ended,
                    response, candidate_id=candidate_id,
                    package_id=resolved['package_id'] if resolved is not None else None,
                    related_skill_ids=use_memory['focus_skill_ids'] if use_memory is not None else ())
                evidence = [before, executed['evidence_id']]
                unknown = has_unknown_motion(response)
                terminal = controller_is_terminal(response)
                if boundary is not None:
                    boundary.execution(executed['evidence_id'], response, unknown=unknown, terminal=terminal)
                if not unknown and not terminal:
                    observation = refresh_observation(attempted_program=arguments,
                        failure_feedback=dict(action=action, execution=response, evidence_ids=evidence))
                    send_observation = True
                    evidence.append(observation['evidence_id'])
                if candidate_id:
                    verification = store.record_verification(candidate_id, list(dict.fromkeys(evidence)),
                                                             'unknown' if unknown else response['status'])
                    report['verifications'].append(verification)
                feedback = dict(action=action, hypothesis=decision['hypothesis'],
                                candidate_id=candidate_id, execution=execution_for_model(response, executed['evidence_id'], compact=continual_learning), evidence_ids=evidence,
                                verification='diagnostic_current_state_not_promotion')
                if (not unknown and not terminal
                        and response.get('result', {}).get('recoverable_admission_error') is True):
                    feedback.update(execution_disposition='admission_rejected', replayed=False,
                        continuation=response['result'].get('continuation'),
                        instruction='This candidate was rejected before its dispatch and the Runtime retained a healthy owner. '
                            'Earlier completed effects remain in the new camera observation. '
                            'Use the quantified admission facts to revise, inspect, optionally preview, or select another skill. '
                            'Choose a new request yourself; the rejected request is not replayed or queued. '
                            'Identical compile input and policy gives no new information; inspect new evidence or change the proposal.')
                if response.get('result', {}).get('recoverable_program_error') is True:
                    feedback.update(execution_disposition='partial_program_error', replayed=False,
                        instruction='The Runtime returned a recoverable program error after recording completed tools. '
                            'Those effects may already be present in the fresh observation. '
                            'Repair your program and choose the remaining work; do not blindly replay the whole request.')
                if development is not None:
                    feedback['execution_elapsed_seconds'] = round(execution_ended - execution_started, 6)
                    feedback['task_result_basis'] = 'awaiting_model_visual_report'
                    memory_entry = dict(turn_index=index, action=action,
                        focus_skill_ids=use_memory['focus_skill_ids'],
                        recall_fingerprint=use_memory['fingerprint'], evidence_ids=evidence,
                        association_basis=('registered_skill_or_exact_source' if use_memory['focus_skill_ids']
                                           else 'unlinked_source_semantics_unknown'))
                    report['skill_use_contexts'].append(memory_entry)
                    feedback['skill_memory'] = memory_entry
                    event('skill_use_context', memory_entry)
                    if resolved is not None:
                        feedback['package_id'] = resolved['package_id']
                        report['skill_calls'].append(dict(package_id=resolved['package_id'],
                            candidate_id=candidate_id, evidence_ids=evidence,
                            execution_status=response.get('status'),
                            elapsed_seconds=feedback['execution_elapsed_seconds']))
                if unknown:
                    report.update(stopped_reason='unknown_motion_no_retry', completed=False)
                    if lifecycle is not None:
                        report['task_status'] = 'stopped_incomplete'
                    event('unknown_motion_stopped', feedback)
                    remember_fault('unknown_motion_no_retry', arguments, failure_feedback=feedback,
                                   case_identity=execution_case)
                    if fault_handler is not None:
                        resume_fault()
                        continue
                    return report
                if terminal:
                    report.update(stopped_reason='native_controller_terminal_no_retry', completed=False)
                    if lifecycle is not None:
                        report['task_status'] = 'stopped_incomplete'
                    event('native_controller_terminal_stopped', feedback)
                    remember_fault('native_controller_terminal_no_retry', arguments,
                                   failure_feedback=feedback, case_identity=execution_case)
                    if fault_handler is not None:
                        resume_fault()
                        continue
                    return report
            event('framework_feedback', feedback)
        except ObservationRecovered:
            # Recovery has supplied a fresh trusted view. Do not repeat the
            # completed action or turn its result into a task-success claim.
            continue
        except (ValueError, SyntaxError) as exc:
            if external_callback_started:
                raise
            feedback = dict(phase='request_rejected', error=str(exc),
                            instruction='Correct your JSON/schema/candidate; the framework did not repair it. '
                            'Validation rejection does not change host permissions.')
            (folder / 'protocol-rejection.json').write_text(canonical(feedback))
            event('request_rejected', feedback)
    report['decision_limit_reached'] = True
    report['stopped_reason'] = 'bounded_episode_decision_limit'
    if lifecycle is not None:
        report['task_status'] = 'stopped_incomplete'
    return report
