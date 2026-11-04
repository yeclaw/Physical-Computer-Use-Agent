"""Model-owned, knowledge-only review after control dispatch is disabled.

The caller must disable dispatch and attempt to close the old controller first.
Optional callbacks read saved diagnostics and the existing recorder cache only.
They never provide controller state or execution authority.
"""
from __future__ import annotations

import hashlib
from itertools import count
import json
from pathlib import Path
import time
import uuid

from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions, KNOWLEDGE_ACTIONS
from astra_robotics_agent.physical_keyboard.records.run_recording import RunRecorder
from astra_robotics_agent.physical_keyboard.records.typing_metrics import validate_case_identity


REVIEW_ACTIONS = (KNOWLEDGE_ACTIONS - {'record_skill_outcome'}) | {
    'save_skill', 'save_lesson', 'inspect_skill', 'finish', 'checkpoint',
}

FAULT_REVIEW_INSTRUCTIONS = '''
FAULT REVIEW ONLY: further physical dispatch is disabled; closing the old host
was attempted. Inspect control_cleanup for its actual receipt or failure; this
does not prove physical release. Continue this SAME session using historical evidence. The
robot and editor's current state are UNKNOWN. last_observation is the OLD image
from BEFORE the failed execution; it does not show the outcome or current pose.
There is no fresh controller state and no task-completion review in this phase.
If inspect_recorder is listed, it reads new cached camera frames only. These may
help diagnose visible effects but never clear the fault or authorize motion.

You decide which unresolved issues to record, merge, prioritize or defer, and
which candidate revisions are justified. The host has not diagnosed a repair.
Distinguish raw fault facts, hypotheses and untested code. Preserve uncertainty
and choose later verification conditions. Saving a program does not run, verify
or promote it, resolve the hardware fault, or establish task success.

Only allowed actions are those in allowed_actions. All other actions and ALL
native tools are rejected without dispatch. In particular observe, run_skill,
verify_skill, call_skill, reset, wait_for_update, confirm_completion and reconnect
are unavailable. record_skill_outcome is also unavailable: an old image cannot
substitute for a post-execution task outcome. No request is queued for later.

Use the existing exact knowledge-action schemas. Selected examples:
{"action":"search_knowledge","arguments":{"query":"capability or fault","kind":"all","limit":8}}
{"action":"inspect_knowledge","arguments":{"identifier":"registered knowledge ID"}}
{"action":"list_improvements","arguments":{"status":"pending","limit":8,"offset":0}}
{"action":"recall_skills","arguments":{"skill_ids":["registered skill/package ID"],"limit":4}}
{"action":"record_improvement","arguments":{"id":"your_issue_id","summary":"evidence-backed opportunity","rationale":"why it matters","status":"open","skill_ids":[],"evidence_ids":["registered ID"]}}
Optional priority (integer 0..100) and revisit_when (list of concise conditions)
are YOUR choices. Updating the same id appends a version. For a duplicate use
status=merged and merged_into=the canonical registered improvement record ID.
This reports your assessment, not independent verification or stable promotion.
{"action":"inspect_evidence","arguments":{"evidence_id":"registered ID"}}
Default evidence inspection returns file metadata. Request file_indices (at most
4) for bounded text/source; include_images=true requests at most 2 selected
historical JPEG/PNG files. Historical images NEVER become current observations.
The registered fault files preserve the raw error and any attempted source and
profile. Inspect their content on demand rather than inferring missing details.
For longer text select exactly one file and use text_offset and text_limit
(1..8000 Unicode codepoints); next_offset continues the text. For JSON, optionally
select a field with json_pointer (RFC6901); paging applies to that selected JSON.
failure_diagnostics, when present, contains saved execution/plan/source facts,
including uncertainty. Inspect its registered full evidence on demand. You own
the repair hypothesis and candidate; a bridge label does not prove physical cause.
Optional read-only actions, ONLY when listed:
{"action":"refresh_fault_diagnostics","arguments":{}}
{"action":"inspect_recorder","arguments":{}}
{"action":"save_lesson","arguments":{"id":"your_lesson","claim":"facts and hypotheses distinguished","applicability":["scope"],"limitations":["not a new physical result"],"evidence_ids":["registered ID"]}}
{"action":"save_skill","arguments":{"id":"your_candidate","source_lines":["def main(robot, profile):","    return None"],"profile":{},"applicability":["scope"],"limitations":["unverified candidate"],"evidence_ids":["registered ID"]}}
{"action":"inspect_skill","arguments":{"candidate_id":"registered Skill candidate ID"}}
save_skill_package retains its existing exact schema: id, source_lines, profile,
contract={description,input_schema,preconditions,postconditions}, applicability,
limitations,evidence_ids; optional parent_id,dependencies,improvement_ids. You
choose the interface and implementation. Saving remains inert and unverified.

To end this REVIEW, use exactly:
{"action":"finish","result":{"status":"partial","reflection":{"summary":"review findings or why none were justified","lessons_saved":[],"skills_saved":[],"limitations":["hardware state and candidate effectiveness remain unknown"]}}}
Only partial or blocked status is allowed; observed_text must be absent/null.
reflection candidate IDs name only candidates saved in THIS fault review.
checkpoint uses the same result shape but continues this review. Neither action
completes the physical task or clears its original stopped_reason. Save only
auditable findings, declarations and plans; do not output private reasoning.
'''


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _excerpt(value, maximum=2400):
    text = _canonical(value)
    return dict(text=text[:maximum], truncated=len(text) > maximum)


def run_fault_review(*, session, store, output: Path, fault: dict, status=None,
                     case_identity=None, max_decisions=None, trace_event=None,
                     diagnostics_provider=None, passive_observe=None,
                     continual_learning=False, learning_sources=None, learning_access=None,
                     stop_requested=None) -> dict:
    """Persist actual fault evidence and let Astra choose inert knowledge actions.

    ``status`` only publishes progress (and may raise an operator-stop signal).
    ``trace_event`` appends model usage to the parent's typing metrics. Neither
    callback is a hardware tool. Model/schema rejections permit a corrected
    model decision; model, recording and operator-stop exceptions propagate.
    """
    if max_decisions is not None and (type(max_decisions) is not int or max_decisions < 1):
        raise ValueError('Explicit max_decisions must be a positive integer, or None')
    if type(continual_learning) is not bool:
        raise ValueError('continual_learning must be an explicit boolean')
    if not continual_learning and (learning_sources is not None or learning_access is not None):
        raise ValueError('learning source/access configuration requires continual learning')
    if stop_requested is not None and not callable(stop_requested):
        raise ValueError('stop_requested must be a callable or None')
    for callback in (diagnostics_provider, passive_observe):
        if callback is not None and not callable(callback):
            raise ValueError('Read-only evidence providers must be callable')
    if (type(fault) is not dict or type(fault.get('stopped_reason')) is not str
            or not fault['stopped_reason'].strip()):
        raise ValueError('fault requires the original nonempty stopped_reason')
    # Strict JSON snapshots prevent caller/model mutation from changing facts.
    raw_fault = _canonical(fault)
    fault = json.loads(raw_fault)
    if continual_learning:
        from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
        learning_access = learning_access or store.learning_access or LearningAccess(store._scope)
        store.learning_access = learning_access
    case_identity = None if case_identity is None else validate_case_identity(case_identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fault_path = output / 'fault.json'
    with fault_path.open('x', encoding='utf-8') as stream:
        stream.write(raw_fault + '\n')
    files = [fault_path]
    attempted = fault.get('attempted_program')
    if type(attempted) is dict:
        if type(attempted.get('source')) is str:
            path = output / 'attempted-source.py'
            with path.open('x', encoding='utf-8') as stream:
                stream.write(attempted['source'])
            files.append(path)
        if type(attempted.get('profile')) is dict:
            path = output / 'attempted-profile.json'
            with path.open('x', encoding='utf-8') as stream:
                stream.write(_canonical(attempted['profile']) + '\n')
            files.append(path)
    fault_id = 'fault-review-' + uuid.uuid4().hex
    registration = store.register_evidence(fault_id, files)
    development = DevelopmentActions(store)
    continual = None
    learning_actions = set()
    learning_instructions = ''
    boundary = None
    boundary_source = None
    if continual_learning:
        from astra_robotics_agent.physical_keyboard.agent.learning import (
            ContinualLearning, LEARNING_ACTIONS, LEARNING_INSTRUCTIONS,
        )
        from astra_robotics_agent.physical_keyboard.records.learning_contracts import digest
        continual = ContinualLearning(development, session, output=output / 'continual-learning',
                                     sources=learning_sources, access=learning_access)
        learning_actions = set(LEARNING_ACTIONS)
        learning_instructions = LEARNING_INSTRUCTIONS
        if 'learning_boundary' in fault:
            boundary = continual.working._facts(fault['learning_boundary'])
            boundary_source = 'frozen_run_boundary'
        else:
            boundary_source = 'fault_review_fallback_not_original_task'
            boundary = dict(task_contract_sha256=digest(dict(boundary_source=boundary_source,
                scope=store._scope, fault_sha256=hashlib.sha256(raw_fault.encode()).hexdigest())),
                observation_evidence_id=None, generation=0, last_execution=None,
                motion_available=False, inflight_rpc=False, inflight_action=False, inflight_model=False,
                stop_requested=False, assistance=None,
                fault=dict(stopped_reason=fault['stopped_reason'], physical_outcome='unknown',
                           execution_status=fault.get('execution_status','unknown')),
                source_policy_sha256=digest(continual.access.descriptor()))
    event = RunRecorder(output, serializer=_canonical).event
    event('fault_review_evidence_registered', registration)

    last = fault.get('last_observation')
    old_id = last.get('evidence_id') if type(last) is dict else None
    registered_ids = [fault_id]
    reported = fault.get('feedback', {})
    reported = reported if type(reported) is dict else {}
    referenced = reported.get('evidence_ids', [])
    referenced = referenced if type(referenced) is list else []
    for identifier in [old_id] + referenced:
        if type(identifier) is str and identifier not in registered_ids:
            try:
                store._evidence(identifier)
            except ValueError:
                continue
            registered_ids.append(identifier)
    historical = dict(evidence_id=old_id if old_id in registered_ids else None,
                      historical=True, timing='pre_execution_not_current', current=False,
                      current_state='unknown', images_available_on_request=True)
    facts = {key: fault[key] for key in ('error_type', 'error', 'execution_status') if key in fault}
    summary = dict(stopped_reason=fault['stopped_reason'],
                   control_cleanup=fault.get('control_cleanup'),
                   exception_facts=_excerpt(facts),
                   execution_response_excerpt=_excerpt(reported.get('execution')))
    report = dict(protocol='physical-keyboard-fault-review-v1',
                  stopped_reason=fault['stopped_reason'], task_completed=False, completed=False,
                  task_status='stopped_incomplete', current_state='unknown',
                  fault_evidence_id=fault_id, evidence_ids=registered_ids,
                  last_observation=historical, saved_candidates=[], improvements=[],
                  skill_packages=[], verifications=[], stable_promoted=False,
                  independently_verified=False, finished_by_model=False, diagnostic_only=True,
                  decision_owner='astra', max_decisions=max_decisions, turns=[])
    if continual is not None:
        report.update(continual_learning=continual.descriptor(),learning_boundary_source=boundary_source,
                      learning_actions=[],learning_commits=[],working_context_operations=[])
    saved, pending_content = {}, []
    feedback = dict(phase='fault_review_begin', fault_evidence_id=fault_id,
                    stopped_reason=fault['stopped_reason'], dispatched=False)

    allowed_actions = set(REVIEW_ACTIONS)
    allowed_actions.update(learning_actions)
    diagnostic_count, passive_count = 0, 0
    if diagnostics_provider is not None:
        allowed_actions.add('refresh_fault_diagnostics')
    if passive_observe is not None:
        allowed_actions.add('inspect_recorder')

    def read_diagnostics():
        nonlocal diagnostic_count
        diagnostic_count += 1
        try:
            result = diagnostics_provider(fault, output / ('diagnostics-%02d' % diagnostic_count))
            if type(result) is not dict or type(result.get('evidence_id')) is not str:
                error = result.get('error') if type(result) is dict else None
                raise ValueError(error or 'Diagnostic provider did not register evidence')
            store._evidence(result['evidence_id'])
            if result['evidence_id'] not in registered_ids:
                registered_ids.append(result['evidence_id'])
            result = {key: result[key] for key in ('success', 'status', 'evidence_id', 'summary', 'error') if key in result}
        except Exception as exc:
            result = dict(success=False, status='diagnostic_unavailable', error=str(exc))
        event('fault_review_saved_diagnostics', result)
        report.setdefault('diagnostics', []).append(result)
        return result

    failure_diagnostics = read_diagnostics() if diagnostics_provider is not None else None

    def read_recorder():
        nonlocal passive_count
        passive_count += 1
        try:
            result = passive_observe(output / ('recorder-%02d' % passive_count))
            if type(result) is not dict or type(result.get('evidence_id')) is not str:
                error = result.get('error') if type(result) is dict else None
                raise ValueError(error or 'Passive recorder provider did not register evidence')
            store._evidence(result['evidence_id'])
            if result['evidence_id'] not in registered_ids:
                registered_ids.append(result['evidence_id'])
            items = result.get('contentItems', [])
            if type(items) is not list or any(type(item) is not dict or item.get('type') not in ('inputText', 'inputImage')
                    or type(item.get('text' if item['type'] == 'inputText' else 'imageUrl')) is not str for item in items):
                raise ValueError('Invalid passive recorder content')
            metadata = {k: v for k, v in result.items() if k != 'contentItems'}
            if continual is not None:
                metadata.update(historical=True,current=False,motion_available=False,
                    current_observation_evidence_id=None,authority='passive_evidence_not_controller_state')
                items = [dict(type='inputText',text='HISTORICAL passive recorder evidence. '
                    'No current controller state, motion permission or fault recovery.')] + items
            return metadata, items
        except Exception as exc:
            result = dict(success=False, status='passive_recorder_unavailable', error=str(exc))
            if continual is not None:
                result.update(historical=True,current=False,motion_available=False,
                              current_observation_evidence_id=None)
            event('fault_review_passive_evidence_unavailable', result)
            return result, []

    def check_stop():
        if stop_requested is not None:
            requested=stop_requested()
            if type(requested) is not bool:
                raise ValueError('stop_requested callback must return an explicit boolean')
            if requested:
                from astra_robotics_agent.physical_keyboard.adapters.applications.chat_service import OperatorStop
                raise OperatorStop('Operator STOP during knowledge-only fault review')

    def learning_facts():
        from astra_robotics_agent.physical_keyboard.records.learning_contracts import digest
        current=json.loads(_canonical(boundary))
        current.update(observation_evidence_id=None,motion_available=False,
                       source_policy_sha256=digest(continual.access.descriptor()))
        if current['fault'] is None:
            current['fault']=dict(stopped_reason=fault['stopped_reason'],physical_outcome='unknown')
        if stop_requested is not None:
            requested=stop_requested()
            if type(requested) is not bool:
                raise ValueError('stop_requested callback must return an explicit boolean')
            current['stop_requested']=current['stop_requested'] or requested
        return continual.working._facts(current)

    def record_learning_commit(result):
        report['learning_commits'].append(result)
        for identifier in result.get('record_ids',[]):
            if identifier.startswith(('skill-','lesson-')):
                candidate=store._candidate(identifier)
                saved[identifier]=candidate['kind']
                descriptor=store._summary(candidate,store.namespace/'candidates'/identifier)
                report['saved_candidates'].append(descriptor)
            elif identifier.startswith('package-'):
                package=development.library.inspect(identifier)
                saved[package['candidate_id']]='skill'
                descriptor=development.library._summary(package)
                report['saved_candidates'].append(descriptor)
                report['skill_packages'].append(descriptor)

    def persist():
        (output / 'report.json').write_text(_canonical(report) + '\n', encoding='utf-8')

    def no_native_tools(name, arguments):
        rejection = dict(phase='request_rejected', requested_tool=name, dispatched=False,
                         error='Native tools are unavailable during knowledge-only fault review')
        event('fault_review_native_tool_rejected', rejection)
        return dict(success=False, contentItems=[dict(type='inputText', text=_canonical(rejection))])

    def usage_event(index, usage, elapsed):
        entry = dict(case=case_identity, turn_index=index, usage=usage, elapsed_seconds=elapsed)
        event('typing_lab_model_turn', entry)
        if trace_event is not None:
            trace_event('typing_lab_model_turn', entry)

    persist()
    try:
        for index in count():
            check_stop()
            if max_decisions is not None and index >= max_decisions:
                report.update(decision_limit_reached=True, review_stopped_reason='decision_limit')
                break
            folder = output / ('turn-%02d' % index)
            folder.mkdir()
            context = dict(phase='fault_review', allowed_actions=sorted(allowed_actions),
                           fault=summary, fault_evidence_id=fault_id,
                           registered_evidence_ids=registered_ids, last_observation=historical,
                           current_observation_evidence_id=None, current_state='unknown',
                           task_completed=False, motion_available=False,
                           remaining_decisions=None if max_decisions is None else max_decisions-index,
                           feedback=feedback, skill_development=(development.context() if continual is None else
                               dict(enabled=True, online_recall=(development.recall.for_skills(development.focus_skill_ids)
                                    if development.focus_skill_ids else {}))))
            if continual is not None:
                context['continual_learning']=continual.context(None)
                context['learning_boundary']=learning_facts()
                context['learning_boundary_source']=boundary_source
            if failure_diagnostics is not None:
                context['failure_diagnostics'] = failure_diagnostics
            items, pending_content = pending_content, []
            images = [item['imageUrl'] for item in items if item['type'] == 'inputImage']
            prompt = _canonical(context) + '\n'
            if index == 0:
                prompt += FAULT_REVIEW_INSTRUCTIONS
                prompt += learning_instructions
            prompt += '\n'.join(item['text'] for item in items if item['type'] == 'inputText')
            (folder / 'request.txt').write_text(prompt, encoding='utf-8')
            event('fault_review_model_request', dict(turn_index=index, prompt=prompt,
                historical_image_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in images],
                evidence_ids=registered_ids))
            if status is not None:
                status(dict(phase='fault_review_model_inference', model_turns=len(report['turns']),
                            stopped_reason=fault['stopped_reason']))
            started = time.monotonic()
            try:
                reply = session.run_turn(prompt, no_native_tools, images=images)
            except BaseException:
                usage_event(index, None, round(time.monotonic()-started, 6))
                raise
            elapsed = round(time.monotonic()-started, 6)
            usage_event(index, reply.get('usage'), elapsed)
            # Persist protocol output only, never provider-internal reasoning.
            public_reply = {key: reply[key] for key in ('turn_id', 'usage', 'final_text') if key in reply}
            (folder / 'reply.json').write_text(_canonical(public_reply), encoding='utf-8')
            event('fault_review_model_reply', dict(turn_index=index, reply=public_reply))
            turn = dict(index=index, turn_id=reply.get('turn_id'), usage=reply.get('usage'),
                        elapsed_seconds=elapsed)
            report['turns'].append(turn)
            development.presented(context)
            finish_review = False
            try:
                # Lazy imports preserve learning_loop's optional integration.
                from astra_robotics_agent.physical_keyboard.agent.learning_loop import parse_decision, validate_finish
                decision = parse_decision(reply['final_text'], skill_development=True, fault_read_only=True,
                                          continual_learning=continual_learning)
                action = decision['action']
                turn['action'] = action
                event('fault_review_model_decision', dict(turn_index=index, decision=decision))
                if action not in allowed_actions:
                    raise ValueError('Action ' + action + ' is unavailable during knowledge-only fault review; no dispatch')
                if continual is not None and action not in learning_actions:
                    continual.validate_legacy(decision.get('arguments',{}))
                if action in learning_actions:
                    facts=learning_facts()
                    maintenance = action == 'manage_working_context'
                    before_usage = json.loads(_canonical(getattr(session, 'total_usage', None)))
                    before_complete = getattr(session, 'total_usage_complete', False)
                    maintenance_started = time.monotonic()
                    operation_status = 'failed'
                    try:
                        feedback,pending_content=continual.handle(action,decision['arguments'],facts=facts)
                        operation_status = feedback['result'].get('status', 'completed')
                    finally:
                        if maintenance:
                            from astra_robotics_agent.physical_keyboard.records.usage import usage_delta
                            cost = dict(case=case_identity, turn_index=index,
                                method=(decision['arguments'].get('method') if type(decision['arguments'].get('method')) is str
                                    and decision['arguments']['method'] in {'native_compact', 'rebuild'} else None), status=operation_status,
                                elapsed_seconds=round(time.monotonic() - maintenance_started, 6),
                                usage=usage_delta(before_usage, getattr(session, 'total_usage', None)),
                                usage_complete=bool(before_complete and getattr(session, 'total_usage_complete', False)))
                            event('fault_review_context_operation_cost', cost)
                            if trace_event is not None:
                                trace_event('typing_lab_context_operation', cost)
                    report['learning_actions'].append(feedback)
                    if action=='commit_learning':record_learning_commit(feedback['result'])
                    if feedback['result'].get('fresh_observation_required') is True:
                        check_stop()
                        if facts!=learning_facts():
                            raise ValueError('Fault authority changed during context operation; no action is resumed')
                        if passive_observe is not None:
                            passive,pending_content=read_recorder()
                            report.setdefault('passive_camera_reads',[]).append(dict(
                                reason='context_refresh',result=passive,current_observation_evidence_id=None,
                                current_state='unknown',motion_available=False))
                            event('fault_review_context_passive_refresh',passive)
                        else:
                            passive=dict(success=False,status='passive_recorder_unavailable',
                                historical=True,current=False,motion_available=False,
                                reason='No passive recorder provider; no current observation or motion authority')
                        feedback.update(dispatched=False,current_state='unknown',motion_available=False,
                            current_observation_evidence_id=None,passive_observation=passive,
                            fresh_observation_required=True)
                        from astra_robotics_agent.physical_keyboard.records.learning_contracts import encoded
                        feedback['receipt'].update(extra_content_json_bytes=len(encoded(pending_content,16*1024*1024)),
                            historical_image_count=sum(item.get('type')=='inputImage' for item in pending_content),
                            passive_refresh_included=True)
                        report['working_context_operations'].append(feedback)
                    feedback.update(dispatched=False,current_state='unknown',motion_available=False,
                                    current_observation_evidence_id=None)
                elif action in ('finish', 'checkpoint'):
                    result = decision['result']
                    validate_finish(result, saved)
                    if result['status'] not in ('partial', 'blocked') or result.get('observed_text') is not None:
                        raise ValueError('Fault review permits only partial/blocked reflection without current observed_text')
                    feedback = dict(action=action, result=result, dispatched=False,
                                    current_state='unknown', task_completed=False)
                    if action == 'finish':
                        report.update(reply=result, finished_by_model=True, review_stopped_reason='model_finish')
                        finish_review = True
                elif action == 'refresh_fault_diagnostics':
                    failure_diagnostics = read_diagnostics()
                    feedback = dict(action=action, result=failure_diagnostics, dispatched=False,
                                    current_state='unknown', motion_available=False)
                elif action == 'inspect_recorder':
                    result, pending_content = read_recorder()
                    feedback = dict(action=action, result=result,
                                    dispatched=False, current_state='unknown', motion_available=False,
                                    current_observation_evidence_id=None)
                    report.setdefault('passive_camera_reads', []).append(feedback)
                elif action in KNOWLEDGE_ACTIONS:
                    if continual is not None and action=='save_skill_package':
                        feedback,pending_content=dict(action=action,result=continual.save_package(decision['arguments']),
                            observation_unchanged=True),[]
                    else:
                        feedback, pending_content = development.knowledge_action(action, decision['arguments'], None)
                    if continual is not None:feedback=continual.published_feedback(feedback)
                    if action == 'save_skill_package':
                        package = feedback['result']
                        saved[package['candidate_id']] = 'skill'
                        report['saved_candidates'].append(package)
                        report['skill_packages'].append(package)
                    elif action == 'record_improvement':
                        report['improvements'].append(feedback['result'])
                elif action in ('save_skill', 'save_lesson'):
                    kind = 'skill' if action == 'save_skill' else 'lesson'
                    candidate = store.propose(kind, decision['arguments'])
                    saved[candidate['candidate_id']] = kind
                    report['saved_candidates'].append(candidate)
                    feedback = dict(action=action, candidate=candidate, dispatched=False)
                else:  # inspect_skill is an inert scoped read.
                    identifier = decision['arguments']['candidate_id']
                    feedback = dict(action=action, skill=store.get_skill(identifier), dispatched=False)
                    development.focus_skill_ids = [identifier]
                event('fault_review_action_result', dict(turn_index=index, feedback=feedback))
            except (ValueError, SyntaxError) as exc:
                feedback = dict(phase='request_rejected', error=str(exc), dispatched=False,
                    instruction='Correct your JSON/schema/knowledge request; no action was dispatched or queued. '
                                'The current robot state remains unknown and permissions remain knowledge-only.')
                (folder / 'protocol-rejection.json').write_text(_canonical(feedback), encoding='utf-8')
                event('fault_review_request_rejected', dict(turn_index=index, feedback=feedback))
            event('fault_review_turn_evidence', dict(turn_index=index, evidence_ids=registered_ids,
                                                    current_observation_evidence_id=None))
            persist()
            if finish_review:
                break
    except BaseException as exc:
        report.update(review_stopped_reason='review_interrupted', review_error_type=type(exc).__name__)
        persist()
        raise
    persist()
    return report
