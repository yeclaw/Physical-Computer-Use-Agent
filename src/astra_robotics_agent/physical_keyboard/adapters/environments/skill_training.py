"""Open-ended training-editor skill development; no per-key plan or live IO."""
import re


PROTOCOL = 'physical-keyboard-skill-development-contract-v1'

SKILL_TRAINING_INSTRUCTIONS = '''
CONTINUOUS CAPABILITY DEVELOPMENT — NONTERMINAL TRAINING PHASES
Choose tasks, useful experiments, program/interface revisions and reuse from
the supplied development objective and actual evidence. The objective is not
a single fixed target sentence. A corrected input and an improved reusable
capability are separate results. Preserve justified knowledge and outstanding
opportunities; decide your own next investigation or use of existing skills.
'''

SKILL_TRAINING_PHASE_PROTOCOL = '''
Use {"action":"checkpoint","result":{"status":"partial","reflection":{
"summary":"evidence and remaining opportunities","lessons_saved":[],
"skills_saved":[],"limitations":["remaining uncertainty"]}}} to record a phase.
reflection lists only candidate IDs newly saved in this episode; reused older
skills can be described in notes. Package saves return a candidate_id for this
list as well as a package_id for parameterized calls.
finish has the same result/reflection schema. All ordinary finish statuses,
including completed, are NONTERMINAL in this training environment. A completed
finish records the current phase without repeating it or ending the session.
No fixed-text completion review is needed. Continue choosing useful work until
the operator stops or a real execution/safety/communication condition stops it.

For a concrete external prerequisite, use
{"action":"request_assistance","arguments":{
"kind":"physical_safety|hardware_fault|operator_action|missing_external_capability",
"reason":"specific observed condition","required_human_action":"needed intervention",
"observation_evidence_id":"current trusted observation ID"}}.
Choose one kind. Ordinary failed input, a failed hypothesis, uncertain geometry
or a finished diagnostic series alone does not establish an external blocker.
This interface does not relax original robot/unknown-motion stop semantics.
'''

SKILL_TRAINING_INSTRUCTIONS += SKILL_TRAINING_PHASE_PROTOCOL

TYPING_LAB_INSTRUCTIONS = '''
CONTINUOUS PHYSICAL TYPING LAB — SUPPLIED TASKS AND ONLINE IMPROVEMENT
The application supplies the current target on screen. Read it using the external
cameras, type that target physically, check the input and press Return to submit.
Do not replace it with a self-chosen word or infer the next target from memory.
Release Return and observe the next screen before entering another case.
You own how to execute, improve, diagnose and verify reusable skills while
continuing these tasks. Keep the same backlog, skills and lessons across cases;
a compensating edit does not resolve the underlying defect. Repeated cases are
opportunities for you to use evidence and improve, not a host-specified repair.
Independent scoring may log editor text and keyboard events, but no text-reader
or synthetic typing tool is provided to you. Case metadata is for records only.
No fixed phrase, completed case or batch summary ends this continuous session.
''' + SKILL_TRAINING_PHASE_PROTOCOL


def validate_document(document, expected_scope):
    expected = {'protocol', 'id', 'version', 'scope', 'objective', 'application',
                'session_mode', 'development_criteria'}
    if type(document) is not dict or set(document) != expected:
        raise ValueError('Skill-development contract requires exact fields')
    if (document['protocol'] != PROTOCOL or document['application'] != 'training_notepad'
            or document['session_mode'] != 'continuous'):
        raise ValueError('Skill development requires explicit continuous training_notepad')
    if (type(document['id']) is not str
            or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', document['id']) is None
            or type(document['version']) is not int or document['version'] < 1):
        raise ValueError('Invalid skill-development contract identity/version')
    if document['scope'] != expected_scope:
        raise ValueError('task/device/layout scope does not match this campaign')
    if (type(document['objective']) is not str or not document['objective'].strip()
            or len(document['objective']) > 4096):
        raise ValueError('Nonempty bounded skill-development objective required')
    criteria = document['development_criteria']
    if (type(criteria) is not list or not 1 <= len(criteria) <= 16
            or any(type(x) is not str or not x.strip() or len(x) > 1024 for x in criteria)):
        raise ValueError('Skill-development criteria must be 1..16 bounded strings')


def contract_text(document_json):
    return ('CONTINUOUS SKILL-DEVELOPMENT TRAINING — MODEL-OWNED IMPROVEMENT\n'
            + document_json
            + '\nChoose useful tasks, experiments, skill/interface revisions and verification '
            'from evidence. Correct results and reusable capability quality are separate. '
            'A completed phrase, a saved program or a phase summary does not end this '
            'training session. finish(completed) records a nonterminal phase; continue '
            'until the operator stops or the original execution contract requires stopping. '
            'You own the improvement choices. No specific key fix, vocabulary, contact '
            'window, ordering or number of failures is prescribed by the host. This '
            'training environment permits exploration and explicit configured reset; '
            'do not assume those privileges in another application.\n')


class SkillTrainingApplication:
    continuous = False  # No conversation/visual-message waiter is needed.
    persistent_completion = True

    def validate_callbacks(self, reset, wait_for_update):
        pass

    def initialize_report(self, report):
        report.update(development_phase_records=[], session_mode='continuous_skill_development')

    def context_fields(self, report):
        return {'skill_training': dict(mode='continuous',
                recorded_phases=len(report['development_phase_records']), session_ended=False)}

    def record_phase(self, result, observation_evidence_id, lifecycle, report):
        lifecycle.invalidate_completion_review()
        entry = dict(result=result, terminal=False, observation_evidence_id=observation_evidence_id)
        report['development_phase_records'].append(entry)
        report['task_status'] = 'developing_skills'
        return entry, dict(phase='skill_development_phase_recorded', **entry,
            instruction='This records a training phase, not the end of skill development. '
            'Use the accumulated evidence and unresolved opportunities to choose your '
            'next task, revision or verification. Do not assume an input correction '
            'resolved its underlying skill defect.')

    def recoverable_assistance(self, request):
        return False
