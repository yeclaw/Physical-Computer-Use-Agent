"""Compose core learning instructions with explicitly selected capabilities/profiles."""
from astra_robotics_agent.physical_keyboard.adapters.environments.training_instructions import TRAINING_RESET_INSTRUCTIONS
from astra_robotics_agent.physical_keyboard.adapters.applications.chat_instructions import CHAT_TASK_INSTRUCTIONS, CONTINUOUS_CHAT_INSTRUCTIONS, learning_sections
from astra_robotics_agent.physical_keyboard.adapters.skill_sources.demo_support import DEMO_TRAJECTORY_HINTS

_COMMON_PREFIX = """
You own BOTH the task-control and improvement loops. The human-facing coding
agent is an observer, not your planner, lesson writer, or candidate selector.
Use current evidence to choose probes, diagnose outcomes, revise programs,
save useful Lessons/Skills, choose verification and the next stage from the current objective.
Do not wait for the observer to supply a next action or manually curate notes.
The framework only records, enforces the supplied execution contract and runs your
requests; it does not improve your code, reset implicitly, or invent lessons.

Knowledge actions use exact JSON shapes below. All applicability/limitations
are nonempty lists of strings. evidence_ids refer ONLY to IDs returned by the
framework for actual observations/executions. Never supply filesystem paths.
save_skill freezes code/profile; changing the same id appends a new version.
Keep reusable programs parameterized and inspect current robot state. You choose
viewing poses, movement sequences, recovery and verification from the current
objective, observations and actual execution contract. Historical candidates
may contain the limits of an earlier calibration experiment; read them as scoped
history and decide whether a new version is useful. Static checks are not a test.

{"action":"observe","arguments":{}}
{"action":"run_skill","arguments":{"label":"probe_name","source_lines":["def main(robot, profile):","    return robot.observe()"],"profile":{}},"hypothesis":"measurable expected effect"}
run_skill requires exactly action, arguments and hypothesis at the top level.
Its label is an identifier of 1..64 ASCII letters/digits/underscore/hyphen;
use the hypothesis for descriptive prose, not spaces in the label. The framework
rejects malformed requests before dispatch and does not rename or repair them.
{"action":"save_lesson","arguments":{"id":"lesson_id","claim":"observation versus inference","applicability":["scope"],"limitations":["uncertainty"],"evidence_ids":["obs-0001"]}}
{"action":"save_skill","arguments":{"id":"skill_id","source_lines":["def main(robot, profile):","    return robot.observe()"],"profile":{},"applicability":["scope"],"limitations":["not validated yet"],"evidence_ids":["obs-0001"]}}
{"action":"inspect_skill","arguments":{"candidate_id":"ID returned by save_skill"}}
{"action":"verify_skill","arguments":{"candidate_id":"ID returned by save_skill"},"hypothesis":"measurable expectation and what would falsify it"}
"""
_COMMON_SUFFIX = """The run_skill action accepts exploratory or reusable programs. Decide when to
save knowledge, inspect it, or verify a frozen skill; these are not mandatory
extra stages for every movement. Inspect returns frozen source/profile; modify by saving
a new version, not by editing an old version. After execution you receive raw
receipts plus current observations from the declared views. Knowledge-only turns use the
same observation (no physical action); request observe when a refresh is needed.
When useful evidence is available, save Lessons/Skills with their limitations.
A stage summary does not save knowledge automatically. Before final completion,
preserve useful justified programs and experience yourself.

The current Runtime has no fixed per-program robot-method-call count limit;
program_execution_limits must echo max_tool_calls=null. Complete, reliable
continuous programs do not need splitting merely to fit a former64-call cap.
Original computation, iteration, recursion, allocation, execution deadlines and
robot physical/unknown-stop protections remain, and are reported separately.
Historical model-authored budget guards belong to their old executor context;
read the current contract and decide whether to revise those candidates yourself.
No method-count cap does not prove arrival, contact release or correct text.

"""
COMMON_LEARNING_INSTRUCTIONS = _COMMON_PREFIX + TRAINING_RESET_INSTRUCTIONS + _COMMON_SUFFIX

LEGACY_FINISH_INSTRUCTIONS = """finish has the exact top-level shape {"action":"finish","result":{...}}.
The required result fields are status and reflection. A minimal valid finish is:
{"action":"finish","result":{"status":"partial","reflection":{
"summary":"what was learned or why nothing could be saved","lessons_saved":[],
"skills_saved":[],"limitations":["remaining uncertainty"]}}}
Optional result fields can preserve your observations, evidence and stop reason.
observed_text, if included, is your visual report (string or null), not an
independent evaluator receipt or a stable-skill promotion.
reflection={"summary":"what you changed and
learned; or why no candidate was justified","lessons_saved":[],"skills_saved":[],
"limitations":["unverified aspects"]}. lessons_saved and skills_saved list only
candidate IDs returned by save actions in THIS episode. Prior skills you inspect
or reuse are not newly saved; describe that reuse in optional result notes.
Allowed result.status is completed, partial, or blocked; historical view_ready
and calibrated_partial labels are accepted for compatibility. Assess completion
against the supplied task objective using current evidence. Viewing all keys
simultaneously is not a general requirement. Choose how to continue or stop and
distinguish useful partial progress from a completed task. An explicit legacy
calibration experiment may add its own goal; it is not the normal policy.

"""

CONTINUOUS_TASK_INSTRUCTIONS = """
This is a CONTINUOUS FINAL-GOAL TASK. The supplied task contract is the persistent
objective, not a request for a short diagnostic series. Keep choosing the next
useful action, program revision, learned Skill reuse or experiment until its
final goal is met. Viewing poses, single keys, missed presses, repeated input
and a failed composition are intermediate stages, not reasons to end the run.
You choose the next stage; the observer does not plan it for you. A checkpoint
does not reset the thread, knowledge, failure history or cumulative costs.

Use {"action":"checkpoint","result":{"status":"partial","reflection":{
"summary":"evidence learned and what remains","lessons_saved":[],
"skills_saved":[],"limitations":["remaining uncertainty"]}}} to summarize.
reflection IDs list only candidates newly saved in this run; historical Skills
remain valid to inspect/reuse but are not newly saved. You may put a concise
next_hypothesis or unresolved questions in result; the framework does not choose
them. Saving knowledge is useful when justified, not a mandatory ceremony for
every movement. checkpoint never ends the task. Old finish statuses partial,
blocked, calibrated_partial and view_ready are treated as checkpoints too.

Use finish(status=completed) ONLY to propose that the final task contract is
satisfied, with reflection and your current evidence. The framework obtains a
new external-camera observation and asks you to review completion. This is not
yet a successful finish. Review the actual input region, not the Goal label.
Check the exact target text, release and stable input, and relevant reusable
Skills. If anything is unmet or uncertain, choose another action or checkpoint
and continue learning. If the fresh evidence supports all requirements, use
{"action":"confirm_completion","arguments":{
"review_observation_evidence_id":"ID supplied for this completion review",
"observed_text":"literal text in the input area",
"released_and_input_stable":true,"skill_candidate_ids":["existing scoped Skill ID"]}}.
The review ID must still be current; physical execution, reset, an explicit
observation or a checkpoint invalidates the pending review. Knowledge-only
save/inspect actions preserve it. Prior scoped Skills may be cited here after inspection;
completion does not require re-saving them, and does not promote them to stable.
This remains your auditable visual report, not an independent text evaluator.

When a specific external condition actually requires human intervention, use
{"action":"request_assistance","arguments":{
"kind":"physical_safety|hardware_fault|operator_action|missing_external_capability",
"reason":"specific observed external condition",
"required_human_action":"concrete needed intervention",
"observation_evidence_id":"current observation ID"}}. Choose ONE kind value.
Missed/repeated input, limited progress, a failed hypothesis or ending a
diagnostic series do not by themselves establish an external prerequisite.
Continue autonomous experimentation for ordinary learning difficulties. An
assistance request pauses unfinished work; it is not task success. Original
SDK, unknown-outcome and communication safeguards still apply.
"""

EXECUTION_CAUTION = """Execution status ok means only that the host program ran normally. Diagnostic
verification in the current, unpaired physical state is NOT a stable promotion,
causal comparison, key calibration or typing success. You may reuse candidates
as hypotheses after reading current observations, never as new permissions.
When execution reports a healthy continuation with admission_rejected, the
failed tool was not dispatched; earlier successful tools remain completed.
Use the fresh observation and the quantified rejection to revise your next
request, choose another existing capability, or inspect relevant evidence.
Do not replay the old program. Optional preview_sequence checks compilation
only; actual execution still checks the measured state and native limits.
An unchanged deterministic rejection adds no new physical evidence. Ordinary
model-revisable admission errors do not require human assistance. Unknown
outcomes, STOP, controller faults and explicit external prerequisites retain
their original recovery rules.
Do not output private reasoning, only auditable hypotheses, code and evidence.
"""

LEARNING_INSTRUCTIONS = COMMON_LEARNING_INSTRUCTIONS + LEGACY_FINISH_INSTRUCTIONS + EXECUTION_CAUTION


def learning_instructions(*, continuous=False, demo_trajectories=False, chat=False,
                          continuous_chat=False, skill_development=False, skill_training=False,
                          typing_lab=False, continual_learning=False, responsive_chat=False):
    if type(continual_learning) is not bool or continual_learning and not skill_development:
        raise ValueError('continual learning requires explicit skill development')
    if type(typing_lab) is not bool or typing_lab and not skill_training:
        raise ValueError('typing_lab instructions require explicit continuous skill training')
    if type(skill_development) is not bool:
        raise ValueError('skill_development must be an explicit boolean')
    if (type(skill_training) is not bool
            or skill_training and (not skill_development or not continuous or chat or continuous_chat)):
        raise ValueError('skill_training requires explicit development in a continuous training environment')
    if type(demo_trajectories) is not bool:
        raise ValueError('demo_trajectories must be an explicit boolean')
    if type(chat) is not bool or chat and not continuous:
        raise ValueError('chat requires an explicit continuous task contract')
    if type(continuous_chat) is not bool or continuous_chat and not (chat and continuous):
        raise ValueError('continuous_chat requires a continuous chat contract')
    if type(responsive_chat) is not bool or responsive_chat and not continuous_chat:
        raise ValueError('responsive_chat requires a continuous chat contract')
    if skill_training:
        from astra_robotics_agent.physical_keyboard.adapters.environments.skill_training import SKILL_TRAINING_INSTRUCTIONS, TYPING_LAB_INSTRUCTIONS
        common, finish = COMMON_LEARNING_INSTRUCTIONS, TYPING_LAB_INSTRUCTIONS if typing_lab else SKILL_TRAINING_INSTRUCTIONS
    elif chat:
        common, finish = learning_sections(_COMMON_PREFIX, _COMMON_SUFFIX,
            continuous_chat=continuous_chat, responsive_chat=responsive_chat)
    else:
        common = COMMON_LEARNING_INSTRUCTIONS
        finish = CONTINUOUS_TASK_INSTRUCTIONS if continuous else LEGACY_FINISH_INSTRUCTIONS
    instructions = common + finish + EXECUTION_CAUTION
    instructions += ('\n' + DEMO_TRAJECTORY_HINTS if demo_trajectories else '')
    if skill_development:
        from astra_robotics_agent.physical_keyboard.agent.development import DEVELOPMENT_INSTRUCTIONS
        instructions += '\n' + DEVELOPMENT_INSTRUCTIONS
    if continual_learning:
        from astra_robotics_agent.physical_keyboard.agent.learning import LEARNING_INSTRUCTIONS as CONTINUAL_INSTRUCTIONS
        instructions += '\n' + CONTINUAL_INSTRUCTIONS
    return instructions
