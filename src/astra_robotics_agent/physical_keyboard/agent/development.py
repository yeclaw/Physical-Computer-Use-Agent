"""Model-owned online improvement with scoped, bounded skill memory."""
import json
import re

from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.recall import SkillRecall


KNOWLEDGE_ACTIONS = frozenset({
    'search_knowledge', 'inspect_knowledge', 'save_skill_package',
    'record_improvement', 'record_skill_outcome', 'inspect_evidence',
    'recall_skills', 'list_improvements', 'save_application_skill',
})
DEVELOPMENT_ACTIONS = KNOWLEDGE_ACTIONS | {'call_skill', 'application_skill'}

DEVELOPMENT_INSTRUCTIONS = '''
OPTIONAL MODEL-OWNED SKILL DEVELOPMENT
You own the improvement decisions as well as current task execution. Judge both
the present result and the reusable capability that produced it. Compensating
for an unwanted effect can finish the present task while leaving a worthwhile
skill defect unresolved. Successful but unnecessarily fragmented or expensive
execution can also offer an improvement opportunity. Choose what is worth
investigating, when, and how from evidence and the task's allowed environment.
Do not wait for the coding observer to choose your experiments or write skills.
There is no fixed failure-count trigger, required optimization order or mandatory
reflection call after each action. Keep the task moving and improve reusable
capabilities when justified. Ordinary uncertainty does not erase prior evidence.

Maintain your own improvement BACKLOG during normal application work. Completing
an input with a compensating action does not erase a reusable defect. Preserve
worthwhile unresolved defects or recurring inefficiency, connect them to the
registered skills involved, and state when they should be revisited. At later
relevant use, actively consider repairing or improving the capability using
that evidence, then inspect the actual effect. Choose the method, priority and
timing yourself; you may perform a short diagnostic where the application permits
it or explain why an item is deferred. Keep working on supplied application
tasks; the improvement loop operates within that work. It does not require a
separate training phase, a new model, or a reflection after every character.

You may consolidate equivalent backlog items by revising one canonical item's
description, skill/evidence links and priority, then explicitly mark duplicates
merged into it. Preserve each original record and uncertainty. Priorities and
revisit conditions come from you, not the observer or an automatic failure count.

You may generate your own skill description, parameter interface, entry/exit
conditions, implementation and dependency composition. The framework only
validates/persists declarations and links your declared programs mechanically;
it does not discover a contact point, choose a repair, or grant new robot tools.
Conditions are hypotheses until supported by actual observations. Never equate
schema validation, SDK ok, saved code, or your own report with independent truth.
Reuse can generalize to new parameter values/combinations where conditions hold;
seeing a complete word before is not a requirement. Choose suitable verification
and watch for regressions in component skills and transitions.

Exact additional JSON actions (enabled only in this run):
{"action":"search_knowledge","arguments":{"query":"your current capability or problem","kind":"all","limit":8}}
Search kinds: all, skill, lesson, package, workflow, improvement, outcome. Empty query is
allowed. Search is across scoped old/new knowledge, not just the recent summary.
{"action":"inspect_knowledge","arguments":{"identifier":"returned ID"}}
Inspect reads full declarations/limitations/evidence or a saved Lesson, not motion.
{"action":"list_improvements","arguments":{"status":"pending","limit":8,"offset":0}}
This pages the CURRENT version of each backlog item, ordered by your priorities.
Statuses include pending (open/reopened/deferred), all, or one individual status.
Use next_offset to see more instead of loading every historical version. Inspect
full records/evidence as needed to decide a merge, reordering or scoped revision.
{"action":"recall_skills","arguments":{"skill_ids":["registered skill or package ID"],"limit":4}}
Recall finds scoped unresolved items, related outcomes and candidate revisions
through your declared versions/parents and downward composition dependencies.
It is read-only, not a command to execute or to replace a skill automatically.
Before relevant reuse, inspect this compact memory and use it in your choice.
For an inline run_skill that adapts/reuses existing skills, include optional
arguments.related_skill_ids (at most 16 registered skill/package IDs). This is
host-side memory metadata and does not change the robot program or permissions.
Exact stored source can also be recognized mechanically. Arbitrary new source
is not semantically understood by the host; declare its relevant skill links.
An unseen related backlog may return phase=related_skill_memory, dispatched=false
before motion. No request is queued: use the memory and choose your next action.
Unchanged memory already presented does not force another review. Seeing memory
does not prove that a hypothesis or a repair works; verify with actual evidence.
{"action":"record_improvement","arguments":{"id":"your_issue_id","summary":"observed opportunity","rationale":"why improving it could matter","status":"open","skill_ids":[],"evidence_ids":["registered ID"]}}
Optional priority is integer 0..100 (higher first, default 50); revisit_when is
a list of up to 8 concise conditions in your words. Update the same id to change
its priority, scope or plan; omitted priority/revisit conditions retain the last
values. To merge an item, use status=merged and merged_into=another registered
improvement record ID after maintaining that canonical item. Merge does not
invent a combined diagnosis/evidence or mark the canonical issue resolved.
Statuses open/deferred/resolved/reopened/merged are your evidence-backed reports, not a
host reliability promotion. Updating the same id appends history. An input fix or
a successful run never automatically closes an improvement. Select and revise
your own issues, including opportunities discovered during successful work.
{"action":"save_skill_package","arguments":{"id":"your_skill_name","source_lines":["def main(robot, profile):","    return robot.observe()"],"profile":{},"contract":{"description":"your capability","input_schema":{"type":"object","properties":{},"additionalProperties":false},"preconditions":["your scoped entry condition"],"postconditions":["your intended effect and release condition"]},"applicability":["scope"],"limitations":["not yet tested"],"evidence_ids":["registered ID"]}}
contract may additionally have side_effects (list of strings). Declare arbitrary
useful input property names/types yourself. Supported schemas: object with
properties/required/additionalProperties, arrays/items/minItems/maxItems,
strings/minLength/maxLength, number/integer/minimum/maximum, boolean/null and enum.
No references, expressions or executable schema. Calls validate arguments then
merge them over the frozen default profile within the existing profile rules.
Optional package fields: parent_id, dependencies, improvement_ids. A parent or
dependency is a registered package/legacy Skill ID. dependencies maps YOUR public
helper names to IDs; your main may call alias(robot, parameters). The linker
isolates dependency helpers and merges their default profiles with parameters.
Internal helper arguments remain your program semantics; the framework does not
statically prove every dynamic call or physical pre/postcondition. Read the saved
programs/contracts before composing. All dependencies freeze before dispatch.
Python fusion removes inter-program model calls; physical trajectory smoothness
still depends on the available backend and actual execution evidence.
{"action":"call_skill","arguments":{"package_id":"saved package ID","arguments":{}},"hypothesis":"measurable expected result and what could falsify it"}
The whole resolved program executes through the original executor once, without
new model decisions inside a held-key segment. Unknown/terminal outcomes stop.
{"action":"record_skill_outcome","arguments":{"id":"your_attempt_id","skill_ids":["package or candidate ID"],"evidence_ids":["actual execution/observation IDs"],"observation_evidence_id":"current trusted observation ID","task_result":"success","correction_performed":false,"remaining_issues":[],"summary":"your visual assessment and limitations"}}
task_result is success/failure/uncertain. This is explicitly model_report, separate
from raw execution facts and stable promotion. Report remaining defects even if
you corrected the current text. Compare versions/related contexts using evidence.
{"action":"inspect_evidence","arguments":{"evidence_id":"registered ID"}}
Default is compact file metadata. To read selected content, add file_indices
(at most 4); include_images=true explicitly requests at most 2 selected JPEG/PNG.
Text is bounded and historic images remain HISTORICAL: they never replace the
current observation or authorize moving from an old state. Request new observe
when needed. Use targeted lookups rather than resending entire histories.
For longer content select exactly one file and add text_offset and text_limit
(1..8000 Unicode codepoints). Continue using next_offset until null. JSON files
may also use json_pointer (RFC6901, empty string selects root); pagination then
applies to that selected value, serialized as compact sorted-key JSON. These
parameters are bounded evidence reads, never execution or current-state updates.
Save_skill_package returns package_id for call_skill and candidate_id for legacy
verification/reflection. All learned content comes from you, never a host-authored
per-key solution. Current SDK, scope, permissions and stop semantics remain fixed.

APPLICATION AND ROBOT KNOWLEDGE
Use the same versioned library. Lessons and program packages may declare optional
knowledge_scope={"domain":"robot"} or
{"domain":"application","application_id":"current declared application"}.
Missing old metadata means unspecified, not robot. Search/directory optionally
filter domain and application_id; inspect applicability before reuse. Device/layout
scope and frozen task permissions remain authoritative. Ordinary conversation
messages, pending replies and current text belong to working context; save only
reusable methods or evidence-backed lessons as long-term knowledge.

save_application_skill arguments: {id,application_id,description,applicability,
limitations,evidence_ids,steps:[{id,instruction,skill_ids,requires_observation}]}.
Optional parent_id and improvement_ids preserve revisions and issue links.
Each step references exact existing executable skill/package IDs (possibly []);
you author the instructions and observation boundaries. Result kind=workflow,
identifier=workflow ID. This is an application Skill, not Python or call_skill.
Read its brief/full via normal knowledge tools. It never automatically runs a
referenced robot skill or decides that text was sent. To reuse it:
{"action":"application_skill","arguments":{"operation":"start","identifier":"workflow ID"}}
Then choose ordinary actions yourself. The current step is in application_skill
context, including its entry observation. At a meaningful stage boundary use:
{"action":"application_skill","arguments":{"operation":"checkpoint","step_id":"active step ID","observation_evidence_id":"CURRENT observation ID","result":"continue","summary":"your observable outcome and limitations"}}
result is continue/retry/blocked/completed/abandon; completed is for the final step
only and does not end a persistent task. An observation-required step needs a new
OUTER camera observation before advancing, retrying or completing; robot.observe
inside Python is SDK state only. Checkpoints are model reports, not independent
success judgements. Read the application contract: drafts still need visual
verification before a separate Return and outgoing-message check.

Improvement/outcome declarations may add knowledge_ids referencing relevant
application workflows or Lessons alongside robot skill_ids. Distinguish physical
key execution from focus/send/delivery effects; preserve uncertain attribution.
You own cross-layer diagnosis, priority, merging and revision. A successful
Backspace compensation or SDK receipt does not automatically resolve an issue.
'''


class DevelopmentActions:
    def __init__(self, store):
        self.library = SkillDevelopment(store)
        self.recall = SkillRecall(self.library)
        self.focus_skill_ids = []
        self._presented = set()

    def context(self):
        with self.library.store.read_snapshot():
            return self._context()

    def _context(self):
        context = self.library.context()
        if self.focus_skill_ids:
            memory = self.recall.for_skills(self.focus_skill_ids)
            context['online_recall'] = memory
            # Do not repeat the same high-information summaries in both views.
            related = {item['record_id'] for field in ('pending_improvements', 'recent_outcomes')
                       for item in memory[field]}
            for field in ('pending_improvements', 'recent_outcomes'):
                context[field] = [item for item in context[field] if item['record_id'] not in related]
            # Preserve the relevant memory before unrelated recent summaries.
            while len(json.dumps(context, ensure_ascii=False, sort_keys=True,
                                 allow_nan=False).encode('utf-8')) > 8192:
                entries = context['recent_outcomes'] or context['pending_improvements']
                if not entries:
                    raise ValueError('Online skill context exceeds byte budget')
                entries.pop(0 if context['recent_outcomes'] else -1)
                context['summary_truncated'] = True
        return context

    def presented(self, context):
        """Track only trusted memory actually supplied to a successful model turn."""
        ids = set()
        def record_ids(record):
            if type(record) is not dict:
                return
            for key in ('record_id', 'identifier', 'package_id', 'candidate_id'):
                item = record.get(key)
                if (type(item) is str
                        and re.fullmatch(r'(?:skill|package|improvement|outcome)-[A-Za-z0-9_-]+-[a-f0-9]{64}', item)):
                    ids.add(item)
        def collect_view(view):
            if type(view) is not dict:
                return
            record_ids(view)
            # These are framework-owned record containers. Never recursively
            # interpret arbitrary profile/contract/program values as receipts.
            for field in ('pending_improvements', 'recent_outcomes', 'candidate_revisions', 'results', 'items', 'lessons', 'linked_improvements', 'revisit_resolved'):
                for record in view.get(field, []):
                    record_ids(record)
            if 'data' in view:
                record_ids(view['data'])
            if 'attention' in view:
                collect_view(view['attention'])
            if 'online_recall' in view:
                collect_view(view['online_recall'])
        collect_view(context.get('skill_development', {}))
        collect_view(context.get('continual_learning', {}))
        feedback = context.get('feedback', {})
        if feedback.get('action') in KNOWLEDGE_ACTIONS - {'inspect_evidence'}:
            collect_view(feedback.get('result', {}))
        elif feedback.get('action') == 'save_skill':
            record_ids(feedback.get('candidate', {}))
        elif feedback.get('action') == 'inspect_skill':
            record_ids(feedback.get('skill', {}))
        new = sorted(ids - self._presented)
        # Require presence in this decision's supplied memory, rather than
        # assuming a long-ago/possibly compacted turn is still available.
        self._presented = ids
        return new

    def before_use(self, skill_ids, source):
        with self.library.store.read_snapshot():
            return self._before_use(skill_ids, source)

    def _before_use(self, skill_ids, source):
        """A read-only briefing for newly encountered memory; never queues motion."""
        references = skill_ids or self.recall.matching_source(source)
        memory = self.recall.for_skills(references)
        self.focus_skill_ids = list(references)
        unseen = sorted(set(memory['attention_record_ids']) - self._presented)
        return memory, unseen

    def knowledge_action(self, action, args, current_observation_id):
        # Mutable knowledge declarations must remain outside the read scope.
        if type(action) is str and action in {'search_knowledge', 'inspect_knowledge', 'inspect_evidence',
                      'recall_skills', 'list_improvements'}:
            with self.library.store.read_snapshot():
                return self._knowledge_action(action, args, current_observation_id)
        return self._knowledge_action(action, args, current_observation_id)

    def _knowledge_action(self, action, args, current_observation_id):
        extra_items = []
        if action == 'search_knowledge':
            if set(args) - {'query', 'kind', 'limit', 'domain', 'application_id'}:
                raise ValueError('Unknown search_knowledge arguments')
            result = self.library.search(**args)
        elif action == 'inspect_knowledge':
            if set(args) != {'identifier'}:
                raise ValueError('inspect_knowledge requires identifier')
            result = self.library.inspect(args['identifier'])
            if result.get('kind') in {'skill', 'package'}:
                self.focus_skill_ids = [args['identifier']]
        elif action == 'recall_skills':
            if set(args) - {'skill_ids', 'limit'} or 'skill_ids' not in args:
                raise ValueError('recall_skills requires skill_ids and optional limit')
            result = self.recall.for_skills(**args)
            self.focus_skill_ids = list(args['skill_ids'])
        elif action == 'list_improvements':
            if set(args) - {'status', 'limit', 'offset'}:
                raise ValueError('Unknown list_improvements arguments')
            result = self.library.list_improvements(**args)
        elif action == 'save_skill_package':
            result = self.library.save_package(args)
            self.focus_skill_ids = [result['package_id']]
        elif action == 'save_application_skill':
            result = self.library.save_application_skill(args)
        elif action == 'record_improvement':
            result = self.library.record_improvement(args)
            if args['skill_ids']:
                self.focus_skill_ids = list(args['skill_ids'])
        elif action == 'record_skill_outcome':
            if args.get('observation_evidence_id') != current_observation_id:
                raise ValueError('Skill outcome must reference the current trusted observation ID')
            result = self.library.record_outcome(args)
        elif action == 'inspect_evidence':
            result = self.library.inspect_evidence(args)
            for entry in result['files']:
                image_data = entry.pop('image_base64', None)
                if image_data is not None:
                    extra_items += [
                        {'type': 'inputText', 'text': 'HISTORICAL evidence ' + result['evidence_id']
                         + ', file ' + str(entry['index']) + '. This is not the current robot state.'},
                        {'type': 'inputImage', 'imageUrl': 'data:' + entry['mime_type'] + ';base64,' + image_data},
                    ]
        else:
            raise ValueError('Unknown skill-development knowledge action')
        return dict(action=action, result=result, observation_unchanged=True), extra_items

    def resolve_call(self, args):
        with self.library.store.read_snapshot():
            return self.library.resolve_call(args)
