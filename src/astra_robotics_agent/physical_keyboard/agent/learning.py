"""Optional model-owned learning routes, separate from physical execution.

Only the caller's explicit construction enables these actions. Registered
historical sources, analysis, knowledge declarations and work checkpoints never
grant motion authority or turn historical images into a current observation.
"""
import base64
import hashlib
from pathlib import Path

from ..adapters.skill_sources.learning_sources import LearningSources, evidence_record_kind
from ..records.evidence_archive import EvidenceArchive
from ..records.learning_access import LearningAccess, LearningAccessDenied
from ..records.learning_contracts import (
    LearningContractError, clone, digest, encoded, exact, identifier, ids, integer, text,
)
from ..runtime.analysis_worker import run_analysis
from ..skills.current_view import CurrentView
from ..skills.learning_changes import LearningChanges
from ..skills.reader import KnowledgeReader
from ..skills.resources import LearningResources
from ..skills.knowledge import _read
from .work_context import WorkContext


PROTOCOL = 'physical-keyboard-continual-learning-v1'
LEARNING_ACTIONS = frozenset({
    'knowledge_directory', 'read_knowledge', 'list_learning_sources',
    'inspect_learning_sources', 'analyze_learning_sources',
    'read_learning_resource', 'commit_learning', 'learning_attention',
    'save_working_context', 'manage_working_context', 'archive_evidence',
})

LEARNING_INSTRUCTIONS = '''
OPTIONAL CONTINUAL LEARNING v1
You choose what to read, analyze, learn, consolidate or defer while working.
Use JSON {"action":"name","arguments":{...}} for these read/learning actions;
there is no top-level hypothesis. A concise task-related reason may be included
in analyze_learning_sources.arguments.reason. No action below moves the robot.
Historical images and model hypotheses are not current state or task truth.
For relevant knowledge, prefer knowledge_directory and read_knowledge's brief
view, then request fields or source slices for missing details. Legacy full
inspect remains available when you need the complete declaration. Choose what
is useful; this does not require a fixed sequence or another model roundtrip.

knowledge_directory: {query?,kind?,limit?,offset?,version_offset?,version_limit?,branch_offset?,branch_limit?,domain?,application_id?};
Default at most 4 versions/branches per logical group; page them explicitly.
Version ordering is reading metadata, never an executable selection. Mechanical logical groups and
explicit condition branches. An unselected capability never means latest wins.
read_knowledge: {identifier,view?,fields?,offset?,limit?,source_form?}; view is
brief (default), fields, source or full; source_form is declared or compiled.
list_learning_sources: {kind?,limit?,offset?,origin?}; origin all (default),
sources or evidence. Sources use their own next_offset; registered_evidence
has an independent metadata page. Nothing opens every old image at startup.
inspect_learning_sources: {source_id,fields?,offset?,limit?,image_indices?,camera?,json_pointer?}.
Use evidence:<registered evidence ID> to inspect a legal own observation or
execution only when needed. Image indices/view IDs come from the source's
historical_images metadata; they never require a particular camera arrangement.
analyze_learning_sources: {id,source_lines,inputs,parameters,reason?}; define
analyze(data, parameters). inputs maps at most 8 aliases to
{source_id,fields?,offset?,limit?,json_pointer?}. This is SDK-free finite JSON analysis in a
bounded subprocess, without imports, arbitrary files, network or robot tools.
It returns an exact inert resource reference/provenance/shape, not a new Skill.
read_learning_resource: {resource_id,view?,fields?,offset?,limit?}; brief by
default, full only explicitly. Missing fields/unknown sources stay unknown.
commit_learning: finite resources/lessons/packages/workflows/improvements/relations/
selections declarations (optional summary). Each declaration may name local_ref;
ID positions accept {"local_ref":"name"}. No expressions, paths or loops.
Resources declare id,value,source_ids and optional kind,evidence_ids,metadata.
Packages use the save_skill_package contract, with profile or the exact
profile_resource_id (choose one). You author every source, condition and claim.
Workflows use save_application_skill declarations; they stay agent-interpreted
application Skills with external observation checkpoints, never fused Python.
Relations are {kind,from_id,to_id,conditions?,reason?}; kinds derived_from,
uses_lesson,uses_resource,addresses_issue,supersedes,contradicts,alias.
Selections are {logical_id,identifier,conditions,reason}. Conditions are your
declarations; the framework does not interpret them to choose an action.
learning_attention: {mode:"skills",skill_ids,limit?} or
{mode:"evidence",cursor?,limit?,offset?}. Evidence/compensation never
automatically closes or reopens an Issue. No new event requires a maintenance call.
save_working_context: {summary,knowledge_ids,evidence_ids,pending_issues,
completed?,remaining?}. These are auditable task facts, not hidden reasoning.
manage_working_context: {checkpoint_id,method,reason}; method native_compact or
rebuild. Host facts/STOP/unknown/in-flight state are authoritative. A successful
operation requires fresh observation; it never replays an old request.
archive_evidence: {evidence_ids,name?}; IDs must already be framework-registered.
Creates a lossless copy with hashes and keeps the originals. No model paths.
'''


class ContinualLearning:
    def __init__(self, development, session, *, output, sources=None, access=None):
        from ..skills.learning_attention import LearningAttention
        self.development, self.library = development, development.library
        self.store, self.session = self.library.store, session
        self.access = access or self.store.learning_access or LearningAccess(self.library.scope)
        if self.access.scope != self.library.scope:
            raise LearningContractError('continual-learning access scope mismatch')
        self.store.learning_access = self.access
        self.output = Path(output)
        self.sources = sources if sources is not None else LearningSources(self.library.scope, access=self.access)
        self.resources = LearningResources(self.store, access=self.access)
        self.changes = LearningChanges(self.library, self.resources, access=self.access)
        self.library.publication_guard = self.changes.is_committed
        self.view = CurrentView(self.library, self.changes)
        self.reader = KnowledgeReader(self.library, resources=self.resources,
            changes=self.changes, view=self.view, access=self.access)
        self.attention = LearningAttention(self.library, self.changes, reader=self.reader)
        self.working = WorkContext(session, self.store, output=self.output / 'work-context', access=self.access)
        self.archives = EvidenceArchive(self.output / 'evidence-archive')

    def descriptor(self):
        return dict(protocol=PROTOCOL, enabled=True, actions=sorted(LEARNING_ACTIONS),
            scope=clone(self.library.scope), reader='physical-keyboard-learning-read-v1',
            default_view='brief', context_bytes=8192, max_analysis_inputs=8,
            source_descriptor=self.sources.describe(), execution_authority=False,
            automatic_semantic_writer=False, mandatory_model_roundtrips=0)

    def validate_legacy(self, arguments):
        with self.store.read_snapshot():
            return self._validate_legacy(arguments)

    def _validate_legacy(self, arguments):
        """Old action names grant no shortcut around atomic publication."""
        def visit(value):
            if type(value) is str and value.startswith(
                    ('skill-', 'lesson-', 'package-', 'workflow-', 'improvement-', 'outcome-', 'resource-')):
                if not self.changes.is_committed(value):
                    raise LearningContractError('uncommitted knowledge is unavailable: ' + value)
            elif type(value) is dict:
                for key, child in value.items():
                    # Source/profile strings are program data, not declarations of IDs.
                    if key not in ('source', 'source_lines', 'profile', 'claim', 'summary', 'rationale'):
                        visit(child)
            elif type(value) is list:
                for child in value:
                    visit(child)
        visit(arguments)

    def published_feedback(self, feedback):
        with self.store.read_snapshot():
            return self._published_feedback(feedback)

    def _published_feedback(self, feedback):
        """Filter list metadata too; exact old inspect contracts stay unchanged."""
        result = feedback.get('result', {})
        if type(result) is dict:
            for key in ('results', 'items', 'pending_improvements', 'recent_outcomes', 'candidate_revisions'):
                if type(result.get(key)) is list:
                    def visible(item):
                        identifier = next((item[k] for k in ('record_id', 'identifier', 'package_id', 'candidate_id')
                                           if type(item.get(k)) is str), None) if type(item) is dict else None
                        return identifier is None or self.changes.is_committed(identifier)
                    before = len(result[key])
                    result[key] = [item for item in result[key] if visible(item)]
                    if len(result[key]) != before:
                        result['unpublished_omitted'] = True
        return feedback

    @staticmethod
    def _shape(value):
        if type(value) is dict:
            return dict(type='object', keys=list(value)[:32], field_count=len(value))
        if type(value) is list:
            return dict(type='array', length=len(value))
        return dict(type=type(value).__name__)

    def _image_references(self, value):
        """Remove known inline image encodings from textual learning views."""
        references = []
        original_sha = digest(value)
        def visit(item):
            if type(item) is list:
                return [visit(child) for child in item]
            if type(item) is not dict:
                return item
            result = {}
            for key, child in item.items():
                if key not in {'image_base64', 'images_base64'}:
                    result[key] = visit(child)
                    continue
                target = 'image_reference' if key == 'image_base64' else 'image_references'
                if target in item:
                    raise LearningContractError('image reference projection would overwrite a field')
                collection = {'image': child} if key == 'image_base64' else child
                projection = self.archives.normalized_view(encoded({'images_base64': collection}, 1024 * 1024))
                result[target] = (projection['value']['image_references']['image'] if key == 'image_base64'
                                  else projection['value']['image_references'])
                references.extend(projection['images'])
                if len(references) > 64:
                    raise LearningContractError('historical image reference budget exceeded')
            return result
        projected = visit(value)
        receipt = (dict(original_json_sha256=original_sha, image_references=references,
            association_kind='same_evidence_encoding_projection', new_visual_evidence=False)
            if references else None)
        return projected, receipt

    def context(self, observation_id=None):
        with self.store.read_snapshot():
            return self._context(observation_id)

    def _context(self, observation_id=None):
        if observation_id is not None:
            identifier(observation_id, 'observation_id')
        directory = self.view.directory(limit=4)
        directory['ordering'] = 'mechanical kind and logical ID; cross-journal recency is unknown'
        # These versions are references only, never duplicate executable text.
        result = dict(protocol=PROTOCOL, directory=directory,
            current_observation_evidence_id=observation_id, execution_authority=False)
        if self.development.focus_skill_ids:
            result['attention'] = self.attention.for_skills(self.development.focus_skill_ids, limit=4)
        while True:
            try:
                return clone(result, 8192)
            except LearningContractError:
                # Keep conditions explicit by omitting whole old groups rather
                # than shortening conditions or silently changing their meaning.
                if result['directory']['items']:
                    result['directory']['items'].pop()
                    result['directory']['context_omitted_groups'] = True
                elif 'attention' in result:
                    result['attention'] = dict(available_on_request=True,
                        skill_ids=list(self.development.focus_skill_ids), omitted_for_byte_budget=True)
                    return clone(result, 8192)
                else:
                    raise

    def _ensure_source(self, source_id):
        identifier(source_id, 'source_id')
        if source_id.startswith('evidence:'):
            evidence_id = source_id[len('evidence:'):]
            identifier(evidence_id, 'evidence_id')
            # Registration performs source, raw-object and audit-only checks.
            self.sources.register_evidence(self.store, evidence_id)
        descriptor = self.sources.describe(source_id)
        if descriptor.get('scope') != self.library.scope:
            raise LearningContractError('learning source scope mismatch')
        self.access.check_sources(descriptor.get('source_ids'), require_provenance=True)
        return descriptor

    def _source_directory(self, args):
        exact(args, set(), {'kind', 'limit', 'offset', 'origin'})
        limit = integer(args.get('limit', 8), 'limit', 1, 16)
        offset = integer(args.get('offset', 0), 'offset', 0, 1000000)
        origin, kind = args.get('origin', 'all'), args.get('kind', 'all')
        if type(origin) is not str or origin not in {'all', 'sources', 'evidence'}:
            raise LearningContractError('source directory origin must be all, sources or evidence')
        if type(kind) is not str or kind not in {'all', 'human_demo', 'robot_attempt', 'device_reference'}:
            raise LearningContractError('recognized source kind required')
        result = clone(self.sources.list_sources(kind=kind, limit=limit, offset=offset), 1024 * 1024) if origin != 'evidence' else {}
        if 'sources' in result:
            if type(result['sources']) is not list or len(result['sources']) > limit:
                raise LearningContractError('bounded source metadata page required')
            visible = []
            for descriptor in result['sources']:
                if descriptor.get('scope') != self.library.scope:
                    raise LearningContractError('source directory scope mismatch')
                try:
                    self.access.check_sources(descriptor.get('source_ids'), require_provenance=True)
                except LearningAccessDenied:
                    continue
                visible.append(descriptor)
            result['sources'] = visible
        if origin == 'sources' or kind not in {'all', 'robot_attempt'}:
            return result
        evidence_ids = []
        for event in self.store._events():
            if event.get('type') == 'evidence_registered' and event['evidence_id'] not in evidence_ids:
                evidence_ids.append(event['evidence_id'])
        items, consumed = [], offset
        # Only requested metadata pages validate underlying registered bytes.
        for evidence_id in evidence_ids[offset:offset + limit]:
            consumed += 1
            try:
                record = self.store._evidence(evidence_id)
                sources = self.access.evidence_sources(evidence_id)
            except LearningAccessDenied:
                continue
            names = [Path(item['source_path']).name for item in record['files']]
            if evidence_record_kind(record, self.store.namespace / 'objects') is None:
                continue
            items.append(dict(source_id='evidence:' + evidence_id, kind='robot_attempt',
                evidence_id=evidence_id, sha256=record['sha256'], source_ids=sources,
                historical=True, registered_metadata_only=True, current_state=False,
                files=[dict(index=i, name=name, sha256=item['sha256'], bytes=item['bytes'])
                       for i, (name, item) in enumerate(zip(names, record['files']))]))
        result['registered_evidence'] = dict(items=items, offset=offset,
            next_offset=consumed if consumed < len(evidence_ids) else None,
            index_count=len(evidence_ids), count_scope='registered_index_not_learning_eligibility',
            content='only explicitly requested evidence is opened as a learning source')
        return result

    def _source_inspect(self, args):
        exact(args, {'source_id'}, {'fields', 'offset', 'limit', 'image_indices', 'camera', 'json_pointer'})
        descriptor = self._ensure_source(args['source_id'])
        request = {key: value for key, value in args.items() if key != 'source_id'}
        result = clone(self.sources.inspect(args['source_id'], request or None), 1024 * 1024)
        if (result.get('scope') != self.library.scope or result.get('source_id') != args['source_id']
                or result.get('sha256') != descriptor['sha256']
                or result.get('source_ids') != descriptor['source_ids']):
            raise LearningContractError('source view identity or provenance changed')
        result, projection = self._image_references(result)
        if projection is not None:
            if 'view_sha256' in result:
                result['original_view_sha256'] = result.pop('view_sha256')
            result['encoding_projection'] = projection
            result['view_sha256'] = digest(result)
        images = result.get('images', [])
        if type(images) is not list or len(images) > 2:
            raise LearningContractError('at most two historical image descriptors required')
        extra = []
        for item in images:
            path = item.pop('path', None)
            if item.get('status') != 'available':
                continue
            if path is None or item.get('historical') is not True:
                raise LearningContractError('trusted historical image descriptor required')
            raw = _read(path, maximum=4 * 1024 * 1024)
            if hashlib.sha256(raw).hexdigest() != item.get('sha256') or len(raw) != item.get('bytes'):
                raise LearningContractError('historical image hash or size mismatch')
            mime = ('image/jpeg' if raw.startswith(b'\xff\xd8\xff') else
                    'image/png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else None)
            if mime is None:
                raise LearningContractError('registered historical JPEG/PNG required')
            extra.extend([dict(type='inputText', text='HISTORICAL source ' + args['source_id']
                + ', index ' + str(item['index']) + '. Not current robot state.'),
                dict(type='inputImage', imageUrl='data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii'))])
        encoded(result, 1024 * 1024)
        return result, extra

    def _analyze(self, args):
        exact(args, {'id', 'source_lines', 'inputs', 'parameters'}, {'reason'})
        identifier(args['id'])
        if 'reason' in args:
            text(args['reason'], 'reason', 2048)
        if type(args['inputs']) is not dict or not 1 <= len(args['inputs']) <= 8:
            raise LearningContractError('analysis requires 1..8 explicit registered inputs')
        with self.store.read_snapshot():
            data, provenance, sources = self._analysis_inputs(args['inputs'])
        worker = run_analysis(args['source_lines'], data, args['parameters'])
        metadata = {key: value for key, value in worker.items() if key != 'result'}
        metadata['inputs'] = provenance
        if 'reason' in args:
            metadata['reason'] = args['reason']
        resource = self.resources.put(args['id'], worker['result'], source_ids=sorted(sources),
            kind='analysis_result', metadata=metadata)
        return dict(resource=resource, provenance=provenance,
            analysis={key: value for key, value in worker.items() if key != 'result'},
            result_shape=self._shape(worker['result']), saved_skill=False, dispatched=False,
            evidence='computed candidate resource; not physical success or execution permission')

    def _analysis_inputs(self, inputs):
        data, provenance, sources = {}, {}, set()
        for alias, request in inputs.items():
            identifier(alias, 'input alias')
            exact(request, {'source_id'}, {'fields', 'offset', 'limit', 'json_pointer'})
            descriptor = self._ensure_source(request['source_id'])
            parameters = {key: value for key, value in request.items() if key != 'source_id'}
            projected = self.sources.analysis_data(request['source_id'], **parameters)
            projected, image_projection = self._image_references(projected)
            data[alias] = projected
            sources.update(descriptor['source_ids'])
            provenance[alias] = dict(source_id=request['source_id'], source_sha256=descriptor['sha256'],
                source_ids=descriptor['source_ids'], projection=clone(parameters),
                projected_sha256=digest(projected), projected_bytes=len(encoded(projected)))
            if image_projection is not None:
                provenance[alias]['encoding_projection'] = image_projection
        return data, provenance, sources

    def save_package(self, args):
        """Old profile saves retain their shape; resource saves explicitly commit."""
        if type(args) is not dict:
            raise LearningContractError('package declaration object required')
        if 'profile_resource_id' in args:
            payload = dict(args)
            if 'local_ref' in payload:
                raise LearningContractError('save_skill_package does not accept local_ref')
            payload['local_ref'] = 'saved_package'
            commit = self.changes.commit({'packages': [payload]})
            result = self.library.inspect(commit['bindings']['saved_package'])
            result = dict(self.library._summary(result), contract=clone(result['contract']),
                          commit_id=commit['commit_id'])
        else:
            references = list(args.get('dependencies', {}).values()) if type(args.get('dependencies', {})) is dict else []
            references += args.get('improvement_ids', []) if type(args.get('improvement_ids', [])) is list else []
            if args.get('parent_id') is not None:
                references.append(args['parent_id'])
            for value in references:
                if not self.changes.is_committed(value):
                    raise LearningContractError('pending package reference is not active')
                self.changes.source_ids(value)
            result = self.library.save_package(args)
        self.development.focus_skill_ids = [result['package_id']]
        return result

    def _archive(self, args):
        exact(args, {'evidence_ids'}, {'name'})
        evidence_ids = ids(args['evidence_ids'], 'evidence_ids', 16)
        if not evidence_ids:
            raise LearningContractError('archive requires registered evidence')
        objects = {}
        for evidence_id in evidence_ids:
            record = self.store._evidence(evidence_id)
            self.access.evidence_sources(evidence_id)
            for item in record['files']:
                objects[item['sha256']] = dict(path=self.store.namespace / 'objects' / item['sha256'],
                    sha256=item['sha256'], bytes=item['bytes'])
        result = self.archives.create(list(objects.values()), name=args.get('name'))
        # Internal archive paths are host artifacts; model follow-up uses IDs.
        result.pop('archive_path', None); result.pop('manifest_path', None)
        return dict(result, evidence_ids=evidence_ids, source='registered_frozen_objects_only')

    def handle(self, action, args, *, facts):
        # Request-local verification only: no cached physical observation,
        # controller state, permission, model operation or knowledge writer.
        if type(action) is str and action in {'knowledge_directory', 'read_knowledge', 'list_learning_sources',
                      'inspect_learning_sources', 'read_learning_resource', 'learning_attention'}:
            with self.store.read_snapshot():
                return self._handle(action, args, facts=facts)
        return self._handle(action, args, facts=facts)

    def _handle(self, action, args, *, facts):
        if type(action) is not str or action not in LEARNING_ACTIONS:
            raise LearningContractError('unknown continual-learning action')
        if type(args) is not dict:
            raise LearningContractError('learning arguments must be an object')
        extra = []
        if action == 'knowledge_directory':
            exact(args, set(), {'query', 'kind', 'limit', 'offset', 'version_offset', 'version_limit', 'branch_offset', 'branch_limit', 'domain', 'application_id'})
            result = self.view.directory(**args)
        elif action == 'read_knowledge':
            result = self.reader.read(args)
            if result['data'].get('kind') in {'skill', 'package'}:
                self.development.focus_skill_ids = [args['identifier']]
        elif action == 'list_learning_sources':
            result = self._source_directory(args)
        elif action == 'inspect_learning_sources':
            result, extra = self._source_inspect(args)
        elif action == 'analyze_learning_sources':
            result = self._analyze(args)
        elif action == 'read_learning_resource':
            exact(args, {'resource_id'}, {'view', 'fields', 'offset', 'limit'})
            identifier(args['resource_id'])
            if not args['resource_id'].startswith('resource-'):
                raise LearningContractError('exact learning resource ID required')
            result = self.reader.read(dict(identifier=args['resource_id'], **{key: value for key, value in args.items() if key != 'resource_id'}))
        elif action == 'commit_learning':
            result = self.changes.commit(args)
            candidates = []
            workflows = []
            for value in result['record_ids']:
                if value.startswith('resource-'):
                    continue
                record = self.library.inspect(value)
                if record['kind'] in {'skill', 'lesson'}:
                    candidates.append(dict(candidate_id=record['candidate_id'], kind=record['kind']))
                elif record['kind'] == 'package':
                    candidates.append(dict(candidate_id=record['candidate_id'], package_id=record['record_id'], kind='skill'))
                elif record['kind'] == 'workflow':
                    workflows.append(self.library._summary(record))
            unique = {}
            for candidate in candidates:
                previous = unique.get(candidate['candidate_id'], {})
                unique[candidate['candidate_id']] = dict(previous, **candidate)
            result['saved_candidates'] = list(unique.values())
            if workflows:
                result['saved_application_skills'] = workflows
        elif action == 'learning_attention':
            mode = args.get('mode', 'skills')
            if mode == 'skills':
                exact(args, {'skill_ids'}, {'mode', 'limit'})
                result = self.attention.for_skills(args['skill_ids'], limit=args.get('limit', 4))
                self.development.focus_skill_ids = list(args['skill_ids'])
            elif mode == 'evidence':
                exact(args, set(), {'mode', 'cursor', 'limit', 'offset'})
                result = self.attention.new_evidence(cursor=args.get('cursor'), limit=args.get('limit', 8), offset=args.get('offset', 0))
            else:
                raise LearningContractError('attention mode must be skills or evidence')
        elif action == 'save_working_context':
            result = self.working.save(args, facts)
        elif action == 'manage_working_context':
            result = self.working.request(args, facts)
        else:
            result = self._archive(args)
        return dict(action=action, result=result, observation_unchanged=True,
            receipt=dict(result_json_bytes=len(encoded(result, 2 * 1024 * 1024)),
                         extra_content_json_bytes=len(encoded(extra, 16 * 1024 * 1024)),
                         historical_image_count=sum(item.get('type') == 'inputImage' for item in extra),
                         accounting='returned JSON/channel bytes; no token or physical-success estimate')), extra
