"""Model-interpreted application skills; no execution or visual judgement here.

A workflow is pinned knowledge. The same agent chooses ordinary actions between
checkpoints; this object neither dispatches its referenced skills nor recognizes
text/focus/delivery. Checkpoints are explicitly model reports, not success truth.
"""
from ..records.learning_contracts import clone, exact, identifier, text


class ApplicationSkillSession:
    def __init__(self, library, application_id, *, read=None):
        self.library = library
        self.application_id = application_id
        self.read = read or library.inspect
        self.active = None

    def context(self):
        if self.active is None:
            return dict(available=self.application_id is not None,
                        application_id=self.application_id, active=False)
        run = self.active
        workflow = run['workflow']
        return clone(dict(application_id=self.application_id, active=True,
            workflow_id=workflow['record_id'], version=workflow['version'],
            status=run['status'], step=workflow['steps'][run['index']],
            step_index=run['index'], step_count=len(workflow['steps']),
            step_started_observation_evidence_id=run['observation_id'],
            basis='agent_interpreted_workflow; checkpoints are model reports',
            execution_authority=False))

    def handle(self, args, current_observation_id):
        if type(args) is not dict:
            raise ValueError('application_skill arguments must be an object')
        if self.application_id is None:
            raise ValueError('No application identity is declared by the current task')
        operation = args.get('operation')
        if operation == 'start':
            exact(args, {'operation', 'identifier'})
            identifier(args['identifier'])
            if self.active is not None and self.active['status'] not in {'completed', 'abandoned'}:
                raise ValueError('Checkpoint or abandon the active application skill before replacing it')
            with self.library.store.read_snapshot():
                workflow = self.read(args['identifier'])
                if workflow.get('kind') != 'workflow' or workflow.get('scope') != self.library.scope:
                    raise ValueError('A scoped application workflow is required, not an executable program')
                if workflow.get('application_id') != self.application_id:
                    raise ValueError('Application skill does not match the current application; rebind explicitly')
                # Resolving validates frozen executable pins and source policy;
                # it executes nothing and does not create a cross-scope route.
                for step in workflow['steps']:
                    for value in step['skill_ids']:
                        self.library._resolve_skill(value)
                self.library.store._evidence(current_observation_id)
            self.active = dict(workflow=clone(workflow), index=0, status='active',
                               observation_id=current_observation_id)
            return dict(operation=operation, started=True, dispatched=False,
                        workflow_id=workflow['record_id'], context=self.context())
        if operation != 'checkpoint':
            raise ValueError('application_skill operation must be start or checkpoint')
        exact(args, {'operation', 'step_id', 'observation_evidence_id', 'result', 'summary'})
        if self.active is None or self.active['status'] in {'completed', 'abandoned'}:
            raise ValueError('No active application skill')
        run = self.active
        steps = run['workflow']['steps']
        step = steps[run['index']]
        if args['step_id'] != step['id']:
            raise ValueError('Checkpoint must name the active application step')
        if args['observation_evidence_id'] != current_observation_id:
            raise ValueError('Application checkpoint must cite the current external observation')
        self.library.store._evidence(current_observation_id)
        result = args['result']
        if type(result) is not str or result not in {'continue', 'retry', 'blocked', 'completed', 'abandon'}:
            raise ValueError('Unknown application checkpoint result')
        summary = text(args['summary'], 'summary', 800)
        if (step['requires_observation'] and result in {'continue', 'retry', 'completed'}
                and run['observation_id'] == current_observation_id):
            raise ValueError('This application step requires a new external observation; SDK state is insufficient')
        if result == 'completed' and run['index'] != len(steps) - 1:
            raise ValueError('Only the final step may complete this application skill')
        if result == 'continue' and run['index'] == len(steps) - 1:
            raise ValueError('Use completed for the final application step')
        entry = dict(operation=operation, workflow_id=run['workflow']['record_id'],
            step_id=step['id'], result=result, summary=summary,
            observation_evidence_id=current_observation_id, basis='model_report',
            independently_verified=False, dispatched=False, task_completed=False)
        if result == 'continue':
            run['index'] += 1
            run['status'] = 'active'
        elif result == 'retry':
            run['status'] = 'active'
        else:
            run['status'] = 'abandoned' if result == 'abandon' else result
        run['observation_id'] = current_observation_id
        return dict(entry, context=self.context())
