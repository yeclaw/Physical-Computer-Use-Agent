"""Strict descriptive knowledge domains and inert application workflow steps.

These declarations neither change physical scope nor grant execution authority.
Missing legacy metadata is projected as unspecified without rewriting records.
"""
from .learning_contracts import LearningContractError, exact, text, ids, clone

DOMAINS = frozenset(('robot', 'application', 'unspecified'))


def validate_knowledge_scope(value):
    exact(value, {'domain'}, {'application_id'})
    domain = value['domain']
    if type(domain) is not str or domain not in DOMAINS:
        raise LearningContractError('knowledge domain must be robot, application or unspecified')
    if domain == 'application':
        if 'application_id' not in value:
            raise LearningContractError('application knowledge requires application_id')
        return dict(domain=domain, application_id=text(value['application_id'], 'application_id', 128))
    if 'application_id' in value:
        raise LearningContractError('application_id is only valid for application knowledge')
    return dict(domain=domain)


def knowledge_scope(record):
    return validate_knowledge_scope(record.get('knowledge_scope', {'domain': 'unspecified'}))


def validate_domain_filter(domain=None, application_id=None):
    if domain is not None and (type(domain) is not str or domain not in DOMAINS):
        raise LearningContractError('recognized knowledge domain filter required')
    if application_id is not None:
        text(application_id, 'application_id', 128)
        if domain not in (None, 'application'):
            raise LearningContractError('application_id filter requires application domain')


def matches_knowledge_scope(record, domain=None, application_id=None):
    validate_domain_filter(domain, application_id)
    scope = knowledge_scope(record)
    return ((domain is None or scope['domain'] == domain)
            and (application_id is None or scope.get('application_id') == application_id))


def validate_workflow_steps(value):
    if type(value) is not list or not 1 <= len(value) <= 32:
        raise LearningContractError('application skill requires 1..32 ordered workflow steps')
    result, seen = [], set()
    for step in value:
        exact(step, {'id', 'instruction', 'skill_ids', 'requires_observation'})
        step_id = text(step['id'], 'step id', 64)
        if step_id in seen:
            raise LearningContractError('workflow step IDs must be unique')
        seen.add(step_id)
        if type(step['requires_observation']) is not bool:
            raise LearningContractError('requires_observation must be an explicit boolean')
        result.append(dict(id=step_id, instruction=text(step['instruction'], 'step instruction', 2048),
            skill_ids=ids(step['skill_ids'], 'step skill_ids', 16),
            requires_observation=step['requires_observation']))
    return clone(result)
