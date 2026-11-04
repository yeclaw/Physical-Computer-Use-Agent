"""Strict JSON primitives shared by learning records, readers and checkpoints.

These values describe evidence and proposals. They confer no execution authority.
"""
import copy
import hashlib
import json
import math
import re

PROTOCOLS = {
    'reader': 'physical-keyboard-learning-read-v1',
    'resource': 'physical-keyboard-learning-resource-v1',
    'commit': 'physical-keyboard-learning-commit-v1',
    'source': 'physical-keyboard-learning-source-v1',
    'analysis': 'physical-keyboard-learning-analysis-v1',
    'checkpoint': 'physical-keyboard-working-context-v1',
    'access': 'physical-keyboard-learning-access-v1',
    'references': 'physical-keyboard-learning-references-v1',
}

class LearningContractError(ValueError):
    pass


def exact(value, required, optional=()):
    if type(value) is not dict or not set(required) <= set(value) or set(value)-set(required)-set(optional):
        raise LearningContractError('unexpected or missing learning fields')
    return value


def identifier(value, name='id'):
    if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}', value) is None:
        raise LearningContractError(name+': registered bounded identifier required')
    return value


def sha256(value):
    if type(value) is not str or re.fullmatch(r'[a-f0-9]{64}',value) is None:
        raise LearningContractError('SHA256 required')
    return value


def text(value, name='text', maximum=4096):
    if type(value) is not str or not value.strip() or len(value)>maximum:
        raise LearningContractError(name+': nonempty bounded text required')
    try:value.encode('utf-8')
    except UnicodeError as exc:raise LearningContractError('invalid Unicode') from exc
    return value


def integer(value, name='count', minimum=0, maximum=65536):
    if type(value) is not int or not minimum<=value<=maximum:
        raise LearningContractError(name+': bounded integer required')
    return value


def ids(value, name='ids', maximum=64):
    if type(value) is not list or len(value)>maximum or len(set(identifier(v,name) for v in value))!=len(value):
        raise LearningContractError(name+': unique bounded ID list required')
    return list(value)


def scope(value):
    exact(value,{'task_id','device_id','layout_id'})
    return {k:text(v,k,128) for k,v in sorted(value.items())}


def encoded(value, maximum=1024*1024):
    def visit(item,depth=0):
        if depth>32:raise LearningContractError('JSON nesting too deep')
        if item is None or type(item) in (str,bool):return
        if type(item) is int:
            if item.bit_length()>128:raise LearningContractError('integer too large')
            return
        if type(item) is float and math.isfinite(item):return
        if type(item) is list:
            for child in item:visit(child,depth+1)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for child in item.values():visit(child,depth+1)
            return
        raise LearningContractError('finite strict JSON required')
    visit(value)
    try:raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    except (ValueError,TypeError,UnicodeError,RecursionError) as exc:raise LearningContractError('invalid JSON') from exc
    if len(raw)>maximum:raise LearningContractError('learning JSON exceeds byte budget; page or project fields')
    return raw


def digest(value):
    return hashlib.sha256(encoded(value,16*1024*1024)).hexdigest()


def clone(value, maximum=1024*1024):
    encoded(value,maximum)
    return copy.deepcopy(value)


def validate_read_request(value):
    exact(value,{'identifier'},{'view','fields','offset','limit','source_form'})
    identifier(value['identifier']);view=value.get('view','brief')
    if view not in {'brief','fields','source','full'}:raise LearningContractError('invalid reader view')
    fields=value.get('fields',[])
    if type(fields) is not list or len(fields)>16 or any(type(f)is not str or re.fullmatch('[a-z][a-z0-9_]{0,63}',f)is None for f in fields):
        raise LearningContractError('explicit top-level fields required')
    if view=='fields' and not fields:raise LearningContractError('fields view requires fields')
    integer(value.get('offset',0),'offset',0,1000000);integer(value.get('limit',32),'limit',1,256)
    if value.get('source_form','declared') not in {'declared','compiled'}:raise LearningContractError('choose one source form')
    return clone(value)


def validate_checkpoint(value):
    exact(value,{'summary','knowledge_ids','evidence_ids','pending_issues'},{'completed','remaining'})
    text(value['summary'],'summary',8192)
    for key in ('knowledge_ids','evidence_ids','pending_issues'):ids(value[key],key)
    for key in ('completed','remaining'):
        if key in value:
            if type(value[key])is not list or len(value[key])>32:raise LearningContractError('bounded task facts required')
            for fact in value[key]:text(fact,key,512)
    return clone(value,16384)
