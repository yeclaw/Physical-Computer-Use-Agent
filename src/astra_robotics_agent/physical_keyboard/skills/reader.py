"""Versioned bounded knowledge views; legacy inspect remains exact and unchanged."""
import copy
from ..records.learning_contracts import (PROTOCOLS, LearningContractError, validate_read_request,
                                        clone, digest, encoded, integer)
from ..records.learning_access import LearningAccessDenied
from ..records.knowledge_domains import knowledge_scope

class KnowledgeReader:
    def __init__(self, library, *, resources=None, changes=None, view=None, access=None,
                 brief_bytes=8192, full_bytes=1024*1024):
        self.library,self.resources,self.changes,self.view=library,resources,changes,view
        self.access=access
        self.brief_bytes=integer(brief_bytes,'brief_bytes',1024,65536)
        self.full_bytes=integer(full_bytes,'full_bytes',8192,8*1024*1024)
        self._reads={}

    def _get(self, identifier):
        if self.changes is not None and not self.changes.is_committed(identifier):
            raise LearningContractError('pending learning commit is not active')
        if self.changes is not None:self.changes.source_ids(identifier)
        elif self.access is not None:self.access.knowledge_sources(identifier)
        if identifier.startswith('resource-'):
            if self.resources is None:raise LearningContractError('resources unavailable')
            return self.resources.get(identifier)
        return self.library.inspect(identifier)

    @staticmethod
    def _shape(value):
        if type(value)is dict:return dict(type='object',keys=list(value),field_count=len(value))
        if type(value)is list:return dict(type='array',length=len(value))
        return dict(type=type(value).__name__)

    @staticmethod
    def _brief(record):
        fields=('record_id','candidate_id','package_id','resource_id','id','kind','version','scope',
                'status','stable','origin','basis','source_sha256','profile_sha256','compiled_source_sha256',
                'compiled_profile_sha256','parent_id','superseded','source_ids','evidence_ids',
                'application_id','execution','knowledge_ids','skill_ids')
        value={k:copy.deepcopy(record[k]) for k in fields if k in record}
        value['knowledge_scope']=knowledge_scope(record)
        if record.get('kind')=='workflow':
            value['step_count']=len(record['steps'])
            value['step_ids']=[step['id'] for step in record['steps']]
            value['skill_pin_count']=len(record['skill_pins'])
        omitted=[]
        for key in ('claim','summary','rationale','description','applicability','limitations','revisit_when'):
            if key in record:
                raw=record[key]
                if type(raw)is str:
                    value[key]=raw[:240]
                    if len(raw)>240:omitted.append(key)
                elif type(raw)is list:
                    value[key]=[str(s)[:240] for s in raw[:3]]
                    if value[key]!=raw:omitted.append(key)
        if 'contract' in record:
            contract=record['contract']
            value['contract']={k:copy.deepcopy(contract[k]) for k in ('description','preconditions','postconditions','side_effects') if k in contract}
            if 'input_schema' in contract:
                value['contract']['input_schema_shape']=KnowledgeReader._shape(contract['input_schema'])
                omitted.append('contract.input_schema')
        for key in ('profile','declared_profile','value'):
            if key in record:value[key+'_shape']=KnowledgeReader._shape(record[key])
        if 'source' in record or 'source_lines' in record:
            source=record.get('source')
            value['source_line_count']=len(source.splitlines()) if type(source)is str else len(record.get('source_lines',[]))
        if 'evidence' in record and 'evidence_ids' not in value:
            value['evidence_ids']=[e['evidence_id'] for e in record['evidence']]
        value['available_fields']=sorted(record)
        value['omitted_or_shortened_fields']=omitted
        return value

    @staticmethod
    def _source(record,form):
        if form=='declared' and 'source_lines' in record:
            return list(record['source_lines'])
        source=record.get('source')
        if type(source)is not str:raise LearningContractError('record has no executable source')
        return source.splitlines()

    @staticmethod
    def _one_form(record,form):
        result=copy.deepcopy(record)
        if record.get('kind')=='package':
            if form=='declared':
                result['source_lines']=KnowledgeReader._source(record,'declared')
                result['profile']=result.pop('declared_profile')
                result.pop('source',None)
            else:
                result.pop('source_lines',None);result.pop('declared_profile',None)
        return result

    def read(self, request):
        request=validate_read_request(request);identifier=request['identifier'];record=self._get(identifier)
        if record.get('scope')!=self.library.scope:raise LearningContractError('reader scope mismatch')
        view=request.get('view','brief');offset=request.get('offset',0);limit=request.get('limit',32)
        form=request.get('source_form','declared');pages={}
        if view=='brief':
            data=self._brief(record)
            if len(encoded(data,self.full_bytes))>self.brief_bytes:
                # Explicit omissions, never pretend shortened conditions are complete.
                if 'contract' in data:
                    original=data['contract'];data['contract']={k:self._shape(v) for k,v in original.items()}
                    data['omitted_or_shortened_fields'].append('contract')
                for key in ('applicability','limitations','revisit_when','claim','summary','rationale'):
                    if key in data:
                        data.pop(key);data['omitted_or_shortened_fields'].append(key)
        elif view=='source':
            lines=self._source(record,form);data=dict(source_lines=lines[offset:offset+limit])
            pages['source_lines']=dict(offset=offset,total=len(lines),next_offset=offset+limit if offset+limit<len(lines) else None)
        elif view=='fields':
            visible=self._one_form(record,form);data={}
            for key in request['fields']:
                if key not in visible:raise LearningContractError('unknown record field: '+key)
                raw=visible[key]
                if type(raw)is list:
                    data[key]=copy.deepcopy(raw[offset:offset+limit]);count=len(raw)
                elif type(raw)is dict:
                    keys=sorted(raw);data[key]={k:copy.deepcopy(raw[k]) for k in keys[offset:offset+limit]};count=len(keys)
                else:data[key]=copy.deepcopy(raw);count=1
                pages[key]=dict(offset=offset,total=count,next_offset=offset+limit if offset+limit<count else None)
        else:data=self._one_form(record,form)
        budget=self.brief_bytes if view=='brief' else self.full_bytes
        raw=encoded(data,budget)
        key=digest(dict(identifier=identifier,view=view,source_form=form,fields=request.get('fields'),offset=offset,limit=limit,record_sha256=digest(record)))
        count=self._reads.get(key,0);self._reads[key]=count+1
        return dict(protocol=PROTOCOLS['reader'],identifier=identifier,scope=clone(self.library.scope),
                    view=view,source_form=form,record_sha256=digest(record),data=data,pages=pages,
                    receipt=dict(read_key=key,repeat_read=count>0,presented_before_count=count,
                                 data_bytes=len(raw),full_record_bytes=len(encoded(record,self.full_bytes)),
                                 evidence='same immutable record, no new physical observation'),
                    execution_authority=False)

    def directory(self, **kwargs):
        if self.view is None:raise LearningContractError('current directory unavailable')
        return self.view.directory(**kwargs)
