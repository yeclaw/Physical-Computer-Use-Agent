"""Explicit source closure policy; descriptions and summaries are not permissions."""
from .learning_contracts import (PROTOCOLS, LearningContractError, exact, ids, identifier,
                                scope as validate_scope, clone, digest)

class LearningAccessDenied(LearningContractError):
    pass

class LearningAccess:
    def __init__(self, scope, config=None, *, own_source_ledger=None):
        self.scope=validate_scope(scope)
        self.restricted=config is not None
        config={} if config is None else clone(config)
        if self.restricted:
            exact(config,{'protocol','scope','allowed_source_ids','allowed_knowledge_ids','allowed_evidence_ids','allow_own_experience'})
            if config['protocol']!=PROTOCOLS['access'] or validate_scope(config['scope'])!=self.scope:
                raise LearningAccessDenied('access scope mismatch')
            if type(config['allow_own_experience'])is not bool:raise LearningContractError('explicit own experience flag required')
        self.allowed_sources=set(ids(config.get('allowed_source_ids',[]),maximum=4096))
        self.allowed_knowledge=set(ids(config.get('allowed_knowledge_ids',[]),maximum=4096))
        self.allowed_evidence=set(ids(config.get('allowed_evidence_ids',[]),maximum=4096))
        self.allow_own_experience=config.get('allow_own_experience',True)
        self._knowledge={};self._evidence={}
        self._own_source_ledger = own_source_ledger
        if own_source_ledger is not None:
            if not self.restricted or own_source_ledger.base_policy != config:
                raise LearningAccessDenied('own source journal starting policy mismatch')
            for evidence_id in sorted(own_source_ledger.evidence_ids):
                self.register_own_evidence(evidence_id)

    def check_sources(self, source_ids, *, require_provenance=False):
        if type(source_ids) not in (list,tuple):raise LearningContractError('source_ids must be an explicit list')
        values=ids(list(source_ids),'source_ids',4096)
        if self.restricted and ((require_provenance and not values) or not set(values)<=self.allowed_sources):
            raise LearningAccessDenied('source closure is not authorized')
        return values

    def register_own_evidence(self, evidence_id):
        """Trusted observation/execution registration only, never model supplied files."""
        identifier(evidence_id)
        if self.restricted and not self.allow_own_experience:raise LearningAccessDenied('own experience is disabled')
        if self._own_source_ledger is not None:
            self._own_source_ledger.record(evidence_id)
        source='own-'+digest(evidence_id)
        self.allowed_sources.add(source);self.allowed_evidence.add(evidence_id)
        self._evidence[evidence_id]=[source]
        return [source]

    def bind_evidence(self, evidence_id, source_ids):
        identifier(evidence_id);values=self.check_sources(source_ids,require_provenance=True)
        if evidence_id in self._evidence and self._evidence[evidence_id]!=values:
            raise LearningAccessDenied('evidence source binding is immutable')
        self._evidence[evidence_id]=values

    def evidence_sources(self,evidence_id):
        identifier(evidence_id)
        values=self._evidence.get(evidence_id)
        if self.restricted and values is None and evidence_id not in self.allowed_evidence:
            raise LearningAccessDenied('evidence is outside source policy')
        if values is not None:self.check_sources(values,require_provenance=True)
        return list(values or [])

    def bind_knowledge(self, knowledge_id, source_ids):
        identifier(knowledge_id);values=self.check_sources(source_ids,require_provenance=True)
        if knowledge_id in self._knowledge and self._knowledge[knowledge_id]!=values:
            raise LearningAccessDenied('knowledge source binding is immutable')
        self._knowledge[knowledge_id]=values

    def knowledge_sources(self, knowledge_id):
        identifier(knowledge_id);values=self._knowledge.get(knowledge_id)
        if self.restricted and values is None and knowledge_id not in self.allowed_knowledge:
            raise LearningAccessDenied('legacy or unbound knowledge is outside source policy')
        if values is not None:self.check_sources(values,require_provenance=True)
        return list(values or [])

    def descriptor(self):
        return dict(protocol=PROTOCOLS['access'],scope=clone(self.scope),restricted=self.restricted,
                    allowed_source_ids=sorted(self.allowed_sources),allowed_knowledge_ids=sorted(self.allowed_knowledge),
                    allowed_evidence_ids=sorted(self.allowed_evidence),allow_own_experience=self.allow_own_experience,
                    meaning='source visibility only; does not grant robot execution permission')
