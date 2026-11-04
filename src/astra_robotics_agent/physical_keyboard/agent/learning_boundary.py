"""Host facts for context operations; never a robot policy or task evaluator."""
from ..records.learning_contracts import clone, digest


class LearningBoundary:
    def __init__(self, store, task_contract=None, stop_requested=None):
        self.store, self.stop_requested = store, stop_requested
        self.task_sha = digest(task_contract.document if task_contract is not None else
                               dict(scope=store._scope, task_contract='legacy_learning'))
        self.generation, self.last_execution = 0, None
        self.inflight_rpc = self.inflight_action = self.inflight_model = False

    def check_stop(self):
        if self.stop_requested is not None and self.stop_requested():
            from ..adapters.applications.chat_service import OperatorStop
            raise OperatorStop('Operator STOP; context operations cannot restore execution')

    def execution(self, evidence_id, response, *, unknown, terminal):
        self.last_execution = dict(evidence_id=evidence_id,
            status=response.get('status', 'unknown'), motion_outcome_uncertain=bool(unknown),
            controller_terminal=bool(terminal))

    def facts(self, observation_id, *, motion_available=True, fault=None, assistance=None):
        policy = self.store.learning_access
        return dict(task_contract_sha256=self.task_sha,
            observation_evidence_id=observation_id, generation=self.generation,
            last_execution=clone(self.last_execution), motion_available=motion_available,
            inflight_rpc=self.inflight_rpc, inflight_action=self.inflight_action,
            inflight_model=self.inflight_model,
            stop_requested=bool(self.stop_requested is not None and self.stop_requested()),
            assistance=clone(assistance), fault=clone(fault),
            source_policy_sha256=digest(policy.descriptor()))

    def require_unchanged(self, before, observation_id):
        self.check_stop()
        if before != self.facts(observation_id):
            raise RuntimeError('Host authority changed during context operation; fresh review required')
