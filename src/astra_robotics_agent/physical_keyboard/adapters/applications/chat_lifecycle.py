"""Physical-chat phase, assistance and waiting policy; never chooses an action."""


class ChatApplication:
    def __init__(self, task_contract):
        self.task_contract = task_contract
        self.continuous = task_contract.is_continuous_chat
        self.application_id = task_contract.document['application']

    def validate_callbacks(self, reset, wait_for_update):
        if reset is not None:
            raise ValueError('Training reset cannot be configured for chat; no callbacks attempted')
        if self.continuous and not callable(wait_for_update):
            raise ValueError('Continuous chat requires wait_for_update before any callbacks')

    def initialize_report(self, report):
        if self.continuous:
            report.update(chat_turn_records=[], assistance_history=[], session_mode='continuous')

    def context_fields(self, report):
        fields = {'application_context': dict(application_id=self.application_id,
            recipient_scope=self.task_contract.document['recipient_scope'],
            send_key=self.task_contract.document['send_key'],
            authority='frozen task contract; learned knowledge cannot override it')}
        if self.continuous:
            fields['chat_service'] = dict(mode='continuous',
                recorded_phases=len(report['chat_turn_records']), session_ended=False)
        return fields

    def record_phase(self, result, observation_evidence_id, lifecycle, report):
        lifecycle.invalidate_completion_review()
        entry = dict(result=result, terminal=False,
                     observation_evidence_id=observation_evidence_id)
        report['chat_turn_records'].append(entry)
        report['task_status'] = 'chatting'
        feedback = dict(phase='chat_turn_recorded', **entry,
            instruction='This message/RSI phase is recorded. Keep this same chat and RSI session alive; wait_for_update when waiting for new conversation or operator feedback. Do not repeat the message or require its old bubble to remain visible.')
        return entry, feedback

    def recoverable_assistance(self, request):
        return self.continuous and request['kind'] in ('operator_action', 'missing_external_capability')

    def validate_wait(self, args, observation_evidence_id):
        if not self.continuous:
            raise ValueError('wait_for_update requires a continuous chat session')
        assessment = args.get('assessment')
        if self.task_contract.is_responsive_chat and assessment is None:
            raise ValueError('responsive_v1 wait requires arguments.assessment with current observation, message summary, reply target, addressing and reason')
        if assessment is not None and assessment['observation_evidence_id'] != observation_evidence_id:
            raise ValueError('wait assessment must reference the current observation evidence ID')
        return assessment
