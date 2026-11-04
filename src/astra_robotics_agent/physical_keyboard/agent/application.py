"""Application policy selection; the task loop owns execution, adapters own semantics."""


class GoalApplication:
    continuous = False
    application_id = None

    def validate_callbacks(self, reset, wait_for_update):
        pass

    def initialize_report(self, report):
        pass

    def context_fields(self, report):
        return {}

    def recoverable_assistance(self, request):
        return False


def application_for(task_contract):
    if task_contract is not None and getattr(task_contract, 'is_skill_development_training', False):
        from astra_robotics_agent.physical_keyboard.adapters.environments.skill_training import SkillTrainingApplication
        application = SkillTrainingApplication()
        application.application_id = 'typing_lab'
        return application
    if task_contract is not None and task_contract.is_chat:
        from astra_robotics_agent.physical_keyboard.adapters.applications.chat_lifecycle import ChatApplication
        return ChatApplication(task_contract)
    return GoalApplication()
