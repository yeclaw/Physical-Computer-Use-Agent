"""Core-only module ownership and dependency validation."""
CORE_MODULES = ('agent', 'runtime', 'skills', 'records')
ADAPTER_GROUPS = ('applications', 'environments', 'skill_sources')
LEGACY_MODULES = {'astra_bridge': 'astra_robotics_agent.physical_keyboard.agent.astra_bridge', 'campaign': 'astra_robotics_agent.physical_keyboard.agent.campaign', 'chat_policy': 'astra_robotics_agent.physical_keyboard.adapters.applications.chat_policy', 'chat_service': 'astra_robotics_agent.physical_keyboard.adapters.applications.chat_service', 'contracts': 'astra_robotics_agent.physical_keyboard.records.contracts', 'demo_support': 'astra_robotics_agent.physical_keyboard.adapters.skill_sources.demo_support', 'episode_review': 'astra_robotics_agent.physical_keyboard.records.episode_review', 'executable': 'astra_robotics_agent.physical_keyboard.runtime.executable', 'hints': 'astra_robotics_agent.physical_keyboard.agent.hints', 'knowledge': 'astra_robotics_agent.physical_keyboard.skills.knowledge', 'learning_loop': 'astra_robotics_agent.physical_keyboard.agent.learning_loop', 'ledger': 'astra_robotics_agent.physical_keyboard.records.ledger', 'notepad_reset': 'astra_robotics_agent.physical_keyboard.adapters.environments.notepad_reset', 'python_worker': 'astra_robotics_agent.physical_keyboard.runtime.python_worker', 'recording_review': 'astra_robotics_agent.physical_keyboard.records.recording_review', 'review_watcher': 'astra_robotics_agent.physical_keyboard.records.review_watcher', 'rpc': 'astra_robotics_agent.physical_keyboard.runtime.rpc', 'skill_language': 'astra_robotics_agent.physical_keyboard.runtime.skill_language', 'synthetic_keyboard': 'astra_robotics_agent.physical_keyboard.adapters.environments.synthetic_keyboard', 'training_reset': 'astra_robotics_agent.physical_keyboard.adapters.environments.training_reset', 'transport': 'astra_robotics_agent.physical_keyboard.runtime.transport', 'trial_runner': 'astra_robotics_agent.physical_keyboard.agent.trial_runner', 'visual_update': 'astra_robotics_agent.physical_keyboard.runtime.visual_update', 'worker': 'astra_robotics_agent.physical_keyboard.runtime.worker'}
MODULE_OWNERS = {'adapters/applications/chat_instructions.py': 'adapters', 'adapters/applications/chat_lifecycle.py': 'adapters', 'adapters/applications/chat_policy.py': 'adapters', 'adapters/applications/chat_service.py': 'adapters', 'adapters/environments/key_state.py': 'adapters', 'adapters/environments/notepad_reset.py': 'adapters', 'adapters/environments/skill_training.py': 'adapters', 'adapters/environments/synthetic_keyboard.py': 'adapters', 'adapters/environments/training_instructions.py': 'adapters', 'adapters/environments/training_reset.py': 'adapters', 'adapters/environments/typing_cases.py': 'adapters', 'adapters/environments/typing_lab.py': 'adapters', 'adapters/environments/typing_resume.py': 'adapters', 'adapters/environments/typing_schedule.py': 'adapters', 'adapters/skill_sources/demo_support.py': 'adapters', 'adapters/skill_sources/learning_sources.py': 'adapters', 'agent/application.py': 'agent', 'agent/application_skill.py': 'agent', 'agent/astra_bridge.py': 'agent', 'agent/campaign.py': 'agent', 'agent/development.py': 'agent', 'agent/execution_feedback.py': 'agent', 'agent/execution_service.py': 'agent', 'agent/fault_review.py': 'agent', 'agent/hints.py': 'agent', 'agent/instructions.py': 'agent', 'agent/learning.py': 'agent', 'agent/learning_boundary.py': 'agent', 'agent/learning_loop.py': 'agent', 'agent/offline_session.py': 'agent', 'agent/providers.py': 'agent', 'agent/runner.py': 'agent', 'agent/trial_runner.py': 'agent', 'agent/work_context.py': 'agent', 'records/admission.py': 'records', 'records/contracts.py': 'records', 'records/episode_review.py': 'records', 'records/evidence_archive.py': 'records', 'records/journal.py': 'records', 'records/knowledge_domains.py': 'records', 'records/learning_access.py': 'records', 'records/learning_contracts.py': 'records', 'records/ledger.py': 'records', 'records/own_sources.py': 'records', 'records/recording_review.py': 'records', 'records/redaction.py': 'records', 'records/review_watcher.py': 'records', 'records/run_recording.py': 'records', 'records/source_inputs.py': 'records', 'records/storage.py': 'records', 'records/trajectory_diagnostics.py': 'records', 'records/typing_metrics.py': 'records', 'records/usage.py': 'records', 'runtime/analysis_worker.py': 'runtime', 'runtime/control_protocol.py': 'runtime', 'runtime/device_adapter.py': 'runtime', 'runtime/executable.py': 'runtime', 'runtime/mock_device.py': 'runtime', 'runtime/observations.py': 'runtime', 'runtime/program_runtime.py': 'runtime', 'runtime/python_worker.py': 'runtime', 'runtime/remote_host.py': 'runtime', 'runtime/request_lifecycle.py': 'runtime', 'runtime/rpc.py': 'runtime', 'runtime/skill_language.py': 'runtime', 'runtime/transport.py': 'runtime', 'runtime/visual_update.py': 'runtime', 'runtime/worker.py': 'runtime', 'skills/current_view.py': 'skills', 'skills/development.py': 'skills', 'skills/knowledge.py': 'skills', 'skills/learning_attention.py': 'skills', 'skills/learning_changes.py': 'skills', 'skills/programs.py': 'skills', 'skills/read_scope.py': 'skills', 'skills/reader.py': 'skills', 'skills/recall.py': 'skills', 'skills/resources.py': 'skills'}
_PACKAGE_INITIALIZERS = ['__init__.py', 'adapters/__init__.py', 'adapters/applications/__init__.py', 'adapters/environments/__init__.py', 'adapters/skill_sources/__init__.py', 'agent/__init__.py', 'records/__init__.py', 'runtime/__init__.py', 'runtime/backends/__init__.py', 'skills/__init__.py']
_ROOT_SUPPORT = {'__init__.py', '_compat.py', '_paths.py', 'framework_layout.py'}
_PREFIX = 'astra_robotics_agent.physical_keyboard'
_FORBIDDEN = {'records': ('agent', 'skills', 'adapters'), 'runtime': ('agent', 'skills'), 'skills': ('agent', 'adapters.applications')}

import ast

import importlib.util

from pathlib import Path

def validate_layout(package_root=None):
    """Return deterministic errors for unregistered files and reversed core dependencies."""
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    errors = []
    expected = set(MODULE_OWNERS) | set(_PACKAGE_INITIALIZERS) | _ROOT_SUPPORT
    expected.update((name + '.py' for name in LEGACY_MODULES))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.py') if '__pycache__' not in p.parts}
    errors.extend(('Unregistered module: ' + p for p in sorted(actual - expected)))
    errors.extend(('Missing registered module: ' + p for p in sorted(expected - actual)))
    for relative, owner in sorted(MODULE_OWNERS.items()):
        path = root / relative
        if not path.is_file():
            continue
        module = _PREFIX + '.' + relative[:-3].replace('/', '.')
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError) as exc:
            errors.append(relative + ': cannot parse: ' + str(exc))
            continue
        for node in ast.walk(tree):
            dependencies = []
            if isinstance(node, ast.ImportFrom):
                dependency = importlib.util.resolve_name('.' * node.level + (node.module or ''), module.rsplit('.', 1)[0]) if node.level else node.module or ''
                dependencies.append(dependency)
                if dependency == _PREFIX:
                    dependencies.extend((_PREFIX + '.' + name.name for name in node.names))
            elif isinstance(node, ast.Import):
                dependencies.extend((name.name for name in node.names))
            for dependency in dependencies:
                if not dependency.startswith(_PREFIX + '.'):
                    continue
                local = dependency[len(_PREFIX) + 1:]
                if local in LEGACY_MODULES:
                    errors.append(relative + ': implementation imports legacy route ' + dependency)
                if any((local == prefix or local.startswith(prefix + '.') for prefix in _FORBIDDEN.get(owner, ()))):
                    errors.append(relative + ': ' + owner + ' cannot depend on ' + local)
    return errors
