"""Public runner/provider glue without a model, SDK or network."""
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.agent.providers import CodexTaskSession
from astra_robotics_agent.physical_keyboard.runtime.device_adapter import load_adapter
from astra_robotics_agent.physical_keyboard.runtime.mock_device import MockDevice
from astra_robotics_agent.physical_keyboard.runtime.program_runtime import ProgramRuntime
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment


class PublicRunnerTests(unittest.TestCase):
    def test_codex_wrapper_is_lazy_and_preserves_explicit_settings(self):
        class Client:
            def __init__(self, **options):
                self.options, self.starts, self.turns, self.closed = options, 0, 0, False
            def start(self):
                self.starts += 1
            def run_turn(self, prompt, callback, images):
                self.turns += 1
                return dict(turn_id=str(self.turns), usage=None, final_text='fixture')
            def close(self):
                self.closed = True
        session = CodexTaskSession({}, instructions='generic', cwd=Path.cwd(), output=Path('/tmp/unused'),
                                   session_type=Client)
        self.assertEqual(session.client.starts, 0)
        self.assertEqual(session.client.options['effort'], 'xhigh')
        self.assertEqual(session.client.options['service_tier'], 'default')
        self.assertTrue(session.client.options['disable_execution_tools'])
        for _ in range(2):
            session.run_turn('fixture', lambda: None, images=[])
        self.assertEqual(session.client.starts, 1)
        session.close()
        self.assertTrue(session.client.closed)

    def test_evidence_ids_are_distinct_across_runs_in_one_knowledge_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            scope = dict(task_id='fixture', device_id='mock', layout_id='same')
            store = KnowledgeStore(root/'knowledge', scope, 'first')
            ids = []
            for run_id in ('first', 'second'):
                runtime = ProgramRuntime(MockDevice(), root/run_id, run_id=run_id)
                try:
                    observation = runtime.observe()
                    ids.append(observation['evidence_id'])
                    store.register_evidence(ids[-1], observation['evidence_files'])
                finally:
                    runtime.close()
            self.assertEqual(len(set(ids)), 2)
            reopened = KnowledgeStore(root/'knowledge', scope, 'third')
            for ident in ids:
                self.assertEqual(SkillDevelopment(reopened).inspect_evidence(
                    {'evidence_id': ident})['evidence_id'], ident)

    def test_adapter_selection_needs_only_a_factory_string(self):
        adapter = load_adapter('astra_robotics_agent.physical_keyboard.runtime.mock_device:factory', {})
        try:
            self.assertEqual(adapter.describe().evidence_mode, 'synthetic')
        finally:
            adapter.close()
