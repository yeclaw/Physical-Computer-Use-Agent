import copy
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.records.own_sources import OwnSourceLedger
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess, LearningAccessDenied
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS, digest
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from test_keyboard_skill_development import contract, SOURCE

SCOPE = dict(task_id='partial-learning-fixture', device_id='robot', layout_id='fixed')

class OwnSourceTests(unittest.TestCase):
    def policy(self):
        return dict(protocol=PROTOCOLS['access'], scope=SCOPE, allowed_source_ids=['seed'],
                    allowed_knowledge_ids=[], allowed_evidence_ids=[], allow_own_experience=True)

    def test_same_condition_reopens_new_program_without_opening_any_old_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'knowledge'; path = root / 'origins.jsonl'
            policy = self.policy()
            first = LearningAccess(SCOPE, policy, own_source_ledger=OwnSourceLedger(path,root,policy))
            store = KnowledgeStore(root,SCOPE,'condition-a01',learning_access=first)
            observation = Path(directory).resolve()/'response.json'; observation.write_text('{"fixture":true}')
            store.register_evidence('condition-own-obs',[observation])
            package = SkillDevelopment(store).save_package(dict(id='new-fixture-program',source_lines=list(SOURCE),
                profile={'text':'new parameter'},contract=contract(),applicability=['fixture'],
                limitations=['software only'],evidence_ids=['condition-own-obs']))
            later = LearningAccess(SCOPE,policy,own_source_ledger=OwnSourceLedger(path,root,policy))
            reopened = SkillDevelopment(KnowledgeStore(root,SCOPE,'condition-a02',learning_access=later))
            self.assertEqual(reopened.resolve_call(dict(package_id=package['package_id'],arguments={'text':'reuse'}))['profile']['text'],'reuse')
            self.assertEqual(later.allowed_sources, {'seed','own-'+digest('condition-own-obs')})
            with self.assertRaises(LearningAccessDenied): later.check_sources(['old-full-keyboard-source'])

    def test_copied_journal_broader_start_and_torn_tail_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()/'first';path=root/'origins.jsonl';policy=self.policy()
            ledger=OwnSourceLedger(path,root,policy);ledger.record('only-own')
            other=Path(directory).resolve()/'other';other.mkdir();(other/'origins.jsonl').write_bytes(path.read_bytes())
            with self.assertRaises(ValueError): OwnSourceLedger(other/'origins.jsonl',other,policy)
            broader=copy.deepcopy(policy);broader['allowed_source_ids'].append('old')
            with self.assertRaises(ValueError): OwnSourceLedger(path,root,broader)
            path.write_bytes(path.read_bytes().rstrip(b'\n'))
            with self.assertRaises(ValueError): OwnSourceLedger(path,root,policy)
