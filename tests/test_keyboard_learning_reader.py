import tempfile
import unittest
from pathlib import Path
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.reader import KnowledgeReader
from astra_robotics_agent.physical_keyboard.records.learning_contracts import LearningContractError

SCOPE=dict(task_id='test',device_id='robot',layout_id='board')
SOURCE=['def main(robot, profile):','    return robot.observe()']

class ReaderTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name).resolve()
        self.store=KnowledgeStore(root/'knowledge',SCOPE,'r')
        raw=root/'evidence';raw.write_text('observed no physical success')
        self.store.register_evidence('e',[raw]);self.library=SkillDevelopment(self.store)
        self.package=self.library.save_package(dict(id='enter',source_lines=SOURCE,profile={'positions':list(range(100))},contract=dict(description='conditional enter',input_schema=dict(type='object',properties={},additionalProperties=False),preconditions=['at hover'],postconditions=['release and inspect']),applicability=['current fixture'],limitations=['candidate'],evidence_ids=['e']))['package_id']
        self.reader=KnowledgeReader(self.library)

    def test_default_brief_does_not_send_source_or_profile_arrays(self):
        response=self.reader.read(dict(identifier=self.package))
        self.assertNotIn('source',response['data']);self.assertNotIn('source_lines',response['data'])
        self.assertNotIn('profile',response['data']);self.assertEqual(response['data']['contract']['preconditions'],['at hover'])
        self.assertFalse(response['execution_authority'])
        self.assertLess(response['receipt']['data_bytes'],response['receipt']['full_record_bytes'])

    def test_source_forms_and_fields_preserve_exact_values(self):
        legacy=self.library.inspect(self.package)
        declared=self.reader.read(dict(identifier=self.package,view='full'))['data']
        self.assertNotIn('source',declared);self.assertNotIn('declared_profile',declared)
        self.assertEqual(declared['profile'],legacy['declared_profile'])
        compiled=self.reader.read(dict(identifier=self.package,view='full',source_form='compiled'))['data']
        self.assertNotIn('source_lines',compiled);self.assertEqual(compiled['source'],legacy['source'])
        page=self.reader.read(dict(identifier=self.package,view='source',offset=1,limit=1))
        self.assertEqual(page['data']['source_lines'],SOURCE[1:]);self.assertIsNone(page['pages']['source_lines']['next_offset'])

    def test_repeated_reads_are_reported_not_silently_suppressed(self):
        args=dict(identifier=self.package)
        first=self.reader.read(args);second=self.reader.read(args)
        self.assertEqual(first['data'],second['data']);self.assertTrue(second['receipt']['repeat_read'])
        first['data']['id']='mutated';self.assertEqual(self.reader.read(args)['data']['id'],'enter')

    def test_exact_legacy_inspect_and_bad_requests(self):
        previous=self.library.inspect(self.package)
        for request in [dict(identifier=self.package,view='bad'),dict(identifier=self.package,view='fields',fields=['notthere']),dict(identifier=self.package,offset=True)]:
            with self.assertRaises(LearningContractError):self.reader.read(request)
        self.assertEqual(self.library.inspect(self.package),previous)

if __name__=='__main__':unittest.main()
