import json,tempfile,unittest
from pathlib import Path
from astra_robotics_agent.physical_keyboard.records.journal import SegmentedJournal
from astra_robotics_agent.physical_keyboard.records.storage import StorageError

class JournalTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.path=self.root/'index.jsonl'
    def test_rollover_retains_legacy_bytes_and_rebuilds_order(self):
        j=SegmentedJournal(self.path,segment_bytes=64)
        first=b'{"first":"unchanged original"}\n';j.append(first)
        for i in range(50):j.append((json.dumps(dict(index=i))+'\n').encode())
        self.assertTrue(self.path.read_bytes().startswith(first))
        self.assertGreater(len(list(j.parts.glob('*.jsonl'))),1)
        self.assertEqual([json.loads(l).get('index') for l in SegmentedJournal(self.path).lines()][1:],list(range(50)))
    def test_large_legacy_index_is_streamed_not_capped_at_eight_megabytes(self):
        line=b'{"padding":"'+b'x'*1000+b'"}\n'
        self.path.write_bytes(line*8400)
        self.assertGreater(self.path.stat().st_size,8*1024*1024)
        self.assertEqual(sum(1 for _ in SegmentedJournal(self.path).lines()),8400)
    def test_missing_segment_or_legacy_is_not_silently_ignored(self):
        j=SegmentedJournal(self.path,segment_bytes=64)
        for i in range(20):j.append((json.dumps(dict(index=i,padding='long enough'))+'\n').encode())
        self.path.unlink()
        with self.assertRaises(StorageError):list(j.lines())
    def test_symlink_and_truncated_record_are_rejected_without_repair(self):
        target=self.root/'target';target.write_bytes(b'unchanged')
        self.path.symlink_to(target)
        with self.assertRaises(StorageError):list(SegmentedJournal(self.path).lines())
        with self.assertRaises(StorageError):SegmentedJournal(self.path).append(b'{}\n')
        self.assertEqual(target.read_bytes(),b'unchanged')
        self.path.unlink();self.path.write_bytes(b'{"partial":')
        with self.assertRaises(StorageError):list(SegmentedJournal(self.path).lines())
        self.assertEqual(self.path.read_bytes(),b'{"partial":')
    def test_record_budget_and_nonregular_files(self):
        j=SegmentedJournal(self.path,max_line_bytes=8)
        with self.assertRaises(StorageError):j.append(b'{"long":1234}\n')
        self.path.mkdir()
        with self.assertRaises(StorageError):list(j.lines())
if __name__=='__main__':unittest.main()
