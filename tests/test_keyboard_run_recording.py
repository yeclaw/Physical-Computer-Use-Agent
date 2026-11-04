"""Record/schema compatibility and real recursive source-freeze checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.records.run_recording import (
    RunRecorder, freeze_python_tree,
)


class RunRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='keyboard-run-recording-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()

    def test_event_bytes_and_schema_match_the_original_learning_loop(self):
        times = iter([123, 456])
        recorder = RunRecorder(self.base, clock_ns=lambda: next(times))
        entries = [('model_decision', {'action': 'observe', 'label': '按键'}),
                   ('framework_feedback', {'status': 'ok', 'text_success': None})]
        expected = []
        for (kind, data), stamp in zip(entries, [123, 456]):
            recorder.event(kind, data)
            expected.append(json.dumps(dict(event=kind, utc_ns=stamp, data=data),
                                       ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        raw = (self.base / 'learning-events.jsonl').read_bytes()
        self.assertEqual(raw, ''.join(expected).encode('utf-8'))
        self.assertEqual([set(json.loads(line)) for line in raw.splitlines()],
                         [{'event', 'utc_ns', 'data'}] * 2)

    def test_custom_serializer_is_used_with_the_unchanged_record(self):
        calls = []
        def serialize(record):
            calls.append(record)
            return json.dumps(record, separators=(',', ':'))
        recorder = RunRecorder(self.base, serializer=serialize, clock_ns=lambda: 900)
        recorder.event('probe', {'id': 'a'})
        expected = {'event': 'probe', 'utc_ns': 900, 'data': {'id': 'a'}}
        self.assertEqual(calls, [expected])
        self.assertEqual((self.base / 'learning-events.jsonl').read_text(),
                         json.dumps(expected, separators=(',', ':')) + '\n')

    def test_constructor_does_not_create_a_directory_or_event_file(self):
        path = self.base / 'not-created'
        RunRecorder(path)
        self.assertFalse(path.exists())

    def test_nonfinite_data_is_rejected_before_creating_an_event_file(self):
        recorder = RunRecorder(self.base, clock_ns=lambda: 1)
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                recorder.event('invalid', {'value': value})
        self.assertFalse((self.base / 'learning-events.jsonl').exists())

    def test_nontext_serializer_result_does_not_create_an_event_file(self):
        recorder = RunRecorder(self.base, serializer=lambda record: b'bytes')
        with self.assertRaises(TypeError):
            recorder.event('invalid', {})
        self.assertFalse((self.base / 'learning-events.jsonl').exists())


class FreezePythonTreeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='keyboard-source-freeze-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / 'package'
        self.source.mkdir()
        self.destination = self.base / 'frozen'

    def put(self, name, raw):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def test_nested_same_names_and_empty_initializers_are_copied_with_hashes(self):
        sources = {'__init__.py': b'', 'agent/__init__.py': b'',
                   'runtime/__init__.py': b'', 'agent/tools.py': b'AGENT = 1\n',
                   'runtime/tools.py': b'RUNTIME = 2\n',
                   'runtime/backends/clock.py': b'def tick():\n    return 3\n'}
        for name, raw in sources.items():
            self.put(name, raw)
        self.put('notes.txt', b'not source')
        self.put('__pycache__/cached.pyc', b'not source')
        manifest = freeze_python_tree(self.source, self.destination)
        self.assertEqual(manifest['source_root'], str(self.source))
        self.assertEqual(set(manifest['files']), set(sources))
        self.assertEqual(json.loads((self.destination / 'source-manifest.json').read_text()), manifest)
        for name, raw in sources.items():
            with self.subTest(name=name):
                self.assertEqual((self.destination / name).read_bytes(), raw)
                self.assertEqual(manifest['files'][name],
                                 {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
                self.assertEqual((self.source / name).read_bytes(), raw)
        self.assertFalse((self.destination / 'notes.txt').exists())

    def test_repeat_and_existing_destinations_are_not_overwritten(self):
        self.put('module.py', b'ORIGINAL = True\n')
        first = freeze_python_tree(self.source, self.destination)
        original_manifest = (self.destination / 'source-manifest.json').read_bytes()
        self.put('module.py', b'CHANGED = True\n')
        with self.assertRaises(FileExistsError):
            freeze_python_tree(self.source, self.destination)
        self.assertEqual((self.destination / 'module.py').read_bytes(), b'ORIGINAL = True\n')
        self.assertEqual((self.destination / 'source-manifest.json').read_bytes(), original_manifest)
        self.assertEqual(json.loads(original_manifest), first)
        existing = self.base / 'existing'
        existing.mkdir()
        (existing / 'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            freeze_python_tree(self.source, existing)
        self.assertEqual((existing / 'keep.txt').read_text(), 'keep')

    def test_destination_inside_source_is_rejected_before_any_write(self):
        self.put('module.py', b'')
        nested = self.source / 'snapshots' / 'copy'
        with self.assertRaisesRegex(ValueError, 'outside'):
            freeze_python_tree(self.source, nested)
        self.assertFalse((self.source / 'snapshots').exists())
        with self.assertRaisesRegex(ValueError, 'outside'):
            freeze_python_tree(self.source, self.source)

    def test_symbolic_source_root_is_rejected(self):
        linked = self.base / 'linked-package'
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            freeze_python_tree(linked, self.destination)
        self.assertFalse(self.destination.exists())

    def test_source_file_directory_and_nonpython_links_are_rejected(self):
        outside = self.base / 'outside'
        outside.mkdir()
        (outside / 'external.py').write_text('OUTSIDE = True\n')
        for name, target in [('linked.py', outside / 'external.py'),
                             ('linked-directory', outside),
                             ('notes.txt', outside / 'external.py')]:
            link = self.source / name
            link.symlink_to(target, target_is_directory=target.is_dir())
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'symbolic'):
                freeze_python_tree(self.source, self.destination)
            self.assertFalse(self.destination.exists())
            link.unlink()

    def test_linked_destination_is_rejected_without_altering_its_target(self):
        outside = self.base / 'other'
        outside.mkdir()
        (outside / 'keep.txt').write_text('keep')
        self.destination.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            freeze_python_tree(self.source, self.destination)
        self.assertEqual((outside / 'keep.txt').read_text(), 'keep')
        self.destination.unlink()
        self.destination.symlink_to(self.base / 'missing', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            freeze_python_tree(self.source, self.destination)
        self.assertFalse((self.base / 'missing').exists())

    def test_linked_source_and_destination_parents_are_rejected(self):
        alias = self.base / 'alias'
        alias.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            freeze_python_tree(alias / self.source.name, self.destination)
        self.assertFalse(self.destination.exists())
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            freeze_python_tree(self.source, alias / self.destination.name)
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
