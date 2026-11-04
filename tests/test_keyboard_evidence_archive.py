"""Lossless byte round trips and malicious-container failures; temporary data."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.records.evidence_archive import EvidenceArchive
from astra_robotics_agent.physical_keyboard.records.learning_contracts import encoded


class EvidenceArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.archive = EvidenceArchive(self.root / 'archive')
        self.original = self.root / 'original.bin'
        self.raw = b'\0binary\xffevidence\r\n' * 137
        self.original.write_bytes(self.raw)

    def descriptor(self, path=None):
        path = self.original if path is None else path
        raw = path.read_bytes()
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    def registration(self, descriptor):
        return json.loads(Path(descriptor['manifest_path']).read_bytes())

    def forged(self, descriptor, make_tar):
        registration = self.registration(descriptor)
        raw = make_tar(encoded(registration['manifest']))
        compressed = gzip.compress(raw, mtime=0)
        archive_hash = hashlib.sha256(compressed).hexdigest()
        archive_id = 'archive-' + archive_hash
        registration.update(archive_id=archive_id, archive_sha256=archive_hash, archive_bytes=len(compressed))
        (self.archive.archives / (archive_id + '.tar.gz')).write_bytes(compressed)
        (self.archive.registrations / (archive_id + '.json')).write_bytes(encoded(registration))
        return archive_id

    def tar(self, manifest, extra, *, include_object=True):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo('manifest.json'); info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            if include_object:
                info = tarfile.TarInfo('objects/' + hashlib.sha256(self.raw).hexdigest()); info.size = len(self.raw)
                archive.addfile(info, io.BytesIO(self.raw))
            if extra is not None:
                info, data = extra
                archive.addfile(info, io.BytesIO(data) if data is not None else None)
        return output.getvalue()

    def test_raw_roundtrip_preserves_original_hash_and_no_action_network_or_extract(self):
        before = self.descriptor()
        with patch('socket.create_connection') as network, patch('subprocess.Popen') as process, \
                patch.object(tarfile.TarFile, 'extractall') as extraction:
            result = self.archive.create([before], name='temporary fixture')
            restored = self.archive.read(result['archive_id'], before['sha256'])
        self.assertEqual(restored, self.raw)
        self.assertEqual(self.descriptor(), before)
        network.assert_not_called(); process.assert_not_called(); extraction.assert_not_called()
        self.assertTrue(result['originals_retained'])
        self.assertEqual(result['association_kind'], 'lossless_byte_copy_same_evidence')
        destination = self.root / 'new-restore'; destination.mkdir()
        (destination / 'restored.bin').write_bytes(restored)
        self.assertEqual(hashlib.sha256((destination / 'restored.bin').read_bytes()).hexdigest(), before['sha256'])

    def test_deterministic_idempotent_archive_and_same_bytes_deduplicate_members(self):
        second = self.root / 'second.bin'; second.write_bytes(self.raw)
        first = self.archive.create([self.descriptor(), self.descriptor(second)])
        again = self.archive.create([self.descriptor(second), self.descriptor()])
        self.assertEqual(first, again)
        self.assertEqual(first['member_count'], 1)
        manifest = self.registration(first)['manifest']
        self.assertEqual(manifest['total_original_bytes'], len(self.raw) * 2)
        self.assertEqual(manifest['total_member_bytes'], len(self.raw))
        self.assertEqual(len(manifest['objects'][0]['originals']), 2)

    def test_trusted_descriptors_must_match_actual_hash_size_and_are_not_model_paths(self):
        for value in (dict(self.descriptor(), sha256='0' * 64),
                      dict(self.descriptor(), bytes=True), dict(self.descriptor(), bytes=1),
                      dict(self.descriptor(), command='anything'),
                      dict(self.descriptor(), path=str(self.root / '..' / 'escape'))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.archive.create([value])
        with self.assertRaises(ValueError): self.archive.create([self.descriptor(), self.descriptor()])
        self.assertEqual(list(self.archive.archives.iterdir()), [])

    def test_missing_corrupted_archive_or_manifest_does_not_fall_back_to_original(self):
        descriptor = self.archive.create([self.descriptor()])
        registration_path = Path(descriptor['manifest_path'])
        archive_path = Path(descriptor['archive_path'])
        original_registration, original_archive = registration_path.read_bytes(), archive_path.read_bytes()
        registration_path.write_text('{}')
        with self.assertRaises(ValueError): self.archive.read(descriptor['archive_id'], self.descriptor()['sha256'])
        registration_path.write_bytes(original_registration)
        archive_path.write_bytes(original_archive[:-1] + bytes([original_archive[-1] ^ 1]))
        with self.assertRaises(ValueError): self.archive.read(descriptor['archive_id'], self.descriptor()['sha256'])
        archive_path.unlink()  # Remove only this temporary test's derived archive.
        with self.assertRaises(ValueError): self.archive.read(descriptor['archive_id'], self.descriptor()['sha256'])
        self.assertEqual(self.original.read_bytes(), self.raw)

    def test_manifest_tampering_or_unregistered_member_fails_closed(self):
        descriptor = self.archive.create([self.descriptor()])
        with self.assertRaises(ValueError): self.archive.read(descriptor['archive_id'], '0' * 64)
        registration = self.registration(descriptor)
        registration['manifest']['objects'][0]['bytes'] += 1
        Path(descriptor['manifest_path']).write_bytes(encoded(registration))
        with self.assertRaises(ValueError): self.archive.read(descriptor['archive_id'], self.descriptor()['sha256'])

    def test_malicious_traversal_absolute_link_duplicate_and_pax_members_are_rejected(self):
        descriptor = self.archive.create([self.descriptor()])
        bad = []
        for name in ('../escape', '/absolute', 'objects/../escape'):
            info = tarfile.TarInfo(name); info.size = 1; bad.append((info, b'x'))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.XHDTYPE):
            info = tarfile.TarInfo('unexpected'); info.type = kind; info.linkname = '../escape'
            bad.append((info, None))
        info = tarfile.TarInfo('objects/' + self.descriptor()['sha256']); info.size = len(self.raw)
        bad.append((info, self.raw))
        for extra in bad:
            with self.subTest(type=extra[0].type, name=extra[0].name):
                archive_id = self.forged(descriptor, lambda manifest: self.tar(manifest, extra))
                with patch('socket.create_connection') as network, patch('subprocess.Popen') as process:
                    with self.assertRaises(ValueError): self.archive.read(archive_id, self.descriptor()['sha256'])
                network.assert_not_called(); process.assert_not_called()
        self.assertFalse((self.root / 'escape').exists())

    def test_huge_header_and_truncated_data_fail_before_unbounded_decompression(self):
        descriptor = self.archive.create([self.descriptor()])
        def huge(manifest):
            header = tarfile.TarInfo('objects/' + self.descriptor()['sha256'])
            header.size = 2 ** 30
            return header.tobuf(format=tarfile.USTAR_FORMAT) + b'\0' * 1024
        archive_id = self.forged(descriptor, huge)
        with self.assertRaisesRegex(ValueError, 'registered byte boundary'):
            self.archive.read(archive_id, self.descriptor()['sha256'])
        archive_id = self.forged(descriptor, lambda manifest: self.tar(manifest, None)[:800])
        with self.assertRaises(ValueError): self.archive.read(archive_id, self.descriptor()['sha256'])

    def test_missing_member_wrong_inner_hash_and_extra_trailing_data_are_rejected(self):
        descriptor = self.archive.create([self.descriptor()])
        makers = [lambda manifest: self.tar(manifest, None, include_object=False),
                  lambda manifest: self.tar(manifest, None).replace(self.raw, b'X' * len(self.raw), 1),
                  lambda manifest: self.tar(manifest, None) + b'not-tar-data',
                  lambda manifest: self.tar(manifest, None) + b'\0' * (tarfile.RECORDSIZE + 1)]
        for maker in makers:
            with self.subTest(maker=maker):
                archive_id = self.forged(descriptor, maker)
                with self.assertRaises(ValueError): self.archive.read(archive_id, self.descriptor()['sha256'])

    def test_symlink_original_archive_and_root_are_rejected_without_writing_through(self):
        original_alias = self.root / 'alias.bin'; original_alias.symlink_to(self.original)
        with self.assertRaises(ValueError): self.archive.create([dict(self.descriptor(), path=str(original_alias))])
        outside = self.root / 'outside'; outside.mkdir()
        alias = self.root / 'linked-root'; alias.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): EvidenceArchive(alias)
        self.assertEqual(list(outside.iterdir()), [])
        descriptor = self.archive.create([self.descriptor()])
        archive_path = Path(descriptor['archive_path']); archive_path.unlink(); archive_path.symlink_to(self.original)
        with self.assertRaises(ValueError): self.archive.read(descriptor['archive_id'], self.descriptor()['sha256'])

    def test_per_operation_budgets_are_explicit_and_do_not_touch_originals(self):
        bounded = EvidenceArchive(self.root / 'bounded', max_member_bytes=2, max_total_bytes=3, max_members=1)
        with self.assertRaises(ValueError): bounded.create([self.descriptor()])
        with self.assertRaises(ValueError): bounded.create([self.descriptor(), self.descriptor()])
        self.assertEqual(self.original.read_bytes(), self.raw)

    def test_normalized_base64_has_exact_raw_and_image_hash_same_evidence_and_keeps_raw(self):
        image = b'fixture image bytes, no new camera observation'
        raw = json.dumps(dict(manifest={'historical': True}, images_base64={
            'right': base64.b64encode(image).decode(), 'left': base64.b64encode(image).decode()})).encode()
        before = bytes(raw)
        with patch('socket.create_connection') as network, patch('subprocess.Popen') as process:
            view = self.archive.normalized_view(raw)
        self.assertEqual(raw, before)
        self.assertEqual(view['raw_json_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertFalse(view['new_visual_evidence'])
        self.assertNotIn('images_base64', view['value'])
        refs = view['value']['image_references']
        self.assertEqual(refs['right']['image_id'], refs['left']['image_id'])
        self.assertTrue(refs['right']['same_evidence'])
        self.assertEqual((self.archive.images / refs['right']['sha256']).read_bytes(), image)
        network.assert_not_called(); process.assert_not_called()

    def test_invalid_json_base64_url_collision_and_existing_blob_corruption_fail(self):
        cases = [b'{"x":1,"x":2}', b'{"x":NaN}',
                 b'{"images_base64":{"v":"https://example.com/image.jpg"}}',
                 b'{"images_base64":["data:text/plain;base64,aGVsbG8="]}',
                 b'{"images_base64":["%%%%"]}', b'{"images_base64":5}',
                 b'{"images_base64":[],"image_references":{}}']
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.archive.normalized_view(raw)
        raw = b'{"images_base64":["aGVsbG8="]}'
        view = self.archive.normalized_view(raw)
        (self.archive.images / view['images'][0]['sha256']).write_bytes(b'bad')
        with self.assertRaises(ValueError): self.archive.normalized_view(raw)


if __name__ == '__main__':
    unittest.main()
