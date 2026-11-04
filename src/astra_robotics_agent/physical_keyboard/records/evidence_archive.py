"""Opt-in lossless evidence copies and encoding-only image projections.

Inputs to create are trusted registered-file descriptors, never model paths.
No SDK, network, extraction, original deletion, or original rewriting exists.
"""
import base64
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import uuid

from .learning_contracts import LearningContractError, encoded, exact, integer, sha256, text
from .storage import parent_fd, read_bytes, write_new


PROTOCOL = 'physical-keyboard-evidence-archive-v1'
_MANIFEST_MAX = 2 * 1024 * 1024
_CHUNK = 65536


class EvidenceArchiveError(LearningContractError):
    pass


def _decode(raw, maximum=_MANIFEST_MAX):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise EvidenceArchiveError('duplicate archive JSON field')
            value[key] = item
        return value
    try:
        result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(EvidenceArchiveError('nonfinite archive JSON')))
        encoded(result, maximum)
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EvidenceArchiveError('invalid bounded archive JSON') from exc


def _mkdir(path):
    """Create owned directories without traversing an existing symlink."""
    path = Path(path)
    if '..' in path.parts:
        raise EvidenceArchiveError('parent traversal forbidden')
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = child
    except OSError as exc:
        raise EvidenceArchiveError('owned archive directories without symlinks required') from exc
    finally:
        os.close(descriptor)
    return path


@contextmanager
def _owned_stream(path, maximum):
    directory, name = parent_fd(path)
    descriptor = None
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise EvidenceArchiveError('bounded regular archive input required')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            yield stream, before.st_size
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise EvidenceArchiveError('archive input changed while being read')
    except OSError as exc:
        raise EvidenceArchiveError('cannot read owned archive input') from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _hash_stream(stream):
    value, size = hashlib.sha256(), 0
    while True:
        block = stream.read(_CHUNK)
        if not block:
            break
        value.update(block); size += len(block)
    return value.hexdigest(), size


class EvidenceArchive:
    """Per-operation byte/member budgets; these impose no learning-run limit."""
    def __init__(self, root, *, max_member_bytes=256 * 1024 * 1024,
                 max_total_bytes=1024 * 1024 * 1024, max_members=256):
        self.max_member_bytes = integer(max_member_bytes, 'member byte budget', 1, 2 ** 40)
        self.max_total_bytes = integer(max_total_bytes, 'archive byte budget', 1, 2 ** 42)
        self.max_members = integer(max_members, 'archive member budget', 1, 4096)
        self.max_archive_bytes = max_total_bytes + max_members * 2048 + _MANIFEST_MAX + 65536
        self.root = _mkdir(root)
        self.archives = _mkdir(self.root / 'archives')
        self.registrations = _mkdir(self.root / 'registrations')
        self.staging = _mkdir(self.root / 'staging')
        self.images = _mkdir(self.root / 'images')

    def _descriptor(self, registration):
        return dict(archive_id=registration['archive_id'], sha256=registration['archive_sha256'],
                    bytes=registration['archive_bytes'], manifest_sha256=registration['manifest_sha256'],
                    member_count=len(registration['manifest']['objects']),
                    raw_bytes=registration['manifest']['total_member_bytes'],
                    archive_path=str(self.archives / (registration['archive_id'] + '.tar.gz')),
                    manifest_path=str(self.registrations / (registration['archive_id'] + '.json')),
                    association_kind='lossless_byte_copy_same_evidence', originals_retained=True)

    def create(self, objects, *, name=None):
        if type(objects) is not list or not 1 <= len(objects) <= self.max_members:
            raise EvidenceArchiveError('explicit bounded trusted file descriptor list required')
        if name is not None:
            text(name, 'archive label', 128)
        members, originals, total = {}, set(), 0
        for item in objects:
            exact(item, {'path', 'sha256', 'bytes'})
            expected = sha256(item['sha256'])
            size = integer(item['bytes'], 'original bytes', 0, self.max_member_bytes)
            if not isinstance(item['path'], (str, Path)):
                raise EvidenceArchiveError('trusted registered path required')
            path = Path(item['path'])
            if '..' in path.parts:
                raise EvidenceArchiveError('original traversal forbidden')
            path = Path(os.path.abspath(path))
            if str(path) in originals:
                raise EvidenceArchiveError('duplicate original path')
            originals.add(str(path)); total += size
            if total > self.max_total_bytes:
                raise EvidenceArchiveError('archive input byte budget exceeded')
            with _owned_stream(path, self.max_member_bytes) as (stream, actual_size):
                actual_hash, actual_read = _hash_stream(stream)
            if actual_size != size or actual_read != size or actual_hash != expected:
                raise EvidenceArchiveError('original hash or size differs from registered descriptor')
            member = members.setdefault(expected, dict(member='objects/' + expected,
                sha256=expected, bytes=size, originals=[]))
            if member['bytes'] != size:
                raise EvidenceArchiveError('conflicting content-addressed sizes')
            member['originals'].append(dict(path=str(path), sha256=expected, bytes=size))
        for member in members.values():
            member['originals'].sort(key=lambda item: item['path'])
        manifest = dict(protocol=PROTOCOL, name=name, objects=[members[key] for key in sorted(members)],
                        total_member_bytes=sum(item['bytes'] for item in members.values()),
                        total_original_bytes=total, association_kind='lossless_byte_copy_same_evidence',
                        originals_retained=True, compression='gzip+ustar')
        manifest_raw = encoded(manifest, _MANIFEST_MAX)
        stage = self.staging / (uuid.uuid4().hex + '.tar.gz')
        directory, basename = parent_fd(stage)
        descriptor = None
        try:
            descriptor = os.open(basename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            with os.fdopen(descriptor, 'wb', closefd=False) as output:
                with gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                        info = tarfile.TarInfo('manifest.json'); info.size = len(manifest_raw); info.mode = 0o400
                        archive.addfile(info, io.BytesIO(manifest_raw))
                        for member in manifest['objects']:
                            with _owned_stream(member['originals'][0]['path'], self.max_member_bytes) as (stream, size):
                                if size != member['bytes']:
                                    raise EvidenceArchiveError('original changed before archive copy')
                                hasher = hashlib.sha256()
                                class Reader:
                                    def read(self, amount):
                                        block = stream.read(amount); hasher.update(block); return block
                                info = tarfile.TarInfo(member['member']); info.size = size; info.mode = 0o400
                                archive.addfile(info, Reader())
                                if stream.read(1) or hasher.hexdigest() != member['sha256']:
                                    raise EvidenceArchiveError('original changed while copying archive')
                output.flush(); os.fsync(descriptor)
            os.close(descriptor); descriptor = None
            with _owned_stream(stage, self.max_archive_bytes) as (stream, size):
                archive_hash, _ = _hash_stream(stream)
            archive_id = 'archive-' + archive_hash
            target = self.archives / (archive_id + '.tar.gz')
            target_directory, target_name = parent_fd(target)
            try:
                try:
                    os.link(basename, target_name, src_dir_fd=directory, dst_dir_fd=target_directory,
                            follow_symlinks=False)
                    os.fsync(target_directory)
                except FileExistsError:
                    with _owned_stream(target, self.max_archive_bytes) as (stream, existing_size):
                        existing_hash, _ = _hash_stream(stream)
                    if existing_hash != archive_hash or existing_size != size:
                        raise EvidenceArchiveError('existing archive content is corrupted')
            finally:
                os.close(target_directory)
            registration = dict(protocol=PROTOCOL, archive_id=archive_id, archive_sha256=archive_hash,
                                archive_bytes=size, manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                                manifest_bytes=len(manifest_raw), manifest=manifest)
            path = self.registrations / (archive_id + '.json')
            raw = encoded(registration, _MANIFEST_MAX)
            if path.exists() or path.is_symlink():
                if read_bytes(path, _MANIFEST_MAX) != raw:
                    raise EvidenceArchiveError('archive registration is corrupted')
            else:
                write_new(path, raw)
            return self._descriptor(registration)
        except (OSError, tarfile.TarError, EOFError) as exc:
            raise EvidenceArchiveError('archive creation failed') from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(basename, dir_fd=directory)  # Only this invocation's staging file.
            except FileNotFoundError:
                pass
            os.close(directory)

    def _registration(self, archive_id):
        if type(archive_id) is not str or not archive_id.startswith('archive-'):
            raise EvidenceArchiveError('content-addressed archive ID required')
        sha256(archive_id[len('archive-'):])
        registration = _decode(read_bytes(self.registrations / (archive_id + '.json'), _MANIFEST_MAX))
        exact(registration, {'protocol', 'archive_id', 'archive_sha256', 'archive_bytes',
                             'manifest_sha256', 'manifest_bytes', 'manifest'})
        if registration['protocol'] != PROTOCOL or registration['archive_id'] != archive_id or registration['archive_sha256'] != archive_id[8:]:
            raise EvidenceArchiveError('archive registration identity mismatch')
        integer(registration['archive_bytes'], 'compressed bytes', 1, self.max_archive_bytes)
        manifest = registration['manifest']
        exact(manifest, {'protocol', 'name', 'objects', 'total_member_bytes', 'total_original_bytes',
                         'association_kind', 'originals_retained', 'compression'})
        if (manifest['protocol'] != PROTOCOL or manifest['association_kind'] != 'lossless_byte_copy_same_evidence'
                or manifest['originals_retained'] is not True or manifest['compression'] != 'gzip+ustar'):
            raise EvidenceArchiveError('invalid archive provenance declaration')
        if manifest['name'] is not None:
            text(manifest['name'], 'archive label', 128)
        integer(manifest['total_member_bytes'], 'total member bytes', 0, self.max_total_bytes)
        integer(manifest['total_original_bytes'], 'total original bytes', 0, self.max_total_bytes)
        if type(manifest['objects']) is not list or not 1 <= len(manifest['objects']) <= self.max_members:
            raise EvidenceArchiveError('bounded unique archive members required')
        seen, paths, total, original_total = set(), set(), 0, 0
        for item in manifest['objects']:
            exact(item, {'member', 'sha256', 'bytes', 'originals'})
            sha256(item['sha256']); integer(item['bytes'], 'member bytes', 0, self.max_member_bytes)
            if item['member'] != 'objects/' + item['sha256'] or item['member'] in seen:
                raise EvidenceArchiveError('invalid or duplicate archive member name')
            seen.add(item['member']); total += item['bytes']
            if type(item['originals']) is not list or not 1 <= len(item['originals']) <= self.max_members:
                raise EvidenceArchiveError('explicit original associations required')
            for original in item['originals']:
                exact(original, {'path', 'sha256', 'bytes'}); text(original['path'], 'original path', 8192)
                integer(original['bytes'], 'original bytes', 0, self.max_member_bytes)
                if original['sha256'] != item['sha256'] or original['bytes'] != item['bytes'] or original['path'] in paths:
                    raise EvidenceArchiveError('invalid original association')
                paths.add(original['path']); original_total += item['bytes']
        if (len(paths) > self.max_members or total > self.max_total_bytes or original_total > self.max_total_bytes
                or manifest['total_member_bytes'] != total or manifest['total_original_bytes'] != original_total):
            raise EvidenceArchiveError('archive size manifest mismatch')
        manifest_raw = encoded(manifest, _MANIFEST_MAX)
        if (registration['manifest_bytes'] != len(manifest_raw)
                or registration['manifest_sha256'] != hashlib.sha256(manifest_raw).hexdigest()):
            raise EvidenceArchiveError('archive manifest hash or size mismatch')
        return registration, manifest_raw

    def read(self, archive_id, member_sha256):
        """Verify the entire bounded container, returning one original byte string.

        A minimal regular-USTAR reader rejects links, longname/PAX extensions,
        traversal, duplicate headers and huge sizes before reading their bodies.
        No extract/extractall API is used or exposed.
        """
        sha256(member_sha256)
        registration, manifest_raw = self._registration(archive_id)
        expected = {item['member']: item for item in registration['manifest']['objects']}
        requested = 'objects/' + member_sha256
        if requested not in expected:
            raise EvidenceArchiveError('member is not registered in this archive')
        result, seen = None, set()
        try:
            with _owned_stream(self.archives / (archive_id + '.tar.gz'), self.max_archive_bytes) as (stream, size):
                actual_hash, _ = _hash_stream(stream)
                if size != registration['archive_bytes'] or actual_hash != registration['archive_sha256']:
                    raise EvidenceArchiveError('compressed archive hash or size mismatch')
                stream.seek(0)
                with gzip.GzipFile(fileobj=stream, mode='rb') as compressed:
                    while True:
                        header = compressed.read(tarfile.BLOCKSIZE)
                        if len(header) != tarfile.BLOCKSIZE:
                            raise EvidenceArchiveError('truncated tar header')
                        if header == b'\0' * tarfile.BLOCKSIZE:
                            if compressed.read(tarfile.BLOCKSIZE) != b'\0' * tarfile.BLOCKSIZE:
                                raise EvidenceArchiveError('tar end marker is incomplete')
                            remaining = 0
                            while True:
                                tail = compressed.read(_CHUNK)
                                if not tail: break
                                remaining += len(tail)
                                if remaining > tarfile.RECORDSIZE or any(tail):
                                    raise EvidenceArchiveError('unexpected or excessive tar trailing data')
                            break
                        if header[257:265] != b'ustar\x0000':
                            raise EvidenceArchiveError('ordinary USTAR header required')
                        info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
                        if (info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or info.linkname
                                or info.name in seen or len(seen) >= self.max_members + 1):
                            raise EvidenceArchiveError('only unique registered regular tar members are allowed')
                        if info.name == 'manifest.json':
                            expected_size, expected_hash = len(manifest_raw), hashlib.sha256(manifest_raw).hexdigest()
                        elif info.name in expected:
                            expected_size, expected_hash = expected[info.name]['bytes'], expected[info.name]['sha256']
                        else:
                            raise EvidenceArchiveError('unregistered or unsafe tar member path')
                        if info.size != expected_size:
                            raise EvidenceArchiveError('tar member exceeds registered byte boundary')
                        seen.add(info.name); hasher = hashlib.sha256(); chunks = []
                        remaining = info.size
                        while remaining:
                            block = compressed.read(min(_CHUNK, remaining))
                            if not block: raise EvidenceArchiveError('truncated tar member')
                            hasher.update(block); remaining -= len(block)
                            if info.name == requested: chunks.append(block)
                        if hasher.hexdigest() != expected_hash:
                            raise EvidenceArchiveError('restored member hash mismatch')
                        padding = (-info.size) % tarfile.BLOCKSIZE
                        if compressed.read(padding) != b'\0' * padding:
                            raise EvidenceArchiveError('invalid tar member padding')
                        if info.name == requested: result = b''.join(chunks)
            if seen != set(expected) | {'manifest.json'} or result is None:
                raise EvidenceArchiveError('archive lacks registered members')
            return result
        except (OSError, tarfile.TarError, EOFError, UnicodeError) as exc:
            raise EvidenceArchiveError('corrupt or unsupported archive container') from exc

    def normalized_view(self, raw):
        """Project JSON images_base64 to same-evidence blob references.

        The trusted caller must first authorize the original registered object.
        This operation neither registers a new observation nor grants visibility.
        """
        if type(raw) is not bytes or len(raw) > 16 * 1024 * 1024:
            raise EvidenceArchiveError('bounded original JSON bytes required')
        value = _decode(raw, 16 * 1024 * 1024)
        references = []
        decoded_total = 0
        def image(value):
            nonlocal decoded_total
            if type(value) is not str or len(references) >= 64:
                raise EvidenceArchiveError('bounded embedded image strings required')
            mime = 'unknown'
            if value.startswith('data:'):
                prefix, separator, value = value.partition(',')
                if not separator or prefix not in {'data:image/jpeg;base64', 'data:image/png;base64', 'data:image/webp;base64'}:
                    raise EvidenceArchiveError('only embedded recognized image data URLs are accepted')
                mime = prefix[5:].split(';')[0]
            try:
                data = base64.b64decode(value, validate=True)
            except (ValueError, UnicodeError) as exc:
                raise EvidenceArchiveError('invalid embedded base64 image') from exc
            if not data or len(data) > self.max_member_bytes:
                raise EvidenceArchiveError('embedded image byte budget exceeded')
            decoded_total += len(data)
            if decoded_total > self.max_total_bytes:
                raise EvidenceArchiveError('embedded image total byte budget exceeded')
            image_hash = hashlib.sha256(data).hexdigest()
            reference = dict(image_id='image-' + image_hash, sha256=image_hash, bytes=len(data),
                             mime_type=mime, same_evidence=True, new_visual_evidence=False)
            target = self.images / image_hash
            if target.exists() or target.is_symlink():
                if read_bytes(target, self.max_member_bytes) != data:
                    raise EvidenceArchiveError('normalized image object is corrupted')
            else:
                write_new(target, data)
            references.append(reference)
            return reference
        def visit(item):
            if type(item) is list: return [visit(child) for child in item]
            if type(item) is not dict: return item
            result = {}
            for key, child in item.items():
                if key != 'images_base64': result[key] = visit(child); continue
                if 'image_references' in item:
                    raise EvidenceArchiveError('image projection would overwrite an existing field')
                if type(child) is dict:
                    result['image_references'] = {key: image(value) for key, value in child.items()}
                elif type(child) is list:
                    result['image_references'] = [image(value) for value in child]
                else:
                    raise EvidenceArchiveError('images_base64 requires an explicit image collection')
            return result
        projected = visit(value)
        return dict(protocol=PROTOCOL + ':normalized-view', raw_json_sha256=hashlib.sha256(raw).hexdigest(),
                    raw_json_bytes=len(raw), value=projected, images=references,
                    association_kind='same_evidence_encoding_projection', new_visual_evidence=False,
                    original_retained=True)
