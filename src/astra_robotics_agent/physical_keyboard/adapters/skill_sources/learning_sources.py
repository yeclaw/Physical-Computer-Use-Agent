"""Registered, bounded historical learning sources; no SDK or control imports.

Registration is trusted host work. Model requests name IDs and fields, never
filesystem paths. Numerical references, raw measurements and achieved effects
remain separate, and every read rechecks its complete source closure.
"""
from __future__ import annotations

import hashlib
import base64
import json
import os
from pathlib import Path
import re
import stat

from astra_robotics_agent.physical_keyboard.records.learning_contracts import (
    PROTOCOLS, LearningContractError, clone, digest, encoded, exact, identifier,
    ids, integer, sha256, scope as validate_scope,
)
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess, LearningAccessDenied


_MAX_SOURCE = 16*1024*1024
_MAX_READ = 1024*1024
_EVIDENCE_FILES = {'response.json', 'request.json', 'skill.py', 'profile.json', 'parent-diff.patch',
                   'head.jpg', 'left_wrist.jpg', 'right_wrist.jpg'}
_OBSERVATION = 'physical-agent-observation-v1'
_EXECUTION = 'physical-agent-execution-v1'


def _learning_data(value):
    forbidden = {'typing_buffer','independent_score','hidden_case_id','ground_truth','case_identity','evaluator_result'}
    if type(value) is dict:
        if forbidden & value.keys():
            raise LearningContractError('audit-only evidence is not a learning source')
        for child in value.values():_learning_data(child)
    elif type(value) is list:
        for child in value:_learning_data(child)


def _json_pointer(value, pointer):
    if type(pointer) is not str or len(pointer)>1024 or (pointer and not pointer.startswith('/')):
        raise LearningContractError('json_pointer must be a bounded RFC6901 string')
    tokens=pointer.split('/')[1:] if pointer else []
    if len(tokens)>16 or any(re.search(r'~(?![01])',token) for token in tokens):
        raise LearningContractError('json_pointer depth or escaping is invalid')
    decoded=[token.replace('~1','/').replace('~0','~') for token in tokens]
    if any(token in {'image_base64','images_base64','jpeg_base64'} for token in decoded):
        raise LearningContractError('inline image encodings require the historical image channel, not scalar JSON projection')
    for token in decoded:
        if type(value) is dict:
            if token not in value:raise LearningContractError('json_pointer member is missing')
            value=value[token]
        elif type(value) is list:
            if re.fullmatch(r'0|[1-9][0-9]*',token) is None or int(token)>=len(value):
                raise LearningContractError('json_pointer array index is invalid')
            value=value[int(token)]
        else:raise LearningContractError('json_pointer cannot descend through scalar data')
    normalized=''.join('/'+token.replace('~','~0').replace('/','~1') for token in decoded)
    return value,normalized




def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise LearningContractError('duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        encoded(value, _MAX_SOURCE)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise LearningContractError('invalid finite JSON source') from exc


def _read(path, maximum=_MAX_SOURCE):
    """Read one regular file without following any symlink component."""
    path = Path(os.path.abspath(path))
    if '..' in Path(path).parts:
        raise LearningContractError('source traversal is forbidden')
    descriptor = directory = None
    try:
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise LearningContractError('source file exceeds regular-file budget')
        chunks, remaining = [], maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        raw = b''.join(chunks)
        if len(raw) > maximum or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise LearningContractError('source changed while reading or exceeded budget')
        return raw
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise LearningContractError('cannot read source without symlinks') from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _relative(root, value):
    if type(value) is not str or '\x00' in value or not value:
        raise LearningContractError('explicit relative source path required')
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts:
        raise LearningContractError('source path escapes registered root')
    return root / relative


def _shape(value):
    if type(value) is list:
        return dict(type='array', count=len(value))
    if type(value) is dict:
        return dict(type='object', fields=sorted(value))
    return dict(type=type(value).__name__)


def _evidence_object(root, item, maximum=_MAX_SOURCE):
    """Read the registered immutable object, never a historical source path."""
    sha256(item['sha256'])
    raw = _read(Path(root) / item['sha256'], maximum)
    if _hash(raw) != item['sha256'] or len(raw) != item['bytes']:
        raise LearningContractError('evidence object SHA or size mismatch')
    return raw


def _generic_manifest(record, object_root):
    """Validate names, identity and pins of a generic Runtime evidence bundle.

    The source_path values only establish the registered sibling relationship.
    The historical filesystem is never reopened: all bytes come from the
    store's no-symlink, content-addressed immutable objects.
    """
    files = record['files']
    names = [Path(item['source_path']).name for item in files]
    if 'record.json' not in names:
        return None
    if len(set(names)) != len(names):
        raise LearningContractError('generic evidence has ambiguous filenames')
    entries = dict(zip(names, files))
    header = _json(_evidence_object(object_root, entries['record.json']))
    if type(header) is not dict or header.get('protocol') not in {_OBSERVATION, _EXECUTION}:
        return None
    required = {'protocol', 'anchor', 'adapter'}
    if header['protocol'] == _OBSERVATION:
        exact(header, required | {'state', 'content_items', 'images'})
    else:
        exact(header, required | {'label', 'source_sha256', 'profile_sha256'})
    adapter = header['adapter']
    exact(adapter, {'id', 'coordinate_frame', 'views', 'evidence_mode'})
    identifier(adapter['id'], 'adapter id')
    if type(adapter['coordinate_frame']) is not str or not adapter['coordinate_frame'].strip():
        raise LearningContractError('generic evidence must declare its command frame')
    if adapter['evidence_mode'] not in {'synthetic', 'physical'}:
        raise LearningContractError('generic evidence mode is unknown')
    views = ids(adapter['views'], 'declared views', 64)
    exact(header['anchor'], {'utc_ns', 'monotonic_ns'}, {'sampling_span_ns'})
    for name, value in header['anchor'].items():
        integer(value, 'anchor ' + name, 0, 2**127 - 1)
    original_parent = Path(entries['record.json']['source_path']).parent
    if any(Path(item['source_path']).parent != original_parent for item in files):
        raise LearningContractError('generic evidence files must be registered siblings')
    if header['protocol'] == _EXECUTION:
        if set(names) != {'record.json', 'response.json', 'skill.py', 'profile.json'}:
            raise LearningContractError('generic execution requires its exact program/profile/response bundle')
        if (sha256(header['source_sha256']) != entries['skill.py']['sha256']
                or sha256(header['profile_sha256']) != entries['profile.json']['sha256']):
            raise LearningContractError('generic execution source/profile pins differ from evidence')
        _learning_data(header)
        return header
    if type(header['state']) is not dict or type(header['content_items']) is not list:
        raise LearningContractError('generic observation requires state and content_items')
    images = header['images']
    if type(images) is not list or len(images) > 15:
        raise LearningContractError('generic observation has too many image files')
    paths = set()
    for index, image in enumerate(images):
        exact(image, {'path', 'sha256', 'view_id'})
        path = image['path']
        if (type(path) is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,96}\.(?:png|jpg)', path) is None
                or Path(path).name != path or path in paths):
            raise LearningContractError('historical image path must name one unique sibling file')
        paths.add(path)
        if path not in entries or sha256(image['sha256']) != entries[path]['sha256']:
            raise LearningContractError('historical image pin does not match the registered object')
        view = identifier(image['view_id'], 'view_id')
        if view not in views and view != 'image_%02d' % index:
            raise LearningContractError('image view is neither declared nor its explicit index fallback')
    if set(names) != {'record.json'} | paths:
        raise LearningContractError('generic observation contains unlisted files')
    _learning_data(header)
    return header


def evidence_record_kind(record, object_root):
    """Recognize eligible frozen evidence for a requested metadata page.

    Generic headers are validated; content extraction is still explicit. Old
    observation/execution envelopes remain readable, while specialized raw
    collection and trajectory diagnostic formats stay outside the public core.
    """
    names = [Path(item['source_path']).name for item in record['files']]
    if 'record.json' in names:
        header = _generic_manifest(record, object_root)
        return None if header is None else header['protocol']
    if ('response.json' in names and len(set(names)) == len(names)
            and all(name in _EVIDENCE_FILES for name in names)):
        return 'legacy-observation-or-execution'
    return None


def _historical_observation(header, entries, object_root):
    """Project image encodings to bounded references without changing evidence."""
    projected = clone(header, _MAX_SOURCE)
    media, image_index = {}, 0
    for item in projected['content_items']:
        if type(item) is not dict:
            raise LearningContractError('generic observation content item must be an object')
        if item.get('type') == 'inputText' and type(item.get('text')) is str:
            continue
        if item.get('type') != 'inputImage' or image_index >= len(header['images']):
            raise LearningContractError('generic observation has an unrecognized content item')
        image = header['images'][image_index]
        raw = _evidence_object(object_root, entries[image['path']], 4 * 1024 * 1024)
        if not (raw.startswith(b'\xff\xd8\xff') or raw.startswith(b'\x89PNG\r\n\x1a\n')):
            raise LearningContractError('registered historical image is not JPEG or PNG')
        url = item.get('imageUrl')
        if type(url) is not str or ',' not in url:
            raise LearningContractError('generic observation image is missing its encoded source')
        prefix, payload = url.split(',', 1)
        if prefix not in {'data:image/jpeg;base64', 'data:image/png;base64'} or len(payload) > 6*1024*1024:
            raise LearningContractError('generic observation image encoding is unsupported or too large')
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (TypeError, ValueError) as error:
            raise LearningContractError('generic observation image encoding is invalid') from error
        if _hash(decoded) != image['sha256'] or decoded != raw:
            raise LearningContractError('encoded observation pixels differ from the registered image')
        camera = image['view_id']
        item.clear()
        item.update(type='inputImage', historical_image_index=image_index,
                    view_id=camera, sha256=image['sha256'], encoding_omitted=True)
        media[image_index] = dict(root=Path(object_root), colors={camera+'_camera': image['sha256']},
                                  hashes={camera: image['sha256']})
        image_index += 1
    if image_index != len(header['images']):
        raise LearningContractError('generic observation image manifest/content count differs')
    return projected, media


class LearningSources:
    def __init__(self, scope, *, access=None, output=None):
        self.scope = validate_scope(scope)
        self.access = LearningAccess(self.scope) if access is None else access
        if self.access.scope != self.scope:
            raise LearningAccessDenied('source registry scope mismatch')
        self.output = Path(output) if output is not None else None
        self._sources = {}
        self._media = {}

    def register(self, source_id, *, kind, data, source_ids=None, metadata=None):
        """Register immutable trusted host JSON; never call with model paths."""
        identifier(source_id, 'source_id')
        if kind not in {'human_demo', 'robot_attempt', 'device_reference'}:
            raise LearningContractError('unsupported source kind')
        sources = ids([source_id] if source_ids is None else source_ids, 'source_ids', 4096)
        self.access.check_sources(sources, require_provenance=True)
        value = dict(protocol=PROTOCOLS['source'], source_id=source_id, source_ids=sources,
                     kind=kind, scope=self.scope, data=clone(data, _MAX_SOURCE),
                     metadata=clone({} if metadata is None else metadata, _MAX_READ))
        value['sha256'] = digest(value)
        if source_id in self._sources and self._sources[source_id] != value:
            raise LearningContractError('registered source is immutable')
        self._sources[source_id] = value
        return self.describe(source_id)

    def register_references(self, document):
        """Publish curated host inputs, not old knowledge records or their graph.

        This is an explicit trusted input boundary, not a semantic scrubber.
        The preparer must audit the complete data/metadata before publication.
        Each reference has its own allowlisted origin; it cannot declare an
        unrelated origin to launder a hidden historical package.
        """
        exact(document, {'protocol', 'scope', 'items'})
        if (document['protocol'] != PROTOCOLS['references']
                or validate_scope(document['scope']) != self.scope):
            raise LearningContractError('reference scope or protocol mismatch')
        items = document['items']
        if type(items) is not list or len(items) > 64:
            raise LearningContractError('bounded explicit reference list required')
        seen = set()
        for item in items:
            exact(item, {'source_id', 'data', 'data_sha256'}, {'metadata'})
            key = identifier(item['source_id'])
            if key in seen:
                raise LearningContractError('duplicate reference source')
            seen.add(key)
            self.access.check_sources([key], require_provenance=True)
            if sha256(item['data_sha256']) != digest(item['data']):
                raise LearningContractError('reference data hash mismatch')
            clone(item.get('metadata', {}), _MAX_READ)
        previous = dict(self._sources)
        try:
            return [self.register(item['source_id'], kind='device_reference',
                                 data=item['data'], metadata=item.get('metadata')) for item in items]
        except Exception:
            self._sources = previous
            raise

    def _get(self, source_id):
        identifier(source_id, 'source_id')
        if source_id not in self._sources:
            raise LearningContractError('unknown registered learning source')
        value = self._sources[source_id]
        self.access.check_sources(value['source_ids'], require_provenance=True)
        unsigned = {k: v for k, v in value.items() if k != 'sha256'}
        if digest(unsigned) != value['sha256']:
            raise LearningContractError('registered source hash mismatch')
        return value

    def describe(self, source_id=None):
        if source_id is None:
            return dict(protocol=PROTOCOLS['source'], scope=clone(self.scope),
                        capabilities=['list_learning_sources', 'inspect_learning_sources', 'analyze_learning_sources'],
                        default_view='brief', max_page_samples=1024, max_images=2,
                        json_pointer=dict(protocol='RFC6901',max_chars=1024,max_depth=16,selection_before_paging=True),
                        input_kinds=['host_registered_JSON', 'generic_observation', 'generic_execution'],
                        meaning='historical candidate evidence; no current observation or motion authority')
        value = self._get(source_id)
        return dict(protocol=PROTOCOLS['source'], source_id=source_id, source_ids=list(value['source_ids']),
                    scope=clone(self.scope), kind=value['kind'], sha256=value['sha256'],
                    fields={key: _shape(item) for key, item in value['data'].items()} if type(value['data']) is dict else _shape(value['data']),
                    metadata=clone(value['metadata']))

    def list_sources(self, kind='all', limit=8, offset=0):
        if kind not in {'all', 'human_demo', 'robot_attempt', 'device_reference'}:
            raise LearningContractError('invalid source kind')
        integer(limit, 'limit', 1, 64)
        integer(offset, 'offset', 0, 1000000)
        visible = []
        for source_id, value in self._sources.items():
            if kind not in ('all', value['kind']):
                continue
            try:
                self._get(source_id)
            except LearningAccessDenied:
                continue
            visible.append(source_id)
        end = min(len(visible), offset+limit)
        return dict(protocol=PROTOCOLS['source'], sources=[self.describe(i) for i in visible[offset:end]],
                    total=len(visible), offset=offset, next_offset=end if end<len(visible) else None)



    def register_evidence(self, store, evidence_id):
        """Open a registered generic/legacy observation or execution only.

        This does not read current device state or grant execution authority.
        The immutable object closure and manifest pins are rechecked on every
        registration. Historical source paths are never reopened.
        """
        identifier(evidence_id, 'evidence_id')
        record = store._evidence(evidence_id)
        if record.get('scope') != self.scope:
            raise LearningAccessDenied('evidence scope mismatch')
        if 'source_ids' in record:
            sources = self.access.check_sources(record['source_ids'], require_provenance=True)
            self.access.bind_evidence(evidence_id, sources)
        sources = self.access.evidence_sources(evidence_id)
        object_root = store.namespace / 'objects'
        kind = evidence_record_kind(record, object_root)
        if kind is None:
            raise LearningContractError('only registered observation/execution bundles may be learning sources')
        entries = {Path(item['source_path']).name: item for item in record['files']}
        frozen, media = {}, {}
        generic = kind in {_OBSERVATION, _EXECUTION}
        for index, item in enumerate(record['files']):
            name = Path(item['source_path']).name
            raw = _evidence_object(object_root, item)
            if name.endswith(('.jpg', '.png')):
                if not generic:
                    camera = name.rsplit('.', 1)[0]
                    media[index] = dict(root=object_root, colors={camera+'_camera': item['sha256']},
                                        hashes={camera: item['sha256']})
                frozen[name] = dict(sha256=item['sha256'], bytes=len(raw), historical=True)
            elif name.endswith('.json'):
                frozen[name] = _json(raw)
            else:
                try:
                    frozen[name] = raw.decode('utf-8')
                except UnicodeError as error:
                    raise LearningContractError('registered program must be UTF-8') from error
        if kind == _OBSERVATION:
            frozen['record.json'], media = _historical_observation(
                frozen['record.json'], entries, object_root)
            association = 'observation_snapshot'
        elif kind == _EXECUTION:
            response = frozen['response.json']
            if (type(response) is not dict or response.get('status') not in {'ok', 'blocked', 'error'}
                    or type(response.get('result')) is not dict):
                raise LearningContractError('generic execution requires a Runtime response envelope')
            association = 'execution_boundaries'
        else:
            response = frozen['response.json']
            observation = type(response) is dict and 'state' in response and 'snapshot' in response
            execution = (type(response) is dict and type(response.get('response')) is dict
                         and {'request.json', 'skill.py', 'profile.json'} <= frozen.keys())
            if not (observation or execution):
                raise LearningContractError('unrecognized legacy observation/execution evidence')
            association = 'execution_boundaries' if execution else 'observation_snapshot'
        _learning_data(frozen)
        source_id = 'evidence:' + evidence_id
        image_refs = [dict(index=index, views=sorted(name[:-7] for name in value['colors']))
                      for index, value in sorted(media.items())]
        result = self.register(source_id, kind='robot_attempt', data=dict(files=frozen),
            source_ids=sources or [source_id], metadata=dict(evidence_id=evidence_id,
                run_id=record['run_id'], evidence_sha256=record['sha256'], historical=True,
                association_kind=association, current_state=False, contact='unknown',
                achieved_text='unknown', historical_images=image_refs))
        self._media[source_id] = media
        return result


    def inspect(self, source_id, request=None):
        value = self._get(source_id)
        result = dict(protocol=PROTOCOLS['source'], source_id=source_id, source_ids=list(value['source_ids']),
                      scope=clone(self.scope), sha256=value['sha256'], data={},
                      provenance=dict(clone(value['metadata']), historical=True, current_observation_unchanged=True))
        if not request:
            result['data'] = self.describe(source_id)['fields']
            result['provenance']['view'] = 'brief'
            return result
        exact(request, set(), {'fields', 'offset', 'limit', 'image_indices', 'camera','json_pointer'})
        if 'camera' in request:
            identifier(request['camera'], 'historical view')
        offset = integer(request.get('offset', 0), 'offset', 0, 1000000)
        limit = integer(request.get('limit', 32), 'limit', 1, 1024)
        data = value['data']
        if 'json_pointer' in request:
            selected,pointer=_json_pointer(data,request['json_pointer'])
            result['provenance'].update(json_pointer=pointer,selection_sha256=digest(selected))
            data=(selected if type(selected) is dict else {'items':selected} if type(selected) is list
                  else {'value':selected})
        if type(data) is not dict:
            raise LearningContractError('source projection requires object data')
        fields = request.get('fields', list(data))
        if (type(fields) is not list or not fields or len(fields)>32
                or any(type(field) is not str for field in fields) or len(set(fields))!=len(fields)):
            raise LearningContractError('unique explicit fields required')
        # For a samples object, selecting sample column names projects rows.
        rows = data.get('samples')
        columns = rows is not None and fields != ['samples'] and all(field not in data for field in fields)
        if columns:
            if any(field not in (rows[0] if rows else {}) for field in fields):
                raise LearningContractError('unknown sample field')
            end = min(len(rows), offset+limit)
            result['data']['samples'] = [{field: clone(row[field]) for field in fields} for row in rows[offset:end]]
            result['provenance'].update(offset=offset, total=len(rows), next_offset=end if end<len(rows) else None)
        else:
            if any(field not in data for field in fields):
                raise LearningContractError('unknown source field')
            for field in fields:
                item = data[field]
                if type(item) is list:
                    if offset>len(item): raise LearningContractError('offset exceeds source total')
                    end = min(len(item), offset+limit)
                    result['data'][field] = clone(item[offset:end], _MAX_READ)
                    result['provenance'].update(offset=offset, total=len(item), next_offset=end if end<len(item) else None)
                elif type(item) is dict and field == 'files':
                    names = sorted(item)
                    end = min(len(names), offset+limit)
                    result['data'][field] = {name: clone(item[name], _MAX_READ) for name in names[offset:end]}
                    result['provenance'].update(offset=offset, total=len(names), next_offset=end if end<len(names) else None)
                else:
                    result['data'][field] = clone(item, _MAX_READ)
        if 'image_indices' in request:
            indices = request['image_indices']
            if (type(indices) is not list or len(indices)>2 or any(type(index) is not int or index<0 for index in indices)
                    or len(set(indices))!=len(indices)):
                raise LearningContractError('at most two unique historical image indices')
            camera = request.get('camera')
            result['images'] = [self._image(source_id, integer(index, 'image_index', 0, 1000000), camera) for index in indices]
        encoded(result, _MAX_READ)
        result['view_sha256'] = digest(result)
        return result

    def _image(self, source_id, index, camera):
        media = self._media.get(source_id, {}).get(index)
        if camera is None and media is not None and media['colors']:
            camera = next(iter(media['colors']))[:-7]
        base = dict(index=index, camera=camera, historical=True, association_kind='same_registered_evidence_image',
                    acquisition_clock='unknown', current_observation_unchanged=True)
        if media is None or camera+'_camera' not in media['colors']:
            return dict(base, status='unknown', reason='image_mapping_unavailable')
        path = _relative(media['root'], media['colors'][camera+'_camera'])
        try:
            raw = _read(path, 4*1024*1024)
        except FileNotFoundError:
            return dict(base, status='unknown', reason='image_missing')
        if not (raw.startswith(b'\xff\xd8\xff') or raw.startswith(b'\x89PNG\r\n\x1a\n')):
            raise LearningContractError('registered image is not JPEG/PNG')
        pin = _hash(raw)
        previous = media.setdefault('hashes', {}).get(camera)
        if previous is not None and previous != pin:
            raise LearningContractError('historical image SHA changed')
        media['hashes'][camera] = pin
        return dict(base, status='available', path=str(path), sha256=pin, bytes=len(raw))

    def analysis_data(self, source_id, fields=None, offset=0, limit=1024, *, json_pointer=None):
        """Host-only convenience: exact finite projection, no image bytes."""
        request = dict(offset=offset, limit=limit)
        if fields is not None:
            request['fields'] = fields
        if json_pointer is not None:
            request['json_pointer']=json_pointer
        return self.inspect(source_id, request)['data']
