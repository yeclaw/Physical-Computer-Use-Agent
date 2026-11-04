"""Resolve frozen, bounded runtime files without executing candidate content."""

from pathlib import PurePosixPath

from astra_robotics_agent.physical_keyboard.records.contracts import digest, integer, sha256
from astra_robotics_agent.physical_keyboard.runtime.transport import ProtocolError, strict_loads


EXECUTABLE_PROTOCOL = 'physical-keyboard-executable-v1'
PYTHON_KIND = 'bounded-python-v1'
MAX_SOURCE_FILES = 16
MAX_SOURCE_BYTES = 64 * 1024


def safe_name(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value or value == '.'):
        raise ValueError('Unsafe frozen runtime path')
    return value


def _file(bundle, name, read_object):
    safe_name(name)
    try:
        entry = bundle['files'][name]
        if not isinstance(entry, dict) or set(entry) != {'sha256', 'bytes'}:
            raise ValueError('Invalid frozen file entry')
        sha256(entry['sha256'])
        integer(entry['bytes'], 'Frozen file byte count')
        content = read_object(entry['sha256'])
        if not isinstance(content, bytes) or digest(content) != entry['sha256'] or len(content) != entry['bytes']:
            raise ValueError('Frozen file integrity mismatch: ' + name)
        return content
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or invalid frozen runtime file: ' + name) from exc


def _json_object(content):
    try:
        return strict_loads(content)
    except ProtocolError as exc:
        raise ValueError('Executable metadata must be a bounded strict JSON object') from exc


def python_source_names(descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {'protocol', 'kind', 'entry', 'includes'}:
        raise ValueError('Exact execution descriptor fields required')
    if descriptor['protocol'] != EXECUTABLE_PROTOCOL or descriptor['kind'] != PYTHON_KIND:
        raise ValueError('Unsupported frozen execution descriptor')
    entry = safe_name(descriptor['entry'])
    includes = descriptor['includes']
    if not isinstance(includes, list) or len(includes) + 1 > MAX_SOURCE_FILES:
        raise ValueError('Too many frozen Python sources')
    names = [safe_name(name) for name in includes] + [entry]
    if len(set(names)) != len(names) or any(not name.endswith('.py') for name in names):
        raise ValueError('Python sources require unique .py paths')
    return names


def resolve_execution(bundle, read_object):
    """Return verified runtime material and its immutable entry fingerprint.

    Legacy actions remain opaque bytes here; their own worker owns action-plan
    syntax validation. Legacy profile bytes remain opaque for compatibility.
    ``sources`` includes the descriptor/profile needed for materialization.
    """
    if not isinstance(bundle, dict) or not isinstance(bundle.get('files'), dict):
        raise ValueError('Frozen bundle needs a files mapping')
    profile_bytes = _file(bundle, 'profile.json', read_object)
    if 'execution.json' not in bundle['files']:
        content = _file(bundle, 'actions.json', read_object)
        return dict(kind='declarative-v1', program_sha256=digest(content), file_name='actions.json',
                    sources={'actions.json': content, 'profile.json': profile_bytes}, profile=None)

    execution_bytes = _file(bundle, 'execution.json', read_object)
    descriptor = _json_object(execution_bytes)
    names = python_source_names(descriptor)
    profile = _json_object(profile_bytes)
    sources = {'execution.json': execution_bytes, 'profile.json': profile_bytes}
    total = 0
    for name in names:
        content = _file(bundle, name, read_object)
        total += len(content)
        if total > MAX_SOURCE_BYTES:
            raise ValueError('Frozen Python sources exceed 64 KiB')
        try:
            content.decode('utf-8')
        except UnicodeError as exc:
            raise ValueError('Python source must be UTF-8') from exc
        sources[name] = content
    return dict(kind=PYTHON_KIND, program_sha256=digest(execution_bytes), file_name='execution.json',
                sources=sources, profile=profile)
