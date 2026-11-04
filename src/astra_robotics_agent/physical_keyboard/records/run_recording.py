"""Run evidence records and complete Python source snapshots.

This module only records supplied data. Importing it does not inspect files,
connect to services, initialize a robot, or claim a task outcome.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class RunRecorder:
    """Append the learning loop's existing event schema without reinterpretation.

    The caller creates and owns the output directory. This preserves the old
    loop's append-only JSONL format; it is not a multi-writer transaction log.
    """

    def __init__(self, output, *, serializer=None, clock_ns=None):
        self.output = Path(output)
        self.serializer = _canonical if serializer is None else serializer
        self.clock_ns = time.time_ns if clock_ns is None else clock_ns

    def event(self, kind, data):
        record = dict(event=kind, utc_ns=self.clock_ns(), data=data)
        # Serialize first: invalid values must not create a partial record.
        line = self.serializer(record)
        if not isinstance(line, str):
            raise TypeError('Event serializer must return text')
        with (self.output / 'learning-events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')


def _python_files(root):
    """Preflight the whole tree, including links that are not Python files."""
    files = []
    for directory, dirs, names in os.walk(str(root), followlinks=False):
        dirs.sort()
        names.sort()
        for name in dirs + names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError('Source snapshot refuses symbolic links: ' + str(path))
            if path.suffix == '.py':
                if not stat.S_ISREG(mode):
                    raise ValueError('Python source must be a regular file: ' + str(path))
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _reject_link_components(path, label):
    # Inspect the supplied path before resolve() would erase linked parents.
    # Callers can supply a canonical path when an OS directory alias is intended.
    path = path.absolute()
    for component in (path,) + tuple(path.parents):
        if component.is_symlink():
            raise ValueError(label + ' must not contain symbolic links: ' + str(component))


def freeze_python_tree(package_root, destination):
    """Freeze every ``*.py`` under a package with relative paths and SHA-256.

    The destination must not exist and must be outside the source tree. Source
    links and linked source/destination path components are rejected. A
    failed copy remains visibly partial, without a completed source manifest;
    this function never deletes source data or replaces a previous snapshot.
    """
    source = Path(package_root)
    target = Path(destination)
    _reject_link_components(source, 'Source root')
    _reject_link_components(target, 'Destination')
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError('Source root must be a directory')
    target = target.resolve()
    try:
        target.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError('Destination must be outside the source tree')
    if target.exists():
        raise FileExistsError(target)
    paths = _python_files(source)
    manifest = dict(protocol='physical-keyboard-source-tree-v1',
                    source_root=str(source), files={})
    target.mkdir(parents=True, exist_ok=False)
    for path in paths:
        # Repeat the type check after preflight; never knowingly copy a link.
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Source changed type during snapshot: ' + str(path))
        raw = path.read_bytes()
        relative = path.relative_to(source).as_posix()
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(raw)
        manifest['files'][relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with (target / 'source-manifest.json').open('x', encoding='utf-8') as stream:
        stream.write(_canonical(manifest) + '\n')
    return manifest
