"""Strict immutable JSON source reads shared by input loading and runtime evidence."""
import json
import math
import os
from pathlib import Path
import stat


MAX_SOURCE_BYTES = 32 * 1024

class ContextError(ValueError):
    """Input is invalid, mismatched, or cannot be frozen without overwriting."""

def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ContextError(f"duplicate JSON key: {key}")
        value[key] = item
    return value

def _constant(value):
    raise ContextError(f"nonfinite JSON constant: {value}")

def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ContextError("nonfinite JSON number")
    return result

def _read_source(path):
    """Open each path component without following symlinks (POSIX only)."""
    path = Path(path)
    if ".." in path.parts:
        raise ContextError("input path must not contain '..'")
    absolute = Path(os.path.abspath(path))
    descriptor = None
    directory = None
    try:
        nofollow = os.O_NOFOLLOW
        directory = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | nofollow)
        for component in absolute.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | nofollow,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(absolute.name, os.O_RDONLY | os.O_NONBLOCK | nofollow,
                             dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
            raise ContextError("context input must be a regular file of at most 32 KiB")
        chunks = []
        remaining = MAX_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > MAX_SOURCE_BYTES:
            raise ContextError("context input exceeds 32 KiB")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ContextError("context input changed while being read")
    except (OSError, AttributeError) as exc:
        raise ContextError(f"cannot read context input without following symlinks: {absolute}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=_constant, parse_float=_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContextError(f"invalid UTF-8 JSON context input: {absolute}") from exc
    return str(absolute), raw, data
