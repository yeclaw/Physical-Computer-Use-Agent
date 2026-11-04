"""Owned regular files: bounded reads and durable append, without symlink traversal."""
import os
from pathlib import Path
import stat
from .learning_contracts import LearningContractError

class StorageError(LearningContractError):pass

def parent_fd(path):
    path=Path(path)
    if '..' in path.parts:raise StorageError('parent traversal forbidden')
    path=Path(os.path.abspath(path));fd=None
    try:
        fd=os.open(path.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        for part in path.parts[1:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=new
        return fd,path.name
    except OSError as exc:
        if fd is not None:os.close(fd)
        raise StorageError('owned directory without symlinks required') from exc


def read_bytes(path,maximum=8*1024*1024):
    directory,name=parent_fd(path);fd=None
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>maximum:raise StorageError('bounded regular file required')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(maximum+1)
        after=os.fstat(fd)
        if len(raw)>maximum or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise StorageError('file changed during read')
        return raw
    except OSError as exc:raise StorageError('cannot read owned regular file') from exc
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)


def write_new(path,raw):
    directory,name=parent_fd(path);fd=None
    try:
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        with os.fdopen(fd,'wb',closefd=False) as stream:stream.write(raw);stream.flush();os.fsync(fd)
        os.fsync(directory)
    except OSError as exc:raise StorageError('immutable new file write failed') from exc
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)


def append_bytes(path,raw):
    directory,name=parent_fd(path);fd=None
    try:
        fd=os.open(name,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise StorageError('append requires regular file')
        with os.fdopen(fd,'ab',closefd=False) as stream:stream.write(raw);stream.flush();os.fsync(fd)
    except OSError as exc:raise StorageError('journal append failed') from exc
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)
