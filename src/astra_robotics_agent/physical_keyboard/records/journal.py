"""Single-owner segmented index; original index bytes remain append-only.

No global record/file-size ceiling: lines are streamed and bounded separately.
Incomplete lines or segment gaps are evidence of failure, never silently erased.
"""
import os
from pathlib import Path
import re
import stat
from .storage import StorageError,parent_fd,append_bytes

class SegmentedJournal:
    def __init__(self,path,*,segment_bytes=256*1024,max_line_bytes=8*1024*1024):
        self.path=Path(path);self.parts=self.path.with_name(self.path.stem+'.segments')
        if type(segment_bytes)is not int or segment_bytes<64:raise StorageError('positive segment budget required')
        self.segment_bytes=segment_bytes;self.max_line_bytes=max_line_bytes

    def _segments(self):
        if self.parts.is_symlink():raise StorageError('symlink journal directory forbidden')
        if not self.parts.exists():return []
        fd,name=parent_fd(self.parts/'probe');os.close(fd)
        values=sorted(self.parts.iterdir())
        for i,path in enumerate(values,1):
            if path.name!='%08d.jsonl'%i or path.is_symlink() or not path.is_file():raise StorageError('invalid or missing journal segment')
        return values

    def lines(self):
        parts=self._segments()
        if parts and not self.path.exists():raise StorageError('original journal index is missing')
        paths=([self.path] if self.path.exists() or self.path.is_symlink() else [])+parts
        for path in paths:
            directory,name=parent_fd(path);fd=None
            try:
                fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode):raise StorageError('journal must be regular')
                with os.fdopen(fd,'rb',closefd=False) as stream:
                    while True:
                        line=stream.readline(self.max_line_bytes+1)
                        if not line:break
                        if len(line)>self.max_line_bytes or not line.endswith(b'\n'):raise StorageError('incomplete or oversized journal record')
                        if line.strip():yield line
                after=os.fstat(fd)
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise StorageError('journal changed while reading')
            except OSError as exc:raise StorageError('journal read failed') from exc
            finally:
                if fd is not None:os.close(fd)
                os.close(directory)

    def append(self,raw):
        if type(raw)is not bytes or not raw.endswith(b'\n') or b'\n' in raw[:-1] or len(raw)>self.max_line_bytes:
            raise StorageError('one complete bounded journal line required')
        parts=self._segments()
        if parts and not self.path.exists():raise StorageError('original journal index is missing')
        target=parts[-1] if parts else self.path
        if target.is_symlink():raise StorageError('symlink journal forbidden')
        size=target.stat().st_size if target.exists() else 0
        if size:
            directory,name=parent_fd(target);fd=None
            try:
                fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
                if not stat.S_ISREG(os.fstat(fd).st_mode):raise StorageError("journal must be regular")
                if os.pread(fd,1,size-1)!=b"\n":raise StorageError("incomplete journal tail; append forbidden")
            finally:
                if fd is not None:os.close(fd)
                os.close(directory)
        if size and size+len(raw)>self.segment_bytes:
            if not self.parts.exists():self.parts.mkdir(mode=0o700)
            target=self.parts/('%08d.jsonl'%(len(parts)+1))
        append_bytes(target,raw)
