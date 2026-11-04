"""Single-operation verified knowledge reads, never a runtime state cache.

Only already-validated historical objects/indices and complete source closures
are memoized. No bytes or decisions survive the outer read boundary. Touched
file identities and authorization are checked again before results leave it.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import stat


_SCOPES = ContextVar('physical_keyboard_knowledge_read_scopes', default=())


def _authorization(access):
    if access is None:
        return None
    return (dict(access.scope), access.restricted, access.allow_own_experience,
            frozenset(access.allowed_sources), frozenset(access.allowed_knowledge),
            frozenset(access.allowed_evidence))


class ReadSnapshot:
    def __init__(self, store, error):
        self.namespace = str(store.namespace)
        self.scope = dict(store._scope)
        self.access = store.learning_access
        self.authorization = _authorization(self.access)
        self.stores = {id(store): store}
        self.error = error
        self.memo, self.active, self.files, self.ancestors = {}, set(), {}, {}
        self.stats = {}

    def check(self, store=None):
        if store is not None and (str(store.namespace) != self.namespace or store._scope != self.scope
                or store.learning_access is not self.access):
            raise self.error('knowledge read snapshot authority changed')
        if _authorization(self.access) != self.authorization:
            raise self.error('knowledge source policy changed during read snapshot')

    def cached(self, category, key, loader):
        self.check()
        full_key = (category, key)
        if full_key in self.memo:
            self.stats[category + '_hits'] = self.stats.get(category + '_hits', 0) + 1
            return self.memo[full_key]
        if full_key in self.active:
            raise self.error('cyclic knowledge read dependency')
        self.stats[category + '_misses'] = self.stats.get(category + '_misses', 0) + 1
        self.active.add(full_key)
        try:
            result = loader()
        finally:
            self.active.remove(full_key)
        self.memo[full_key] = result
        return result

    def watch(self, path, *, missing=False):
        path = Path(os.path.abspath(path))
        if '..' in Path(path).parts:
            raise self.error('parent traversal is forbidden in a knowledge read')
        current = Path(path.anchor)
        for part in path.parts[1:-1]:
            current = current / part
            try:
                status = current.lstat()
            except FileNotFoundError:
                if missing:
                    break
                raise
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise self.error('knowledge read ancestor is not a real directory')
            identity = (status.st_dev, status.st_ino, status.st_mode)
            previous = self.ancestors.setdefault(str(current), identity)
            if previous != identity:
                raise self.error('knowledge read ancestor changed')
        try:
            status = path.lstat()
        except FileNotFoundError:
            if not missing:
                raise
            signature = None
        else:
            if stat.S_ISLNK(status.st_mode) or not (stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)):
                raise self.error('knowledge read requires a regular file or directory')
            signature = (status.st_dev, status.st_ino, status.st_mode, status.st_size,
                         status.st_mtime_ns, status.st_ctime_ns)
            if stat.S_ISDIR(status.st_mode):
                # Linux can create another entry within one coarse clock tick
                # without changing the directory's allocated size. Membership
                # is part of the snapshot, independently of stat timestamps.
                signature += (tuple(sorted(os.listdir(path))),)
        key = str(path)
        if key in self.files and self.files[key] != signature:
            raise self.error('knowledge file changed during read snapshot')
        self.files[key] = signature

    def watch_journal(self, path):
        path = Path(path)
        self.watch(path, missing=True)
        parts = path.with_name(path.stem + '.segments')
        self.watch(parts, missing=True)
        if parts.exists():
            for segment in parts.iterdir():
                self.watch(segment)

    def finish(self):
        self.check()
        # Recheck permissions, symlinks, inode/size/time and directory membership
        # before the enclosing read operation can return its result.
        for name in list(self.files):
            self.watch(name, missing=True)
        for name, expected in self.ancestors.items():
            status = Path(name).lstat()
            if (status.st_dev, status.st_ino, status.st_mode) != expected:
                raise self.error('knowledge read ancestor changed before publication')


def active_snapshot(store=None):
    for snapshot in reversed(_SCOPES.get()):
        if store is None or (hasattr(store, 'namespace') and str(store.namespace) == snapshot.namespace
                and store._scope == snapshot.scope and store.learning_access is snapshot.access):
            return snapshot
    return None


def assert_writable(store):
    for snapshot in reversed(_SCOPES.get()):
        if hasattr(store, 'namespace') and str(store.namespace) == snapshot.namespace:
            raise snapshot.error('writes are forbidden inside a knowledge read snapshot')


def assert_policy_writable(store):
    # A fresh copy may install a narrower independent policy before its first
    # read. It receives a separate snapshot/cache. Existing participant objects
    # cannot change the authority to which their cached results were bound.
    for snapshot in reversed(_SCOPES.get()):
        if snapshot.stores.get(id(store)) is store:
            raise snapshot.error('writes are forbidden inside a knowledge read snapshot')


@contextmanager
def read_snapshot(store, error):
    current = active_snapshot(store)
    owner = current is None
    snapshot = ReadSnapshot(store, error) if owner else current
    snapshot.check(store)
    snapshot.stores[id(store)] = store
    token = _SCOPES.set(_SCOPES.get() + (snapshot,))
    try:
        # Pin index and registration membership too, including absent paths.
        if owner:
            snapshot.watch_journal(store.namespace / 'index.jsonl')
            snapshot.watch_journal(store.namespace / 'development/index.jsonl')
            for suffix in ('learning/changes/commits', 'learning/changes/intents',
                           'learning/changes/publications', 'learning/resources/registrations'):
                snapshot.watch(store.namespace / suffix, missing=True)
        yield snapshot
        if owner:
            snapshot.finish()
        else:
            snapshot.check(store)
    finally:
        _SCOPES.reset(token)
        if owner:
            snapshot.memo.clear()
            snapshot.active.clear()


def read_operation(function):
    """Automatic boundary for real-store reads; overlays retain their semantics."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        store = getattr(self, 'store', self)
        context = getattr(store, 'read_snapshot', None)
        if not callable(context):
            return function(self, *args, **kwargs)
        with context():
            return function(self, *args, **kwargs)
    return wrapped
