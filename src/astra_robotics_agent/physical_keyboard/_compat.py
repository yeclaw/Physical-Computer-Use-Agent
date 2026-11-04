"""Legacy module identity, including callers loading a file with an arbitrary spec."""
import importlib
import sys


def export_legacy(namespace, target):
    module = importlib.import_module(target)
    metadata = {'__name__', '__loader__', '__package__', '__spec__', '__file__',
                '__cached__', '__builtins__'}
    namespace.update((name, value) for name, value in vars(module).items()
                     if name not in metadata)
    # Normal imports/patches must address the actual implementation, not a copy
    # of its globals. Direct spec users retain the exported public/private API.
    if namespace['__name__'] in sys.modules:
        sys.modules[namespace['__name__']] = module
    return module
