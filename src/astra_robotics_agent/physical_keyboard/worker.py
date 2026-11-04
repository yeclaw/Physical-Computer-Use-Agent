"""Compatibility entry point; implementation: astra_robotics_agent.physical_keyboard.runtime.worker."""
from astra_robotics_agent.physical_keyboard._compat import export_legacy

if __name__ == '__main__':
    from importlib import import_module
    raise SystemExit(import_module('astra_robotics_agent.physical_keyboard.runtime.worker').main())
else:
    export_legacy(globals(), 'astra_robotics_agent.physical_keyboard.runtime.worker')
