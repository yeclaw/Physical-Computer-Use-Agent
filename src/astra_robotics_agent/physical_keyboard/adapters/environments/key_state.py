"""Read-only X11 keyboard state. No input injection or robot dependencies."""
import ctypes


class X11KeyState:
    def __init__(self, *, library=None):
        self._lib = library if library is not None else ctypes.CDLL('libX11.so.6')
        self._lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._lib.XOpenDisplay.restype = ctypes.c_void_p
        self._lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.XQueryKeymap.restype = ctypes.c_int
        self._lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._display = self._lib.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError('X11 keyboard state unavailable')

    def __call__(self):
        if self._display is None:
            raise RuntimeError('X11 keyboard state provider is closed')
        keys = (ctypes.c_ubyte * 32)()
        if not self._lib.XQueryKeymap(self._display, keys):
            raise RuntimeError('X11 keyboard state query failed')
        return [code for code in range(256) if keys[code // 8] & (1 << (code % 8))]

    def close(self):
        if self._display is not None:
            self._lib.XCloseDisplay(self._display)
            self._display = None
