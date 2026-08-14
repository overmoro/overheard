"""Compatibility surface over :mod:`overheard.settings`.

The schema, the defaults and the file I/O all live in ``settings``. This module
exists so the many ``from overheard import config as cfg`` / ``cfg.get(...)``
call sites keep reading naturally, and so there is one obvious place to look
when following an old reference.

``CONFIG_DIR`` and ``CONFIG_PATH`` deliberately do *not* appear here any more.
Re-exporting them would create two names for one path, and patching the copy
would silently fail to redirect the real reads. They live in ``settings``.
"""

from overheard.settings import (  # noqa: F401  (re-exported on purpose)
    DEFAULTS,
    Settings,
    get,
    load,
    load_dict,
    save,
    set_value,
)
