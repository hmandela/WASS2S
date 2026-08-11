"""Deprecated wass2s symbols, kept for backward compatibility.

Not part of the public API in the sense of new development: nothing here
should gain features or be relied on going forward. Each symbol is a thin
wrapper around its replacement so bug fixes to the replacement apply here
too, and each warns via ``FutureWarning`` on use so callers can migrate
before removal.
"""

from wass2s._lifecycle import warn_deprecated
from wass2s.was_analog import WAS_Analog

__all__ = ["WAS_Analog__"]


class WAS_Analog__(WAS_Analog):
    """Deprecated alias for :class:`wass2s.was_analog.WAS_Analog`.

    .. deprecated:: 0.5.0
        Use :class:`wass2s.was_analog.WAS_Analog` instead. Will be removed
        in v0.6.0.
    """

    def __init__(self, *args, **kwargs):
        warn_deprecated("WAS_Analog__", "WAS_Analog", removed_in="0.6.0")
        super().__init__(*args, **kwargs)
