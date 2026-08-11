"""Shared helper for deprecating public wass2s symbols.

Not part of the public API. See ``wass2s/_deprecated.py`` for the module
where deprecated classes/functions live once superseded.
"""

import warnings


def warn_deprecated(old: str, new: str, *, removed_in: str) -> None:
    """Warn that ``old`` is deprecated in favor of ``new``.

    Uses ``FutureWarning`` rather than ``DeprecationWarning`` because Python
    silences ``DeprecationWarning`` by default outside ``__main__``/tests, so
    end users calling this from a notebook or script would otherwise never
    see it.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in v{removed_in}; "
        f"use {new} instead.",
        FutureWarning,
        stacklevel=3,
    )
