"""Shared lazy exports for the thematic public namespaces.

The historical implementation modules remain the single source of algorithms.
Importing a namespace alone loads no numerical or optional dependency.
"""
from importlib import import_module


def resolve(namespace, exports, name):
    module = exports.get(name)
    if module is None:
        raise AttributeError(
            f"module {namespace['__name__']!r} has no attribute {name!r}"
        )
    value = getattr(import_module(f"{__package__}.{module}"), name)
    namespace[name] = value
    return value


def names(namespace, exports):
    return sorted(set(namespace) | set(exports))
