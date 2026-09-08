"""Architecture adapters, and the one place the shipped ones are registered.

Importing this package registers nothing. It used to: the concrete adapter
called `register_adapter` at module scope, so whether `get_adapter` resolved a
family depended on whether some earlier import had happened to pull that module
in. The initialization cutover removed the package `__init__` that had been
doing so as a side effect, and the failure surfaced as test files that passed or
failed according to the order the suite happened to run in.

Registration is now something a caller DOES:

    from aadistill.initialization.adapters import build_registry
    registry = build_registry()                 # its own, shared explicitly
    adapter = registry.get("qwen3")

or, for an application content to use the process default:

    register_builtin_adapters()

A family defined elsewhere ships its own `register_<family>(registry)` beside
its adapter and never edits this file.
"""

from aadistill.initialization.adapters.qwen3 import (
    QWEN3_ADAPTER, Qwen3Adapter, register_qwen3)
from aadistill.initialization.specs.arch import AdapterRegistry

__all__ = ["QWEN3_ADAPTER", "Qwen3Adapter", "register_qwen3",
           "BUILTIN_FAMILIES", "build_registry", "register_builtin_adapters"]

#: family -> its registration function. The only list of what ships.
BUILTIN_FAMILIES = {"qwen3": register_qwen3}


def build_registry(families=None) -> AdapterRegistry:
    """A NEW registry holding the named families. Nothing global is touched."""
    registry = AdapterRegistry()
    for name in (families if families is not None else BUILTIN_FAMILIES):
        if name not in BUILTIN_FAMILIES:
            raise KeyError(f"no builtin adapter for family {name!r}; "
                           f"known: {sorted(BUILTIN_FAMILIES)}")
        BUILTIN_FAMILIES[name](registry)
    return registry


def register_builtin_adapters(families=None) -> tuple[str, ...]:
    """Fill the process-default registry. Idempotent; returns the families.

    Idempotent because registering the same family at the same version returns
    the existing entry, so several entry points may call this in one process. A
    version CHANGE still raises -- that is the check which stops a manifest
    describing an adapter that is no longer what ran.
    """
    names = tuple(families if families is not None else BUILTIN_FAMILIES)
    for name in names:
        if name not in BUILTIN_FAMILIES:
            raise KeyError(f"no builtin adapter for family {name!r}; "
                           f"known: {sorted(BUILTIN_FAMILIES)}")
        BUILTIN_FAMILIES[name](None)
    return names
