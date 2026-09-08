"""Architecture adapters, and the one place the shipped ones are registered.

Importing this package does NOT register anything. It used to: the concrete
adapter called `register_adapter` at module scope, so whether `get_adapter`
resolved a family depended on whether some earlier import had happened to pull
that module in. The initialization cutover removed the package `__init__` that
had been doing so as a side effect, and the failure surfaced as a test that
passed or failed according to the order the suite happened to run in.

So registration is an explicit call with a single owner. A caller that needs to
resolve adapters calls `register_builtin_adapters()` first; a family defined
elsewhere calls `aadistill.initialization.specs.arch.register_adapter` itself.
"""

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER, Qwen3Adapter
from aadistill.initialization.specs.arch import register_adapter

__all__ = ["QWEN3_ADAPTER", "Qwen3Adapter", "register_builtin_adapters"]


def register_builtin_adapters() -> tuple[str, ...]:
    """Register the adapters this project ships. Idempotent; returns families.

    Idempotent because `register_adapter` returns the existing entry when the
    same family is re-registered at the same version, so calling this from
    several entry points in one process is safe. A version CHANGE still raises,
    which is the check that stops a manifest describing an adapter that is no
    longer what ran.
    """
    register_adapter(QWEN3_ADAPTER)
    return (QWEN3_ADAPTER.family,)
