"""The one place the shipped operator implementations are registered.

Each implementation module defines its instance and registers nothing. Importing
one used to register it, so which operators `get_implementation` could resolve
depended on who had imported what first -- the same coupling the adapter
bootstrap removed, and the same failure mode: a driver resolving an operator
because of an unrelated import three modules away.

An application that needs the shipped operators calls `register_builtin_operators()`.
An operator defined elsewhere registers itself by calling
`aadistill.initialization.operators.base.register_implementation`, and never
edits this file.

`attention.activation_importance_v1` is deliberately absent. It is registered by
`attention_activation.enable()` and unregistered by `disable()`, because it is a
treatment under test rather than a shipped default, and a search that enumerates
the registry must not pick it up by accident.
"""
from __future__ import annotations

from aadistill.initialization.operators.attention import ATTENTION_WEIGHT_PROXY_V0
from aadistill.initialization.operators.base import register_implementation
from aadistill.initialization.operators.composite import (
    COMPOSITE_STAGE1_SANDWICH_V0)
from aadistill.initialization.operators.depth import (
    DEPTH_CAUSAL_KL_GREEDY_V1, DEPTH_POSITIONAL_V0)
from aadistill.initialization.operators.ffn import FFN_ACTIVATION_IMPORTANCE_V0
from aadistill.initialization.operators.width import WIDTH_GLOBAL_PCA_V0

__all__ = ["BUILTIN_OPERATORS", "register_builtin_operators"]

#: Every implementation this project ships, in no meaningful order.
BUILTIN_OPERATORS = (
    ATTENTION_WEIGHT_PROXY_V0,
    COMPOSITE_STAGE1_SANDWICH_V0,
    DEPTH_POSITIONAL_V0,
    DEPTH_CAUSAL_KL_GREEDY_V1,
    FFN_ACTIVATION_IMPORTANCE_V0,
    WIDTH_GLOBAL_PCA_V0,
)


def register_builtin_operators() -> tuple[str, ...]:
    """Register the shipped implementations. Idempotent; returns their ids.

    Idempotent because `register_implementation` returns the existing entry when
    the same id is re-registered identically, so several entry points may call
    this in one process. A CHANGED implementation under an existing id still
    raises -- that check is what stops a search manifest describing an operator
    that is no longer the one that ran.
    """
    for impl in BUILTIN_OPERATORS:
        register_implementation(impl)
    return tuple(impl.impl_id for impl in BUILTIN_OPERATORS)
