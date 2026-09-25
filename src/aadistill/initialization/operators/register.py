"""The one place the shipped operator implementations are registered.

Each implementation module defines its instance and registers nothing. Importing
one used to register it, so which operators `get_implementation` could resolve
depended on import order; under randomized test ordering the same test resolved
an operator in one run and saw an empty registry in the next.

An application that needs the shipped operators calls `register_builtin_operators()`.
An operator defined elsewhere registers itself by calling
`aadistill.initialization.operators.base.register_implementation`, and never by
being imported.

**The package layout changes nothing about this.** Importing
`aadistill.initialization.operators`, `...operators.attention` or
`...operators.attention.gqa` registers nothing: every package `__init__` exposes
names and mutates no global state. That is load-bearing rather than tidy —
`BeamSearch._allowed_impl_ids` falls back to *every registered implementation*
when `SearchConfig.allowed_impls` is None, so a registration triggered by an
import would silently add a branch to an unrelated search.

`attention.activation_importance_v1` is deliberately absent from
`BUILTIN_OPERATORS`. It is registered by
`attention.gqa.activation_importance.register()` and removed again by
`unregister()`, because it is an experimental treatment whose presence in a
search is a decision its consumer makes explicitly.
"""

from __future__ import annotations

from aadistill.initialization.operators.base import register_implementation
from aadistill.initialization.operators.attention.gqa.weight_proxy import (
    ATTENTION_WEIGHT_PROXY_V0)
from aadistill.initialization.operators.composite.stage1_sandwich import (
    COMPOSITE_STAGE1_SANDWICH_V0)
from aadistill.initialization.operators.depth.causal_kl_greedy import (
    DEPTH_CAUSAL_KL_GREEDY_V1)
from aadistill.initialization.operators.depth.positional import DEPTH_POSITIONAL_V0
from aadistill.initialization.operators.ffn.dense.activation_importance import (
    FFN_ACTIVATION_IMPORTANCE_V0)
from aadistill.initialization.operators.width.residual.global_pca import (
    WIDTH_GLOBAL_PCA_V0)

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
