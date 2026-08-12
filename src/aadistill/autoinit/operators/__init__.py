"""Operator kinds and implementations.

Importing this package registers the v1 library. An implementation defined
elsewhere joins by calling ``register_implementation`` (and ``register_kind``
first, if it introduces a new structural dimension) — no edit here, and none to
the search engine.
"""

from .attention import ATTENTION_WEIGHT_PROXY_V0
from .base import (
    ATTENTION,
    DEPTH,
    FFN,
    RESIDUAL_WIDTH,
    CalibrationNeed,
    ContractViolation,
    OperatorContext,
    OperatorError,
    OperatorImplementation,
    OperatorKindSpec,
    OperatorOutcome,
    OperatorPlan,
    applicable_implementations,
    get_implementation,
    get_kind,
    implementations_for_kind,
    register_implementation,
    register_kind,
    registered_implementations,
    registered_kinds,
    registry_ledger,
    rejected_implementations,
    verify_ledger,
    write_ledger,
)
from .composite import COMPOSITE_STAGE1, COMPOSITE_STAGE1_SANDWICH_V0
from .depth import DEPTH_CAUSAL_KL_GREEDY_V1, DEPTH_POSITIONAL_V0
from .ffn import FFN_ACTIVATION_IMPORTANCE_V0
from .width import WIDTH_GLOBAL_PCA_V0

#: The v1 search library. Deliberately small: the first question is whether
#: conditional operator order plus calibration choice plus beam selection works
#: at all, not which FFN algorithm is best.
V1_IMPLEMENTATIONS = (
    DEPTH_POSITIONAL_V0,
    DEPTH_CAUSAL_KL_GREEDY_V1,
    WIDTH_GLOBAL_PCA_V0,
    FFN_ACTIVATION_IMPORTANCE_V0,
    ATTENTION_WEIGHT_PROXY_V0,
    COMPOSITE_STAGE1_SANDWICH_V0,
)

#: The four structural kinds a decomposed path composes. `COMPOSITE_STAGE1` is
#: excluded: it reaches the target in one step and does not compose with them.
V1_DECOMPOSED_KINDS = ("DEPTH", "RESIDUAL_WIDTH", "FFN", "ATTENTION")

__all__ = [
    "ATTENTION", "ATTENTION_WEIGHT_PROXY_V0", "COMPOSITE_STAGE1",
    "COMPOSITE_STAGE1_SANDWICH_V0", "CalibrationNeed", "ContractViolation", "DEPTH",
    "DEPTH_CAUSAL_KL_GREEDY_V1", "DEPTH_POSITIONAL_V0", "FFN",
    "FFN_ACTIVATION_IMPORTANCE_V0", "OperatorContext", "OperatorError",
    "OperatorImplementation", "OperatorKindSpec", "OperatorOutcome", "OperatorPlan",
    "RESIDUAL_WIDTH", "V1_DECOMPOSED_KINDS", "V1_IMPLEMENTATIONS",
    "WIDTH_GLOBAL_PCA_V0", "applicable_implementations", "get_implementation",
    "get_kind", "implementations_for_kind", "register_implementation", "register_kind",
    "registered_implementations", "registered_kinds", "registry_ledger",
    "rejected_implementations", "verify_ledger", "write_ledger",
]
