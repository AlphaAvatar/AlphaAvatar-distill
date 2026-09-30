"""How an operator runs, as distinct from what it computes.

`OperatorStep.identity()` feeds `compute_state_id`, and it reads `impl_id`, the
implementation signature, the profile hash, the operator's `config_hash` and the
seed. Everything reachable from that tuple is a claim about *what the science
is*. A knob that only decides how many calibration items share a forward pass is
not such a claim: the estimand, the aggregation and the selection are identical
at any batch size, so two runs that differ only there must be the same
scientific state.

The tempting shortcut is to put the knob in `operator_config` and exclude its
key when hashing. That works exactly once. The second such knob makes the
exclusion list the real definition of "scientific", spelled as a negative, in a
module that has no reason to enumerate execution concerns — and a knob someone
forgets to exclude silently forks every state id downstream.

So the boundary is positive and structural instead:

    OperatorContext.config      WHAT is computed   -> hashed into the state id
    OperatorContext.execution   HOW it is computed -> never hashed, always traced

`Deadline` already lived on the second side of that line ("Runtime only. Never
hashed"); this gives it a named home rather than a comment. Adding a future
execution knob means a field here and nothing at all in the hashing path, which
is the property the exclusion list could not offer.

Execution settings are still **evidence**: an operator records the batch size it
actually used in its `trace`, and `trace` is deliberately not part of
`OperatorStep.identity()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aadistill.initialization.calibration.packing import (
    ORIGINAL_ORDER_V1, PACKING_POLICIES,
)

#: The default when no caller states one. Small because calibration items are
#: long and the accumulators are large, and because a value that fits the
#: smallest accelerator this project rents is worth more than one that has to be
#: lowered on arrival. `1` reproduces one-item-per-forward execution exactly and
#: is the reference path.
DEFAULT_MICRO_BATCH_SIZE = 4


class ExecutionError(ValueError):
    """An execution setting is not usable."""


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime knobs. Never enters a hash, a state id or a manifest identity.

    Frozen, so an operator cannot reach back and change how a later operator in
    the same path executes.
    """

    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE

    #: WHICH items share a forward, as a named deterministic policy. The second
    #: knob this module's docstring promised: a field here and nothing at all in
    #: the hashing path.
    #:
    #: It is execution and not science for the same reason the batch size is —
    #: the estimand, the aggregation and the selection are identical under any
    #: grouping. What it is NOT is numerically free: a different grouping pads
    #: to different widths, and this project has measured bf16 GEMMs reducing
    #: shape-dependently, so a packing change can move a selection even though
    #: it cannot move a definition. That makes it an execution knob whose
    #: equivalence is an empirical question, not an assumed one.
    #:
    #: An operator whose RESULT IS DEFINED BY batch composition — a causal
    #: scorer, whose score is computed over a batched forward — must still carry
    #: the policy in its hashed config instead. `packing.py` names the policies
    #: and deliberately does not decide who hashes them.
    calibration_batch_packing: str = ORIGINAL_ORDER_V1

    def __post_init__(self) -> None:
        if self.calibration_batch_packing not in PACKING_POLICIES:
            raise ExecutionError(
                f"unknown calibration_batch_packing "
                f"{self.calibration_batch_packing!r}; known: "
                f"{list(PACKING_POLICIES)}")
        size = self.micro_batch_size
        if isinstance(size, bool) or not isinstance(size, int):
            raise ExecutionError(
                f"micro_batch_size must be an int, not {type(size).__name__}")
        if size < 1:
            raise ExecutionError(
                f"micro_batch_size must be >= 1, got {size}; 1 is the "
                "one-item-per-forward reference path")

    def as_trace(self) -> dict[str, Any]:
        """What an operator records about how it ran. Evidence, not identity."""
        return {"micro_batch_size": int(self.micro_batch_size),
                "calibration_batch_packing": str(self.calibration_batch_packing)}


#: The value every caller gets unless it says otherwise.
DEFAULT_EXECUTION = ExecutionConfig()
