"""The hashed operator config, and the hash. ONE owner, two callers.

`config_hash` is what forks a search's whole subtree: two operators that reduce
over different vocabulary partitions, or protect different scoring positions,
compute different things and must not share a state id. So the dict this builds
is scientific identity, not bookkeeping.

WHY IT IS HERE RATHER THAN IN THE BEAM. `BeamSearch` built it from two private
methods, and `materialize_fixed_path` -- which exists to REPLAY a path the beam
produced -- built its own:

    operator_config = {"n_calibration_items": n, **step.config}

with no position policy and no distribution support. That is not a smaller
config; it is a DIFFERENT one. A replay therefore ran its operators under the
full-vocabulary default and the incumbent position policy whatever the run it
replayed had declared, and the executor's own agreement check could not see it:
`apply_checked` compares the config's support declaration against the context's
support OBJECT, and with neither set the two consistently agreed on "full
vocabulary" -- with each other, and with nothing else.

The only symptom was a digest mismatch, reported after the operator had run. A
replay that reconstructs its inputs differently from the run it replays is not
a replay, and a mismatch it produces is evidence about the harness rather than
about the artifact. There is now one function, both callers use it, and a
divergence is caught by `FixedPathStep.expected_config_hash` before any
operator starts rather than by a digest comparison after one finishes.

WHAT IS OMITTED, AND WHY OMISSION IS PART OF THE CONTRACT. Both contributors
are absent under their incumbent setting -- `{}` at the full vocabulary, `{}`
at the incumbent position policy or for an operator that consumes no
calibration data. Those omissions preserve recorded identities rather than
tidying them away: committed states hashed configs without those keys, and
emitting them now, even as explicit defaults, would move every one of those
hashes and the `measurement_protocol_id`s beside them.

The timings, hashes and costs behind all of this belong to the runs that
produced them; see `docs/maintenance/core-provenance.md`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aadistill.infrastructure.manifest import sha256_json
from aadistill.initialization.calibration.profiles import consumes_calibration
from aadistill.initialization.scoring.content import scoring_content_config
from aadistill.initialization.scoring.positions import policy_config

#: Keys the EXECUTOR owns and a caller may not declare. `n_calibration_items`
#: describes the run rather than operator policy, and a step that could
#: override it would plan as if there were 8 items while executing against 67.
EXECUTOR_DERIVED_CONFIG_KEYS = frozenset({"n_calibration_items"})


def position_policy_config(implementation: Any, policy: Any,
                           items: Sequence[Any]) -> dict[str, Any]:
    """The scoring-position policy AND its content, as hashed config.

    The policy id alone is not enough, and that is not a refinement. A policy
    that reads positions consumes metadata no other identity covers:
    `profile_hash` pins the profile's spec, which pins one `content_sha256`,
    which hashes only item ids and token ids. Two assets with identical tokens
    and different supervised masks therefore agreed on every term above --
    including the policy hash -- while producing different operator decisions.
    `scoring_content_config` binds what the policy actually reads.

    `{}` for an implementation that consumes no calibration data: a weight-only
    operator has no mechanism by which a position policy could change its
    output, and branching it would manufacture byte-identical states.
    """
    #: `None` means NO DECLARATION, which under the historical contract is the
    #: incumbent -- and `policy_config(ALL_POSITIONS_V1)` is `{}` too, so the
    #: two reach the same config. Handled here rather than pushed onto callers
    #: because a fixed path legitimately declares no policy and
    #: `policy_config(None)` raises on `.policy_hash`.
    if policy is None or not consumes_calibration(implementation):
        return {}
    named = policy_config(policy)
    if not named:
        return {}
    return {**named, **scoring_content_config(items, policy)}


def distribution_support_config(support: Any) -> dict[str, Any]:
    """The vocabulary partition as hashed config, or `{}` at full vocabulary.

    Not restricted to calibrated operators, unlike the position policy: a
    support reaches an operator through `OperatorContext` whether or not that
    operator consumes calibration items, and an operator that ignores it is
    free to. What must not happen is a declaration that disagrees with the
    object, which `OperatorImplementation.execute` refuses.
    """
    if support is None or support.is_full_vocab:
        return {}
    return {"distribution_support": support.as_dict()}


def hashed_operator_config(*, implementation: Any, policy: Any,
                           support: Any, items: Sequence[Any],
                           declared: Mapping[str, Any] | None = None
                           ) -> dict[str, Any]:
    """Everything an operator is applied with, including `n_calibration_items`.

    `declared` is a caller's extra configuration -- a fixed path's per-step
    config -- merged OVER the derived part, and refused if it names an
    executor-owned key. "Last writer wins" was a fixed path's first version and
    it let a step lie about the corpus it was running against, which is the one
    thing a plan must not be able to do.
    """
    extra = dict(declared or {})
    overlap = EXECUTOR_DERIVED_CONFIG_KEYS & extra.keys()
    if overlap:
        raise ValueError(
            f"declared operator config names executor-owned keys "
            f"{sorted(overlap)}; these describe the run, not the operator, and "
            "a caller that could override them would plan against a different "
            "corpus than it executes on")
    return {
        "n_calibration_items": len(items),
        **position_policy_config(implementation, policy, items),
        **distribution_support_config(support),
        **extra,
    }


def operator_config_hash(operator_config: Mapping[str, Any]) -> str:
    """The `config_hash` a state carries.

    `n_calibration_items` is EXCLUDED, and that exclusion is the historical
    contract rather than a choice available here: it is a property of the
    corpus a run was given, so including it would make two otherwise identical
    expansions over different-sized mixtures incomparable, and would move every
    committed hash.
    """
    return sha256_json({k: v for k, v in operator_config.items()
                        if k != "n_calibration_items"})
