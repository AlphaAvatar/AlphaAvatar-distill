"""Which positions a structural decision is allowed to care about.

One contract, consumed by every calibration-derived operator objective and by
global state evaluation, so an operator and the beam that prunes it cannot
disagree about which predictions matter.

See :mod:`aadistill.initialization.scoring.positions`.
"""

from aadistill.initialization.scoring.positions import (
    ALL_POSITIONS_V1,
    FORM_ALL,
    FORM_SELECT,
    FORM_WEIGHTED,
    POLICY_CONFIG_KEY,
    POLICY_HASH_CONFIG_KEY,
    PREDICTION_AXIS,
    PositionAxis,
    PositionWeights,
    ScoringPositionError,
    ScoringPositionPolicy,
    SUPERVISED_TARGET_V1,
    TOKEN_AXIS,
    get_position_policy,
    normalized_prediction_tags,
    policy_config,
    register_position_policy,
    registered_position_policies,
    resolve_position_policy,
    unregister_position_policy,
    weights_for_items,
)

__all__ = [
    "ALL_POSITIONS_V1",
    "FORM_ALL",
    "FORM_SELECT",
    "FORM_WEIGHTED",
    "POLICY_CONFIG_KEY",
    "POLICY_HASH_CONFIG_KEY",
    "PREDICTION_AXIS",
    "PositionAxis",
    "PositionWeights",
    "SUPERVISED_TARGET_V1",
    "ScoringPositionError",
    "ScoringPositionPolicy",
    "TOKEN_AXIS",
    "get_position_policy",
    "normalized_prediction_tags",
    "policy_config",
    "register_position_policy",
    "registered_position_policies",
    "resolve_position_policy",
    "unregister_position_policy",
    "weights_for_items",
]
