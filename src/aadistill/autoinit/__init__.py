"""Teacher-Adaptive AutoInitializer.

Compress an arbitrary larger teacher into a *requested* smaller target
architecture by searching over initialization operators, their order, and the
calibration mixture each one is measured on — rather than by committing to one
fixed recipe.

    teacher + target architecture + calibration pool + search config + budget
      -> search over initialization paths
      -> complete target-size leaves
      -> Beam Top-N
      -> fixed low-budget recovery probe
      -> Top-1
      -> full recovery

Why it exists: E8a found that a full-width depth-ablation proxy preserved the
teacher's output distribution 3.11x better and then initialized 2.8 nats *worse*
once composed with width, FFN and attention compression. An operator's own
objective does not predict the composed result, so the composition has to be
measured — conditionally, in order, remeasuring after every step.

Load-bearing invariants, all mechanical:

* intermediate checkpoints are search states only and can never enter recovery
  Top-N (``state.require_recovery_admissible``);
* every Top-N leaf matches the requested target architecture exactly;
* every produced checkpoint is materialized, reloaded, hashed, validated and
  measured, and its metrics bind to its hash — nothing is inherited from a
  parent (``state.attach_evaluation``);
* an operator's local objective can never become the beam metric
  (``metrics.require_state_metric``);
* the final promotion battery is not reachable from the search
  (``datasets.SEARCH_VISIBLE_ROLES``).

Importing this package registers the shipped adapters and the v1 operator
library. Nothing here launches compute.
"""

from . import adapters as _adapters  # noqa: F401  (registers the Qwen3 adapter)
from . import operators as _operators  # noqa: F401  (registers the v1 library)
from .arch import (
    ArchitectureAdapter,
    ArchSpec,
    Capability,
    UnsupportedCapability,
    adapter_for_config,
    get_adapter,
    register_adapter,
    registered_families,
)
from .artifact import CheckpointIdentity, identify_checkpoint
from .calibration import (
    NO_CALIBRATION,
    V1_PROFILES,
    CalibrationProfile,
    CalibrationSource,
    get_profile,
)
from .cost import (
    A100_80GB_ESTIMATED,
    L40S_MEASURED,
    HardwareProfile,
    branching_estimate,
    price_search,
)
from .datasets import DatasetRole, check_role_isolation
from .manifest import build_manifest, verify_manifest, write_manifest
from .metrics import (
    MetricLevel,
    OperatorLocalMetrics,
    ReferenceStrategy,
    StateEvaluation,
    StateEvaluator,
    StateEvalSuite,
    SuiteItem,
)
from .ranking import PARETO_V1, SCHEDULE_V1, BeamRankingPolicy, BeamSchedule, Objective
from .recovery import E1_KD_HEAVY_0860K, SuccessiveHalvingPlan, admit_leaves
from .stats import DEFAULT_STATS_SPEC, StatsCache, StatsSpec, stats_cache_key
from .search import BeamSearch, SearchConfig, SearchResult
from .state import (
    InitializationState,
    OperatorStep,
    StateValidity,
    make_control_state,
)

__all__ = [
    "A100_80GB_ESTIMATED", "ArchSpec", "ArchitectureAdapter", "BeamRankingPolicy",
    "BeamSchedule", "CheckpointIdentity", "DEFAULT_STATS_SPEC", "NO_CALIBRATION",
    "ReferenceStrategy", "SCHEDULE_V1", "StatsCache", "StatsSpec",
    "identify_checkpoint", "make_control_state", "stats_cache_key",
    "BeamSearch", "CalibrationProfile", "CalibrationSource", "Capability",
    "DatasetRole", "E1_KD_HEAVY_0860K", "HardwareProfile", "InitializationState",
    "L40S_MEASURED", "MetricLevel", "Objective", "OperatorLocalMetrics",
    "OperatorStep", "PARETO_V1", "SearchConfig", "SearchResult", "StateEvalSuite",
    "StateEvaluation", "StateEvaluator", "StateValidity", "SuccessiveHalvingPlan",
    "SuiteItem", "UnsupportedCapability", "V1_PROFILES", "adapter_for_config",
    "admit_leaves", "branching_estimate", "build_manifest", "check_role_isolation",
    "get_adapter", "get_profile", "price_search", "register_adapter",
    "registered_families", "verify_manifest", "write_manifest",
]
