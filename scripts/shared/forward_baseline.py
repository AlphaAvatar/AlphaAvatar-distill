"""The project's FORWARD BASELINE, resolved. The one entry point for forward work.

    from shared import forward_baseline as FB
    baseline = FB.selected()            # checked against the core at load
    FB.register_operators(baseline)     # registry + signature check
    execution = baseline.execution      # bsz 3, length_sorted_v1, scored_v1

A new initialization experiment builds its session from this rather than from
a historical experiment's constants. Two different questions live in two
different places, and both keep working:

* "what does NEW work run by default?" -- this module, resolved from
  `configs/shared/baselines/selection.json` (maintainer decision 2026-10-17:
  D1's method with `prediction_rows=scored_v1`);
* "which checkpoint did a PAST round challenge?" -- `stages.d_series.incumbent.
  standing_incumbent()`, unchanged, which still answers B for every historical
  record that asks.

The record owns every term; nothing here restates one. The generic mechanism is
`aadistill.initialization.baseline`, which knows no experiment; this module only
knows where this project keeps its selection.

    PYTHONPATH=src:scripts python -m shared.forward_baseline     # print + check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The selection file, and the role this project's initialization work reads.
SELECTION = "configs/shared/baselines/selection.json"
ROLE = "initialization"


def selected(repo_root: str | Path = REPO_ROOT):
    """The baseline in force for initialization work, checked against the core."""
    from aadistill.initialization.baseline import resolve_selection

    return resolve_selection(Path(repo_root) / SELECTION, ROLE)


def register_operators(baseline=None) -> dict[str, str]:
    """Register what a session needs and check each named implementation.

    Registration is explicit in this codebase -- `attention.activation_
    importance_v1` is deliberately not a shipped default -- so a session that
    assumed somebody else registered it would refuse on a pod. Returns
    KIND -> impl_id once every signature matches the record.
    """
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )

    register_builtin_adapters()
    register_builtin_operators()
    activation_importance.register()
    return (baseline or selected()).require_registered_operators(
        get_implementation)


def session_terms(baseline=None) -> dict[str, Any]:
    """The objects a forward search session is built from, by name.

    What an experiment still supplies itself, because the baseline leaves it
    free: the target geometry, the calibration profiles and operator order its
    registered protocol searches over, the state-evaluation suite, the seed and
    the run's identities.
    """
    b = baseline or selected()
    return {
        "execution": b.execution,
        "numerics": b.numerics,
        "position_policy": b.position_policy,
        "distribution_support": b.distribution_support,
        "schedule": b.schedule,
        "ranking_policy": b.ranking_policy,
        "allowed_impls": tuple(sorted(b.implementation_ids().values())),
        "operator_kl_reference": {k: b.kl_reference_config(k)
                                  for k in b.operators},
        "baseline": b.as_record(),
    }


def main() -> int:
    b = selected()
    register_operators(b)
    hist = [a.as_dict() | {"applies_to_forward_profile":
                           a.numerical_execution_fingerprint
                           == b.execution_fingerprint}
            for a in b.reference_artifacts]
    fwd = b.forward_materialization()
    print(json.dumps({"selection": SELECTION, **b.as_record(),
                      "forward_materialization": fwd.as_dict() if fwd else None,
                      "reference_artifacts": hist}, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    raise SystemExit(main())
