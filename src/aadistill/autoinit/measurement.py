"""The bounded causal-depth measurement: its plan and its authorization SCHEMA.

Separate from Phase A on purpose. This session measures a **rate** and validates
a **backend**; it runs no greedy search, selects no depth map and writes no
checkpoint, and its authorization is the ordinary
:class:`~aadistill.autoinit.authorization.SpendAuthorization`, whose
``allows_phase_a`` is a hard ``False``. A measurement pointed at the wrong
artifact therefore *cannot* start Phase A — a property, not a promise.

The schema carries caps, stages and scope. It carries **no grant**: who permitted
what, at what cumulative spend, is a one-use maintainer decision that belongs in
a document the issuer hashes, not in executable source where it goes stale
silently and still reads as though it applies.
"""

from __future__ import annotations

from .authorization import SpendAuthorization
from .phase_a import GRANT_PROSE_REQUIRED
from .recovery import PreflightPlan, PreflightStage

#: One stage. The plan exists because an authorization binds to a plan hash, and
#: a measurement borrowing another session's hash would be claiming to be that
#: session.
MEASUREMENT_PLAN_V1 = PreflightPlan(
    plan_id="autoinit.causal_depth_measurement",
    version=1,
    stages=(
        PreflightStage(
            stage=0, name="bounded causal-depth runtime and backend measurement",
            blocking=True,
            purpose=("time 3 deterministic evaluations at each skip cardinality "
                     "1-8 against the frozen mixture, extrapolate the 260-"
                     "evaluation runtime with the real 36,35,...,29 weights, "
                     "record the production reference-cache decision and peak "
                     "VRAM, sample GPU utilization, and compare the repaired "
                     "port against E8a per item on the same accelerator"),
            produces=("mean seconds per cardinality and evaluations/min",
                      "weighted 260-evaluation runtime, and the deliberately "
                      "wrong flat-cardinality-8 figure beside it",
                      "production cache decision and production peak VRAM",
                      "comparison peak VRAM, reported separately",
                      "GPU utilization distribution",
                      "E8a-vs-port per-item deltas for |skip|=1 and |skip|=8"),
            stop_conditions=(
                "any per-item backend delta is non-zero -> STOP: the port and "
                "E8a disagree, and that is the finding",
                "the production cache falls back to recompute -> STOP: the cost "
                "basis this measurement establishes no longer holds",
                "no CUDA device -> STOP: a host run re-measures the defect")),),
)

#: $1.6294 hard, from `logs/autoinit_causal_depth_pricing_bound.json`: 24 timed
#: evaluations, one production reference pass and two E8a paired checks with the
#: reference recomputed, at E8a's measured 12.0 evaluations/min, plus teacher
#: load, pod setup and overhead, times three. The recomputed 3x basis is $1.3571;
#: the higher figure is kept because it is more conservative.
MEASUREMENT_AUTHORIZATION = SpendAuthorization(
    authorization_id="autoinit.measurement.UNISSUED",
    granted_utc="",
    granted_by=GRANT_PROSE_REQUIRED,
    plan_id=MEASUREMENT_PLAN_V1.plan_id,
    plan_hash=MEASUREMENT_PLAN_V1.plan_hash,
    expected_usd=0.4524,
    hard_cap_usd=1.6294,
    per_launch_hard_usd=1.6294,
    authorized_stages=(0,),
    stage_conditions={
        "0": ("bounded measurement only: no greedy search, no depth-map "
              "selection, no checkpoint, no follow-on"),
    },
    scope_note=(
        "ONE bounded causal-depth runtime and backend measurement on a single "
        "L40S. NOT Phase-A attempt 11. Trains nothing, searches nothing, selects "
        "nothing, writes no checkpoint. `allows_phase_a` is False by type, so "
        "this artifact cannot start Phase A whatever it is pointed at. The "
        "measured throughput, VRAM and cache decision are inputs to a separate "
        "repricing and a separate cumulative-budget decision; they authorize "
        "nothing by themselves."),
)
