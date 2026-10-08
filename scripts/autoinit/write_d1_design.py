#!/usr/bin/env python3
"""The D1 design: target-aware search, derived end to end from what exists.

    PYTHONPATH=src:scripts:scripts/data python scripts/autoinit/write_d1_design.py --write

**What D1 is.** The first of three GLOBAL scoring/search experiments. Every
calibration-derived operator objective and the global state metric read positions
through one hash-bound scoring-position policy instead of over every token:

    search (target-aware) -> Top-K -> screening -> confirmation -> promotion

The four structural implementations are frozen at the current best per kind, so
the scoring semantics are the only experimental variable.

**Why this is a writer and not a hand-typed document.** Every number it turns on
is a measurement that lives elsewhere: the space is enumerated from the registry,
the per-expansion minutes come from two committed telemetry files, the per-probe
minutes from C2's and C3's own pricing records, the seed-level noise from A3's
three published per-seed deltas, and the evidence capacity from the prompt pools
themselves. Typing any of them here would create a second copy to keep in step
with the first, and this repository has already had a plan assert a bootstrap
seed the computation did not use.

**It authorizes nothing, and it says so where a reader will see it.** Its
fresh behavioural evidence cannot be built under the frozen mixture, it is not
fundable at the current balance, and its search session alone exceeds the
package's per-session envelope — which binds separately, so a grant that moved
only the cumulative cap would still not authorize the search. All three are
derived by `open_blockers()` from figures that live elsewhere, and every
statement of the count follows from that list rather than being typed: this
document read "two blockers" and "either blocker" for a round after the third
appeared. They are stated as blockers rather than caveats, because a design that
reads as ready when it is not is how a grant gets requested for an experiment
that could not have produced a valid result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1  # noqa: E402
from aadistill.initialization.scoring.positions import (  # noqa: E402
    ALL_POSITIONS_V1, SUPERVISED_TARGET_V1,
)

OUT = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
SCHEMA = "aadistill.autoinit.phase_d1_design/v1"

#: Inputs, by path, so a reader can check every figure against its owner.
C0_PREREG = "logs/stages/stage-1/phase_c1/plans/phase_c0_preregistration.json"
C1_BATTERY = "logs/stages/stage-1/phase_c1/plans/battery.json"
A3_COMPARISON = "logs/stages/stage-1/phase_a3/analyses/a3_comparison.json"
CAPACITY = "logs/stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json"
#: THE LIVE behavioural-evidence owners, which supersede CAPACITY above.
D_SERIES_FAMILY = "logs/shared/analyses/autoinit_d_series_battery_family.json"
D_SERIES_MANIFEST = "logs/shared/analyses/autoinit_d_series_family_manifest.json"
BUDGET_TERMS = "configs/experiments/phase_c1/authorization.json"

#: THE INCUMBENT IS DERIVED, NOT TYPED. MAINTAINER DECISION 2026-10-08.
#:
#: These were hand-typed as `fe9683e6a9c783bbc6fe276a78c851c6` / `c313d1b4081b`
#: for the whole of D1's design and search. That is C1's INCUMBENT arm,
#: `attention.weight_proxy_v0` -- the arm C1 measured and BEAT. C1 returned GO
#: at +0.013725 against a SESOI of 0.010, so the checkpoint that stands is C1's
#: TREATMENT, `53e30566...`, exactly as `phase_c2.baseline.B_ARTIFACT_DIGEST`
#: and the C3 stage-E gate have said all along.
#:
#: The error was larger than its subject: C1's delta between the two arms
#: EXCEEDS the SESOI the D1 decision rule tests against, so a candidate measured
#: against the beaten arm is credited with more than the amount that decides --
#: identically for every arm. Owner of the finding:
#: `logs/stages/stage-1/phase_d1/analyses/d1_control_arm_identity.json`.
#:
#: So nothing here states the identity. `standing_incumbent()` derives it from
#: C1's recorded verdict and C1's MEASURED arm identities, which is the only
#: form that cannot go stale: if a later round promotes something else, the
#: answer moves because the record moved.
def incumbent() -> dict[str, Any]:
    """The arm every D-series round challenges, DERIVED from C1's verdict.

    The promoted arm was built as a FIXED PATH and has no search state id --
    `c1_arm_identities.json` records `state_id: null` for it -- so the block
    carries the four CONTENT identities instead. A round that keyed its control
    on a state id would be keying on something the promoted arm does not have.
    """
    from experiments.phase_d_series.incumbent import (
        IDENTITY_FIELDS, standing_incumbent,
    )

    standing = standing_incumbent(REPO)
    return {
        "label": "B",
        #: null, and that is a fact about the arm rather than a gap. See the
        #: docstring: a fixed-path arm has no search state id.
        "state_id": standing.get("state_id"),
        **{f: standing[f] for f in IDENTITY_FIELDS},
        "c1_arm": standing["c1_arm"],
        "impl_id": standing["impl_id"],
        "profile_id": standing["profile_id"],
        "num_parameters": standing["num_parameters"],
        "config_sha256": standing["config_sha256"],
        "_what_it_is": (
            "B, the frozen C1 TREATMENT -- the arm C1's GO verdict promoted. C2 "
            "closed without promotion and C3 returned NO_GO, so B still stands."),
        "_derived_not_declared": (
            "from C1's recorded verdict and C1's measured arm identities, by "
            "scripts/experiments/stage-1/phase_d_series/incumbent.py. This "
            "block was hand-typed until 2026-10-08 and named the arm C1 BEAT; "
            "see logs/stages/stage-1/phase_d1/analyses/"
            "d1_control_arm_identity.json."),
        "_how_it_reaches_a_pod": (
            "materialized ON THE POD from its frozen construction spec -- "
            "`phase_c2.baseline.frozen_baseline_spec`, the one owner of that "
            "construction -- because a fixed-path arm has no state-id directory "
            "and a 1.19 GiB checkpoint fits neither transport to a pod."),
        "_not_recovered": (
            "B is an INITIALIZATION. No formal Stage-2/Stage-3 recovery "
            "evidence exists for it or for anything else."),
    }

#: The recovery recipe every behavioural probe uses, unchanged from C1/C2/C3 so
#: a D1 delta is comparable with the deltas those rounds measured.
RECOVERY_RECIPE = "E1_KD_HEAVY_0860K"


def _load(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        raise SystemExit(
            f"{rel} is missing, and this design restates none of its figures. "
            "Build it first; a design that invented them would be a second "
            "copy of a number with one owner.")
    return json.loads(path.read_text())


def _d_series_protocol() -> dict[str, Any]:
    """The D-series scoring protocol, from its owner rather than retyped."""
    from experiments.phase_d_series.scoring_protocol import describe

    return describe()


def hypothesis() -> dict[str, Any]:
    return {
        #: CORRECTED 2026-10-04, and the correction narrows what D1 may claim.
        #:
        #: The full-vocabulary GPU qualification measured that
        #: `all_v1 @ bsz=1` vs `all_v1 @ bsz=3` moves THREE of four fixed-path
        #: operator selections, while `all_v1 @ bsz=3` vs
        #: `supervised_target_v1 @ bsz=3` moved none of that path's selected
        #: digests. So the position policy is not D1's only changed axis relative
        #: to the historical incumbent -- the adopted bsz=3 execution protocol is
        #: another, and it is the one that moved those selections. The maintainer
        #: decision of 2026-10-04 adds a third: `reference_topk_tail_v1` at K=200.
        #:
        #: D1 is therefore an OPTIMIZATION/CHALLENGER experiment against incumbent
        #: B over the combined protocol. It is NOT a clean causal attribution of
        #: any improvement to position weighting, and a reading that treats it as
        #: one would be attributing to the policy an effect three changes could
        #: have produced.
        "question": (
            "Does the scalable D-series scoring protocol -- "
            "reference_topk_tail_v1 at K=200, the adopted bsz=3 / "
            "length_sorted_v1 execution protocol, and supervised-target-aware "
            "scoring -- produce an initialization that, after the frozen recovery "
            "recipe, outperforms the incoming incumbent B?"),
        "claim_boundary": {
            "it_is": ("an optimization/challenger experiment against incumbent B "
                      "over the combined protocol"),
            "it_is_NOT": ("a causal attribution of any improvement to position "
                          "weighting alone"),
            "why": ("three axes differ from the historical incumbent at once: the "
                    "distribution support, the calibration batch size and the "
                    "position policy. The qualification measured that the BATCH "
                    "SIZE moves three of four fixed-path selections and the "
                    "position policy moved none of them, so the policy is "
                    "demonstrably not the dominant axis."),
            "evidence": ("logs/stages/stage-1/phase_d1/validations/"
                         "gpu-qualification/v1/closeout.json"),
            "_what_would_be_needed_for_attribution": (
                "a design that varies ONE axis at a time against a common "
                "baseline. D1 does not do that and must not be read as if it did."),
        },
        "why_it_might": (
            "every calibration-derived objective in the incumbent search is an "
            "expectation over EVERY token position of the mixture. For the 51 of "
            "67 templated items in calib.domain_balanced@v1, most of those "
            "positions are prompt, system and user text the student is never "
            "asked to predict. The structure that survives compression is "
            "therefore chosen to protect predictions the objective does not "
            "care about, which is a plausible misallocation of a fixed capacity "
            "budget — and it has never been varied, so there is no evidence "
            "either way."),
        "scoring_protocol": _d_series_protocol(),
        "what_D1_changes": [
            "DISTRIBUTION SUPPORT: KL is reduced over the reference's Top-200 "
            "entries plus one aggregate tail bucket, not the full vocabulary",
            "DEPTH: the causal-KL mean is taken over supervised target "
            "positions only",
            "FFN: E[|a_j|] is an expectation with respect to supervised target "
            "positions only",
            "RESIDUAL_WIDTH: the residual second moments accumulate over the "
            "token positions that feed supervised predictions",
            "ATTENTION: mean_t ||W_o,h a_h(t)||^2 is a mean over the same set",
            "GLOBAL STATE EVALUATION: teacher KL, worst-domain, critical-token "
            "and top-1 agreement are weighted means over the same set",
            "BEAM RANKING: consumes that state evaluation, so a candidate "
            "selected on supervised positions cannot be pruned on all of them",
        ],
        "what_D1_does_NOT_change": [
            "the calibration DATA: the same two frozen mixtures, the same "
            "items, the same tokens. Only which of their positions the "
            "objectives read.",
            "the operator set: one frozen implementation per structural kind",
            "the beam width, the beam schedule and the ranking policy",
            "the target geometry, the teacher and the tokenizer",
            "the recovery recipe and the scoring contract",
        ],
        "untemplated_items": (
            "an item with no `assistant` tag is untemplated raw text with no "
            "assistant turn, and every one of its real prediction positions "
            "stays active. That keeps D1's DATA identical to the incumbent "
            "search's and changes only the scoring semantics; removing the "
            "raw-LM exception belongs to the later Stage-0 teacher-native-v2 "
            "experiment, not here."),
        "measured_restriction": (
            "the policy admits 44,746 of 59,763 prediction positions on "
            "calib.domain_balanced@v1 (74.9%) and 46,825 of 59,763 on "
            "calib.reasoning_heavy@v2 (78.4%), and 54,014 of 74,022 on the "
            "state_eval_v1 suite (73.0%). Derived from each asset's own "
            "`assistant` tag; see `scoring_policy.verification`."),
    }


def scoring_policy() -> dict[str, Any]:
    return {
        "treatment": SUPERVISED_TARGET_V1.declare(),
        "control": ALL_POSITIONS_V1.declare(),
        "_control_is_the_incumbent": (
            "`positions.all_v1` is not a new object introduced to be the "
            "control: it is the semantics every committed operator result was "
            "computed under, now named and hashed so a record can state it. It "
            "is numerically inert — the reducers detect it and take their "
            "untouched path — which is what keeps a historical checkpoint "
            "reproducible."),
        "where_it_is_bound": {
            "operator": "OperatorStep.config_hash, via the hashed operator "
                        "config, so the whole subtree forks",
            "search": "SearchConfig.config_hash",
            "state_metric": "StateEvaluation.detail.position_policy_hash",
            "statistics_cache": "stats_cache_key.numerical_config",
        },
        "verification": (
            "tests/initialization/test_scoring_position_policy.py derives the "
            "restriction independently from each frozen mixture's own fields "
            "and compares; tests/initialization/"
            "test_target_aware_scoring_end_to_end.py asserts the incumbent "
            "policy changes no artifact and the treatment moves all four "
            "structural decisions."),
    }


def _execution_wiring_status() -> str:
    """Whether the D1 execution surface exists, DERIVED from the tree.

    This read "CONTRACT ONLY. No D1 launcher exists and none is authorized" for
    two rounds after the launcher existed. A status that outlives its subject is
    how a reader concludes a design is less ready than it is -- and the previous
    round's review had to point it out rather than read it.

    Derived from what the contract itself names: the modules that discharge the
    requirements, and the experiment-suite test that checks them.
    """
    surface = {
        "session": "scripts/experiments/stage-1/phase_d1/d1_session.py",
        "authorization": "scripts/experiments/stage-1/phase_d1/d1_authorization.py",
        "driver": "scripts/pod/autoinit_d1_driver.py",
        "launcher": "scripts/pod/autoinit_d1_launch.py",
        "contract test": ("scripts/experiments/stage-1/phase_d1/tests/"
                          "test_d1_execution_contract.py"),
        "runner-interface test": ("scripts/experiments/stage-1/phase_d1/tests/"
                                  "test_d1_runner_interface.py"),
    }
    missing = [name for name, rel in surface.items()
               if not (REPO / rel).is_file()]
    if missing:
        return (f"PARTIALLY IMPLEMENTED: {sorted(missing)} do not exist. Until "
                "every one does, this record is a CONTRACT and nothing here is "
                "validated.")
    return (
        "IMPLEMENTED AND CHECKED AT $0. Every requirement below is discharged in "
        "`d1_session.build_session`, in one function rather than left to a driver "
        "to remember, and `d1_session.assert_session_contract` verifies the result "
        "against THIS DESIGN before the first expansion -- so a miswired session "
        "costs nothing instead of producing records that look valid and are "
        "scientifically unusable. The experiment suite drives the checker through "
        "every requirement broken on its own, and a separate test constructs the "
        "REAL SessionRunner to prove the authorization interface is satisfied "
        "before any provider work. NOT a launch authorization: see "
        "`open_blockers` and the one-use artifact.")


def execution_wiring_required() -> dict[str, Any]:
    """What a D1 execution entry point must wire, recorded before one exists.

    No D1 launcher is authorized, so this is a CONTRACT and not an
    implementation. It is written down now because every item is a thing the
    core permits a caller to omit, and omitting any of them produces a run
    whose records are valid-looking and scientifically unusable.
    """
    return {
        "status": _execution_wiring_status(),
        "must_be_wired": [
            {"requirement": (
                "the frozen state-eval manifest's `content_sha256` is passed "
                "into `StateEvaluator(suite_content_sha256=...)`"),
             "why": (
                 "the evaluator falls back to `suite.content_sha256` and then "
                 "to `UNBOUND_SUITE_CONTENT`. A frozen D1 run must NOT execute "
                 "with `suite_content_identity = \"unbound\"`: the whole point "
                 "of binding content separately from the structural "
                 "`suite_hash` is lost, and the resulting "
                 "`measurement_protocol_id` would not distinguish two suites "
                 "with the same structure and different items."),
             "mechanism": "aadistill.initialization.planning.metrics.StateEvaluator"},
            {"requirement": (
                "the evaluator is constructed with D1's position policy "
                "(`positions.supervised_target_v1` for the treatment, "
                "`positions.all_v1` for the control) and the declared "
                "`NumericalEnvironment`"),
             "why": (
                 "both are terms of `measurement_protocol_id`. A treatment arm "
                 "measured under the control's policy, or under an unrecorded "
                 "execution environment, produces a protocol id that does not "
                 "describe what ran."),
             "mechanism": "aadistill.initialization.scoring.protocol_identity"},
            {"requirement": (
                "`SearchConfig.measurement_protocol_id` is SET to the "
                "evaluator's protocol id before the first expensive expansion "
                "— declared, not learned from the first measurement"),
             "why": (
                 "the search already refuses a declared id that disagrees with "
                 "its measurer, at construction, because every state measured "
                 "in between would have to be discarded. Learning it instead "
                 "means the config hash does not carry the protocol until "
                 "after work has been done under it, and a resume cannot tell "
                 "whether the records it is adopting were measured the same "
                 "way."),
             "mechanism": "aadistill.initialization.planning.search"},
        ],
        "the_test_that_was_owed": (
            "a small D1 EXPERIMENT-suite contract test, written when the driver "
            "is — asserting the evaluator receives the frozen content hash, the "
            "declared policy and environment, and that the search's declared "
            "protocol id is non-null and equals the evaluator's before any "
            "expansion. It belongs in "
            "scripts/experiments/stage-1/phase_d1/tests/, not the core suite: "
            "it checks THIS experiment's wiring, not a reusable mechanism."),
        "_why_it_was_written_before_an_implementation": (
            "so the requirement was not rediscovered after a paid run produced "
            "unusable records. Each item is a thing the core PERMITS a caller to "
            "omit."),
    }


def materialization_prerequisite() -> dict[str, Any]:
    """The A3 blocker, and the state of its resolution."""
    return {
        "requirement": (
            "A-bsz3 may not enter D1/D2/D3 execution until the repository binds "
            "the numerical execution fingerprint to materialization/resume "
            "identity. Owner: logs/stages/stage-1/phase_c3/plans/"
            "a_bsz3_adoption.json :: identity_semantics."),
        "why": (
            "A3 built 7dd2f6f6980b where the bsz=1 protocol built 53e30566c5f7 "
            "— reproducibly, on three machines, with an identical "
            "result_spec_hash. `compute_state_id` binds neither the "
            "ExecutionConfig nor the artifact digest, so two differing "
            "artifacts collided on one resumable, deduplicable state id."),
        "resolution": {
            "mechanism": "aadistill.initialization.specs.materialization",
            "identities": {
                "semantic_state_id": "the scientific/path identity; still "
                                     "deliberately blind to execution",
                "numerical_execution_fingerprint": "batch size, packing, device "
                                                   "class, compute dtype, "
                                                   "accumulation dtype",
                "materialization_id": "semantic + fingerprint; what resume, "
                                      "dedup and checkpoint ownership key on",
                "artifact_digest": "the bytes, observed and bound once",
            },
            "enforcement": [
                "BeamSearch._restore declines a journal entry whose "
                "materialization differs",
                "BeamSearch._restore declines an evaluation measured under "
                "another scoring policy",
                "BeamSearch._materialize_and_measure refuses a measurer that "
                "scored other positions",
                "stats_cache_key carries the execution fingerprint and the "
                "policy hash",
                "MaterializationIdentity.bind refuses a second, different "
                "digest",
            ],
            "regression": "tests/initialization/test_materialization_identity.py and "
                          "tests/initialization/test_target_aware_scoring_end_to_end.py"
                          "::TestResumeRefusals",
            "status": "IMPLEMENTED AND VERIFIED at `$0` on CPU. The A3 "
                      "precondition is met.",
        },
        "not_cleared_by": (
            "a passing behavioural result. It is an engineering correctness "
            "property, which is why the A3 closeout refuses to let one stand "
            "in for it."),
    }


def execution_protocol() -> dict[str, Any]:
    """The batching protocol, by maintainer instruction, and what prices it."""
    return {
        "micro_batch_size": 3,
        "calibration_batch_packing": "length_sorted_v1",
        "_decision": _batching_decision(),
        "_the_value_is_config_not_core": (
            "`3` is an experiment-policy number carried by ExecutionConfig. The "
            "core accepts 2, 4, 8 or a future token-budget batching policy with "
            "no edit; `aadistill.initialization.execution` owns the knob and "
            "`calibration.packing` owns the policies."),
        "paths_covered": [
            "DEPTH causal-KL scoring", "FFN activation statistics",
            "RESIDUAL_WIDTH residual second moments",
            "ATTENTION residual-write statistics",
            "global state evaluation (forwards batched; the reduction stays the "
            "certified per-item `distortion` call)",
            "beam candidate evaluation, through the same evaluator",
        ],
        "state_eval_bound": {
            "max_group_width_tokens": 2002,
            "peak_logit_bytes_both_models_bf16": 3_651_993_600,
            "padding_over_valid": 0.0211,
            "padding_over_valid_at_original_order": 0.2335,
            "budget_bytes": 12 * 2 ** 30,
            "_why_a_bound_exists": (
                "a batched state evaluation materializes two [B, T_max, V] "
                "logit blocks at a 151,936 vocabulary. StateEvaluator refuses "
                "before the first forward when the widest group would exceed "
                "the budget, because the alternative is discovering the limit "
                "as an OOM mid-search — which is how the causal-depth "
                "rehearsal died."),
            "_why_length_sorted": (
                "2.1% padding against 23.4% at the mixture's own order, derived "
                "from the suite's lengths at `$0`."),
        },
        "numerical_invariant": (
            "score = sum_t(w_t * value_t) / sum_t(w_t), with w_t = 0 at a "
            "padded position. Padding never enters a denominator; a per-item "
            "mean is formed per row so batch composition cannot change domain "
            "weighting; and there is no batch-size-specific scientific branch."),
    }


def _ensure_the_frozen_operators_are_registered() -> None:
    """Register the frozen D1 operator set. Idempotent, and called by every
    function that needs it rather than by one that happens to run first.

    `attention.activation_importance_v1` registers from its own module instead
    of as a shipped default, so a bare `register_c2_operators()` leaves the
    frozen set incomplete. Both are needed and both are safe to repeat.

    **This exists because the call order was load-bearing and undeclared.**
    `budget()` reads `d1.chain_cost`, which searches the frozen space, and it
    worked only because `search_stage()` appeared earlier in `build()`'s dict
    literal and registered as a side effect. Hoisting `budget()` to derive the
    blocker list broke it immediately -- which is the good version of that bug,
    since the alternative is a caller that imports the sections in another
    order and gets an unregistered-operator failure far from its cause.
    """
    from experiments.phase_c2.search_space import register_c2_operators
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )

    register_c2_operators()
    activation_importance.register()


def search_stage() -> dict[str, Any]:
    from experiments.phase_d1 import search_space as d1

    _ensure_the_frozen_operators_are_registered()
    size = d1.size_report()
    #: ONE CURRENT SEARCH COST. `d1.search_cost()` on the FROZEN model is the
    #: full-vocabulary planning figure -- $31.1577 -- and `budget.chain.sessions
    #: .search` carries the measured production Top-K one. Both lived here under
    #: generic names, so the live design stated two current prices for one session
    #: and this field held the superseded one.
    adjusted = _rebuilt_cost_model()
    if adjusted is None:
        cost = d1.search_cost()
        historical = None
    else:
        with _measured_cost_model(adjusted):
            cost = d1.search_cost()
        #: Kept for PROVENANCE under a name that cannot be mistaken for the live
        #: price: a reader comparing the measured cell against what it replaced
        #: needs the old figure, and deleting it would lose the comparison.
        historical = {
            **d1.search_cost(),
            "_this_is_HISTORICAL": (
                "the superseded FULL-VOCABULARY planning basis for this session, "
                "retained so the measured cell can be compared against what it "
                "replaced. It is NOT the current price and must not be read as "
                "one; `cost` above and `budget.chain.sessions.search` are, and a "
                "regression asserts those two agree."),
        }
    return {
        "coverage": d1.coverage(),
        "frozen_implementations": size["frozen_implementations"],
        "exclusions": size["exclusions"],
        "profiles": size["profiles"],
        "free_variables": [
            "operator ORDER (4! = 24 orderings)",
            "calibration PROFILE per operator (2^4 = 16 assignments)",
        ],
        "_why_calibration_assignment_stays_free": (
            "the incumbent assignments were chosen by searches that scored over "
            "every position. Pinning them would inherit a choice made under the "
            "semantics D1 is testing, which is the same mistake C2b records "
            "about Search-1's pinned mixtures."),
        "reachable_leaves": size["d1_frozen_set"]["total_leaves"],
        "leaves_if_nothing_excluded":
            size["if_no_operator_were_excluded"]["total_leaves"],
        "beam": size["beam"],
        "ranking_policy": {"id": PARETO_V1.qualified_id,
                           "hash": PARETO_V1.policy_hash},
        "schedule": {"id": SCHEDULE_V1.schedule_id,
                     "width": SCHEDULE_V1.width,
                     "warmup_levels": SCHEDULE_V1.warmup_levels},
        "cost": cost,
        **({} if historical is None
           else {"superseded_full_vocab_planning_cost": historical}),
        "_cost_is_the_budget_chains": (
            "`cost` is the same session `budget.chain.sessions.search` carries: "
            "both come from `search_cost` on the SAME cost model, measured when a "
            "production Top-K basis exists and frozen when none does. Two "
            "generic current-cost fields for one session is how this document came "
            "to state $31.1577 and $21.4897 at the same time."),
        "_cost_basis_status": _cost_basis_status(),
        "stops_at": (
            "commit_top_k. The search trains nothing, measures no behaviour and "
            "has no code path into a behavioural stage — the same boundary C2's "
            "driver holds."),
    }


#: The design, FIXED HERE rather than selected by a rule.
#:
#: An earlier version of this file chose by filtering the priced grid on
#: `bias_under_sesoi` and maximizing the advance probability. That filter
#: encoded a validity condition that does not exist: the winner's curse lives in
#: the SCREENING estimate, and a confirmation rung on fresh disjoint prompts and
#: fresh seeds is unbiased under the global null however inflated the screening
#: number was. Choosing a design by comparing that inflation to the SESOI was
#: arithmetic in the service of a wrong argument.
#:
#: So the three numbers are stated as a judgement, with the reasons recorded,
#: and the arithmetic is reported beside them as planning and sensitivity
#: analysis rather than as the thing that picked them.
#: WIDENED FROM 2 TO 4 by a maintainer decision of 2026-10-07, after the
#: historical full-search records were re-read. The search science is untouched;
#: this is the downstream finalist/recovery policy and nothing else.
#:
#: WHY. Top-2 was never an inherited retention convention -- Phase-B attempt5
#: and C2 attempt4 both committed `n_selected = 5` -- it was a later D1
#: screening-cost choice. And the quantity it gives elimination power to is the
#: one this design itself marks UNKNOWN: `P(a behaviourally good candidate is in
#: the Top-K)`. C2's evidence is that the cheap state-eval ranking is a weak
#: predictor of recovered behavioural quality, so a candidate ranked 3rd or 4th
#: by the proxy may still be the best after recovery. Shrinking the recovery
#: field to 2 hands the proxy a decision the proxy has not earned.
#:
#: WHAT IT COSTS, both directions, because only one of them is an improvement:
#: a wider field raises the unmeasured P(the good candidate is in it), and at a
#: FIXED two screening seeds it lowers the probability of ordering the field
#: correctly once it is in -- `advance_probability` 0.7808 -> 0.5839. The chain
#: hard ceiling moves $60.7509 -> $69.3365. The trade was accepted knowing
#: both; see `_what_the_arithmetic_says_about_it` below, which reports the
#: second number rather than only the first.
D1_TOP_K = 4
D1_SCREENING_SEEDS = 2
D1_CONFIRMATION_SEEDS = 3


#: THE SCIENTIFIC PREIMAGE: a POSITIVE list of what `design_hash` covers.
#:
#: It was `{k: v for k, v in doc.items() if not k.startswith("_")}` -- every
#: non-underscore key, which swept in `budget`. So the "frozen scientific design
#: identity" was a function of the project's wallet: booking a session's spend
#: moved `budget.position` and `provisional_shortfall_usd`, which moved the hash,
#: which invalidated the committed record, which reddened two tests in D1's own
#: pod gate and blocked every launch until the record was regenerated and the
#: authorization re-issued against a new hash. That cycle was paid three times in
#: one round. A scientific design does not change because money was spent.
#:
#: A BLACKLIST WOULD HAVE BEEN THE WRONG FIX. `budget` is not the only
#: money-bearing key -- `search_stage.cost` and `behavioural_design.priced_grid`
#: are priced too, and `open_blockers` is partly derived from funding -- and the
#: next money-bearing field added anywhere in this document would silently join
#: the preimage. A positive list fails the other way: a new SCIENTIFIC field is
#: absent from the hash until somebody adds it here, which is visible in review
#: rather than invisible in a digest.
#:
#: MOVES THE HASH: the hypothesis; the scoring protocol; the search objectives,
#: space, exclusions and free variables; the position policy and distribution
#: support (inside `scoring_policy`); beam semantics (schedule, ranking policy);
#: the finalist-retention rule; Top-K; the recovery recipe; screening and
#: confirmation seeds; the decision rule; the contamination/evidence protocol;
#: the incumbent being challenged; the materialization prerequisite.
#:
#: MUST NOT MOVE IT: spent dollars, remaining balance, live GPU price, project /
#: formal / package headroom, resource state, run status, ledger position. Money
#: stays independently hash-bound where it belongs -- in each run's one-use
#: authorization, whose own self-hash covers its `money` block, its live quote
#: and its derived ceiling.
SCIENTIFIC_TOP_LEVEL: tuple[str, ...] = (
    "schema", "experiment_id", "stage_id",
    "hypothesis", "scoring_policy", "incumbent", "recovery_recipe",
    "contamination_protection", "evidence", "execution_protocol",
    "materialization_prerequisite", "what_this_may_not_be_used_to_claim",
)

#: Scientific sub-keys of `search_stage`. `cost` and
#: `superseded_full_vocab_planning_cost` are priced and are deliberately out.
SCIENTIFIC_SEARCH_STAGE: tuple[str, ...] = (
    "beam", "schedule", "ranking_policy", "frozen_implementations", "profiles",
    "free_variables", "exclusions", "stops_at", "coverage", "reachable_leaves",
    "leaves_if_nothing_excluded",
)

#: Scientific sub-keys of `behavioural_design`. `priced_grid` is priced and is
#: deliberately out; every probe count, seed count, retention rule and decision
#: rule is in.
SCIENTIFIC_BEHAVIOURAL: tuple[str, ...] = (
    "top_k", "finalist_retention", "screening_seeds", "confirmation_seeds",
    "screening_probes", "confirmation_probes", "total_probes", "decision_rule",
    "protocol_uniformity", "sesoi", "seed_sd", "seed_sd_n",
    "seed_sd_interval_95", "noise_model", "screening_estimate_inflation",
    "advance_probability", "advance_probability_sensitivity",
    "pipeline_probability_at_assumed_values",
)


def scientific_preimage(doc: dict[str, Any]) -> dict[str, Any]:
    """What `design_hash` covers: the science and the protocol, never the money.

    Raises if a declared scientific key is absent, because a preimage that
    silently skips a missing field hashes a smaller document and still looks
    like a valid identity.
    """
    out: dict[str, Any] = {}
    for key in SCIENTIFIC_TOP_LEVEL:
        if key not in doc:
            raise KeyError(
                f"scientific_preimage declares {key!r} and the design does not "
                "carry it; a preimage that skips a missing scientific field "
                "hashes a smaller document and still reads as an identity")
        out[key] = doc[key]
    for parent, keys in (("search_stage", SCIENTIFIC_SEARCH_STAGE),
                         ("behavioural_design", SCIENTIFIC_BEHAVIOURAL)):
        block = doc[parent]
        missing = [k for k in keys if k not in block]
        if missing:
            raise KeyError(f"{parent} is missing scientific keys {missing}")
        out[parent] = {k: block[k] for k in keys}
    return out


def behavioural_design() -> dict[str, Any]:
    """Top-K, screening seeds and confirmation seeds, with their rationale."""
    from experiments.phase_d1 import search_space as d1
    from experiments.phase_d1.selection_noise import (
        CLAIM_BOUNDARY,
        P_IN_TOP_K_SENSITIVITY,
        SEED_SD,
        SEED_SD_INTERVAL,
        SESOI,
        advance_probability,
        advance_probability_sensitivity,
        pipeline_detection_probability,
        report as noise,
        screening_estimate_inflation,
    )

    grid = d1.designs()
    chosen = next(row for row in grid
                  if row["top_k"] == D1_TOP_K
                  and row["screening_seeds"] == D1_SCREENING_SEEDS)
    return {
        "top_k": D1_TOP_K,
        #: THE POST-SEARCH RETENTION RULE, hash-bound because changing it changes
        #: which candidates reach behavioural recovery. Standing maintainer
        #: policy of 2026-10-07 for every full-search experiment: lineage
        #: diversity is an EXPLORATION mechanism used during beam pruning, and
        #: the final Top-K is the search objectives' own quality ordering alone.
        #:
        #: The completed 2026-10-06 search reused the beam's rule for Stage D and
        #: committed a finalist at quality position 11 of 12 -- over twice the
        #: best leaf's objective value, worse than seven leaves it excluded --
        #: because that leaf was the sole member of its lineage, while the
        #: candidates at quality positions 2 and 4 were excluded for sharing one.
        #: That run stays valid under the semantics it ran; this changes the rule
        #: prospectively.
        "finalist_retention": PARETO_V1.RETENTION_QUALITY_ONLY,
        "finalist_retention_note": (
            "post-search Top-K is epsilon-Pareto fronts, best first, with the "
            "policy's deterministic tie-break inside each front, concatenated, "
            "first K taken. NO lineage rotation. Diversity remains in force "
            "during intermediate beam levels, where a state is a partial "
            "hypothesis and one early proxy measurement must not extinguish a "
            "structural family. It is NOT a new scalar score: the three "
            "objectives and the epsilon are unchanged."),
        "screening_seeds": D1_SCREENING_SEEDS,
        "confirmation_seeds": D1_CONFIRMATION_SEEDS,
        "screening_probes": chosen["screening_probes"],
        "confirmation_probes": chosen["confirmation_probes"],
        #: THE ARM COUNTS. Carried because every arm's checkpoint must be
        #: materialized on the pod before any probe can train from it, and that
        #: cost scales with ARMS rather than with probes -- it sat inside a
        #: fixed session overhead as one rebuild until 2026-10-08, so the
        #: screening session was priced for a fifth of the work it requires.
        #:
        #: Not in `SCIENTIFIC_BEHAVIOURAL`: `screening_arms` is `top_k + 1` and
        #: `top_k` is hashed, so the identity already covers it. A derived
        #: quantity in a hash preimage adds nothing and is one more thing to
        #: keep in step.
        "screening_arms": chosen["screening_arms"],
        "confirmation_arms": chosen["confirmation_arms"],
        "total_probes": chosen["total_probes"],
        "screening_estimate_inflation": chosen["screening_estimate_inflation"],
        "advance_probability": chosen["advance_probability"],
        "advance_probability_sensitivity": advance_probability_sensitivity(
            D1_TOP_K, D1_SCREENING_SEEDS),
        "pipeline_probability_at_assumed_values": {
            str(p): pipeline_detection_probability(
                D1_TOP_K, D1_SCREENING_SEEDS, p_in_top_k=p)
            for p in P_IN_TOP_K_SENSITIVITY},
        "sesoi": SESOI,
        "seed_sd": round(SEED_SD, 6),
        "seed_sd_n": 3,
        "seed_sd_interval_95": [round(SEED_SD_INTERVAL[0], 6),
                                round(SEED_SD_INTERVAL[1], 6)],
        "noise_model": noise(),
        "priced_grid": grid,
        "_why_top_k_is_4": (
            "MAINTAINER DECISION 2026-10-07, amending the downstream "
            "finalist/recovery policy only. The beam search, its measurement "
            "protocol and every frozen scientific term are unchanged.\n\n"
            "Top-2 was not an inherited full-search retention convention: "
            "Phase-B attempt5 and C2 attempt4 each committed n_selected=5. It "
            "was a later D1 screening-cost choice, and it gave elimination "
            "power to the exact quantity this design marks UNKNOWN -- "
            "P(a behaviourally good candidate is in the Top-K). C2 measured the "
            "cheap state-eval ranking to be a weak predictor of recovered "
            "behavioural quality, so a candidate the proxy ranks 3rd or 4th may "
            "still be the best after recovery, and a field of 2 lets the proxy "
            "decide that on its own.\n\n"
            "BOTH DIRECTIONS, because only one of them is an improvement: a "
            "wider field raises the unmeasured P(it contains the good "
            "candidate); at a fixed two screening seeds it LOWERS the "
            "probability of ordering that field correctly once it does, from "
            "0.7808 to 0.5839. Screening-estimate inflation rises 0.005149 to "
            "0.009394. The chain hard ceiling moves $60.7509 to $69.3365 and "
            "the probe count 12 to 16. The trade was accepted knowing all of "
            "it; the arithmetic below reports the losses as well as the gain.\n\n"
            "NOT changed: the search, the beam, K=200, bsz=3, "
            "length_sorted_v1, the operator space, both calibration profiles, "
            "the position policy, the state-eval metrics, the screening seed "
            "count, the confirmation seed count, or the per-session envelope. "
            "Already measured search states remain valid, because none of this "
            "touches the search's measurement protocol."),
        "_how_these_three_numbers_were_chosen": (
            "as a pragmatic balance, not as the optimum of a formula. Top-K=4 "
            "keeps the recovery field wide enough that the search's own cheap "
            "ranking is not the thing that eliminates a candidate it cannot "
            "reliably rank -- see `_why_top_k_is_4`; two screening seeds make "
            "the ordering "
            "stable rather than a single-draw coin flip, which is the specific "
            "weakness of C2's one-seed rung; three confirmation seeds are the "
            "same count A3 used, on a FRESH disjoint battery, which keeps the "
            "confirmation estimate independent of everything the screening rung "
            "saw. The cost of the whole chain is what caps all three.\n\n"
            "NOT claimed: that this is a formally demonstrated optimum, or "
            "that it was derived from a bias-versus-SESOI comparison. The "
            "numbers below describe the design; they did not select it."),
        "_what_the_arithmetic_says_about_it": (
            "the winner's curse on the screening estimate is "
            f"{screening_estimate_inflation(D1_TOP_K, D1_SCREENING_SEEDS):.6f} "
            "— a property of the screening number, which is never reported as "
            "D1's effect estimate and never promotes anything by itself. The "
            "probability of advancing a candidate that is better by the SESOI, "
            "GIVEN that it is among the two, is "
            f"{advance_probability(D1_TOP_K, D1_SCREENING_SEEDS):.4f} at the "
            "point estimate of the per-seed spread — and "
            f"{advance_probability_sensitivity(D1_TOP_K, D1_SCREENING_SEEDS)['p_at_sd_high']:.4f}"
            " to "
            f"{advance_probability_sensitivity(D1_TOP_K, D1_SCREENING_SEEDS)['p_at_sd_low']:.4f}"
            " across that spread's own 95% sampling interval, because it was "
            "estimated from THREE A3 deltas. The interval is wider than the "
            "differences between the candidate designs, which is why no design "
            "was selected by maximizing it."),
        "_claim_boundary_of_the_noise_model": CLAIM_BOUNDARY,
        "_what_the_derivation_does_NOT_cover": (
            "WHETHER A GOOD CANDIDATE IS IN THE TOP-K AT ALL. Both figures "
            "above condition on the better candidate being inside the screened "
            "field; neither says how often the search's own cheap metric puts "
            "it there. C2's evidence is that the cheap metric predicts "
            "behaviour poorly — four of its five committed candidates sat in a "
            "better eps-Pareto front than the incumbent and two dominated it on "
            "all three ranked objectives, and the behavioural confirmation came "
            "back negative — so for D1 the probability that the top 2 of the "
            "~12 leaves the beam visits contains the behaviourally best one is "
            "UNKNOWN and is not claimed to be high.\n\n"
            "That is a real limitation and it is not repairable by widening K. "
            "Widening it worsens both the bias and the discrimination, so the "
            "screening rung cannot be made simultaneously broad and reliable at "
            "this noise level. The levers that would actually address it are "
            "reducing the noise (more seeds per screened candidate, which costs "
            "probes linearly) or improving the search metric's correlation with "
            "recovered behaviour (which is a research question, not a design "
            "parameter). D1 takes neither: it screens narrowly and reliably, "
            "and the claim boundary records that a candidate the cheap metric "
            "ranked third is never behaviourally tested."),
        "_claim_boundary": (
            "a GO confirms THE ADVANCING CANDIDATE against B, conditional on "
            "the three confirmation seeds, having been selected on disjoint "
            "screening prompts and disjoint seeds. It is not a statement about "
            "the other Top-K candidate, about the leaves the beam did not "
            "visit, about a population of recovery seeds, or about target-aware "
            "scoring in general — only about this path under this policy."),
        "_what_this_replaces": {
            "c2_design": "K=5 at ONE screening seed",
            "c2_screening_estimate_inflation": screening_estimate_inflation(5, 1),
            "c2_advance_probability": advance_probability(5, 1),
            "reading": (
                "C2's screening rung reported a screening delta inflated by "
                "about 1.5x the effect it was looking for, and at one seed it "
                "was more likely to advance a candidate that was NOT the better "
                "one (0.42 conditional on the better one being in the field). "
                "The design above trades breadth for a more stable ordering at "
                "the same noise level. This is a diagnosis of C2's SCREENING "
                "rung's discriminating power, derived from the project's own "
                "measurements — it is NOT a re-analysis of C2's result, it "
                "changes no C2 figure, and it does not say C2's confirmation "
                "estimate was biased: C2's confirmation ran on its own "
                "disjoint battery."),
        },
        "protocol_uniformity": {
            "requirement": (
                "every probe in one comparison field is GENERATED and EVALUATED "
                "under one protocol identity. C2's confirmation field mixed "
                "them, which is the limitation its closure records."),
            "how": (
                "all probes of a rung are trained and evaluated in ONE session "
                "on one host, under one image digest, one engine version and "
                "one scoring contract; the comparison verifies each probe's own "
                "admission record asserts comparability under "
                "generation_runtime_comparability@v2 and serializes the "
                "verdict."),
            "_not_a_between_session_comparison": (
                "A3's remaining unquantified alternative explanation is that "
                "its treatment and controls were measured in different "
                "sessions. D1 does not inherit that: both arms of each rung are "
                "measured together."),
        },
        "decision_rule": {
            "primary_endpoint": "correct_overall",
            "estimand": ("prompt-mean of the seed-mean paired difference "
                         "(candidate - B) over the three fixed confirmation "
                         "seeds"),
            "inference": ("stratified PROMPT-cluster bootstrap, seeds as fixed "
                          "blocks; the interval is conditional on those three "
                          "seed pairs and is not a seed-population claim"),
            "rule": "three-way GO / NO-GO / INCONCLUSIVE, no forced winner",
            "guardrails": ("usable_rollout vetoes only, reported with every "
                           "component and never positive ranking credit"),
            "_inherited_deliberately": (
                "the endpoint, the estimand, the inference and the SESOI are "
                "C0's and are NOT re-derived: D1 must be comparable with C1's "
                "GO and C2's and C3's negatives, and a new endpoint would make "
                "it a separate programme. What is re-derived is the part C0 "
                "left to each round — how many candidates are screened and on "
                "how many seeds."),
        },
    }


#: The qualification's own closeout is the single owner of whether the owed GPU
#: validation has run. This writer derives from it rather than carrying a typed
#: status, because a hand-written "NOT RUN" is false exactly when the run
#: succeeds — the one moment a reader is most likely to trust it.
QUALIFICATION_CLOSEOUT = ("logs/stages/stage-1/phase_d1/validations/"
                          "gpu-qualification/v1/closeout.json")
#: The Top-K adoption validation, whose closeout is under corrective review.
TOPK_ADOPTION = ("logs/stages/stage-1/phase_d1/validations/topk-adoption/v1/"
                 "closeout.json")


def _qualification_state() -> dict[str, Any]:
    """`status` for `gpu_validation_owed`, read off the qualification closeout."""
    path = REPO / QUALIFICATION_CLOSEOUT
    if not path.is_file():
        return {
            "status": "OWED, NOT RUN. Nothing was created and $0 was spent.",
            "_status_owner": (f"derived: no closeout at {QUALIFICATION_CLOSEOUT}. "
                              "Writing one moves this line with no edit here."),
        }
    raw = path.read_bytes()
    doc = json.loads(raw)
    #: TWO GPU ROUNDS NOW, and the status must not read as though both were
    #: settled. The full-vocabulary qualification passed; the Top-K adoption was
    #: reopened by an independent review over the tail arithmetic. A bare
    #: "RUN -- PASSED" here would be the kind of status that is true of one thing
    #: and read as true of everything.
    #: DERIVED, through the one function that owns the adoption's live state, so
    #: this line cannot say "under corrective review" after the review closed.
    topk_state = _topk_adoption_state()["status"]
    answers = doc.get("answers") or {}
    recon = answers.get("incumbent_reconstruction") or {}
    decisions = answers.get("discrete_decisions") or {}
    return {
        "status": (f"full-vocab qualification RUN -- {doc['verdict']}; "
                   f"Top-K adoption: {topk_state}"),
        "_status_owner": QUALIFICATION_CLOSEOUT,
        "ran": {
            #: The closeout's PATH and CONTENT HASH, so this design is bound to
            #: the exact bytes it read, plus ONLY the conclusions the design
            #: consumes. It used to inline the whole `answers` object, which made
            #: this one key 1.48 MB of a 3.9 MB design -- a third copy of
            #: selection lists that `runs/` already owned.
            "closeout": QUALIFICATION_CLOSEOUT,
            "closeout_sha256": hashlib.sha256(raw).hexdigest(),
            "verdict": doc["verdict"],
            "gpu": doc.get("gpu"),
            "price_per_hour_usd": doc.get("price_per_hour_usd"),
            "cost_usd": doc.get("cost_usd"),
            "reconstructed_the_frozen_incumbent":
                recon.get("matched_the_frozen_incumbent"),
            "n_operator_selections_moved": decisions.get("n_moved"),
            "what_moved_them": sorted({
                s.get("attribution", "").split(":")[0]
                for s in (decisions.get("steps_that_moved") or [])}),
            "_everything_else_is_in_the_closeout": (
                "the per-arm answers, the provenance, the agreement across pods "
                "and the source run ids live in the closeout named above; its "
                "raw evidence lives in that directory's runs/. This design "
                "carries the hash and the conclusions it uses, not a copy."),
            "_authorizes": ("nothing. A passed engineering qualification is not "
                            "formal D1 authorization, and it closes none of the "
                            "open blockers: see `open_blockers`."),
        },
    }


def _topk_adoption_state() -> dict[str, Any]:
    """The Top-K adoption's live state, from its closeout and the priced basis.

    This was typed, and it said "UNDER CORRECTIVE REVIEW ... the adoption evidence
    is being re-measured" for two rounds after the re-measurement finished. A
    status that outlives its own subject is worse than no status: a reader
    concludes the evidence is in flight.

    Derived from the same predicate that drives
    `per_session_envelope_compatibility`, `price_status` and `open_blockers`, so
    the four cannot disagree.
    """
    path = REPO / TOPK_ADOPTION
    if not path.is_file():
        return {"status": "NOT RUN -- no adoption closeout exists",
                "owner": TOPK_ADOPTION,
                "_status_owner": "derived: the closeout's absence"}
    doc = json.loads(path.read_text())
    verdict = doc.get("verdict")
    verdict = verdict if isinstance(verdict, dict) else {"_status": str(verdict)}
    basis = _topk_production_basis()
    out = {
        "owner": TOPK_ADOPTION,
        "_status_owner": (f"derived from {TOPK_ADOPTION} and the production basis "
                          "it names; nothing here is typed"),
        "closeout_status": verdict.get("_status") or verdict.get("_prior"),
        "remaining_blockers_it_names": doc.get("remaining_blockers"),
    }
    if basis is None:
        out["status"] = (
            "RUN, and the production cost is NOT yet a valid pricing basis: the "
            "closeout exists but no record carries a whole-invocation measurement "
            "under the production path.")
        return out
    out["status"] = (
        f"CLOSED. The production operator cost is MEASURED -- one real "
        f"`impl.execute` invocation of {basis['operator_invocation_seconds_max']} s "
        f"({basis['operator_minutes_at_max']} min), the max over both DEPTH "
        f"calibration profiles -- and the per-session envelope resolves on it. "
        "Nothing about the Top-K protocol is owed or in flight.")
    out["measured_basis"] = {
        "owner": basis["owner"],
        "operator_invocation_seconds_max":
            basis["operator_invocation_seconds_max"],
        "candidate_subsets_measured": basis["candidate_subsets_measured"],
    }
    return out


def _what_is_not_priced() -> list[str]:
    """What a GPU still owes a price for. Derived, and EMPTY is a valid answer.

    It used to be a typed list headed "the PRODUCTION Top-K D1 search cost ... and
    therefore the per-session envelope, which stays UNRESOLVED until a Top-K-only
    representative timing exists". Both entries were satisfied by a8 and neither
    knew it.
    """
    basis = _topk_production_basis()
    if basis is None:
        return [
            "the PRODUCTION Top-K D1 search cost. An adoption validation that "
            "computes BOTH reducers from the same forwards includes work the "
            "formal path will not do, and a per-candidate figure is not the "
            "whole-invocation quantity the cost table holds.",
            "and therefore the per-session envelope, which stays UNRESOLVED until "
            "a whole-invocation production measurement exists. The full-vocab "
            "provisional basis is NOT the Top-K price.",
        ]
    return [
        "nothing on the DEPTH path: its cell is a measured whole invocation under "
        "the production protocol, and the per-session envelope resolves on it.",
        "the three non-KL structural cells keep their committed UNBATCHED basis. "
        "That OVERSTATES, which is the safe direction for a ceiling, so no GPU "
        "time is owed for them -- refreshing them would lower a bound that is "
        "already satisfied.",
    ]


def gpu_validation_owed() -> dict[str, Any]:
    """What a GPU must answer before D1 executes, and what it must not re-ask.

    Scoped here rather than run, because D1 cannot execute while ANY of its
    open blockers stands — see `open_blockers` — and validating code for an
    experiment that cannot start is spending a paid resource on a question
    nothing is waiting for. AGENTS.md P8.2 asks a
    hardware request to state what gate it is intended to pass; this is that
    statement, for the maintainer to decide alongside the blockers.
    """
    return {
        **_qualification_state(),
        "why_a_gpu_is_required": (
            "the questions are CUDA numerical behaviour and real memory, which "
            "no CPU substitute reaches. AGENTS.md P8.2: a CPU rehearsal that "
            "cannot reach the behaviour under test is a more expensive way of "
            "learning nothing."),
        "what_the_cpu_round_ALREADY_settled": [
            "the incumbent policy is numerically inert: a four-operator toy "
            "chain rebuilt against the pre-change tree produces the same "
            "artifact digest and the same four structural decisions",
            "`distortion` unweighted is bit-identical to an inlined copy of the "
            "previous arithmetic at three chunk sizes",
            "a 0/1 weight equals subsetting the selected rows exactly, and "
            "`positions` and `weight` were driven apart so neither can stand in "
            "for the other unnoticed",
            "the batched state evaluation equals the reference path exactly on "
            "CPU float32, under both reference strategies and both policies",
            "handing `distortion` bf16 rows is bit-identical to pre-upcasting, "
            "which is what the memory bound depends on",
            "DEVICE PLACEMENT, via the `meta` device: every mask and weight is "
            "placed from the batch it describes rather than defaulting to the "
            "host. Meta performs no arithmetic, so it answers placement and "
            "nothing else — which is the only question it is asked.",
        ],
        "what_the_gpu_rounds_ANSWERED": {
            "full_vocab_qualification": {
                "status": "COMPLETE -- baseline engineering evidence",
                "owner": QUALIFICATION_CLOSEOUT,
                "answered": [
                    "the incumbent fixed path reconstructs the frozen incumbent "
                    "artifact identity on real CUDA, agreed across three pods",
                    "the real state-eval peak memory at the real 151,936 "
                    "vocabulary, and that the derived logit bound holds for what "
                    "it claims while understating the reduction's transients",
                    "that the CALIBRATION BATCH SIZE, not the position policy, "
                    "moves three of four fixed-path operator selections",
                    "the batched direction, which was UNKNOWN: batching costs "
                    "x1.97 on the dominant DEPTH operator at all positions and "
                    "x1.08 at the target-aware policy, so the policy's smaller "
                    "reduction very nearly cancels the batching penalty",
                ],
            },
            "topk_adoption": _topk_adoption_state(),
        },
        "what_is_not_priced": _what_is_not_priced(),
        "surface_that_owes_it": {
            "_what": ("files on the historically CUDA-validated surface that "
                      "this round changed, so the 2026-09-10 evidence does not "
                      "cover them. Owner: tests/architecture/"
                      "test_cuda_surface_preserved.py :: "
                      "HISTORICAL_SURFACE_CHANGED_PENDING_REVALIDATION."),
            "files": [
                "src/aadistill/initialization/planning/fixed_path.py",
                "src/aadistill/initialization/operators/attention/gqa/"
                "_statistics.py",
                "src/aadistill/initialization/operators/attention/gqa/"
                "activation_importance.py",
            ],
        },
        "cheapest_sufficient_shape": (
            "ONE short session on the smallest card that holds the teacher at "
            "the real vocabulary: run the incumbent fixed path once under the "
            "incumbent policy and gate the artifact digest against the frozen "
            "incumbent, then once under the target-aware policy at the D1 "
            "execution protocol, recording the state evaluation's peak against "
            "`batch_plan` and both runs' per-expansion timings. It needs no "
            "recovery training, no battery and no behavioural measurement."),
        "gate_it_is_intended_to_pass": (
            "that the incumbent policy still rebuilds the frozen incumbent "
            "digest on CUDA, and that the batched target-aware path runs inside "
            "its derived memory bound. Neither is a scientific result and "
            "neither authorizes D1."),
        "funding": (
            "the GPU engineering allowance, which is a different book from the "
            "formal one and does not transfer into it — so paying for this "
            "validation moves D1's funding blocker not at all. It is a "
            "PREREQUISITE of pricing rather than a consequence of funding: the "
            "chain figures cannot become authorization prices until it runs."),
    }


def contamination() -> dict[str, Any]:
    capacity = _load(CAPACITY)
    return {
        "requirement": (
            "D1/D2/D3 are an adaptive sequence. Each round's confirmation must "
            "rest on prompts that round's design has not been tuned against, "
            "and D2's and D3's must not reuse D1's."),
        "why_disjoint_seeds_are_not_enough": (
            "C0's inferential unit is the PROMPT and C0 measured substantial "
            "same-prompt cross-seed dependence: ICC 0.25 +/- 0.095, and "
            "P(correct | correct on another seed) = 0.257 against a 0.022 "
            "marginal, an 11.7x lift. Screening and confirming on the same "
            "prompts would let the selection leak into the confirmation through "
            "that dependence."),
        "batteries_D1_requires": 2,
        "batteries_available": capacity["batteries_remaining"],
        "binding_stratum": capacity["binding_stratum"],
        "short_by_items":
            capacity["strata"][capacity["binding_stratum"]]["short_by"],
        "owner": CAPACITY,
        "BLOCKER": (
            "ZERO further disjoint batteries can be drawn under the frozen C1 "
            "mixture. math_verified holds 70 eligible items of the 150 a "
            "battery needs: MATH-500's 500 rows are already committed to the "
            "five isolation roles, c1_confirmation_v1 and c2_screening_v1. D1 "
            "cannot be executed as designed until this is resolved, and neither "
            "can D2 or D3."),
        #: THE FAMILY OWNS THE RESOLUTION, and it corrects the first option
        #: below. See `logs/shared/analyses/autoinit_d_series_battery_family.json`.
        "d_series_battery_family": {
            "owner": "scripts/experiments/stage-1/phase_d_series/battery_family.py",
            "record": "logs/shared/analyses/autoinit_d_series_battery_family.json",
            "family_id": "d_series_behavioural_v1",
            "what": ("six roles - D1/D2/D3 x screening/confirmation - allocated "
                     "by ONE rule frozen before any D1 outcome exists, each "
                     "disjoint from the others and from every historical role "
                     "by stable id AND normalized prompt content. It carries "
                     "its own behavioural-distribution identity and is NOT the "
                     "c1_confirmation distribution."),
            "_it_corrects_the_first_option_below": (
                "extending math_verified alone unblocks D1 and leaves the "
                "FAMILY short. Six roles need 6x the mixture at once, and at "
                "that scale three strata are short rather than one: "
                "math_verified by 830 items, code by 321 and gsm8k by 11. The "
                "capacity record's 'zero batteries remaining, binding on "
                "math_verified' is the right answer to a different question."),
        },
        "resolutions_for_the_maintainer": [
            {"option": "extend the verified-math source",
             "what": ("draw the stratum from the full Hendrycks MATH test set "
                      "rather than its 500-problem verified subset, as a "
                      "`math_verified_v2` source with the same count and the "
                      "same sampling discipline"),
             "cost": ("a data decision: license, the fact that MATH-500 is a "
                      "curated subset, and a difficulty distribution that would "
                      "shift. `correct_overall` would remain a mean over the "
                      "same stratum BALANCE but over a different population, so "
                      "the SESOI's transfer needs an explicit argument."),
             "unblocks": ("potentially D1's two batteries. The full MATH test "
                          "set holds ~4,500 additional upstream candidate rows "
                          "BEFORE exclusions -- not 4,500 eligible items. How "
                          "many survive is an OWED MEASUREMENT: the source has "
                          "not been pinned and the exclusion/contamination "
                          "chain that produced the current pools (committed "
                          "roles, c1_confirmation_v1, c2_screening_v1, "
                          "near-duplicate and leakage screening) has not been "
                          "run against it. Until it is, `4,500` is an upstream "
                          "row count and the eligible count is unknown. NOT the "
                          "full six-role family either: see "
                          "`d_series_battery_family` above - code and gsm8k are "
                          "short too at six roles, and both are answerable from "
                          "files of repositories already pinned."),
             "owed_measurement": (
                 "run the exclusion/contamination chain against the pinned "
                 "full-MATH source and report the ELIGIBLE count per stratum. "
                 "Only that number says whether this option unblocks D1's two "
                 "batteries; the upstream row count does not."),},
            {"option": "reduce the math_verified count per battery",
             "what": "e.g. 70 instead of 150, with the other strata unchanged",
             "cost": ("changes the MIXTURE, and `correct_overall` and its SESOI "
                      "are DEFINED on the mixture. Every C1/C2/C3/A3 number "
                      "becomes incomparable; this is a C0-level redesign."),
             "unblocks": "one further battery, then exhausted again"},
            {"option": "reuse an existing battery for D1's confirmation",
             "what": "confirm on c1_confirmation_v1, as A3 did",
             "cost": ("the contamination the directive forbids. B was PROMOTED "
                      "on that battery and C2's and C3's negatives were "
                      "measured on it, so each further round's design is "
                      "informed by the previous round's numbers on the same "
                      "prompts. The leak is cumulative across rounds, not "
                      "within one."),
             "unblocks": "D1 only, at a recorded and growing selection risk"},
        ],
        "_not_a_resolution": (
            "drawing the screening battery from a different source than the "
            "confirmation battery. A screening delta informs a confirmation "
            "delta only if both are means over the same distribution, which is "
            "exactly why c2_screening_v1 preserves C1's mixture."),
    }


def _derived_budget() -> dict[str, Any]:
    """The live position, from the DERIVER rather than restated.

    `derive_budget.py` reads each attempt's own closeout and is the owner; a
    hand-copied balance expires the next time anything spends, and this file is
    generated often enough that it would expire quietly. Loaded by path because
    it is a script rather than a package module.
    """
    import importlib.util

    path = REPO / "scripts/consolidate/derive_budget.py"
    spec = importlib.util.spec_from_file_location("_derive_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.derive()


#: The DEPTH implementation whose cell is rebuilt.
DEPTH_IMPL = "depth.causal_kl_greedy_v1"

#: The two calibration profiles D1's DEPTH step may run under. Both must be timed
#: before a MAX exists: historical full-vocab evidence showed materially different
#: DEPTH timings between them, so one profile's max is not the cell's max.
D1_DEPTH_PROFILES = ("calib.domain_balanced@v1", "calib.reasoning_heavy@v2")

#: The measured PRODUCTION Top-K evidence. One owner, read not retyped.
#:
#: a8, which timed ONE REAL `impl.execute` invocation per DEPTH profile. a7 is NOT
#: this pointer: it timed a reimplemented candidate loop that omitted the position
#: weights, the `values.tolist()` host transfer, the per-subtype aggregation,
#: `domain_balanced_score` and the reference-cache fill, and `candidates x
#: per-candidate` is not the quantity `_expand_one` records as `operator_seconds`.
#: `_topk_production_basis` refuses a record without `measurement_path`, so
#: repointing this at a5, a6 or a7 yields no basis rather than a wrong price.
TOPK_PRODUCTION = ("logs/stages/stage-1/phase_d1/validations/topk-adoption/v1/"
                   "runs/a8/adoption.json")


def non_operator_expansion_overhead() -> dict[str, Any] | None:
    """The non-operator part of one DEPTH expansion, at its committed MAX.

    Derived from the SAME telemetry and the SAME root/deeper resolution the
    published cost table uses -- `full_search_space.non_operator_observations` --
    so a rebuilt cell cannot disagree with the table about which observations are
    roots. No GPU time is spent on phases already in committed telemetry.

    MAX, not mean: the cell this feeds is a ceiling, and `CostModel`'s own
    docstring says a ceiling built on a mean is not a ceiling.
    """
    from experiments.phase_c2.full_search_space import non_operator_observations

    obs = non_operator_observations(REPO)
    out: dict[str, Any] = {"_derived_from": ("the committed search telemetry, via "
                                            "full_search_space."
                                            "non_operator_observations")}
    for scope in ("root", "deeper"):
        values = obs.get((DEPTH_IMPL, scope)) or []
        if not values:
            return None
        out[f"{scope}_max"] = round(max(values), 4)
        out[f"{scope}_n"] = len(values)
        out[f"{scope}_mean"] = round(sum(values) / len(values), 4)
    out["_phases"] = ("parent_load, materialize, identify, canonical_reload, "
                      "validation, state_evaluation -- everything an "
                      "operator-only figure omits")
    return out


def a_deeper_parent_is_a_smaller_model() -> dict[str, Any]:
    """Why a ROOT invocation prices the deeper DEPTH cell too.

    MECHANICAL, not asserted, and generic -- no model family appears in it. A
    deeper parent has already had other structural operators applied, and every
    operator in the frozen set only SHRINKS the geometry it modifies, so its DEPTH
    invocation runs a smaller model than the root's.

    This needs establishing rather than reasoning about, because it is a property
    of the operator/adapter contract rather than of this operator.
    `tests/initialization/test_operators_only_shrink.py` applies every frozen
    operator to a toy parent and asserts no structural field increases.

    **It used to claim more than this, and no longer needs to.** While the
    operator term was `candidates x per-candidate-max` measured in round 0, two
    further premises were load-bearing -- that a growing skip set executes
    non-increasingly many blocks, and that round 0 evaluates the most candidates.
    The operator term is now ONE MEASURED INVOCATION covering every round, so
    those two premises price nothing. They remain true and remain tested; they are
    simply no longer part of this claim. Keeping them here would be a derivation
    asserting more than its consumer needs.
    """
    return {
        "claim": ("an invocation measured at the ROOT parent bounds the same "
                  "operator at any deeper parent"),
        "because": ("every frozen operator only shrinks the geometry it "
                    "modifies, so a deeper parent runs a smaller model"),
        "established_by": "tests/initialization/test_operators_only_shrink.py",
        "_not_a_family_shortcut": (
            "checked against the operator/adapter contract on a toy geometry, not "
            "asserted for one model family, and nothing about it lives in "
            "src/aadistill."),
        "_no_longer_claimed_because_nothing_needs_it": (
            "that round 0 bounds the later rounds. The priced operator term is a "
            "whole invocation over all rounds, measured."),
    }


import contextlib


def _price_status_detail() -> str:
    """Cell by cell: what is measured, what is a planning basis, what is owed.

    DERIVED from the production basis, like the three fields it has to agree with.
    It used to end "STILL NOT MEASURED: the PRODUCTION Top-K cost ... WHAT MUST
    REFRESH IT: a Top-K-only representative expansion timing. Until that exists
    these numbers size a grant request; they do not price one" -- in the same
    `budget` object as a `price_status` reading "MEASURED PRODUCTION TOP-K BASIS".
    One object, two answers, and the detailed one was the stale one.
    """
    batched = (
        "MEASURED, by the full-vocabulary GPU qualification: the batched "
        "direction, which this field once called UNKNOWN. Batching costs x1.97 on "
        "the dominant DEPTH operator at all positions and x1.08 at the "
        "target-aware policy, so the policy's smaller reduction very nearly "
        "cancels the penalty. Against the frozen per-cell table it does NOT bound "
        "the all-positions bsz=3 case (39.117 over a 34.354 maximum) and DOES "
        "bound the protocol D1 runs (21.420 under a 29.248 mean).")
    live_quote = (
        "AND `securePrice` is re-queried live at authorization, so every dollar "
        "figure here is a derived consequence of an hour-old quote.")
    basis = _topk_production_basis()
    if basis is None:
        return (
            "every figure in `chain` is a PROVISIONAL PLANNING BASIS, NOT a "
            f"finalized authorization price.\n\n{batched}\n\n"
            "NOT MEASURED: the PRODUCTION Top-K cost. These per-expansion minutes "
            "come from FULL-VOCABULARY telemetry and D1 runs "
            "reference_topk_tail_v1. An adoption validation's wall clock is not a "
            "substitute -- it computes BOTH reducers from the same forwards -- and "
            "neither is a per-candidate figure, because the cost table holds "
            f"whole-invocation `operator_seconds`.\n\n{live_quote}\n\n"
            "WHAT WOULD REFRESH IT: one timed `impl.execute` invocation per DEPTH "
            "calibration profile under the production protocol. Until that exists "
            "these numbers size a grant request; they do not price one.")
    return (
        "MIXED, and the dominant cell is MEASURED. The Top-K protocol owes "
        "nothing: its production cost is settled evidence.\n\n"
        f"MEASURED, under the production protocol: {DEPTH_IMPL}. One real "
        f"`impl.execute` invocation of {basis['operator_invocation_seconds_max']} s "
        f"over {basis['candidate_subsets_measured']} candidate subsets -- the max "
        f"over both DEPTH calibration profiles -- rebuilt end to end into "
        f"{basis['root_max_minutes']} min at the root with the committed "
        f"non-operator MAX. This is the same quantity `BeamSearch._expand_one` "
        f"records as `operator_seconds`, so it is comparable with the frozen "
        f"table it replaces.\n\n"
        "A PLANNING BASIS, deliberately left alone: the three non-KL structural "
        "cells keep their committed UNBATCHED whole-expansion figures. They reduce "
        "activation statistics rather than KL, so the distribution support cannot "
        "move them, and an unbatched basis OVERSTATES -- which is the safe "
        f"direction for a ceiling.\n\n{batched}\n\n{live_quote}\n\n"
        "SO: the search session's figure is a BOUND rather than a plan. The chain "
        "total remains provisional in SIZE because three cells are unmeasured, and "
        "it is not a finalized authorization price -- what the project cap should "
        "become is a maintainer decision this document does not take.")


def _measured_depth_cell() -> dict[str, Any] | None:
    """The rebuilt DEPTH cell, or None while no production measurement exists."""
    basis = _topk_production_basis()
    if basis is None:
        return None
    return {"root_max": basis["root_max_minutes"],
            "deeper_max": basis["deeper_max_minutes"],
            "operator_minutes": basis["operator_minutes_at_max"],
            "invocation_seconds": basis["operator_invocation_seconds_max"]}


def _batching_decision() -> str:
    """The batching decision, with its measurement status DERIVED.

    This used to end "the direction of the correction is unknown, so the figure
    sizes a grant request rather than capping one; only the owed GPU qualification
    can turn it into a price". Both halves became false — the qualification ran and
    the DEPTH cell is now a measured invocation — and a sentence that describes an
    owed measurement after it has been taken is how a reader concludes a design is
    less ready than it is.
    """
    head = (
        "maintainer engineering decision after A3: batched execution is adopted "
        "across every batchable path from D1 onward, not only ATTENTION. A3 "
        "measured that it is NOT an optimization on "
        "attention.activation_importance_v1 -- 8-10% slower on the scorer with no "
        "detectable correctness effect -- so this is a uniformity decision rather "
        "than a performance claim. ")
    cell = _measured_depth_cell()
    if cell is None:
        return head + (
            "The search ceiling below is derived from UNBATCHED telemetry, which "
            "makes it a PROVISIONAL PLANNING BASIS and NOT a bound: unbatched "
            "timing does not upper-bound the batched implementation in either "
            "direction, and the net effect on an expansion running all four "
            "operators is unmeasured. The figure sizes a grant request rather "
            "than capping one until a production measurement exists.")
    return head + (
        f"THE DEPTH CELL IS NOW MEASURED, batched and under the production "
        f"protocol: one real `impl.execute` invocation of "
        f"{cell['invocation_seconds']} s ({cell['operator_minutes']} min), the max "
        f"of both calibration profiles, rebuilt end to end into "
        f"{cell['root_max']} min at the root. The other three cells keep their "
        "unbatched committed basis, which overstates rather than understates, so "
        "the search session's price is a bound and not a plan.")


def _cost_basis_status() -> str:
    """Which cells are measured and which are a planning basis. Derived."""
    cell = _measured_depth_cell()
    tail = (
        "A3 measured the ATTENTION scorer 8.2-10.0% SLOWER at batch 3 on three "
        "separate pods with the sign never flipping, at +29.6% peak VRAM, while "
        "length-sorted packing won 1.1884x on causal-KL. Batching moves different "
        "operators in different directions, so no cell is adjusted on a predicted "
        "speed-up: a ceiling derived from a prediction is a prediction.")
    if cell is None:
        return ("PROVISIONAL. Every cell of the per-expansion table was measured "
                "with an UNBATCHED state evaluation and a one-item-per-forward "
                "statistics pass, and D1 runs both batched. " + tail)
    return (f"MIXED, and the dominant cell is MEASURED. {DEPTH_IMPL} is "
            f"{cell['root_max']} min at the root, rebuilt from one real batched "
            f"`impl.execute` invocation under the production protocol. The other "
            f"three cells keep their UNBATCHED committed basis: they are a "
            f"planning basis that overstates, which is the safe direction for a "
            f"ceiling. " + tail)


def _rebuilt_cost_model():
    """The frozen cost model with the DEPTH cell rebuilt, or None if unmeasured.

    One builder, called by the session price and by the whole-chain price, so the
    two cannot be computed from different models. It is deliberately NOT a record
    field: a `CostModel` is not serializable, and what a reader needs -- the cell
    before, the cell after and which cells were left alone -- is recorded already.
    """
    import dataclasses

    from experiments.phase_d1 import search_space as d1

    basis = _topk_production_basis()
    if basis is None:
        return None
    _ensure_the_frozen_operators_are_registered()
    frozen = d1.cost_model()
    minutes = {k: dict(v) for k, v in frozen.minutes.items()}
    before = dict(minutes[DEPTH_IMPL])
    rebuilt = {"root_max": basis["root_max_minutes"],
               "deeper_max": basis["deeper_max_minutes"],
               #: THE MAX IN BOTH SLOTS: an expected figure must not weaken a
               #: ceiling, and two invocations is not a distribution.
               "root_mean": basis["root_max_minutes"],
               "deeper_mean": basis["deeper_max_minutes"]}
    minutes[DEPTH_IMPL] = {k: rebuilt[k] for k in before if k in rebuilt}
    return dataclasses.replace(
        frozen, minutes=minutes,
        source=(f"{frozen.source} -- with {DEPTH_IMPL} REBUILT END TO END as ONE "
                f"MEASURED production Top-K operator invocation "
                f"({basis['operator_invocation_seconds_max']} s over "
                f"{basis['candidate_subsets_measured']} candidate subsets, the max "
                f"of both DEPTH profiles) PLUS the committed non-operator MAX for "
                f"the scope, from {TOPK_PRODUCTION}"))


def _evidence_authorizes_note() -> str:
    """What the behavioural evidence does and does not settle.

    It said "the funding scope and the per-session envelope are separate and still
    open" -- true when both were. The envelope resolved on measurement, so the
    sentence survived its own subject and described a blocker that had closed.
    """
    #: NO `budget()` CALL HERE. `budget()` reaches `evidence()`, so asking it what
    #: is open is a cycle -- it hung the writer for ten minutes before a
    #: faulthandler dump named it. The derived list already exists, once, in
    #: `open_blockers`; this field points at it instead of restating it, which is
    #: how it came to describe a blocker that had closed.
    return ("nothing. The evidence exists and settles only that the behavioural "
            "family can be built. Every other precondition is separate and is "
            "listed, derived, in `open_blockers`.")


def _shortfall_note(chain: dict[str, Any], project_remaining: float) -> str:
    """The gap between the chain ceiling and the live balance, and WHAT it is.

    Derived, because the answer changed. While every cell came from unbatched
    telemetry the gap was a planning figure whose direction was unknown. The DEPTH
    cell is now a measured batched invocation and the other three keep a basis that
    OVERSTATES, so the chain ceiling is a bound -- and the gap is the minimum a
    grant would have to cover, not a number whose sign is unknown.
    """
    gap = round(chain["hard_ceiling_usd"] - project_remaining, 4)
    cell = _measured_depth_cell()
    if cell is None:
        return (f"THE SHORTFALL IS PROVISIONAL. `provisional_shortfall_usd` = "
                f"${gap} is the gap between a planning basis derived from "
                "UNBATCHED telemetry and the live balance. It is NOT the "
                "finalized amount by which the project cap must increase: the "
                "direction of the batched correction is unknown. This document "
                "does not claim the cap must move by that amount.")
    return (f"THE SHORTFALL'S DIRECTION IS NOW KNOWN AND ITS SIZE IS STILL "
            f"PROVISIONAL. `provisional_shortfall_usd` = ${gap} is the gap between "
            f"the chain hard ceiling of ${chain['hard_ceiling_usd']} and the live "
            f"balance. The dominant cell is MEASURED -- {DEPTH_IMPL} at "
            f"{cell['root_max']} min from one real batched invocation -- and the "
            "three unmeasured cells keep an unbatched basis that OVERSTATES, so "
            "the ceiling is a bound and a grant smaller than this gap cannot fund "
            "the chain. It remains PROVISIONAL because those three cells are not "
            "measured, so it is NOT the finalized amount by which the project cap "
            "must increase: this document does not claim what the cap should "
            "become, which is a maintainer decision.")


@contextlib.contextmanager
def _measured_cost_model(adjusted):
    """Price through the EXISTING machinery with the rebuilt DEPTH cell in place.

    A context manager rather than two copies of the swap, because the search
    session and the whole chain must be priced on ONE cost model: a chain summed
    from a frozen-model screening rung and a measured-model search session is
    still one number, and nothing would say which model produced it.
    """
    from experiments.phase_d1 import search_space as d1

    original = d1.cost_model
    d1.cost_model = lambda *a, **k: adjusted
    try:
        yield
    finally:
        d1.cost_model = original


def _topk_production_basis() -> dict[str, Any] | None:
    """The measured Top-K DEPTH cell, END TO END, or None if not yet valid.

    **An operator-only number is not an expansion cell, and a mean is not a
    ceiling.** Both of those were wrong here and an independent review caught
    them:

    * `CostModel` defines every cell as ONE EXPANSION end to end -- operator,
      parent load, materialize, identify, canonical reload, validation and state
      evaluation. The first version of this function replaced the whole DEPTH cell
      with the operator's candidate work alone, which DELETED about 3.07 min of
      non-operator cost per expansion. The comment claiming those phases stayed in
      the frozen cell was false: the cell was replaced entire.
    * `CostModel`'s own docstring says a ceiling built on `mean` is not a ceiling.
      a5 timed 12 round-0 candidates and stored `total / 12`, and that mean was
      promoted into `root_max` and `deeper_max`.

    So this now REFUSES to produce a basis unless the record carries what a
    ceiling needs: a per-candidate MAX, measured for BOTH D1 DEPTH calibration
    profiles, plus the non-operator overhead derived from committed telemetry. Any
    older record -- a5's included -- yields None, and the envelope is therefore
    UNRESOLVED rather than resolved on an invalid figure.
    """
    path = REPO / TOPK_PRODUCTION
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    P = doc.get("P_production_timing") or {}
    profiles = P.get("per_profile") or {}
    #: WHAT A VALID CEILING NEEDS, and the marker is the first of them. A record
    #: without `measurement_path == "operator_execute_v1"` came from the
    #: superseded shadow loop, which reimplemented the candidate inner loop and so
    #: never paid the position weights, the `values.tolist()` host transfer, the
    #: per-subtype aggregation, `domain_balanced_score` or the reference-cache
    #: fill. Those records -- a5's, a6's and a7's -- yield None here, and the
    #: envelope is therefore UNRESOLVED rather than resolved on a shadow.
    required = {"per_profile", "measurement_path",
                "operator_invocation_seconds_max"}
    if not required <= set(P) or not profiles:
        return None
    if P.get("measurement_path") != "operator_execute_v1":
        return None
    #: A SYNCED RUN CANNOT PRICE. Splitting forward from reduction needs a
    #: synchronize after each phase, which inflates the per-candidate totals --
    #: the DEPTH operator's own source says so. Such a record describes the split
    #: and must not reach a ceiling.
    #: FAIL CLOSED: the field must be PRESENT and true. A record that does not
    #: say whether it synced cannot be trusted to price -- a6's does not say, and
    #: a6 synced unconditionally.
    if P.get("_valid_for_pricing") is not True or P.get("sync_split_enabled"):
        return None
    if set(profiles) != set(D1_DEPTH_PROFILES):
        return None
    overhead = non_operator_expansion_overhead()
    if overhead is None:
        return None

    #: ONE MEASURED INVOCATION, not a candidate count times a per-candidate max.
    #: `BeamSearch._expand_one` records `operator_seconds` around
    #: `impl.execute(ctx)`, so a whole invocation IS the quantity the committed
    #: cost table holds -- packing, the reference-cache fill, every candidate of
    #: every round, the greedy bookkeeping and the child construction. Nothing is
    #: extrapolated and no fixed term has to be added back.
    operator_minutes = float(P["operator_invocation_seconds_max"]) / 60.0
    per_profile_invocations = {
        k: v.get("operator_invocation_seconds") for k, v in profiles.items()}
    candidates = max((int(v.get("candidate_subsets") or 0)
                      for v in profiles.values()), default=0)
    if not candidates:
        return None
    return {
        "owner": TOPK_PRODUCTION,
        "operator_invocation_seconds_max": float(
            P["operator_invocation_seconds_max"]),
        "operator_invocation_seconds_per_profile": per_profile_invocations,
        "candidate_subsets_measured": candidates,
        "per_profile": profiles,
        "operator_minutes_at_max": round(operator_minutes, 4),
        "non_operator_overhead_minutes": overhead,
        "root_max_minutes": round(operator_minutes + overhead["root_max"], 4),
        "deeper_max_minutes": round(operator_minutes + overhead["deeper_max"], 4),
        "_the_cell_is_END_TO_END": (
            "ONE MEASURED OPERATOR INVOCATION at the max over both DEPTH "
            "calibration profiles, plus the non-operator phases of one expansion "
            "at their committed-telemetry MAX. The operator term is not "
            "`candidates x per-candidate`: that product omits the reference-cache "
            "fill, the packing, the greedy bookkeeping and the child "
            "construction, all of which historical `operator_seconds` includes."),
        "_why_the_root_bounds_the_deeper_cell": (
            "see `bounds_the_deeper_parents`. The whole invocation is measured, "
            "so nothing rests on round 0 bounding the later rounds any more."),
    }


def _envelope_compatibility(topk: dict[str, Any] | None,
                            envelope_usd: float) -> str:
    """`RESOLVED_FITS` / `RESOLVED_NEEDS_RAISE` / `UNRESOLVED`, from measurement.

    Three states and no fourth. A measured session price inside the envelope
    resolves it; outside, the envelope would have to move; and with no production
    Top-K timing there is nothing to resolve it with.
    """
    if topk is None:
        return "UNRESOLVED"
    priced = topk["search_session"]["hard_ceiling_usd"]
    return "RESOLVED_FITS" if priced <= envelope_usd else "RESOLVED_NEEDS_RAISE"


def topk_search_cost() -> dict[str, Any] | None:
    """The SEARCH session priced with a correctly rebuilt END-TO-END DEPTH cell.

    Runs the EXISTING cost machinery -- `search_space.search_cost`, which calls
    `search_cost_model.bound` -- against a cost model whose DEPTH cell is
    rebuilt, and implements no second pricing formula.

    THE CELL IS END TO END, which is the correction an independent review
    required. `CostModel` defines a cell as one whole expansion: operator, parent
    load, materialize, identify, canonical reload, validation, state evaluation.
    So the rebuilt cell is

        measured Top-K operator MAX  +  committed non-operator MAX for that scope

    and the earlier version -- which replaced the whole cell with the operator
    term alone -- deleted about 3.07 min of real cost from every root expansion.

    The other three cells are untouched. They reduce activation statistics rather
    than KL, so the distribution support cannot move them, and keeping their
    frozen values overstates rather than understates.

    EXPECTED pricing uses the MAX too. An expected-cost figure is not allowed to
    weaken an authorization ceiling, and no justified Top-K mean exists: the
    measured distribution is over round-0 candidates at the root, which is the
    bound rather than the average case.
    """
    basis = _topk_production_basis()
    if basis is None:
        return None
    from experiments.phase_c2.search_space import PRICE_PER_HOUR_LAST_QUOTED
    from experiments.phase_d1 import search_space as d1

    _ensure_the_frozen_operators_are_registered()
    before = dict(d1.cost_model().minutes[DEPTH_IMPL])
    adjusted = _rebuilt_cost_model()
    with _measured_cost_model(adjusted):
        priced = d1.search_cost(price_per_hour=PRICE_PER_HOUR_LAST_QUOTED)
    return {
        "basis": basis,
        "bounds_the_deeper_parents": a_deeper_parent_is_a_smaller_model(),
        "depth_cell_before": before,
        "depth_cell_after": dict(adjusted.minutes[DEPTH_IMPL]),
        "search_session": priced,
        "_priced_by": ("experiments.phase_d1.search_space.search_cost, which "
                       "calls search_cost_model.bound. No second pricing formula "
                       "exists here."),
        "_other_three_cells_unchanged": (
            "ffn.activation_importance_v0, width.global_pca_v0 and "
            "attention.activation_importance_v1 reduce activation statistics, "
            "not KL, so the distribution support cannot move them."),
    }


def budget() -> dict[str, Any]:
    from experiments.phase_d1 import search_space as d1
    from experiments.phase_c2.search_space import PRICE_PER_HOUR_LAST_QUOTED

    #: `chain_cost` searches the frozen space, so the operators must be there.
    _ensure_the_frozen_operators_are_registered()

    design = behavioural_design()
    #: The measured production Top-K basis, or None before it is measured.
    topk = topk_search_cost()
    #: THE WHOLE CHAIN ON THE MEASURED MODEL, through `chain_cost` -- which calls
    #: the same `search_cost` the session price came from, and leaves the screening
    #: and confirmation rungs exactly as they were, since a behavioural probe
    #: reduces no KL and a distribution support cannot move it. Derived, never
    #: typed: the chain total is `sum(session hard ceilings)` computed by
    #: `chain_cost`, not a figure restated here.
    if topk is not None:
        with _measured_cost_model(_rebuilt_cost_model()):
            chain = d1.chain_cost(
                screening_probes=design["screening_probes"],
                confirmation_probes=design["confirmation_probes"],
                screening_arms=design["screening_arms"],
                confirmation_arms=design["confirmation_arms"])
    else:
        chain = d1.chain_cost(
            screening_probes=design["screening_probes"],
            confirmation_probes=design["confirmation_probes"],
            screening_arms=design["screening_arms"],
            confirmation_arms=design["confirmation_arms"])
    terms = _load(BUDGET_TERMS)["execution_package"]
    pricing = _load(BUDGET_TERMS)["accepted_pricing"]
    live = _derived_budget()
    project_remaining = float(live["project"]["remaining_usd"])
    funded = tuple(terms.get("funds_formal_sessions_of", {})
                   .get("experiment_ids", ()))
    return {
        "price_basis": {
            "usd_per_hour": PRICE_PER_HOUR_LAST_QUOTED,
            "gpu": "NVIDIA L40S securePrice",
            "_a_launch_requotes": (
                "an hour-old price is not a price. securePrice is re-queried "
                "live immediately before authorization and every ceiling "
                "re-derived from it."),
        },
        "chain": chain,
        "price_status": (
            "MEASURED PRODUCTION TOP-K BASIS for the search session: "
            f"${topk['search_session']['hard_ceiling_usd']:.4f} hard ceiling from "
            f"an END-TO-END DEPTH cell whose operator term is ONE MEASURED "
            f"`impl.execute` invocation "
            f"({topk['basis']['operator_minutes_at_max']:.4f} min, the max of both "
            f"DEPTH profiles) plus the committed non-operator MAX. The chain's "
            "other sessions keep their full-vocab basis."
            if topk else
            "PROVISIONAL FULL-VOCAB BASIS -- NOT THE TOP-K PRICE. The batched "
            "direction is now MEASURED; the production Top-K cost is not. A "
            "per-candidate measurement does not supply it: the cost table holds "
            "whole-invocation `operator_seconds`."),
        "_price_status": _price_status_detail(),
        "sessions": 3,
        "_why_three_sessions": (
            "the search commits a candidate set and stops; the screening rung "
            "cannot be bound until that set exists, and the confirmation rung "
            "cannot be bound until screening names the one candidate that "
            "advances. Each is separately priced and separately authorized."),
        "per_session_envelope_usd": terms["per_attempt_hard_ceiling_usd"],
        #: PROVISIONAL IN THE NAME, not only in a docstring. These compare a
        #: planning basis derived from FULL-VOCABULARY telemetry against a real
        #: envelope, while D1 will run `reference_topk_tail_v1` -- so the
        #: comparison is against the wrong protocol, not merely an unmeasured
        #: direction. A field called `fits_per_session_envelope` would be read
        #: later as a finalized authorization fact; this one cannot be.
        "provisional_basis_fits_per_session_envelope":
            chain["max_session_hard_ceiling_usd"]
            <= terms["per_attempt_hard_ceiling_usd"],
        "provisional_per_session_excess_usd": round(
            chain["max_session_hard_ceiling_usd"]
            - terms["per_attempt_hard_ceiling_usd"], 4),
        "per_session_envelope_compatibility": _envelope_compatibility(
            topk, terms["per_attempt_hard_ceiling_usd"]),
        "topk_production_basis": topk,
        "_per_session_envelope_compatibility": (
            "DERIVED from the measured production Top-K session price against "
            f"${terms['per_attempt_hard_ceiling_usd']:.2f}. RESOLVED_FITS means "
            "the measured session ceiling is inside the envelope; "
            "RESOLVED_NEEDS_RAISE means it is outside and the envelope would have "
            "to move; UNRESOLVED means no production Top-K timing exists yet, "
            "which blocks authorization because authorization needs a figure it "
            "can bind -- not because incompatibility is proven.\n\n"
            "THE PROVISIONAL FULL-VOCAB FIGURES BELOW DO NOT DECIDE THIS. They "
            "are historical planning evidence from UNBATCHED telemetry for a "
            "protocol D1 will not run. `open_blockers` used to require both this "
            "field AND that the provisional basis fit, so a measurement proving "
            "D1 fits would have stayed blocked by an estimate it supersedes."),
        #: DERIVED from the live compatibility state. It used to assert
        #: UNRESOLVED unconditionally, which contradicts the measured value
        #: whenever one exists.
        "_SECOND_BLOCKER_THE_PER_SESSION_CEILING": _envelope_blocker_note(
            budget_section_compatibility=_envelope_compatibility(
                topk, terms["per_attempt_hard_ceiling_usd"]),
            envelope_usd=terms["per_attempt_hard_ceiling_usd"],
            topk=topk),
        "position": {
            "project_cap_usd": float(live["project"]["cap_usd"]),
            "project_remaining_usd": project_remaining,
            "formal_remaining_usd": float(live["formal"]["remaining_usd"]),
            "engineering_remaining_usd":
                float(live["engineering"]["remaining_usd"]),
            "package_remaining_usd": float(live["package"]["remaining_usd"]),
            "full_ceiling_sessions_fundable":
                live.get("full_ceiling_sessions_fundable"),
            "_derived_by": ("scripts/consolidate/derive_budget.py, CALLED by "
                            "this writer rather than copied from it. A "
                            "hand-copied balance expires the next time "
                            "anything spends, and this document is "
                            "regenerated often enough that it would expire "
                            "quietly."),
            "_cap_cross_check": pricing["cumulative_cap_usd"],
        },
        #: PROVISIONAL: the difference between a planning basis and the live
        #: balance. NOT the finalized amount by which the project cap must
        #: increase -- that figure does not exist until the GPU qualification
        #: reprices the chain.
        "provisional_shortfall_usd": round(
            chain["hard_ceiling_usd"] - project_remaining, 4),
        "funds_formal_sessions_of": list(funded),
        "d1_is_in_the_funded_list": "phase_d1" in funded,
        "BLOCKER": (
            "D1 is NOT FUNDABLE, and the DEFINITE reason is authorization scope "
            "rather than arithmetic: `phase_d1` is not in the C1 execution "
            "package's `funds_formal_sessions_of` list, so NO existing "
            "allowance covers it at any price. A maintainer grant is required "
            "for the phase. That alone blocks D1 and does not depend on any "
            "cost estimate.\n\n"
            + _shortfall_note(chain, project_remaining)),
        "d_series_extrapolation": {
            "_what": ("D2 and D3 repeat this shape by the maintainer's "
                      "instruction — each a full search, freeze, recovery, "
                      "behavioural evaluation and promotion decision."),
            "three_chains_hard_ceiling_usd":
                round(chain["hard_ceiling_usd"] * 3, 4),
            "_caveat": ("an extrapolation, not a price. D2 adds a reference "
                        "forward per operator-local comparison and D3 adds a "
                        "student-confidence component; neither is measured, so "
                        "neither is priced here. The figure is the floor of the "
                        "D-series, not its cost."),
        },
    }


#: The three constraints that independently prevent a D1 launch, each named by
#: the section that owns its figures. DERIVED, not counted: this document said
#: "two blockers" and "either blocker" for a round after the per-session ceiling
#: became the third, because the count was prose in four places while the facts
#: lived in `budget` and `contamination_protection`. A fourth blocker now adds
#: one entry here and every statement follows.
#: The realized D-series family D1's behavioural evidence comes from. Read, not
#: assumed: the evidence blocker closes because these roles EXIST and are
#: verified, and it would reopen if the manifest vanished.
FAMILY_MANIFEST = "logs/shared/analyses/autoinit_d_series_family_manifest.json"

#: The two roles D1 itself consumes. D2 and D3 own the other four; D1's evidence
#: readiness must not depend on roles it never reads.
D1_ROLES = ("d1_screening", "d1_confirmation")


def d_series_evidence() -> dict[str, Any]:
    """Whether D1's behavioural evidence exists, from the realized family.

    **This replaces the capacity question.** `contamination_protection` asked how
    many further batteries the frozen C1 mixture could support and answered zero,
    which was the right question while no D-series family existed and the sources
    were undecided. The family is built now, so the live question is whether the
    roles D1 consumes exist and are identified -- and the old capacity analysis is
    kept below as the historical reasoning that led to extending the sources,
    where it no longer drives a blocker.
    """
    path = REPO / FAMILY_MANIFEST
    if not path.is_file():
        return {
            "status": "OPEN",
            "why": (f"{FAMILY_MANIFEST} does not exist, so D1 has no fresh "
                    "disjoint behavioural battery to confirm on."),
            "roles_required": list(D1_ROLES),
            "roles_available": [],
        }
    doc = json.loads(path.read_text())
    built = sorted(doc.get("roles", {}))
    missing = [role for role in D1_ROLES if role not in built]
    if missing:
        return {
            "status": "OPEN",
            "why": f"the realized family is missing {missing}",
            "roles_required": list(D1_ROLES),
            "roles_available": built,
        }
    return {
        "status": "CLOSED",
        "why": ("the realized D-series family provides both roles D1 consumes, "
                "built before any D1 outcome, pairwise disjoint and isolated "
                "from every historical reserved population. Closed on "
                "independent maintainer review 2026-10-03."),
        "roles_required": list(D1_ROLES),
        "roles_available": built,
        "allocation_rule_id": doc.get("allocation_rule_id"),
        "family_content_id": doc.get("family_content_id"),
        "construction_commit": (doc.get("code_state") or {}).get("git_commit"),
        "owner": FAMILY_MANIFEST,
        "_authorizes": _evidence_authorizes_note(),
    }


BLOCKER_SPECS: tuple[tuple[str, str], ...] = (
    ("evidence", "d_series_evidence"),
    ("funding authorization", "budget.BLOCKER"),
    ("per-session envelope", "budget._SECOND_BLOCKER_THE_PER_SESSION_CEILING"),
    #: A FOURTH, and it is the only one about the SCIENCE rather than the money
    #: or the evidence supply.
    #:
    #: The design's control identity was HAND-TYPED until 2026-10-08 and named
    #: C1's INCUMBENT arm -- `attention.weight_proxy_v0` at `c313d1b4` -- which
    #: is the arm C1 measured and BEAT. C1 returned GO at +0.013725 against a
    #: SESOI of 0.010, so the checkpoint that stands is C1's TREATMENT at
    #: `53e30566`, and C2's four frozen B constants and C3's stage-E gate both
    #: said so throughout. The consequence was not cosmetic: a candidate
    #: measured against the beaten arm carries C1's already-banked effect on top
    #: of its own, and that effect is LARGER than the SESOI the decision rule
    #: tests against -- every candidate looks better than it is, by more than
    #: the amount that decides.
    #:
    #: CORRECTED on the maintainer decision of 2026-10-08, and the blocker is
    #: KEPT rather than deleted. `incumbent()` now derives the identity, so this
    #: check compares a derivation against the same derivation and should stay
    #: closed -- which is exactly why it is worth leaving armed: it is the thing
    #: that would notice if the design ever went back to stating the identity,
    #: or if a later round promoted something else and this document did not
    #: follow.
    ("incumbent identity", "phase_d_series.incumbent.standing_incumbent"),
)


def declared_incumbent() -> dict[str, Any]:
    """What this design declares its control arm to be.

    Read from the emitted block rather than from constants, because there are
    no constants any more: `incumbent()` derives it. Kept as a separate
    function so `incumbent_identity_check` compares the DOCUMENT against the
    owner rather than comparing the owner with itself -- a committed design can
    be edited after it is written, and the check should notice.
    """
    return {k: v for k, v in incumbent().items() if not k.startswith("_")}


def incumbent_identity_check() -> dict[str, Any]:
    """Does the declared control arm agree with the one C1's verdict selected?

    DERIVED on both sides. The standing incumbent comes from C1's recorded
    verdict and C1's MEASURED arm identities, so this closes when the design is
    corrected and not when someone edits a sentence -- and it cannot be closed
    by restating the identity, which is how the wrong one got here.
    """
    try:
        from experiments.phase_d_series.incumbent import (
            IncumbentUndetermined, disagreements, standing_incumbent,
        )
    except ImportError as exc:                                 # noqa: BLE001
        return {"status": "UNRESOLVED",
                "why": f"the standing-incumbent owner is not importable: {exc}"}
    declared = declared_incumbent()
    try:
        standing = standing_incumbent()
        differ = disagreements(declared)
    except IncumbentUndetermined as exc:
        #: UNRESOLVED, not AGREES. "Which arm stands cannot be derived" is a
        #: different finding from "the declared arm is right", and defaulting
        #: the first to the second is exactly how an unchecked identity passes.
        return {"status": "UNRESOLVED", "why": str(exc), "declared": declared}
    return {
        "status": "AGREES" if not differ else "DISAGREES",
        "declared": declared,
        "standing": {k: standing[k] for k in (
            "c1_arm", "verdict", "impl_id", "profile_id", "artifact_digest",
            "weights_digest", "single_shard_sha256", "arch_signature")},
        "disagreements": differ,
        "owner": "scripts/experiments/stage-1/phase_d_series/incumbent.py",
        "finding": ("logs/stages/stage-1/phase_d1/analyses/"
                    "d1_control_arm_identity.json"),
        "why_it_matters": (
            "C1's measured delta between these two arms is +0.013725 and the "
            "D-series SESOI is 0.010. A candidate compared against the arm C1 "
            "beat is credited with that delta on top of its own effect, which "
            "is more than the amount the decision rule tests for -- so the "
            "error does not merely add noise, it can manufacture a GO."),
        "_corroborated_by": [
            "phase_c2.baseline.B_ARTIFACT_DIGEST and its three sibling "
            "constants, all four equal to the derived standing identity",
            "the C3 stage-E gate, `GATE: incumbent == 53e30566...`",
            "logs/state/current.json :: accepted_incumbent, 'B = frozen C1 "
            "treatment'",
        ],
    }


def open_blockers(budget_section: dict[str, Any],
                  contamination_section: dict[str, Any]) -> tuple[str, ...]:
    """Which of :data:`BLOCKER_SPECS` are open, from the derived figures.

    Each test reads a field another function already computed from the live
    balance, the frozen capacity analysis or the package's own terms — so a
    blocker closes here when the underlying fact changes, and not when someone
    remembers to edit a sentence.

    **A blocker being open is not the same as its cause being settled, and the
    three differ.** Evidence is definite: the batteries do not exist. Funding is
    definite for a reason that needs no cost estimate — `phase_d1` is outside
    the package's `funds_formal_sessions_of`, so no allowance covers it at any
    price — while the shortfall figure beside it is provisional. The per-session
    envelope is open because compatibility is UNRESOLVED, not because
    incompatibility is proven: the only comparison available rests on
    FULL-VOCABULARY telemetry, and D1 will run `reference_topk_tail_v1`.

    **Once measured compatibility exists it is authoritative.** The envelope test
    reads `per_session_envelope_compatibility` and nothing else. It used to ALSO
    require that the provisional full-vocab basis fit, which would have let a
    superseded planning figure veto a measurement proving D1 fits.

    **What a GPU round can and cannot settle.** It settles the cost-dependent
    figures: real timing, the resulting project funding requirement, and the
    per-session envelope compatibility -- and for D1 that timing must come from a
    PRODUCTION Top-K path, not from the adoption validation's dual-reducer wall
    clock, which includes work the formal path will not do. It CANNOT settle
    the funding AUTHORIZATION blocker — `phase_d1` being outside
    `funds_formal_sessions_of` is a scope fact, not a price, and no measurement
    puts an experiment inside a package's funded list. That needs an explicit
    maintainer funding decision. Saying the qualification "settles the last two"
    conflated the two, and would have let a repricing read as closing a blocker
    it cannot touch.
    """
    open_: list[str] = []
    #: EVIDENCE is now about whether the realized family EXISTS, not about how
    #: many batteries the original pins could have supported. That capacity
    #: question was answered by extending the sources and building the family; a
    #: blocker still driven by it would reopen a closed problem on every run.
    if d_series_evidence()["status"] != "CLOSED":
        open_.append("evidence")
    #: Either is sufficient, and they are different kinds of fact. The funded
    #: list is categorical; the shortfall is a provisional comparison that a
    #: repricing could move to zero while the list still blocked D1.
    if (not budget_section["d1_is_in_the_funded_list"]
            or budget_section["provisional_shortfall_usd"] > 0):
        open_.append("funding authorization")
    #: MEASURED COMPATIBILITY IS AUTHORITATIVE, and the provisional comparison
    #: does not get a veto over it.
    #:
    #: This used to require BOTH `RESOLVED_FITS` and that the provisional basis
    #: fit -- so a future Top-K measurement proving D1 fits would have stayed
    #: blocked by a FULL-VOCABULARY planning figure it supersedes. That is a
    #: superseded estimate outvoting a measurement, which is backwards.
    #:
    #: The provisional field remains historical planning evidence and is still
    #: reported beside the resolution; it simply no longer decides. While
    #: compatibility is UNRESOLVED it is the only thing there is, so the blocker
    #: is open then -- "not proven to fit" rather than "proven not to".
    compatibility = budget_section["per_session_envelope_compatibility"]
    if compatibility == "RESOLVED_FITS":
        pass
    elif compatibility == "RESOLVED_NEEDS_RAISE":
        open_.append("per-session envelope")
    else:
        open_.append("per-session envelope")
    #: THE CONTROL ARM. `AGREES` is the only state that closes it: a
    #: `DISAGREES` names the wrong checkpoint and an `UNRESOLVED` cannot say
    #: whether it does, and a challenger round must not run against a control
    #: nobody can show was selected.
    if incumbent_identity_check()["status"] != "AGREES":
        open_.append("incumbent identity")
    return tuple(open_)


def _envelope_blocker_note(*, budget_section_compatibility: str,
                           envelope_usd: float,
                           topk: dict[str, Any] | None) -> str:
    """The per-session-envelope note, written from the live state.

    Three states, three notes. The constraint it describes -- that
    `per_attempt_hard_ceiling_usd` binds each session independently of the
    cumulative cap -- is true in all three and is stated in all three; what
    changes is whether it is satisfied, unknown, or violated.
    """
    separate = (
        "This is a SEPARATE constraint from the project-level cap: "
        "`per_attempt_hard_ceiling_usd` binds each session independently, and "
        "the C1 authorization shows the grant issuer refusing a session price "
        "that disagreed with its pricing file -- so a D1 grant that moved only "
        "the cumulative cap would still not authorize the search session.")
    if budget_section_compatibility == "RESOLVED_FITS":
        priced = topk["search_session"]["hard_ceiling_usd"]
        return (f"SATISFIED ON MEASUREMENT: the corrected production Top-K search "
                f"session prices at ${priced:.4f} against the ${envelope_usd:.2f} "
                f"envelope, so this is no longer a blocker. " + separate)
    if budget_section_compatibility == "RESOLVED_NEEDS_RAISE":
        priced = topk["search_session"]["hard_ceiling_usd"]
        return (f"VIOLATED ON MEASUREMENT: the corrected production Top-K search "
                f"session prices at ${priced:.4f} against the ${envelope_usd:.2f} "
                f"envelope, so the envelope would have to move -- a maintainer "
                f"decision, not something this design may take. " + separate)
    return ("WHAT IS DEFINITE: per-session envelope compatibility is UNRESOLVED, "
            "and an unresolved compatibility blocks authorization because "
            "authorization needs a figure it can bind -- not because "
            "incompatibility is proven. What resolves it is a PRODUCTION Top-K "
            "measurement: an un-synced per-candidate MAX over both DEPTH "
            "calibration profiles, rebuilt into an END-TO-END cell with the "
            "committed non-operator overhead. " + separate)


def _blocker_causes(open_: tuple[str, ...],
                    budget_section: dict[str, Any]) -> str:
    """One clause per LIVE blocker, in the contract line.

    Each clause states the live reason rather than a historical one, so a closed
    blocker cannot leave its cause behind as prose.
    """
    clauses = []
    if "evidence" in open_:
        clauses.append("the behavioural evidence does not exist")
    if "funding authorization" in open_:
        clauses.append(
            "`phase_d1` is outside the package's `funds_formal_sessions_of`, so "
            "no allowance covers it at any price")
    if "per-session envelope" in open_:
        state = budget_section["per_session_envelope_compatibility"]
        clauses.append(
            "the search session's envelope compatibility is "
            f"{state}, and authorization needs a figure it can bind"
            if state == "UNRESOLVED" else
            f"the measured search session does not fit the package's "
            f"per-session envelope ({state}), which binds separately from the "
            "cumulative cap")
    return "Live causes: " + "; ".join(clauses) + "."


def _blocker_phrase(open_: tuple[str, ...]) -> str:
    """`THREE INDEPENDENT BLOCKERS: evidence, funding, per-session ceiling`."""
    words = {0: "NO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR"}
    n = len(open_)
    if not n:
        return "NO OPEN BLOCKER"
    return (f"{words.get(n, str(n))} INDEPENDENT BLOCKER"
            f"{'' if n == 1 else 'S'}: {', '.join(open_)}")


def build() -> dict[str, Any]:
    budget_section = budget()
    contamination_section = contamination()
    open_ = open_blockers(budget_section, contamination_section)
    phrase = _blocker_phrase(open_)
    any_one = ("it" if len(open_) == 1
               else "any one of them alone")
    doc = {
        "schema": SCHEMA,
        #: DERIVED, clause by clause. This used to state all three causes
        #: unconditionally -- including "the behavioural evidence cannot be built
        #: under the frozen mixture", which is false since the family was built,
        #: and "the search session alone exceeds the envelope", which contradicts
        #: measured compatibility whenever that resolves. A contract that
        #: enumerates causes must enumerate the LIVE ones.
        "_contract": (
            "The derived D1 protocol. AUTHORIZES NOTHING: it is a design, and it "
            f"records {phrase.lower()}"
            + (" -- each independently sufficient to prevent a launch."
               if len(open_) > 1 else
               ", sufficient on its own to prevent a launch." if open_ else
               #: NO CLAUSE ABOUT MEMBERS OF AN EMPTY SET. The previous wording
               #: read "NO OPEN BLOCKER, and any one of them alone prevents it".
               ". A launch still needs the one-use authorization, which is issued "
               "against the live rate and the derived harness, and which this "
               "document does not grant.")
            + (
                "" if not open_ else " " + _blocker_causes(open_, budget_section))),
        "open_blockers": list(open_),
        "_open_blockers": (
            "DERIVED by `open_blockers()` from the figures in `budget` and "
            "`contamination_protection`, not transcribed. Owners: "
            + "; ".join(f"{name} -> {owner}" for name, owner in BLOCKER_SPECS)),
        #: OUTSIDE THE HASH PREIMAGE, deliberately, and the placement is the
        #: point. `incumbent` IS in the preimage, so putting this check inside
        #: that block would move `design_hash` -- and `design_hash` is the
        #: preimage the behavioural seeds are derived from. Recording the
        #: DISAGREEMENT would then have moved the seeds by itself, before anyone
        #: decided whether to correct the identity, and the correction would
        #: arrive at a design whose seeds had already changed for an unrelated
        #: reason. The finding is reported here; only the fix moves the hash.
        "_incumbent_identity_check": incumbent_identity_check(),
        "experiment_id": "phase_d1",
        "stage_id": "1",
        "_stage_id_meaning": (
            "Stage 1 — projection and structural initialization. D1 varies how "
            "a Stage-1 initialization is SCORED; the 0.86M recovery probes it "
            "trains are the measuring instrument for that question, not the "
            "subject."),
        "status": f"DESIGNED / NOT AUTHORIZED / BLOCKED -- {phrase}",
        "incumbent": incumbent(),
        "recovery_recipe": RECOVERY_RECIPE,
        "hypothesis": hypothesis(),
        "scoring_policy": scoring_policy(),
        "evidence": d_series_evidence(),
        "materialization_prerequisite": materialization_prerequisite(),
        "execution_wiring_required": execution_wiring_required(),
        "execution_protocol": execution_protocol(),
        "gpu_validation_owed": gpu_validation_owed(),
        "search_stage": search_stage(),
        "behavioural_design": behavioural_design(),
        "contamination_protection": {
            "_status": (
                "HISTORICAL REASONING, no longer a live blocker. This analysis "
                "asked how many further disjoint batteries the ORIGINAL pins "
                "could support and answered zero -- which is what prompted the "
                "2026-10-03 source decision. The realized family superseded it; "
                "live evidence readiness is `evidence` above."),
            **contamination_section,
        },
        "budget": budget_section,
        #: THE LIVE OWNERS FIRST, AND THE HISTORICAL ONE NAMED AS HISTORICAL.
        #: This map used to list `evidence_capacity` as a peer of the others,
        #: and a reader who followed it found `batteries_remaining: 0` --
        #: correct for the question that record answers, which is how many
        #: further batteries the ORIGINAL C1 source populations can yield. It
        #: is not the live blocker: the realized D-series family superseded it
        #: on 2026-10-03, `evidence.status` is CLOSED and `open_blockers` is
        #: empty. All three facts were in the same document, and a key that
        #: reads as a live input still sent a reader to the wrong one.
        "inputs": {"c0_preregistration": C0_PREREG, "c1_battery": C1_BATTERY,
                   "a3_comparison": A3_COMPARISON,
                   "budget_terms": BUDGET_TERMS,
                   "behavioural_evidence_family": D_SERIES_FAMILY,
                   "behavioural_evidence_realized": D_SERIES_MANIFEST,
                   "_superseded_capacity_record": CAPACITY,
                   "_why_that_key_is_underscored": (
                       "it answers a HISTORICAL question -- how many further "
                       "batteries the original C1 pool can yield, which is "
                       "zero, binding on MATH-500 -- and is kept as the "
                       "provenance showing why the D-series family was "
                       "necessary. It is NOT the live evidence owner; "
                       "`evidence` above is, and it is CLOSED.")},
        #: HASHED, so no entry here may carry RUN STATE.
        #:
        #: The launch-readiness entry used to interpolate `phrase` -- the live
        #: blocker list -- into a field inside `scientific_preimage`. So
        #: `design_hash` moved every time a blocker opened or closed, and the
        #: behavioural seeds are `SHA256(design_hash + ...)`: opening a FUNDING
        #: blocker silently redrew the scientific seeds. That is the same defect
        #: `_design_hash_covers` describes having fixed for `budget`, surviving
        #: through a derived sentence. The live list stays in `open_blockers`
        #: and in the unhashed `_contract`, where run state belongs; this entry
        #: states the standing rule, which is what is actually scientific about
        #: it and does not move.
        "what_this_may_not_be_used_to_claim": [
            "that target-aware scoring is better. Nothing has been measured; "
            "this is a design.",
            "that D1 is ready to launch. `open_blockers` is the live list and "
            "any open blocker alone prevents it; an empty list is still not an "
            "authorization, which the one-use artifact grants and this document "
            "does not.",
            "that the C2 diagnosis re-opens C2. C2 is CLOSED WITHOUT PROMOTION "
            "and no figure of its is restated, re-analysed or revised here.",
        ],
        "_authorizes": "nothing",
    }
    doc["design_hash"] = sha256_json(scientific_preimage(doc))
    doc["_design_hash_covers"] = (
        "THE SCIENCE AND THE PROTOCOL, and nothing about money or run state. "
        "Composed by `scientific_preimage` as an explicit POSITIVE list rather "
        "than by excluding underscore keys, which used to sweep in `budget` and "
        "made a frozen scientific identity a function of the project's wallet. "
        "See that function's docstring for what moves this hash and what must "
        "not. Money stays hash-bound in each run's one-use authorization.")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    doc = build()
    if args.write:
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    else:
        print(json.dumps(doc, indent=1, sort_keys=True))

    design = doc["behavioural_design"]
    print(f"\ndesign_hash {doc['design_hash'][:16]}")
    cov = doc["search_stage"]["coverage"]
    print(f"  search         : {doc['search_stage']['reachable_leaves']} leaves, "
          f"{doc['search_stage']['cost']['expansions_max']} expansions, "
          f"${doc['search_stage']['cost']['hard_ceiling_usd']:.4f}")
    print(f"  coverage       : {cov['complete_leaves_visited']} of "
          f"{cov['reachable_leaves']} leaves visited "
          f"({cov['fraction_of_leaves_visited'] * 100:.1f}%); every root child "
          f"survives level 0")
    print(f"  behavioural    : Top-{design['top_k']}, "
          f"{design['screening_seeds']} screening seed(s), "
          f"{design['confirmation_seeds']} confirmation seeds, "
          f"{design['total_probes']} probes")
    sens = design["advance_probability_sensitivity"]
    print(f"  screening      : inflation "
          f"{design['screening_estimate_inflation']:.6f} on the SCREENING "
          f"estimate (not a validity condition); P(advance | in Top-K) "
          f"{design['advance_probability']:.4f} "
          f"[{sens['p_at_sd_high']:.3f}–{sens['p_at_sd_low']:.3f}]")
    print(f"  UNKNOWN factor : P(a good candidate is in the Top-{design['top_k']}"
          f") is unmeasured; pipeline probability "
          + ", ".join(f"{k}->{v}" for k, v in
                      design['pipeline_probability_at_assumed_values'].items()))
    print(f"  chain ceiling  : ${doc['budget']['chain']['hard_ceiling_usd']:.4f}")
    #: Printed from the DERIVED list, so the console cannot disagree with the
    #: document about how many blockers are open -- which it did, showing two
    #: while the record carried three.
    detail = {
        "evidence": (
            f"{doc['contamination_protection']['batteries_available']} of "
            f"{doc['contamination_protection']['batteries_D1_requires']} fresh "
            "batteries available"),
        "funding authorization": (
            "phase_d1 is not in funds_formal_sessions_of (definite); "
            f"provisional shortfall ${doc['budget']['provisional_shortfall_usd']:.4f}"),
        "per-session envelope": (
            f"compatibility {doc['budget']['per_session_envelope_compatibility']}; "
            f"provisional ${doc['budget']['chain']['max_session_hard_ceiling_usd']:.4f} "
            f"vs ${doc['budget']['per_session_envelope_usd']:.2f} envelope"),
    }
    print()
    for name in doc["open_blockers"]:
        print(f"  BLOCKER ({name}) : {detail.get(name, 'see the record')}")
    if not doc["open_blockers"]:
        print("  no open blocker")
    print("\n  AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
