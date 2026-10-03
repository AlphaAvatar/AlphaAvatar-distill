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
BUDGET_TERMS = "configs/experiments/phase_c1/authorization.json"

#: The incumbent the comparison is against, carried forward from C1. C2 closed
#: without promotion and C3 returned NO_GO, so B still stands.
INCUMBENT_STATE_ID = "fe9683e6a9c783bbc6fe276a78c851c6"
INCUMBENT_DIGEST = "c313d1b4081b"

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


def hypothesis() -> dict[str, Any]:
    return {
        "question": (
            "Does making structural scoring and global candidate evaluation "
            "SUPERVISED-TARGET-AWARE produce a better initialization than the "
            "incumbent full-sequence scoring, under the same operator set, the "
            "same calibration data, the same beam and the same recovery recipe?"),
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
        "what_D1_changes": [
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


def execution_wiring_required() -> dict[str, Any]:
    """What a D1 execution entry point must wire, recorded before one exists.

    No D1 launcher is authorized, so this is a CONTRACT and not an
    implementation. It is written down now because every item is a thing the
    core permits a caller to omit, and omitting any of them produces a run
    whose records are valid-looking and scientifically unusable.
    """
    return {
        "status": ("CONTRACT ONLY. No D1 launcher exists and none is "
                   "authorized. Nothing here is implemented or validated."),
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
        "owed_test": (
            "a small D1 EXPERIMENT-suite contract test, written when the driver "
            "is — asserting the evaluator receives the frozen content hash, the "
            "declared policy and environment, and that the search's declared "
            "protocol id is non-null and equals the evaluator's before any "
            "expansion. It belongs in "
            "scripts/experiments/stage-1/phase_d1/tests/, not the core suite: "
            "it checks THIS experiment's wiring, not a reusable mechanism."),
        "_not_a_gate": ("this record blocks nothing today. It exists so the "
                        "requirement is not rediscovered after a paid run "
                        "produced unusable records."),
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
        "_decision": (
            "maintainer engineering decision after A3: batched execution is "
            "adopted across every batchable path from D1 onward, not only "
            "ATTENTION. A3 measured that it is NOT an optimization on "
            "attention.activation_importance_v1 — 8-10% slower on the scorer "
            "with no detectable correctness effect — so this is a uniformity "
            "decision rather than a performance claim. The search ceiling below "
            "is derived from UNBATCHED telemetry, which makes it a PROVISIONAL "
            "PLANNING BASIS and NOT a bound: unbatched timing does not "
            "upper-bound the batched implementation in either direction. A3 "
            "measured the ATTENTION scorer 8.2-10.0% SLOWER at batch 3 while "
            "causal-KL's length-sorted packing won 1.1884x, and the net effect "
            "on an expansion running all four operators is unmeasured. The "
            "direction of the correction is unknown, so the figure sizes a "
            "grant request rather than capping one; only the owed GPU "
            "qualification can turn it into a price."),
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
    cost = d1.search_cost()
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
        "_cost_is_PROVISIONAL": (
            "every cell of the per-expansion table was measured with an "
            "UNBATCHED state evaluation and a one-item-per-forward statistics "
            "pass. D1 runs both batched, and batching does NOT reliably reduce "
            "the time per expansion: A3 measured the ATTENTION scorer 8.2-10.0% "
            "SLOWER at batch 3 on three separate pods with the sign never "
            "flipping, at +29.6% peak VRAM, while length-sorted packing won "
            "1.1884x on causal-KL, whose 60,099 forwards each carry a fixed "
            "ablation setup to amortize. Batching moves different operators in "
            "different directions and the net effect on a D1 expansion is "
            "UNMEASURED. These minutes are a planning ceiling to be refreshed "
            "by the owed short GPU qualification, not a finalized authorization "
            "price; no cell is adjusted on a predicted speed-up, because a "
            "ceiling derived from a prediction is a prediction."),
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
D1_TOP_K = 2
D1_SCREENING_SEEDS = 2
D1_CONFIRMATION_SEEDS = 3


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
        "screening_seeds": D1_SCREENING_SEEDS,
        "confirmation_seeds": D1_CONFIRMATION_SEEDS,
        "screening_probes": chosen["screening_probes"],
        "confirmation_probes": chosen["confirmation_probes"],
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
        "_how_these_three_numbers_were_chosen": (
            "as a pragmatic balance, not as the optimum of a formula. Top-K=2 "
            "buys some behavioural breadth over the search's own ranking while "
            "keeping the screening field small enough that two seeds per arm "
            "give a stable ordering; two screening seeds make the ordering "
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
        "status": "OWED, NOT RUN. Nothing was created and $0 was spent.",
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
        "what_only_a_GPU_can_answer": [
            "whether the batched state evaluation's measured peak matches the "
            "derived `peak_logit_bytes` at the real 151,936 vocabulary, and "
            "whether `batch_plan`'s budget is the right bound",
            "whether the bf16 reductions move a SELECTION under the "
            "target-aware policy, as they were measured to do under the "
            "batching protocol — the operator decisions are integer choices "
            "over float scores and a near-tie can flip",
            "the real per-expansion time under batched statistics and batched "
            "state evaluation. The unbatched cost table is a PROVISIONAL "
            "PLANNING ESTIMATE whose direction relative to the batched "
            "implementation is UNKNOWN -- it neither bounds nor describes it, "
            "because A3 measured batching slower on the ATTENTION scorer and "
            "packing faster on causal-KL, and the net across four operators "
            "has never been measured",
        ],
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
            "formal one and does not transfer into it. It is not requested here: "
            "D1 cannot execute while any of its open blockers stands, so the "
            "validation is owed at authorization time rather than now. It is "
            "nonetheless a PREREQUISITE of pricing, not a consequence of "
            "funding: the chain figures cannot become authorization prices "
            "until it runs."),
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


def budget() -> dict[str, Any]:
    from experiments.phase_d1 import search_space as d1
    from experiments.phase_c2.search_space import PRICE_PER_HOUR_LAST_QUOTED

    #: `chain_cost` searches the frozen space, so the operators must be there.
    _ensure_the_frozen_operators_are_registered()

    design = behavioural_design()
    chain = d1.chain_cost(
        screening_probes=design["screening_probes"],
        confirmation_probes=design["confirmation_probes"])
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
        "price_status": "PROVISIONAL PLANNING BASIS -- DIRECTION UNKNOWN",
        "_price_status": (
            "every figure in `chain` is a PROVISIONAL PLANNING BASIS, NOT a "
            "finalized authorization price and NOT a proven upper bound on "
            "the batched implementation. Two reasons, and the first is the "
            "one that matters:\n\n"
            "1. the per-expansion minutes were measured on an UNBATCHED state "
            "evaluation and a one-item-per-forward statistics pass, and D1 runs "
            "both batched. A3 measured the ATTENTION scorer 8.2-10.0% SLOWER at "
            "batch 3; causal-KL's length-sorted packing won 1.1884x. The net "
            "effect on a D1 expansion running all four operators is UNMEASURED, "
            "so the direction of the correction is unknown, not merely its "
            "size.\n"
            "2. `securePrice` is re-queried live at authorization, so every "
            "dollar figure here is a derived consequence of an hour-old quote.\n\n"
            "WHAT MUST REFRESH IT: a short GPU qualification measuring real "
            "CUDA/bf16 execution, the real state-eval memory peak, the "
            "target-aware batched path's correctness, and the actual timing of "
            "a representative expansion. Until that runs, these numbers size a "
            "grant request; they do not price one."),
        "sessions": 3,
        "_why_three_sessions": (
            "the search commits a candidate set and stops; the screening rung "
            "cannot be bound until that set exists, and the confirmation rung "
            "cannot be bound until screening names the one candidate that "
            "advances. Each is separately priced and separately authorized."),
        "per_session_envelope_usd": terms["per_attempt_hard_ceiling_usd"],
        #: PROVISIONAL IN THE NAME, not only in a docstring. These compare a
        #: planning basis derived from UNBATCHED telemetry against a real
        #: envelope, and the comparison's direction relative to measured batched
        #: D1 is unknown. A field called `fits_per_session_envelope` would be
        #: read later as a finalized authorization fact; this one cannot be.
        "provisional_basis_fits_per_session_envelope":
            chain["max_session_hard_ceiling_usd"]
            <= terms["per_attempt_hard_ceiling_usd"],
        "provisional_per_session_excess_usd": round(
            chain["max_session_hard_ceiling_usd"]
            - terms["per_attempt_hard_ceiling_usd"], 4),
        "per_session_envelope_compatibility": "UNRESOLVED",
        "_per_session_envelope_compatibility": (
            "UNRESOLVED, and it BLOCKS AUTHORIZATION while it is. It is not "
            "RESOLVED-INCOMPATIBLE: nothing has measured the batched D1 search, "
            "so it is NOT established that the real session exceeds "
            f"${terms['per_attempt_hard_ceiling_usd']:.2f}. The provisional "
            "basis says it would, by "
            f"${round(chain['max_session_hard_ceiling_usd'] - terms['per_attempt_hard_ceiling_usd'], 4)}, "
            "and an unresolved compatibility is a blocker because authorization "
            "needs a figure it can bind — not because the incompatibility is "
            "proven. THE OWED GPU QUALIFICATION IS WHAT RESOLVES THIS."),
        "_SECOND_BLOCKER_THE_PER_SESSION_CEILING": (
            "WHAT IS DEFINITE: per-session envelope compatibility is "
            "UNRESOLVED, and an unresolved compatibility blocks authorization. "
            "This is a SEPARATE constraint from the project-level cap — "
            "`per_attempt_hard_ceiling_usd` binds each session independently, "
            "and the C1 authorization shows the grant issuer refusing a session "
            "price that disagreed with its pricing file — so a D1 grant that "
            "moved only the cumulative cap would still not authorize the search "
            "session.\n\n"
            "WHAT IS NOT ESTABLISHED: that the real batched D1 search exceeds "
            f"${terms['per_attempt_hard_ceiling_usd']:.2f}. The provisional "
            "basis puts the search session at "
            f"${chain['max_session_hard_ceiling_usd']:.4f}, over by "
            f"${round(chain['max_session_hard_ceiling_usd'] - terms['per_attempt_hard_ceiling_usd'], 4)}, "
            "but that basis comes from UNBATCHED telemetry whose direction "
            "relative to the batched implementation is unknown. The measured "
            "session could land either side of the envelope, so this document "
            "does NOT claim the search has been shown not to fit.\n\n"
            "WHAT RESOLVES IT: the owed short GPU qualification, which prices a "
            "representative batched expansion. ONLY THEN is the envelope "
            "question answerable, and only then can a grant choose between "
            "raising the per-session envelope for the phase, splitting the "
            "search into sessions that each fit, or a cheaper search protocol. "
            "That resolution is a maintainer decision recorded as one, never "
            "inferred from a cap change."),
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
            "THE SHORTFALL IS PROVISIONAL. "
            f"`provisional_shortfall_usd` = ${round(chain['hard_ceiling_usd'] - project_remaining, 4)} "
            "is the gap between a planning basis derived from UNBATCHED "
            "telemetry and the live balance. It is NOT the finalized amount by "
            "which the project cap must increase: the direction of the batched "
            "correction is unknown, so the real figure is unknown until the "
            "owed GPU qualification reprices the chain. This document does not "
            "claim the cap must move by that amount."),
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
BLOCKER_SPECS: tuple[tuple[str, str], ...] = (
    ("evidence", "contamination_protection.BLOCKER"),
    ("funding", "budget.BLOCKER"),
    ("per-session ceiling", "budget._SECOND_BLOCKER_THE_PER_SESSION_CEILING"),
)


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
    incompatibility is proven: the comparison rests on unbatched telemetry whose
    direction relative to batched D1 is unknown. The owed GPU qualification is
    what settles the last two figures; it cannot change the first.
    """
    open_: list[str] = []
    if (contamination_section["batteries_available"]
            < contamination_section["batteries_D1_requires"]):
        open_.append("evidence")
    #: Either is sufficient, and they are different kinds of fact. The funded
    #: list is categorical; the shortfall is a provisional comparison that a
    #: repricing could move to zero while the list still blocked D1.
    if (not budget_section["d1_is_in_the_funded_list"]
            or budget_section["provisional_shortfall_usd"] > 0):
        open_.append("funding")
    #: Open while compatibility is unresolved. Under the provisional basis it
    #: does not fit; a resolved-and-fits answer needs the qualification, so the
    #: condition is deliberately "not proven to fit" rather than "proven not to".
    if (budget_section["per_session_envelope_compatibility"] != "RESOLVED_FITS"
            or not budget_section["provisional_basis_fits_per_session_envelope"]):
        open_.append("per-session ceiling")
    return tuple(open_)


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
        "_contract": (
            "The derived D1 protocol. AUTHORIZES NOTHING: it is a design, and "
            f"{phrase.lower()} are recorded below, each independently "
            "sufficient to prevent a launch. The behavioural evidence cannot "
            "be built under the frozen mixture; the chain is not fundable at "
            "the current balance; and the search session alone exceeds the "
            "package's per-session envelope, which binds separately from the "
            "cumulative cap."),
        "open_blockers": list(open_),
        "_open_blockers": (
            "DERIVED by `open_blockers()` from the figures in `budget` and "
            "`contamination_protection`, not transcribed. Owners: "
            + "; ".join(f"{name} -> {owner}" for name, owner in BLOCKER_SPECS)),
        "experiment_id": "phase_d1",
        "stage_id": "1",
        "_stage_id_meaning": (
            "Stage 1 — projection and structural initialization. D1 varies how "
            "a Stage-1 initialization is SCORED; the 0.86M recovery probes it "
            "trains are the measuring instrument for that question, not the "
            "subject."),
        "status": f"DESIGNED / NOT AUTHORIZED / BLOCKED -- {phrase}",
        "incumbent": {"state_id": INCUMBENT_STATE_ID,
                      "artifact_digest": INCUMBENT_DIGEST,
                      "_what_it_is": "B, the frozen C1 treatment. C2 closed "
                                     "without promotion and C3 returned NO_GO, "
                                     "so B still stands.",
                      "_not_recovered": "B is an INITIALIZATION. No formal "
                                        "Stage-2/Stage-3 recovery evidence "
                                        "exists for it or for anything else."},
        "recovery_recipe": RECOVERY_RECIPE,
        "hypothesis": hypothesis(),
        "scoring_policy": scoring_policy(),
        "materialization_prerequisite": materialization_prerequisite(),
        "execution_wiring_required": execution_wiring_required(),
        "execution_protocol": execution_protocol(),
        "gpu_validation_owed": gpu_validation_owed(),
        "search_stage": search_stage(),
        "behavioural_design": behavioural_design(),
        "contamination_protection": contamination_section,
        "budget": budget_section,
        "inputs": {"c0_preregistration": C0_PREREG, "c1_battery": C1_BATTERY,
                   "a3_comparison": A3_COMPARISON,
                   "evidence_capacity": CAPACITY,
                   "budget_terms": BUDGET_TERMS},
        "what_this_may_not_be_used_to_claim": [
            "that target-aware scoring is better. Nothing has been measured; "
            "this is a design.",
            "that D1 is ready to launch. " + phrase + ", and " + any_one + " "
            "prevents it.",
            "that the C2 diagnosis re-opens C2. C2 is CLOSED WITHOUT PROMOTION "
            "and no figure of its is restated, re-analysed or revised here.",
        ],
        "_authorizes": "nothing",
    }
    doc["design_hash"] = sha256_json(
        {k: v for k, v in doc.items() if not k.startswith("_")})
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
        "funding": (
            "phase_d1 is not in funds_formal_sessions_of (definite); "
            f"provisional shortfall ${doc['budget']['provisional_shortfall_usd']:.4f}"),
        "per-session ceiling": (
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
