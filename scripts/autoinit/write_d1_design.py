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

**It authorizes nothing, and it says so where a reader will see it.** D1 is not
fundable at the current balance and its fresh behavioural evidence cannot be
built under the frozen mixture. Both are derived below and both are stated as
blockers rather than as caveats, because a design that reads as ready when it is
not is how a grant gets requested for an experiment that could not have produced
a valid result.
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
            "tests/autoinit/test_scoring_position_policy.py derives the "
            "restriction independently from each frozen mixture's own fields "
            "and compares; tests/autoinit/"
            "test_target_aware_scoring_end_to_end.py asserts the incumbent "
            "policy changes no artifact and the treatment moves all four "
            "structural decisions."),
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
            "regression": "tests/autoinit/test_materialization_identity.py and "
                          "tests/autoinit/test_target_aware_scoring_end_to_end.py"
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
    """The batching protocol, by maintainer instruction, and what bounds it."""
    return {
        "micro_batch_size": 3,
        "calibration_batch_packing": "length_sorted_v1",
        "_decision": (
            "maintainer engineering decision after A3: batched execution is "
            "adopted across every batchable path from D1 onward, not only "
            "ATTENTION. A3 measured that it is NOT an optimization on "
            "attention.activation_importance_v1 — 8-10% slower on the scorer "
            "with no detectable correctness effect — so this is a uniformity "
            "decision rather than a performance claim, and the search ceiling "
            "below is derived from UNBATCHED telemetry and therefore bounds it "
            "either way."),
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


def search_stage() -> dict[str, Any]:
    from experiments.phase_c2.search_space import register_c2_operators
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from experiments.phase_d1 import search_space as d1

    register_c2_operators()
    activation_importance.register()
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
        "_cost_is_conservative": (
            "every cell of the per-expansion table was measured with an "
            "UNBATCHED state evaluation and a one-item-per-forward statistics "
            "pass. D1 runs both batched, which can only reduce the time per "
            "expansion, so these minutes bound D1 rather than describe it. They "
            "are deliberately not adjusted downward: a ceiling derived from a "
            "predicted speed-up is a prediction."),
        "stops_at": (
            "commit_top_k. The search trains nothing, measures no behaviour and "
            "has no code path into a behavioural stage — the same boundary C2's "
            "driver holds."),
    }


def behavioural_design() -> dict[str, Any]:
    """Top-K and the screening seeds, DERIVED from the measured noise."""
    from experiments.phase_d1 import search_space as d1
    from experiments.phase_d1.selection_noise import (
        SEED_SD, SESOI, advance_probability, expected_max_bias, report as noise,
    )

    grid = d1.designs()
    admissible = [row for row in grid if row["bias_under_sesoi"]]
    #: The choice: among designs whose selection bias is under the SESOI, the one
    #: with the HIGHEST advance probability. It is also the cheapest of those
    #: that screen at all, which is a property of the arithmetic rather than a
    #: compromise — see `_why_narrow_and_replicate`.
    chosen = max((row for row in admissible if row["screening_seeds"] > 0),
                 key=lambda row: (row["advance_probability"],
                                  -row["hard_ceiling_usd"]))
    return {
        "top_k": chosen["top_k"],
        "screening_seeds": chosen["screening_seeds"],
        "confirmation_seeds": 3,
        "screening_probes": chosen["screening_probes"],
        "confirmation_probes": chosen["confirmation_probes"],
        "total_probes": chosen["total_probes"],
        "selection_bias": chosen["selection_bias"],
        "advance_probability": chosen["advance_probability"],
        "sesoi": SESOI,
        "seed_sd": round(SEED_SD, 6),
        "noise_model": noise(),
        "priced_grid": grid,
        "_derivation": (
            "the screening rung advances the best of K candidates on m seeds. "
            "The maximum of K noisy estimates is biased upward even when every "
            "candidate is exactly as good as the anchor, so a design has TWO "
            "requirements: the bias must sit under the SESOI, and the "
            "probability of advancing a candidate that really is better by the "
            "SESOI must be worth the probes. Both are derived from A3's "
            "measured per-seed spread of the paired delta."),
        "_why_narrow_and_replicate": (
            "bias and discrimination pull opposite ways in K, and at this noise "
            "level the narrow design wins on both: K=2 at two screening seeds "
            "has a lower bias AND a higher advance probability than K=3 at two "
            "seeds, at fewer probes. More behavioural breadth is actively worse "
            "unless the seeds grow with it."),
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
            "c2_selection_bias": expected_max_bias(5, 1),
            "c2_bias_over_sesoi": round(expected_max_bias(5, 1) / SESOI, 3),
            "c2_advance_probability": advance_probability(5, 1),
            "reading": (
                "C2's screening rung inflated whichever candidate it advanced "
                "by about 1.5x the effect it was looking for, and was more "
                "likely to advance a candidate that was NOT the better one "
                "(0.42). At the SAME probe count the design above has a bias of "
                "0.52x the SESOI and an advance probability of 0.78. This is a "
                "quantified diagnosis of C2's behavioural stage, derived from "
                "the project's own measurements — it is NOT a re-analysis of "
                "C2's result and it changes no C2 figure."),
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

    Scoped here rather than run, because D1 cannot execute while either blocker
    is open and validating code for an experiment that cannot start is spending
    a paid resource on a question nothing is waiting for. AGENTS.md P8.2 asks a
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
            "state evaluation, which the conservative unbatched cost table "
            "bounds but does not describe",
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
            "D1 cannot execute while either blocker is open, so the validation "
            "is owed at authorization time rather than now."),
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
             "unblocks": "D1, D2 and D3 (about 4,500 further eligible items)"},
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
        "sessions": 3,
        "_why_three_sessions": (
            "the search commits a candidate set and stops; the screening rung "
            "cannot be bound until that set exists, and the confirmation rung "
            "cannot be bound until screening names the one candidate that "
            "advances. Each is separately priced and separately authorized."),
        "per_session_envelope_usd": terms["per_attempt_hard_ceiling_usd"],
        "fits_per_session_envelope": chain["max_session_hard_ceiling_usd"]
            <= terms["per_attempt_hard_ceiling_usd"],
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
        "shortfall_usd": round(chain["hard_ceiling_usd"] - project_remaining, 4),
        "funds_formal_sessions_of": list(funded),
        "d1_is_in_the_funded_list": "phase_d1" in funded,
        "BLOCKER": (
            "D1 is NOT FUNDABLE. The chain's hard ceiling exceeds the project's "
            "entire remaining headroom, and `phase_d1` is not in the C1 "
            "execution package's `funds_formal_sessions_of` list, so no "
            "existing allowance covers it. A maintainer grant is required for "
            "the phase, and the cap would have to move with it."),
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


def build() -> dict[str, Any]:
    doc = {
        "schema": SCHEMA,
        "_contract": (
            "The derived D1 protocol. AUTHORIZES NOTHING: it is a design, and "
            "two independent blockers are recorded below — the behavioural "
            "evidence cannot be built under the frozen mixture, and the chain "
            "is not fundable at the current balance."),
        "experiment_id": "phase_d1",
        "stage_id": "1",
        "_stage_id_meaning": (
            "Stage 1 — projection and structural initialization. D1 varies how "
            "a Stage-1 initialization is SCORED; the 0.86M recovery probes it "
            "trains are the measuring instrument for that question, not the "
            "subject."),
        "status": "DESIGNED / NOT AUTHORIZED / BLOCKED ON EVIDENCE AND FUNDING",
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
        "execution_protocol": execution_protocol(),
        "gpu_validation_owed": gpu_validation_owed(),
        "search_stage": search_stage(),
        "behavioural_design": behavioural_design(),
        "contamination_protection": contamination(),
        "budget": budget(),
        "inputs": {"c0_preregistration": C0_PREREG, "c1_battery": C1_BATTERY,
                   "a3_comparison": A3_COMPARISON,
                   "evidence_capacity": CAPACITY,
                   "budget_terms": BUDGET_TERMS},
        "what_this_may_not_be_used_to_claim": [
            "that target-aware scoring is better. Nothing has been measured; "
            "this is a design.",
            "that D1 is ready to launch. Two blockers are open and either one "
            "alone prevents it.",
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
    print(f"  selection bias : {design['selection_bias']:.6f} against a SESOI "
          f"of {design['sesoi']} (C2's design: "
          f"{design['_what_this_replaces']['c2_selection_bias']:.6f})")
    print(f"  chain ceiling  : ${doc['budget']['chain']['hard_ceiling_usd']:.4f}")
    print(f"\n  BLOCKER (evidence) : "
          f"{doc['contamination_protection']['batteries_available']} of "
          f"{doc['contamination_protection']['batteries_D1_requires']} fresh "
          f"batteries available")
    print(f"  BLOCKER (funding)  : short by "
          f"${doc['budget']['shortfall_usd']:.4f}")
    print("\n  AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
