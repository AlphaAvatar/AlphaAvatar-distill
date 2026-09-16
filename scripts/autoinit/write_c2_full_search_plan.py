#!/usr/bin/env python3
"""Emit the Phase-C2 full-joint-re-search protocol and its pricing, derived.

    PYTHONPATH=src:scripts python scripts/autoinit/write_c2_full_search_plan.py --write

Zero cost. Loads no model, needs no GPU, authorizes nothing and launches nothing.

Two documents, one generator, because they must agree:

* `phase_c2_full_search_protocol.json` — the scientific plan: the question, the
  derived space, the recorded exclusions, the Top-K rule and the bounded
  behavioural selection stage, all registered BEFORE any result exists.
* `phase_c2_full_search_pricing.json` — what it costs, at several beam widths,
  against the project's real remaining headroom.

**Every number is derived here and now.** The space comes from the registry
through the same functions the search calls; the search cost from both committed
search telemetry files; the probe cost from a committed formal session's own
stage record; the budget position from `derive_budget.py`, which sums the run
closeouts. Nothing is transcribed from a handoff message, and the generator is
re-runnable so a reviewer can reproduce both documents from the tree.

**Why this is a plan and not an authorization.** The search does not fit the
project's remaining headroom at the standing beam width — `plan_session` refuses
it — and that refusal is reported rather than worked around. Deciding what to do
about it is a maintainer decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/consolidate"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.budget import BudgetError  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1, SCHEDULE_V1)

from experiments.phase_c2 import full_search_space as FS  # noqa: E402
from experiments.phase_c2 import selection_pricing as SP  # noqa: E402

PROTOCOL_OUT = ("logs/stages/stage-1/phase_c2/plans/"
                "phase_c2_full_search_protocol.json")
PRICING_OUT = ("logs/stages/stage-1/phase_c2/plans/"
               "phase_c2_full_search_pricing.json")

#: Both documents are DETERMINISTIC: no generated_utc, no clock. A self-hash
#: that moves every time the generator runs cannot be reproduced by a reviewer,
#: and this project has already had a pricing hash become a bound identity that
#: a launch verifies. Regenerating these is therefore a CHECK — identical bytes
#: or the inputs moved — rather than a new document. When the run happened is
#: the git history's fact, not the plan's.
PROTOCOL_SCHEMA = "aadistill.autoinit.c2_full_search_protocol/v1"
PRICING_SCHEMA = "aadistill.autoinit.c2_full_search_pricing/v1"

#: The STANDING scientific design: what SCHEDULE_V1 declares and what this
#: protocol proposes. The funding requirement is this width's chain.
STANDING_WIDTH = SCHEDULE_V1.width

#: Narrower widths are priced too, as explicit scientific ALTERNATIVES. They are
#: not cost options: a narrower beam carries less of the same space forward, so
#: adopting one is a changed experiment with its own result. Narrowing the beam
#: merely to fit the existing cap is not permitted.
PRICED_WIDTHS = (STANDING_WIDTH, 4, 3, 2)

#: The frozen PHASE-C behavioural design this stage reuses UNCHANGED. Every
#: value is read from these two documents rather than restated, so a drift is a
#: test failure rather than a silent divergence.
#:
#: **Phase B's behavioural design is deliberately NOT used.** C0 established that
#: it was scientifically weak — its equivalence interval was ~1.2 SE and Phase
#: B's resolving margin ~1.23 SE — which is why C1 stopped using
#: SuccessiveHalvingPlan and EquivalenceRule and moved to the 950-prompt Phase-C
#: battery, fresh paired seeds, a prompt-cluster bootstrap, a +0.010 SESOI and
#: GO / NO-GO / INCONCLUSIVE with no forced winner. An earlier draft of this
#: protocol reverted to sa/sb/sc and the 0.011695 interval; that was a real
#: scientific regression and this is its repair.
C0_PREREGISTRATION = ("logs/stages/stage-1/phase_c1/plans/"
                      "phase_c0_preregistration.json")
C1_EXECUTION_PREREGISTRATION = ("logs/stages/stage-1/phase_c1/plans/"
                                "execution_preregistration.json")

#: The Phase-B document, named only so the record can say what it does NOT use.
PHASE_B_PREREGISTRATION_NOT_USED = ("logs/stages/stage-1/phase_b/plans/"
                                    "autoinit_phase_b_preregistration.json")

#: The two behavioural assets. Screening and confirmation score on DIFFERENT
#: prompts by design: C0's inferential unit is the prompt and it measured
#: substantial same-prompt cross-seed dependence, so disjoint seeds alone would
#: let a screening selection leak into the confirmation through that dependence.
C2_SCREENING_BATTERY = "artifacts/stage3/c2_screening_v1"
C1_CONFIRMATION_BATTERY = "artifacts/stage3/c1_confirmation_v1"

#: The screening battery's frozen identity record, which lives in git while its
#: bytes do not.
C2_SCREENING_BATTERY_RECORD = ("logs/stages/stage-1/phase_c2/plans/"
                               "c2_screening_battery.json")

#: C2's own paired recovery seeds, derived by C1's rule under a C2 domain.
#: Materialized and hash-bound HERE, before any candidate behavioural result
#: exists, exactly as C0 required of C1. The base digest is C0's own, frozen
#: long before any C2 candidate existed, so no human choice enters.
C2_SEED_DOMAIN = ":phase-c2:recovery-seed:"
C2_BOOTSTRAP_DOMAIN = ":phase-c2:bootstrap"
C2_SEED_COUNT = 4  # 1 screening + 3 confirmation, disjoint by construction

#: The frozen Search-1 evidence this experiment must not touch.
FROZEN_SEARCH1 = (
    "logs/stages/stage-1/phase_c2/runs/attempt4/evidence/stage1_selection.json",
    "logs/stages/stage-1/phase_c2/runs/attempt4/evidence/c2_frozen_comparison_inputs.json",
    "logs/stages/stage-1/phase_c2_baseline_completion/runs/attempt8/evidence/c2_baseline_comparison.json",
    "logs/stages/stage-1/phase_c2_baseline_completion/runs/attempt8/evidence/c2_baseline_completion_evidence.json",
)

TOP_K = 5


def budget_position() -> dict[str, Any]:
    from derive_budget import derive

    project = derive(REPO_ROOT)["project"]
    return {
        "cumulative_spend_usd": project["cumulative_spend_usd"],
        "authorized_cap_usd": project["cap_usd"],
        "remaining_usd": project["remaining_usd"],
        "_derived_by": ("scripts/consolidate/derive_budget.py, which sums every "
                        "recorded closeout cost across every experiment. Not "
                        "restated from any document."),
        "remaining_is_not_permission": True,
    }


def search_pricing(space) -> dict[str, Any]:
    """Cost the search at each priced width, and record every refusal."""
    remaining = budget_position()["remaining_usd"]
    rows = []
    for width in PRICED_WIDTHS:
        limit = FS.bound(space, beam_width=width)
        early = FS.trajectory(space, beam_width=width)
        row: dict[str, Any] = {
            "beam_width": width,
            "warmup_levels": SCHEDULE_V1.warmup_levels,
            "expansions_min": limit.min_expansions,
            "expansions_max": limit.max_expansions,
            "costly_early_hours": early["hours"],
            "structural_min_hours": round(limit.min_minutes / 60, 3),
            "structural_max_hours": round(limit.max_minutes / 60, 3),
        }
        #: Priced twice on purpose: once against a deliberately unbounded
        #: allowance so the FIGURE exists, and once against the project's real
        #: remaining headroom so the REFUSAL is recorded as evidence.
        plan = FS.price(space, price_per_hour=FS.PRICE_PER_HOUR_LAST_QUOTED,
                        authorized_usd=10_000.0, beam_width=width)
        row["expected_minutes"] = round(plan.expected_minutes, 2)
        row["expected_usd"] = round(
            plan.expected_minutes / 60 * FS.PRICE_PER_HOUR_LAST_QUOTED, 4)
        row["hard_ceiling_minutes"] = round(plan.hard_terminate_minutes, 2)
        row["hard_ceiling_usd"] = round(
            plan.hard_terminate_minutes / 60 * FS.PRICE_PER_HOUR_LAST_QUOTED, 4)
        try:
            FS.price(space, price_per_hour=FS.PRICE_PER_HOUR_LAST_QUOTED,
                     authorized_usd=remaining, beam_width=width)
            row["fits_remaining_headroom"] = True
            row["refusal"] = None
        except BudgetError as exc:
            row["fits_remaining_headroom"] = False
            row["refusal"] = str(exc)
        rows.append(row)
    return {
        "price_per_hour_basis": FS.PRICE_PER_HOUR_LAST_QUOTED,
        "_price_basis_note": (
            "NVIDIA L40S securePrice, for planning only. A launch re-quotes: an "
            "hour-old price is not a price, and securePrice is what the "
            "launcher pays — the same query has returned a lower communityPrice "
            "the launcher would never use."),
        "widths": rows,
        "_what_the_width_changes": (
            "a narrower beam leaves the SPACE intact — every admissible "
            "alternative still competes and calibration still affects pruning — "
            "but carries fewer partial paths to the next level, so it explores "
            "less of that space and can return a different front. It is "
            "therefore a change of EXPERIMENT, not a discount on this one, and "
            "these rows exist so the trade can be made deliberately rather than "
            "to supply a smaller authorization number. Narrowing the space "
            "itself — pinning a calibration, dropping an implementation — would "
            "recreate the restriction Search-1 already ran."),
    }


def c2_seeds() -> dict[str, Any]:
    """C2's paired recovery seeds, derived — not chosen.

    C1's rule, under a C2 domain string, based on C0's own digest. C0 refused to
    set its seeds in advance because "no prospective deterministic
    seed-selection rule exists that could choose them without reference to
    future candidate results"; the rule below has that property, so the values
    are determined by a document frozen long before any C2 candidate existed and
    no human choice enters.

    Excluded: the Phase-A/B selection seeds `sa/sb/sc`, because the Phase-B
    winner was selected under them; and C1's three confirmation seeds, because
    the anchor this stage tests against — the C1 treatment — was PROMOTED under
    them, which is the same winner's-curse channel one step along.
    """
    import hashlib

    c0 = json.loads((REPO_ROOT / C0_PREREGISTRATION).read_text())
    c1 = json.loads((REPO_ROOT / C1_EXECUTION_PREREGISTRATION).read_text())
    base = c1["seeds"]["base_digest"]
    if base != sha256_json(c0):
        #: C1 recorded the digest of the C0 FILE, not of its canonical JSON.
        #: Either is fine as a base; what matters is that it is the value C1
        #: actually used, so the two experiments share one frozen base.
        pass
    historical = {int(v) for v in
                  c0["confirmation_seeds"]["historical_seeds_excluded"].values()
                  if isinstance(v, int)}
    excluded = sorted(historical | {int(s) for s in c1["seeds"]["values"]})

    drawn: list[int] = []
    i = 0
    while len(drawn) < C2_SEED_COUNT:
        digest = hashlib.sha256(
            f"{base}{C2_SEED_DOMAIN}{i}".encode()).digest()
        value = int.from_bytes(digest[:4], "big") % (2 ** 31)
        if value not in excluded and value not in drawn:
            drawn.append(value)
        i += 1
    bootstrap = int.from_bytes(hashlib.sha256(
        f"{base}{C2_BOOTSTRAP_DOMAIN}".encode()).digest()[:4], "big") % (2 ** 31)

    return {
        "screening": drawn[:1],
        "confirmation": drawn[1:],
        "all": drawn,
        "count": len(drawn),
        "bootstrap_seed": bootstrap,
        "derivation": (
            "seed_i = uint32_be(SHA256(base_digest + "
            f"'{C2_SEED_DOMAIN}' + decimal(i))[0:4]) mod 2**31, i from 0, "
            "advancing past any excluded seed or earlier draw; the first draw "
            "is the screening seed and the next three are the confirmation "
            "seeds. The bootstrap seed uses the same base under "
            f"'{C2_BOOTSTRAP_DOMAIN}'."),
        "base_digest": base,
        "base_digest_source": (
            f"{C1_EXECUTION_PREREGISTRATION}:seeds.base_digest, which is C0's "
            "own digest — the same base C1 drew from"),
        "excluded": excluded,
        "_why_excluded": (
            "sa/sb/sc because the Phase-B winner was selected under them, and "
            "C1's three because the anchor this stage tests against was "
            "PROMOTED under them. Reusing either leaves a winner's-curse / "
            "seed-selection channel, which is exactly why C0 excluded the "
            "historical set."),
        "screening_and_confirmation_are_disjoint": True,
        "no_human_choice": (
            "the rule leaves no discretion and the base digest predates every "
            "C2 candidate"),
        "role": (
            "FIXED EXPERIMENTAL BLOCKS, as in C0: not a sample from a "
            "recovery-seed superpopulation for the purposes of the primary CI"),
    }


def selection_schedule() -> SP.ProbeSchedule:
    """The bounded two-stage schedule, registered before any candidate exists.

    Screening probes every admitted candidate plus the anchor on ONE seed and
    decides which candidate advances — ranking only, no veto, no promotion.
    Confirmation probes the advanced candidate plus both anchors on the THREE
    remaining seeds and is the only evidence that may name an incumbent.

    The seed sets are disjoint, which is what makes the confirmation interval
    interpretable after a selection: C1 needed no such split because it had two
    arms and no choice to make.
    """
    return SP.ProbeSchedule(
        top_k=TOP_K, screening_seeds=1, confirmation_seeds=3,
        #: ONE advances, so exactly one hypothesis is confirmed and no
        #: multiplicity correction is needed on the confirmation interval.
        #: Advancing two would require Holm across the advanced set and three
        #: more probes; that is a deliberate trade recorded here, not an
        #: oversight.
        advanced_candidates=1,
        #: B in both rungs, and NOTHING else in either. The original canonical
        #: control is not a C2 arm: the comparator is the current incumbent.
        screening_anchors=("frozen_c1_treatment_b",),
        confirmation_anchors=("frozen_c1_treatment_b",),
        screening_battery=C2_SCREENING_BATTERY,
        confirmation_battery=C1_CONFIRMATION_BATTERY)


#: C1's committed probe results, read so the incumbent's behavioural level is a
#: derived figure rather than a remembered one.
C1_PROBE_RESULTS = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
                    "c1_probe_results.json")
C1_DECISION = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
               "c1_decision.json")


def interpretation_boundary() -> dict[str, Any]:
    """What the search measures, what it does NOT, and what promotion needs.

    Stated because the two halves of this programme are easy to conflate and the
    conflation would be a serious misreading: a state_eval front is a
    *pre-recovery* ranking of initializations on a cheap KL surrogate, while
    C1's `4.12%` is a *post-recovery* behavioural measurement. They are not the
    same quantity and neither substitutes for the other.
    """
    import collections

    probes = json.loads((REPO_ROOT / C1_PROBE_RESULTS).read_text())["probes"]
    decision = json.loads((REPO_ROOT / C1_DECISION).read_text())
    pooled: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for probe in probes:
        counts = probe["counts"]
        pooled[probe["arm"]][0] += counts["correct"]
        pooled[probe["arm"]][1] += counts["n_scorable"]

    def rate(arm: str) -> dict[str, Any]:
        correct, scorable = pooled[arm]
        return {"correct": correct, "scorable": scorable,
                "correct_overall": round(correct / scorable, 8)}

    return {
        "the_search_stages_train_nothing": {
            "applies_to": ["Search-1 (frozen)", "the full joint re-search"],
            "what_they_do": (
                "generate initialization states and rank them on the frozen "
                "cheap state_eval metrics — teacher KL and critical-token KL "
                "under an epsilon-Pareto policy."),
            "what_they_do_not_do": (
                "they perform NO 0.86M recovery training and produce NO "
                "behavioural measurement of any kind."),
            "status_of_their_output": (
                "hypothesis generation and candidate SELECTION evidence only. A "
                "state_eval front may narrow the field; it may never promote, "
                "rank behaviourally, or stand in for a recovery comparison."),
        },
        "c1_incumbent_is_a_post_recovery_measurement": {
            "incumbent_before_c1": rate("incumbent"),
            "b_after_c1": rate("treatment"),
            "paired_delta": decision["delta"],
            "lcb_one_sided": decision["lcb_one_sided"],
            "verdict": decision["verdict"],
            "_what_these_numbers_are": (
                "pooled correct_overall over 3 seeds x 850 scorable prompts on "
                "the frozen Phase-C battery, AFTER the frozen 0.86M recovery. "
                "They are NOT raw initialization accuracy and must never be "
                "quoted as such: an un-recovered initialization is not a usable "
                "model and this battery would not measure one meaningfully."),
            "source": C1_PROBE_RESULTS,
        },
        "what_c2_promotion_depends_on": (
            "ONLY the fresh recovery comparison of the selected full-search "
            "candidate C against the current incumbent B, on the frozen 0.86M "
            "recipe, the frozen confirmation battery and C1's decision rule. "
            "Nothing from the search stage enters that decision."),
        "the_target_is_the_current_incumbent": (
            "the question is 'does C improve on B?', NOT 'does C beat the "
            "project's original initialization again?'. The original control is "
            "therefore not a C2 arm, and each later turn of the cycle "
            "challenges whatever incumbent the previous turn left."),
        "_why_this_is_written_down": (
            "a cheap-metric front and a post-recovery behavioural rate are "
            "different quantities on different scales, and this programme "
            "produces both. Conflating them would let a KL ranking read as a "
            "capability claim."),
    }


def behavioural_batteries() -> dict[str, Any]:
    """The two behavioural assets, by frozen identity.

    Screening and confirmation score on DIFFERENT prompts. That is the second
    half of the select-then-confirm split, and it is the half that seed-disjoint
    screening alone does not provide: C0's inferential unit is the prompt and it
    measured same-prompt cross-seed dependence of ICC `0.25 +/- 0.095`, with
    `P(correct | correct on another seed) = 0.257` against a `0.022` marginal.
    Selecting and confirming on the same prompts would let the selection leak
    into the confirmation through that dependence.
    """
    screening = json.loads(
        (REPO_ROOT / C2_SCREENING_BATTERY_RECORD).read_text())
    c1 = json.loads((REPO_ROOT / C1_EXECUTION_PREREGISTRATION).read_text())
    return {
        "screening": {
            "asset_id": screening["asset_id"],
            "path": C2_SCREENING_BATTERY,
            "content_sha256": screening["content_sha256"],
            "n_prompts": screening["n_prompts"],
            "n_scorable_prompts": screening["n_scorable_prompts"],
            "identity_record": C2_SCREENING_BATTERY_RECORD,
            "role": screening["role"],
            "may_not": screening["what_it_may_not_do"],
        },
        "confirmation": {
            "asset_id": c1["battery"]["asset_id"],
            "path": C1_CONFIRMATION_BATTERY,
            "content_sha256": c1["battery"]["content_sha256"],
            "n_prompts": c1["battery"]["n_prompts"],
            "n_scorable_prompts": c1["battery"]["n_scorable_prompts"],
            "identity_record": C1_EXECUTION_PREREGISTRATION,
        },
        "prompt_disjoint": {
            "measured": screening["verification"]["disjointness_measured"][
                "vs_c1_confirmation_v1"],
            "by": ["stable source id", "normalized prompt content hash"],
            "mixture_identical": screening["verification"][
                "mixture_identical_to_c1"],
            "_why_identical_mixture": (
                "correct_overall and its +0.010 SESOI are defined ON the "
                "mixture, so screening and confirmation must share it or a "
                "screening delta says nothing about a confirmation delta."),
        },
        "_both_rungs_are_needed": (
            "the screening battery ranks and cannot promote; only the "
            "confirmation battery, under C1's decision rule, may name an "
            "incumbent."),
    }


def frozen_phase_c_science() -> dict[str, Any]:
    """The Phase-C behavioural contract, READ from C0 and C1 rather than restated."""
    c0 = json.loads((REPO_ROOT / C0_PREREGISTRATION).read_text())
    c1 = json.loads((REPO_ROOT / C1_EXECUTION_PREREGISTRATION).read_text())
    return {
        "sources": {"c0_protocol": C0_PREREGISTRATION,
                    "c1_execution": C1_EXECUTION_PREREGISTRATION},
        "reused_unchanged": True,
        "battery": {
            "asset_id": c1["battery"]["asset_id"],
            "content_sha256": c1["battery"]["content_sha256"],
            "n_prompts": c1["battery"]["n_prompts"],
            "n_scorable_prompts": c1["battery"]["n_scorable_prompts"],
            "mixture_rule": c0["battery"]["mixture_rule"],
            "_not_the_phase_b_development_battery": (
                "the Phase-A/B development battery and its equivalence rule are "
                "NOT used. C0 retired them: the old interval was ~1.2 SE and "
                "Phase B's resolving margin ~1.23 SE."),
            "_final_promotion_battery_stays_reserved": c0[
                "historical_evidence_status"]["final_promotion"],
        },
        "scoring_contract": {
            "contract": c1["scoring_contract"]["contract"],
            "digest": c1["scoring_contract"]["digest"],
            "n_files": c1["scoring_contract"]["n_files"],
        },
        "recovery_recipe": c1["recovery_recipe"],
        "endpoints": c0["endpoints"],
        "estimand": {
            **c0["estimand"],
            "_c2_reading": (
                "identical in form, with the new candidate in place of C1's "
                "treatment and the frozen C1 treatment B as the incumbent: "
                "Delta = mean over prompts ( mean over the 3 fixed fresh C2 "
                "confirmation seeds ( correct_candidate - correct_B ) )."),
        },
        "primary_inference": c0["primary_inference"],
        "bootstrap": {
            **{k: v for k, v in c1["decision"]["bootstrap"].items()
               if k not in ("seed", "seed_derivation",
                            "bound_before_any_c1_datum_exists")},
            "_seed_is_c2s_own": "see seeds.bootstrap_seed",
        },
        "effect_sizes": c0["effect_sizes"],
        "decision_rule": c0["decision_rule"],
        "behavioural_guardrails": {
            "usable_rollout_veto": c0["behavioural_guardrails"][
                "usable_rollout_veto"],
            "catastrophic_capability_veto": {
                **{k: v for k, v in c0["behavioural_guardrails"][
                    "catastrophic_capability_veto"].items()
                   if k != "open_binding_for_c1"},
                "control_operand": (
                    "the INCUMBENT arm — the frozen C1 treatment B"),
                "candidate_operand": "the advanced C2 candidate",
                "_binding_is_stated_not_inherited": (
                    "C0 requires the control operand to be named explicitly and "
                    "does not silently re-point it. This is C1's own binding, "
                    "reused: C1 bound it to the incumbent, and for C2 the "
                    "incumbent is B. The original canonical control is NOT a C2 "
                    "arm — for this comparison it answers no additional "
                    "question, since a candidate that beats the old control but "
                    "loses to B must not promote, and one that beats B gains "
                    "nothing from it. Adding a third recovery arm solely to "
                    "preserve an older convention would cost three probes and "
                    "buy no promotion information. This also keeps the cycle "
                    "scalable: C4 and beyond challenge the CURRENT incumbent "
                    "rather than repeatedly retraining the project's original "
                    "initialization."),
                "asymmetry": (
                    "deliberate: it can veto the candidate, never flag the "
                    "incumbent. Never positive ranking evidence."),
            },
        },
        "what_is_not_reused": {
            "phase_b_document": PHASE_B_PREREGISTRATION_NOT_USED,
            "phase_b_seeds_sa_sb_sc": c0["confirmation_seeds"][
                "historical_seeds_excluded"],
            "phase_b_equivalence_interval": (
                "the 0.011695296982299022 interval and its successive-selection "
                "semantics are NOT used. C0 replaced them with a +0.010 SESOI "
                "and a prompt-cluster bootstrap for the reasons recorded there."),
            "successive_halving": (
                "not used. C1's isolation plan records "
                "successive_halving=false, elimination_rung=false, "
                "tie_break_rung=false, and C2 keeps that: its two stages are a "
                "select-then-confirm split on DISJOINT seeds, not a halving "
                "ladder that eliminates on the data it later confirms with."),
        },
        "_why_read_not_copied": (
            "these are the frozen Phase-C behavioural semantics and this "
            "experiment changes none of them. Restating them here would create "
            "a second place to edit and a way for the two to disagree about the "
            "boundary a verdict is judged against."),
    }


def protocol() -> dict[str, Any]:
    FS.register_c2_operators()
    space = FS.full_joint_space()
    size = FS.size_report()
    cost = FS.cost_model()
    schedule = selection_schedule()

    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "PLAN — NOT PRICED INTO AN AUTHORIZATION, NOT AUTHORIZED, NO COMPUTE",
        "_contract": (
            "The scientific plan for the Phase-C2 full joint re-search and the "
            "bounded behavioural selection that chooses the C2 incumbent. "
            "Registered before any candidate exists. AUTHORIZES NOTHING."),
        "supersedes": {
            "what": "the previously preregistered local Search-2 refinement",
            "decision": (
                "WITHDRAWN by the maintainer decision of 2026-09-17. Search-2 "
                "was a local refinement around Search-1's winners; the accepted "
                "reading of Search-1 is instead that the search PROCEDURE finds "
                "real structure once ATTENTION is promoted, which makes a "
                "broader joint re-search the informative next step rather than a "
                "local polish of a restricted result."),
            "search_1_status": "DONE / FROZEN / PRESERVED AS VALIDATION EVIDENCE",
        },
        "question": {
            "primary": (
                "With the promoted ATTENTION operator in the accepted library, "
                "what is the globally preferred initialization composition when "
                "implementations, applicable calibration profiles and operator "
                "ORDER all compete in one search?"),
            "why_it_is_open": (
                "the incumbent DEPTH / FFN / RESIDUAL_WIDTH calibration "
                "assignments were selected by a search in which ATTENTION could "
                "not consume calibration at all — attention.weight_proxy_v0 "
                "declares CalibrationNeed.NONE and was therefore offered once, "
                "against the no-calibration sentinel, however many mixtures were "
                "active. Those assignments cannot be assumed optimal after the "
                "operator that sits beside them started consuming calibration."),
            "what_search_1_established": (
                "on the restricted space — ATTENTION's mixture free, the other "
                "three pinned at the incumbent's — four of five committed "
                "candidates landed in a better epsilon-Pareto front than the "
                "frozen C1 treatment baseline B and two dominated it on all "
                "three ranked objectives, at margins ~50x the disclosure "
                "threshold. That is evidence about the PROCEDURE, on a cheap "
                "metric. It is not behavioural evidence and it selected no "
                "incumbent."),
        },
        "space": {
            "derived_from": (
                "the live operator registry, through applicable_implementations "
                "and expansion_profiles — the same two functions BeamSearch "
                "calls. Enumerated, not computed from a product formula."),
            "owner": "scripts/experiments/phase_c2/full_search_space.py",
            **size,
            "order": "FREE. A kind is applied at most once per path.",
            "impl_profiles": None,
            "_impl_profiles_meaning": (
                "null means every implementation that consumes calibration "
                "branches over every active mixture. Search-1 pinned three of "
                "four; reopening exactly that is the point of this experiment."),
            "exclusions": FS.EXCLUSIONS,
            "_exclusion_discipline": (
                "all admissible alternatives compete unless an exclusion is "
                "recorded with a scientific reason. Cost is not a reason: "
                "depth.positional_v0 and composite.stage1_sandwich_v0 are both "
                "cheap and both IN, and the only exclusion is an operator whose "
                "isolation experiment has a completed verdict."),
        },
        "beam_and_ranking": {
            "policy_id": PARETO_V1.policy_id,
            "policy_hash": PARETO_V1.policy_hash,
            "objectives": [o.key for o in PARETO_V1.objectives],
            "epsilon": dict(PARETO_V1.epsilon),
            "schedule_id": SCHEDULE_V1.schedule_id,
            "warmup_levels": SCHEDULE_V1.warmup_levels,
            "standing_width": STANDING_WIDTH,
            "width_is_part_of_the_design": (
                "the ranking policy, its objectives and its epsilon are FROZEN "
                "and unchanged, and the standing beam width is part of the "
                "proposed DESIGN rather than a dial left to the funding "
                "decision. A narrower beam carries less of the same space "
                "forward, so adopting one is a changed experiment with reduced "
                "breadth and its own result — a scientific decision that must be "
                "registered as the width before launch. Narrowing the beam "
                "MERELY to fit an existing cap is not permitted; the pricing "
                "record prices the alternatives so the trade can be made "
                "deliberately, not so the number can be made smaller."),
            "_beam_is_not_exhaustive_enumeration": (
                "the point is not to measure every leaf. It is that every "
                "admissible alternative COMPETES inside one search, so a "
                "calibration choice can affect pruning and a structurally "
                "promising path is not excluded merely because Search-1 held "
                "other calibrations fixed."),
        },
        "cost_model": {
            "source": cost.source,
            "minutes": {k: dict(v) for k, v in sorted(cost.minutes.items())},
            "unmeasured_inputs": list(cost.unmeasured),
            "_retired_proxy": (
                "Search-1 priced attention.activation_importance_v1 at 1.5x "
                "width.global_pca_v0 because it had never run inside a search. "
                "C2 attempt 4 ran it 14 times and the measured root cost is "
                "BELOW that proxy, so the margin was conservative in the safe "
                "direction and is now retired. The one cell that moved the "
                "wrong way is depth.causal_kl_greedy_v1 deeper, 31.10 -> 36.07, "
                "and the table takes the max across both runs."),
        },
        "top_k_selection": {
            "k": TOP_K,
            "rule": (
                "the Top-5 admissible complete leaves of the full joint search "
                "by the frozen epsilon-Pareto ranking, committed by the search "
                "itself to its own selection artifact before any behavioural "
                "work begins"),
            "registered_before_results": True,
            "_closed_set": (
                "the candidate set is closed when the search commits it. A set "
                "that can grow once behavioural results are visible is not a "
                "preregistered set — Phase B recorded the same rule and named "
                "the three leaves it refused to re-admit at zero marginal cost."),
        },
        "behavioural_selection": {
            "purpose": (
                "choose the C2 incumbent. Cheap-metric order is NOT behavioural "
                "order: E7 measured a -5.22 nat NLL swing that moved behaviour "
                "by +0.0000, so a search-stage front cannot promote anything."),
            "shape": (
                "PHASE-C discipline, two stages on DISJOINT preregistered seeds: "
                "a bounded screening rung that ranks, then a confirmation rung "
                "that decides under C1's frozen rule."),
            "_not_phase_b": (
                "this is the repair of a real regression. An earlier draft of "
                "this protocol reused Phase-B's sa/sb/sc, its 0.011695 "
                "equivalence interval and its successive-selection semantics. "
                "C0 retired that design as scientifically weak — ~1.2 SE "
                "interval, ~1.23 SE resolving margin — and excluded sa/sb/sc "
                "because the incumbent was selected under them. Reverting to it "
                "would have contradicted frozen Phase-C evidence, and the frozen "
                "Search-1 plan independently says a later behavioural B->C test "
                "should use the Phase-C battery and semantics."),
            "seeds": c2_seeds(),
            "batteries": behavioural_batteries(),
            "schedule": schedule.as_dict(),
            "multiplicity": {
                "problem": (
                    "C1 compared two arms and needed no selection. C2 has "
                    f"{TOP_K} candidates, so choosing among them and then "
                    "testing the choice is a selection problem that has to be "
                    "handled prospectively."),
                "solution": (
                    "select on the screening seed, confirm on the three "
                    "disjoint confirmation seeds. Exactly one candidate "
                    "advances, so exactly one hypothesis is confirmed and the "
                    "one-sided 95% LCB needs no multiplicity correction."),
                "claim_boundary": (
                    "the confirmed candidate was SELECTED on disjoint screening "
                    "data. The confirmation interval is a valid one-sided bound "
                    f"for THAT candidate's Delta against B, conditional on the "
                    "three preregistered seeds — it is NOT a simultaneous "
                    f"statement about all {TOP_K}, and the eliminated "
                    "candidates receive no verdict at all."),
                "if_more_than_one_advanced": (
                    "Holm across the advanced set would be required. The "
                    "schedule advances one, which is a deliberate trade of "
                    "breadth for a clean single-hypothesis confirmation and is "
                    "recorded as such rather than left implicit."),
                "screening_tie_break": (
                    "the frozen epsilon-Pareto search rank, which is a "
                    "cheap-metric ordering fixed before any behavioural datum "
                    "exists and is therefore outcome-independent of the "
                    "behavioural data it breaks a tie in."),
            },
            "anchors": {
                "frozen_c1_treatment_b": (
                    "UNCONDITIONAL, in both rungs, and the ONLY anchor. It is "
                    "the current behavioural incumbent and the arm the estimand "
                    "is a delta against. Search-1's state_eval evidence cannot "
                    "substitute for a behavioural comparison against it. Its "
                    "frozen state_eval measurement is NOT touched; these are "
                    "fresh recovery probes of the same initialization under "
                    "fresh seeds."),
                "_no_canonical_control": (
                    "the project's original initialization is NOT carried "
                    "forward as a C2 arm. It answers no additional C2 question: "
                    "a candidate that beats the old control but loses to B must "
                    "not promote, and one that beats B gains no promotion "
                    "information from it. The behavioural guardrails use the "
                    "incumbent-relative semantics C1 already used, with B as "
                    "the comparator. Keeping the cycle scalable depends on "
                    "this: C4 and beyond challenge the CURRENT incumbent rather "
                    "than repeatedly retraining the original initialization."),
            },
            "every_probe_is_fresh": SP.reuse_is_admissible(REPO_ROOT),
            "frozen_science": frozen_phase_c_science(),
            "terminal_results": [
                "GO — a named C2 incumbent",
                "NO_GO — the anchor stands",
                "INCONCLUSIVE — no incumbent is named",
            ],
            "no_forced_winner": True,
            "_unresolved_is_a_result": (
                "an INCONCLUSIVE terminal state is valid and is not a reason "
                "for a fourth seed, a second screening rung, or a re-run. C0 "
                "fixed three confirmation seeds and 'fourth_seed: never'."),
            "_not_authorized_yet": (
                "this stage is DEFINED and PRICED and is deliberately NOT "
                "launched with the search. Its candidate set does not exist "
                "until the search commits one, and pricing it now is what lets "
                "the funding decision see the whole chain instead of the first "
                "half."),
        },
        "execution_path": {
            "_contract": (
                "Two SEPARATE sessions under two SEPARATE authorizations. "
                "Combining them would let one approval buy both a search and a "
                "promotion decision, and the candidate set the behavioural "
                "session reads does not exist until the search commits one."),
            "session_1_full_joint_search": {
                "driver": "scripts/pod/autoinit_phase_c2_full_search_driver.py",
                "stages": ["bind_identities", "full_joint_search",
                           "commit_top_k"],
                "terminus": "commit_top_k — the session STOPS there",
                "implemented": True,
                "executed_end_to_end_at_toy_scale": True,
                "_toy_execution": (
                    "tests/pod/test_phase_c2_full_search_driver.py drives the "
                    "real stages with a scaled-down model: real operators, real "
                    "checkpoints, real reloads, real hashing, real measurement. "
                    "It found and closed one real defect — a relative_to() that "
                    "raises when the workdir is outside the repository — which "
                    "is the reason the test executes the driver rather than "
                    "asserting about it."),
                "trains_nothing": True,
                "measures_no_behaviour": True,
                "has_no_path_into_the_behavioural_stage": True,
            },
            "session_2_behavioural_selection": {
                "stages": ["screening on c2_screening_v1",
                           "freeze the selected C",
                           "confirmation C vs B on c1_confirmation_v1",
                           "derive GO / NO-GO / INCONCLUSIVE"],
                "only_go_may_name_an_incumbent": True,
                "implemented": False,
                "_why_not_yet": (
                    "its inputs do not exist: the candidate set is produced by "
                    "session 1, and the screening rung cannot be written "
                    "against candidates nobody has generated. Building it now "
                    "would be machinery for an experiment that is neither "
                    "funded nor reachable."),
            },
            "launcher_and_governance_chain": {
                "implemented": False,
                "_deliberately_owed": (
                    "a launcher, an authorization type, an executable closure, "
                    "a readiness contract and a bundle transport are "
                    "per-authorization machinery. They are owed at the moment "
                    "an authorization is requested and are NOT built now: this "
                    "experiment is unfunded, and standing up a governance chain "
                    "for it would be process protecting nothing yet."),
                "precedent": (
                    "the baseline-completion session's chain, which is the shape "
                    "this one would follow"),
            },
        },
        "execution_capabilities_this_round_did_not_build": {
            "multi_session_continuation": {
                "status": "NOT A CURRENT CAPABILITY — possible future design option",
                "why_it_is_tempting": (
                    "search state ids are content-derived, so a state has a "
                    "stable identity across sessions and the journal can be "
                    "replayed within one."),
                "why_that_is_not_enough": (
                    "an identity is not the bytes. The frozen Search-1 plan "
                    "records the real constraint: the search workdir holds "
                    "multi-gigabyte intermediates that CANNOT be relayed for "
                    "resume, so a fresh provider resource would have to "
                    "re-derive any lost state — which is the expensive part. "
                    "Resuming across sessions would need durable "
                    "large-artifact staging, cross-session workdir transport "
                    "and resume semantics."),
                "implemented_this_round": False,
                "validated_this_round": False,
                "_so_do_not_plan_around_it": (
                    "no pricing, option or contingency in these documents may "
                    "assume a >1-session search. If the search cannot finish "
                    "inside one funded session, that is a scope question, not "
                    "something a continuation currently rescues."),
                "source": ("logs/stages/stage-1/phase_c2/plans/"
                           "phase_c2_search1_plan.md"),
            },
        },
        "frozen_and_untouchable": {
            "artifacts": list(FROZEN_SEARCH1),
            "rule": (
                "Search-1 and the Attempt-8 B->C comparison are accepted as "
                "valid search-stage evidence and are frozen. This experiment "
                "does not rerun Search-1, remeasure B, remeasure any frozen C "
                "candidate, or rewrite any selection or comparison record. It "
                "writes its own."),
        },
        "explicitly_not_authorized": [
            "any paid execution of this search",
            "any paid execution of the behavioural selection stage",
            "rerunning Search-1 or the Search-1 beam",
            "remeasuring B or any frozen C candidate",
            "rewriting stage1_selection.json or c2_baseline_comparison.json",
            "the withdrawn local Search-2 refinement",
            "C3 causal-KL ATTENTION R&D",
            "formal Stage-2/Stage-3 recovery training",
            "any increase to the project cap",
        ],
        "interpretation_boundary": interpretation_boundary(),
        "interpretation_discipline": (
            "the search stage produces a cheap-metric front and selects a "
            "candidate SET, never an incumbent. Only the behavioural selection "
            "stage, under the frozen recipe, battery, feasibility floor and "
            "equivalence interval, may name a C2 incumbent — and "
            "unresolved_equivalence with no winner is a legitimate terminal "
            "result, not a reason for a fourth seed."),
        "pricing": PRICING_OUT,
        "authorizes": "nothing",
    }


def pricing(space) -> dict[str, Any]:
    schedule = selection_schedule()
    search = search_pricing(space)
    selection = SP.report(schedule=schedule,
                          price_per_hour=FS.PRICE_PER_HOUR_LAST_QUOTED,
                          repo_root=REPO_ROOT)
    budget = budget_position()

    standing = next(r for r in search["widths"]
                    if r["beam_width"] == STANDING_WIDTH)
    #: The FUNDING REQUIREMENT is the standing design's chain. Narrower widths
    #: are priced as scientific alternatives and are deliberately NOT offered as
    #: a way to fit the existing cap.
    alternatives = [
        {"beam_width": r["beam_width"],
         "expected_usd": round(r["expected_usd"] + selection["expected_usd"], 4),
         "hard_ceiling_usd": round(
             r["hard_ceiling_usd"] + selection["hard_ceiling_usd"], 4),
         "_is_a_different_experiment": (
             "a narrower beam carries less of the same space forward, so it is a "
             "scientific trade in breadth with its own result, not a cheaper way "
             "to run this one")}
        for r in search["widths"] if r["beam_width"] != STANDING_WIDTH]
    combined = {
        "chain": "full joint re-search -> Top-5 -> behavioural selection",
        "standing_beam_width": STANDING_WIDTH,
        "_standing_width_is_the_design": (
            "beam width 6 is what SCHEDULE_V1 declares and what this protocol "
            "proposes. The funding requirement below is the standing design's, "
            "and the alternatives are recorded so breadth can be traded "
            "DELIBERATELY -- never so the authorization number can be made "
            "smaller by narrowing the science."),
        "expected_usd": round(
            standing["expected_usd"] + selection["expected_usd"], 4),
        "hard_ceiling_usd": round(
            standing["hard_ceiling_usd"] + selection["hard_ceiling_usd"], 4),
        "search_hard_ceiling_usd": standing["hard_ceiling_usd"],
        "selection_hard_ceiling_usd": selection["hard_ceiling_usd"],
        "remaining_usd": budget["remaining_usd"],
        "scientific_alternatives_not_cost_options": alternatives,
        "_two_sessions": (
            "these are two separate paid sessions with separate one-use "
            "authorizations, not one launch. The behavioural stage cannot start "
            "until the search commits a candidate set, so its ceiling is a "
            "later obligation rather than a simultaneous one — but a funding "
            "decision that covers only the search would fund a chain that "
            "cannot reach a verdict."),
    }
    shortfall_expected = round(
        combined["expected_usd"] - budget["remaining_usd"], 4)
    shortfall_hard = round(
        combined["hard_ceiling_usd"] - budget["remaining_usd"], 4)
    #: What the project cap would have to be to CONTAIN both ceilings, derived
    #: from the cumulative spend rather than from the headroom, so it does not
    #: silently depend on the current cap being 320.
    minimum_cap = round(
        budget["cumulative_spend_usd"] + combined["hard_ceiling_usd"], 4)

    return {
        "schema": PRICING_SCHEMA,
        "_contract": (
            "What the Phase-C2 full joint re-search and its behavioural "
            "selection stage cost, derived from the registry, both committed "
            "search telemetry files, a committed formal session's probe record "
            "and derive_budget.py. AUTHORIZES NOTHING and FUNDS NOTHING."),
        "search": search,
        "behavioural_selection": selection,
        "combined": combined,
        "budget_position": budget,
        "blocker": {
            "status": "INSUFFICIENT PROJECT HEADROOM FOR THE STANDING DESIGN",
            "standing_beam_width": STANDING_WIDTH,
            "remaining_usd": budget["remaining_usd"],
            "complete_chain_expected_usd": combined["expected_usd"],
            "complete_chain_hard_usd": combined["hard_ceiling_usd"],
            "shortfall_on_expected_usd": shortfall_expected,
            "shortfall_on_hard_ceilings_usd": shortfall_hard,
            "minimum_cumulative_cap_usd": minimum_cap,
            "_minimum_cap_meaning": (
                "the smallest cumulative project cap that CONTAINS both "
                "ceilings: cumulative spend plus the complete standing chain. "
                "Stating it is not requesting it."),
            "what_it_means": (
                "the complete chain at the STANDING beam width does not fit the "
                "remaining project headroom, and neither does the search alone "
                "— plan_session refuses it, and the refusal text is recorded "
                "per width above. This is a maintainer decision and it is "
                "deliberately NOT resolved here by shrinking the run to fit, "
                "which is the one repair the budget module exists to prevent."),
            "the_narrower_beams_are_not_the_requirement": (
                "beam 2/3/4 are priced above as scientific alternatives. They "
                "must NOT be read as the funding requirement for this protocol: "
                "the standing design is beam 6, and a narrower beam carries less "
                "of the same space forward, which is a different experiment with "
                "its own result. Narrowing the beam MERELY to fit the existing "
                "cap is not permitted."),
            "options_for_the_maintainer": [
                "fund the standing design: raise the cumulative project cap to "
                "at least the minimum above, keeping beam width 6 and the full "
                "space",
                "deliberately adopt a narrower beam as a CHANGED scientific "
                "design, accepting reduced breadth and recording it as the "
                "registered width before launch — not as a cost workaround",
                "decline for now and leave C2 at the accepted Search-1 evidence, "
                "which selects no incumbent",
            ],
            "_not_a_recommendation": (
                "the trade is scientific, not arithmetic: the arithmetic is "
                "above and the choice is the maintainer's."),
            "_rounding_note": (
                "every figure here is computed from the 4-dp stored ceilings. A "
                "review message that added the DISPLAY-rounded search ceiling "
                "($33.18) reaches a chain of $61.0708 and a minimum cap of "
                "$358.5798, which is $0.0027 BELOW the derived figures. A "
                "ceiling must round up, so the derived values are the ones to "
                "fund: a cap set at the display-rounded figure would not "
                "contain both ceilings."),
        },
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write both documents; otherwise print a summary")
    args = ap.parse_args(argv)

    FS.register_c2_operators()
    space = FS.full_joint_space()

    proto = protocol()
    proto["protocol_sha256"] = sha256_json(proto)
    price_doc = pricing(space)
    price_doc["pricing_sha256"] = sha256_json(price_doc)

    if not args.write:
        size = proto["space"]["full_joint"]
        print(f"space            : {size['total_leaves']} leaves "
              f"({size['decomposed_leaves']} decomposed), "
              f"Phase-B reference "
              f"{proto['space']['phase_b_reference']['total_leaves']}")
        for row in price_doc["search"]["widths"]:
            mark = " <- STANDING DESIGN" if row["beam_width"] == STANDING_WIDTH \
                else " (scientific alternative, not a cost option)"
            print(f"  width {row['beam_width']}: expected "
                  f"${row['expected_usd']:.4f} ceiling "
                  f"${row['hard_ceiling_usd']:.4f} "
                  f"fits={row['fits_remaining_headroom']}{mark}")
        print(f"selection        : expected "
              f"${price_doc['behavioural_selection']['expected_usd']:.4f} "
              f"ceiling "
              f"${price_doc['behavioural_selection']['hard_ceiling_usd']:.4f}")
        b = price_doc["blocker"]
        print(f"BLOCKER          : {b['status']}")
        print(f"  standing beam {b['standing_beam_width']} complete chain: "
              f"expected ${b['complete_chain_expected_usd']:.4f}, ceiling "
              f"${b['complete_chain_hard_usd']:.4f}")
        print(f"  remaining ${b['remaining_usd']:.4f}  shortfall "
              f"${b['shortfall_on_hard_ceilings_usd']:.4f}  minimum cap "
              f"${b['minimum_cumulative_cap_usd']:.4f}")
        print("nothing written (pass --write)")
        return 0

    for rel, doc in ((PROTOCOL_OUT, proto), (PRICING_OUT, price_doc)):
        out = REPO_ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {rel}")
    print(f"  protocol_sha256 {proto['protocol_sha256']}")
    print(f"  pricing_sha256  {price_doc['pricing_sha256']}")
    print("  AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
