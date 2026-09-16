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

#: The frozen behavioural science this stage reuses UNCHANGED. Every value is
#: read from the Phase-B preregistration rather than restated, so a drift in
#: either document is a test failure rather than a silent divergence.
PHASE_B_PREREGISTRATION = ("logs/stages/stage-1/phase_b/plans/"
                           "autoinit_phase_b_preregistration.json")

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


def selection_schedule() -> SP.ProbeSchedule:
    """The probe schedule, registered before any candidate exists.

    `sa` probes every admitted candidate plus both anchors; `sb` advances the
    two globally best feasible candidates plus both anchors, which advance
    unconditionally because the comparison is *against* them; `sc` is
    conditional and bounded by the number of candidates that can be inside the
    equivalence interval at once.
    """
    anchors = ("canonical_control", "frozen_c1_treatment_b")
    return SP.ProbeSchedule(
        top_k=TOP_K, anchors=anchors,
        sa_probes=TOP_K + len(anchors),
        sb_probes=2 + len(anchors),
        sc_probes_worst_case=2 + len(anchors))


def frozen_behavioural_science() -> dict[str, Any]:
    """The Phase-B behavioural contract, READ rather than restated."""
    doc = json.loads((REPO_ROOT / PHASE_B_PREREGISTRATION).read_text())
    plan = doc["science_plan"]
    return {
        "source": PHASE_B_PREREGISTRATION,
        "reused_unchanged": True,
        "recipe": plan["recipe"],
        "equivalence_interval": plan["equivalence_interval"],
        "feasibility_floor": plan["feasibility_floor"],
        "catastrophic_capability_rule": plan["catastrophic_capability_rule"],
        "seeds": doc["procedure"]["seeds"],
        "selection_rule": doc["procedure"]["selection"],
        "tie_break_authority": doc["procedure"]["tie_break_authority"],
        "terminal_results": doc["procedure"]["terminal_results"],
        "_why_read_not_copied": (
            "these are the frozen Phase-A/B behavioural semantics and this "
            "experiment changes none of them. Restating them here would create "
            "a second place to edit and a way for the two to disagree about "
            "the interval a verdict is judged against."),
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
            "shape": "Phase-B-style rungs over paired seeds",
            "schedule": schedule.as_dict(),
            "anchors": {
                "canonical_control": (
                    "advances unconditionally; it is the floor every candidate "
                    "must clear and the control arm of the catastrophic rule"),
                "frozen_c1_treatment_b": (
                    "the frozen C1 treatment baseline, artifact 53e30566. It "
                    "advances unconditionally because the question is whether "
                    "re-optimizing composition beats it BEHAVIOURALLY, which "
                    "the Search-1 cheap-metric front cannot answer. Its "
                    "state_eval measurement is frozen and is NOT remeasured; "
                    "this is a fresh recovery probe of the same initialization."),
            },
            "every_probe_is_fresh": SP.reuse_is_admissible(REPO_ROOT),
            "frozen_science": frozen_behavioural_science(),
            "_not_authorized_yet": (
                "this stage is DEFINED and PRICED and is deliberately NOT "
                "launched with the search. Its candidate set does not exist "
                "until the search commits one, and pricing it now is what lets "
                "the funding decision see the whole chain instead of the first "
                "half."),
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
