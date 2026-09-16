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

#: The beam widths priced. The standing schedule first, then progressively
#: narrower ones, because the only lever that does not change the SPACE is how
#: much of it the beam carries forward.
PRICED_WIDTHS = (SCHEDULE_V1.width, 4, 3, 2)

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
            "the beam width is the only lever that narrows COST without "
            "narrowing the SPACE: every admissible alternative still competes "
            "and calibration still affects pruning, but fewer partial paths are "
            "carried to the next level. Narrowing the space instead — pinning a "
            "calibration, dropping an implementation — would change the "
            "experiment into a restricted search, which is the thing Search-1 "
            "already did."),
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
            "standing_width": SCHEDULE_V1.width,
            "width_is_a_budget_decision": (
                "the ranking policy, its objectives and its epsilon are FROZEN "
                "and unchanged. The beam WIDTH is the one parameter this plan "
                "leaves to the funding decision, because it trades cost against "
                "how much of the same space is carried forward. The chosen "
                "width must be registered before launch."),
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

    cheapest = min(search["widths"], key=lambda r: r["hard_ceiling_usd"])
    standing = next(r for r in search["widths"]
                    if r["beam_width"] == SCHEDULE_V1.width)
    combined = {
        "chain": "full joint re-search -> Top-5 -> behavioural selection",
        "expected_usd_at_standing_width": round(
            standing["expected_usd"] + selection["expected_usd"], 4),
        "hard_ceiling_usd_at_standing_width": round(
            standing["hard_ceiling_usd"] + selection["hard_ceiling_usd"], 4),
        "expected_usd_at_cheapest_width": round(
            cheapest["expected_usd"] + selection["expected_usd"], 4),
        "hard_ceiling_usd_at_cheapest_width": round(
            cheapest["hard_ceiling_usd"] + selection["hard_ceiling_usd"], 4),
        "remaining_usd": budget["remaining_usd"],
        "_two_sessions": (
            "these are two separate paid sessions with separate one-use "
            "authorizations, not one launch. The behavioural stage cannot start "
            "until the search commits a candidate set, so its ceiling is a "
            "later obligation rather than a simultaneous one — but a funding "
            "decision that covers only the search would fund a chain that "
            "cannot reach a verdict."),
    }
    shortfall_expected = round(
        combined["expected_usd_at_cheapest_width"] - budget["remaining_usd"], 4)
    shortfall_hard = round(
        combined["hard_ceiling_usd_at_cheapest_width"]
        - budget["remaining_usd"], 4)

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
            "status": "INSUFFICIENT PROJECT HEADROOM",
            "remaining_usd": budget["remaining_usd"],
            "cheapest_complete_chain_expected_usd":
                combined["expected_usd_at_cheapest_width"],
            "cheapest_complete_chain_hard_usd":
                combined["hard_ceiling_usd_at_cheapest_width"],
            "shortfall_on_expected_usd": shortfall_expected,
            "shortfall_on_hard_ceilings_usd": shortfall_hard,
            "what_it_means": (
                "the complete chain does not fit the remaining project headroom "
                "at ANY priced beam width, and the search alone does not fit at "
                "the standing width — plan_session refuses it, and the refusal "
                "text is recorded per width above. This is a maintainer "
                "decision: raise the cap, or reduce the scientific scope. It is "
                "deliberately NOT resolved here by shrinking the run to fit, "
                "which is the one repair the budget module exists to prevent."),
            "options_for_the_maintainer": [
                "raise the project cap by at least the shortfall on hard "
                "ceilings, keeping the standing beam width and the full space",
                "fund the search alone at a narrower registered beam width and "
                "decide the behavioural stage separately once a candidate set "
                "exists — accepting that a cheap-metric front promotes nothing",
                "reduce the scientific scope, which means accepting a restricted "
                "search again and losing the joint-pruning property this "
                "experiment exists to obtain",
            ],
            "_not_a_recommendation": (
                "the trade is scientific, not arithmetic: the arithmetic is "
                "above and the choice is the maintainer's."),
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
            print(f"  width {row['beam_width']}: expected "
                  f"${row['expected_usd']:.2f} ceiling "
                  f"${row['hard_ceiling_usd']:.2f} "
                  f"fits={row['fits_remaining_headroom']}")
        print(f"selection        : expected "
              f"${price_doc['behavioural_selection']['expected_usd']:.2f} "
              f"ceiling "
              f"${price_doc['behavioural_selection']['hard_ceiling_usd']:.2f}")
        b = price_doc["blocker"]
        print(f"BLOCKER          : {b['status']} — remaining "
              f"${b['remaining_usd']:.4f}, cheapest complete chain "
              f"${b['cheapest_complete_chain_hard_usd']:.4f}, shortfall "
              f"${b['shortfall_on_hard_ceilings_usd']:.4f}")
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
