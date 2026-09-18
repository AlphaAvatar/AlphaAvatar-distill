#!/usr/bin/env python3
"""Generate the Phase-C2 full-search GRANT PROPOSAL from the live tree.

    PYTHONPATH=src:scripts python \
      scripts/autoinit/write_c2_full_search_grant_proposal.py [--write]

A PROPOSAL, not a grant. It states everything a maintainer would be approving
for one formal full-joint-re-search session, with every machine identity derived
and every figure recomputed, so a launch review has the numbers in front of it.

It is deliberately not promotable by an agent: it states no `granted_by`,
carries no `one_use`, and declares a schema the grant contract does not name, so
`issue_c2_full_search_authorization.py` refuses it. A test exercises that.

Generated rather than hand-written because the first version was hand-written
and went stale the moment the storage derivation was corrected -- and a review
reading a stale proposal is reading another tree's numbers. Deterministic: no
clock, no random, no `generated_utc`, so re-running it on an unchanged tree
produces byte-identical output and a diff means something changed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from experiments.phase_c2 import full_search as FSG  # noqa: E402
from experiments.phase_c2 import full_search_authorization as FA  # noqa: E402

OUT = ("logs/stages/stage-1/phase_c2/plans/"
       "phase_c2_full_search_grant_proposal.json")
SCHEMA = "aadistill.autoinit.c2_full_search_grant_proposal/v1"

#: The rate the proposal is stated at. A launch RE-QUOTES: the issuer queries
#: securePrice and refuses when it no longer matches what a grant approved
#: against, printing what the ceiling would become. So this is the rate that was
#: live when the proposal was generated, not a permission to pay it forever.
RATE_AT_GENERATION = 1.09


def budget_position() -> dict:
    """The project's position, derived by the module that owns it."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import consolidate.derive_budget as D

    pkg_doc = D.load(D.PACKAGE, REPO_ROOT)
    return D.project_balance(
        REPO_ROOT, pkg_doc.get("execution_package", pkg_doc),
        pkg_doc["accepted_pricing"]["cumulative_cap_usd"])


def proposal() -> dict:
    rate = RATE_AT_GENERATION
    total = FSG.total_ceiling_usd(rate, REPO_ROOT)
    provision = FSG.provision_gb(REPO_ROOT)
    storage = FSG.peak_resident_gib(REPO_ROOT)
    environment = FSG.teacher_and_environment_gib(REPO_ROOT)
    identities = {k: v for k, v in FA.live_identities(REPO_ROOT).items()
                  if not k.startswith("_")}
    position = budget_position()
    pricing = FSG.pricing(REPO_ROOT)
    behavioural = pricing["behavioural_selection"]
    behavioural_hours = behavioural["hard_ceiling_minutes"] / 60.0
    behavioural_disk = FSG.storage_cost_usd(behavioural_hours, REPO_ROOT)["usd"]
    chain_total = round(total["total_hard_ceiling_usd"]
                        + behavioural["hard_ceiling_usd"]
                        + behavioural_disk, 4)
    expected_usd = round(FSG.expected_usd(REPO_ROOT)
                         / FSG.price_per_hour_basis(REPO_ROOT) * rate, 4)

    return {
        "schema": SCHEMA,
        "_contract": (
            "A PROPOSAL, not a grant. It states everything a maintainer would "
            "be approving for one formal Phase-C2 full-joint-re-search session, "
            "with every machine identity DERIVED from this tree and every "
            "figure recomputed. It states no `granted_by`, carries no "
            "`one_use`, and its schema is not the one the grant contract names, "
            "so the issuer refuses it -- exercised by a test, not asserted. "
            "AUTHORIZES NOTHING AND APPROVES NOTHING."),
        "_what_it_is_for": (
            "the maintainer's execution sequence ends 'return for FINAL FORMAL "
            "SEARCH LAUNCH REVIEW'. This is what that review reads. If it "
            "approves, the decision supplies `granted_by`, `covers`, "
            "`explicitly_not_authorized`, `approved_money` and `one_use`, and "
            "the result is committed as the run's grant.json under the real "
            "grant schema. Nothing here can be promoted by an agent."),
        "_generated_by": ("scripts/autoinit/write_c2_full_search_grant_proposal.py, "
                          "deterministically from the live tree. Re-run it: an "
                          "unchanged tree gives byte-identical output, so a diff "
                          "means a figure moved."),

        "proposed_for": {
            "experiment_id": "phase_c2_full_search",
            "session": (
                "ONE beam over the derived full joint space at the standing "
                "width, ending at a committed Top-5. It trains nothing, scores "
                "no battery, produces no correct_overall and cannot name an "
                "incumbent."),
            "terminates_at": "commit_top_k",
            "run_id": ("NOT ASSIGNED. The attempt id is chosen when a grant is "
                       "written; every artifact in the chain resolves from it."),
        },

        "money": {
            "_the_ceiling_is_not_the_gpu_price_times_the_minutes": (
                "the provider bills Container Disk separately from the GPU, and "
                "this session provisions "
                f"{provision['provision_gb']} GB of it. An earlier version of "
                "this proposal stated a GPU-only ceiling while the launcher "
                "provisioned hundreds of GB, and no figure in it disagreed with "
                "any other -- review found it, not a gate. A live GPU quote is "
                "not evidence that separately priced storage is free."),
            "gpu": {
                "card": "NVIDIA L40S",
                "securePrice_usd_per_hour": rate,
                "_requoted": ("live at generation, and RE-QUOTED again by the "
                              "issuer immediately before it writes an "
                              "authorization; a moved rate refuses and prints "
                              "the re-derived ceiling"),
                "provider_api_exposes_this": True,
                "hard_ceiling_usd": total["gpu_usd"],
            },
            "container_disk": {
                **total["container_disk"],
                "provider_api_exposes_this": False,
                "_basis": FSG.STORAGE_PRICING,
            },
            "network_volume_usd": total["network_volume_usd"],
            "_no_volume": total["_no_volume"],
            "hard_ceiling_minutes": total["hard_ceiling_minutes"],
            "effective_rate_usd_per_hour": total["effective_rate_usd_per_hour"],
            "expected_usd": expected_usd,
            "TOTAL_hard_ceiling_usd": total["total_hard_ceiling_usd"],
            "max_price_usd_per_hour": rate,
            "_the_boundary": (
                "a launch is refused above this GPU rate, by the launcher's "
                "runtime gate and by the issuer's live re-quote. A higher rate "
                "needs the ceiling re-derived and a new decision; it is not a "
                "launcher flag, and beam width 6 is not narrowed to absorb it."),
        },

        "provisioning": {
            "peak_resident_search_states_gib": storage["peak_resident_gib"],
            "peak_at_level": storage["peak_at_level"],
            "final_resident_gib": storage["final_resident_gib"],
            "teacher_and_environment_gib": environment["total_gib"],
            "required_gib": provision["required_gib"],
            "required_gb": provision["required_gb"],
            "provision_gb": provision["provision_gb"],
            "headroom_multiple": provision["headroom_multiple"],
            "_residency_is_cumulative": storage["_model"],
            "_why_not_search_1s_200": (
                "Search-1 provisions for a 87.4 GiB peak: five retained level-0 "
                "states expanded into eighteen children. Here every operator "
                "kind may go first, so level 0 generates eleven children of "
                "which nine continue and level 1 expands those nine into SIXTY "
                "-- and because the search releases ONLY the partial children a "
                "level prunes, the root, every expanded ancestor, every dead end "
                "and every completed leaf stay resident for the whole run. An "
                "earlier derivation modelled parents + children + leaves and "
                "missed the retained ancestors, giving 243.4 GiB; review caught "
                "that it was not an upper bound before a volume was provisioned "
                "from it."),
            "_cross_checked": (
                "the derivation puts a finished leaf at "
                f"{storage['target_state_gib']} GiB and the CUDA engineering "
                "validation weighed the real 596M student's checkpoint at "
                "exactly that."),
            "_units": provision["_unit"],
        },

        "budget_position": {
            "cumulative_spend_usd": position["cumulative_spend_usd"],
            "authorized_cap_usd": position["cap_usd"],
            "remaining_usd": position["remaining_usd"],
            "search_total_hard_ceiling_usd": total["total_hard_ceiling_usd"],
            "remaining_after_the_search_usd": round(
                position["remaining_usd"] - total["total_hard_ceiling_usd"], 4),
            "behavioural_gpu_hard_ceiling_usd": behavioural["hard_ceiling_usd"],
            "behavioural_storage_upper_bound_usd": behavioural_disk,
            "_behavioural_storage_is_bounded_not_derived": (
                "that session's launcher and provision do not exist yet, so its "
                "disk term is bounded ABOVE by the search's own "
                f"{provision['provision_gb']} GB -- certainly more than twelve "
                "probes on a 596M student need. The chain total is therefore an "
                "over-estimate rather than an unstated omission, and the figure "
                "should fall when that session is bound."),
            "chain_total_hard_ceiling_usd": chain_total,
            "remaining_after_the_whole_chain_usd": round(
                position["remaining_usd"] - chain_total, 4),
            "_derived_by": (
                "scripts/consolidate/derive_budget.py :: project_balance, from "
                "the package anchor plus every recorded run closeout. The cap's "
                "canonical owner is configs/experiments/phase_c1/"
                "authorization.json :: accepted_pricing.cumulative_cap_usd."),
            "_fitting_is_not_permission": (
                "the cap rose to 370.00 on 2026-09-17 as an ACCOUNTING "
                "ENVELOPE, explicitly not a spend authorization and not "
                "transferable to C3/C4. Both sessions fit inside it at their "
                "corrected totals. That is a feasibility statement."),
            "_two_authorizations_never_one": (
                "the behavioural figures are shown so a review can see both "
                "halves at once, NOT because this proposal covers them. That "
                "session's candidate identities do not exist until this search "
                "commits a Top-5, so it cannot be authorized yet even in "
                "principle."),
        },

        "bound_identities_the_issuer_will_reproduce": identities,
        "_identities_note": (
            "every one is DERIVED by full_search_authorization.live_identities "
            "and re-derived at issuance; a grant that asserted a different "
            "value, omitted one, or introduced one the issuer cannot derive is "
            "refused. The space figures are what make a Search-1 grant unusable "
            "here: Search-1 fixed three operators and varied one, this searches "
            "all four jointly with calibration unpinned."),

        "what_a_grant_would_still_have_to_supply": [
            "granted_by -- only a person can say who approved",
            "covers -- the scope sentence carried into the authorization",
            "explicitly_not_authorized -- what the approval refuses",
            "approved_money -- expected, the TOTAL ceiling, the live rate and "
            "the rate boundary, which the issuer recomputes from the pricing "
            "record's minutes at the stated rate plus the derived storage bound",
            "one_use -- how many provider resources the session may draw, which "
            "becomes the enforceable resource scope",
        ],

        "engineering_evidence_this_rests_on": {
            "cuda_validation": ("logs/stages/stage-1/phase_c2/validations/"
                                "full-search-cuda/v1/closeout.json"),
            "verdict": (
                "PASS. The full-search driver reached C2_FULL_SEARCH_ALL_DONE on "
                "a real L40S over the real 578-leaf joint space; all 35 "
                "materialized states reloaded on cuda in bfloat16; the real 596M "
                "student reloaded canonically with its rope base and tied head "
                "intact. $0.1453 of a $0.9000 ceiling, every teardown "
                "provider-confirmed."),
            "setup_dispatch": (
                "the shared setup script's c2_full_search authorization branch "
                "is EXECUTED at $0 against real artifacts -- admitting a valid "
                "one and refusing each governance boundary. It was missing "
                "entirely at 5535c6a, which would have refused a formal pod "
                "after paid setup: the same class as SESSION_KIND=phase_b, which "
                "cost $0.2300."),
            #: The price rests on this one, so it is cited beside the driver
            #: validation rather than left in the analysis: the cost cells were
            #: refreshed from ITS measurement, and a reviewer reading the money
            #: above needs to know which campaign produced the reduction and
            #: what equivalence was established at the same time.
            "performance_validation": ("logs/stages/stage-1/phase_c2/"
                                       "validations/full-search-performance/"
                                       "v1/closeout.json"),
            "performance_verdict": (
                "COMPLETE, and NOT recorded as PASS: no subrun exited 0, and "
                "each failure was in the instrumentation rather than in the "
                "optimizations. What it measured on the real pinned teacher: "
                "the state_eval reduction 76.0x, the DEPTH forward-KL-only "
                "path 1.10x, worst RELATIVE drift 3.03e-05 on pooled per-item "
                "KL over four calibration items, with item ordering identical, "
                "top-1 exact and DEPTH's removal order unchanged. $0.2252 of a "
                "$1.5000 ceiling, three subruns, every teardown "
                "provider-confirmed."),
            "_that_was_a_kernel_result_not_a_decision_result": (
                "the earlier version of this field called 3.03e-05 '257x below "
                "the search's own 0.007782 decision threshold and under "
                "float32's own floor'. Review corrected both halves: 0.007782 "
                "is C2's pre-B numerical-sensitivity DISCLOSURE trigger and not "
                "a decision threshold, the Pareto epsilon is 1e-4 ABSOLUTE, "
                "dividing an absolute gap by a relative drift is not a margin "
                "in any units, and sqrt(V)*eps is an error-scale heuristic "
                "rather than a floor below which agreement is impossible. The "
                "decision-level claim comes from the certification below."),
            "state_eval_certification": ("logs/stages/stage-1/phase_c2/"
                                         "validations/state-eval-certification/"
                                         "v1/"),
            "_why_a_second_validation": (
                "review made launch NO-GO for one narrow reason: the four-item "
                "benchmark demonstrated the speedup and exercised none of the "
                "aggregation the beam ranks on -- the unweighted two-level "
                "domain mean, the worst-domain maximum, the unweighted mean "
                "over critical-token classes -- and contained no rare tag. "
                "The certification runs the COMPLETE frozen suite, 74,022 "
                "positions over 5 domains, 7 sub-types and 4 critical-token "
                "classes, reconstructs the full StateEvaluation under both "
                "implementations on provably identical logits, and checks the "
                "PARETO_V1 decisions including deliberately close cases at the "
                "epsilon boundary. Its targets were predeclared."),
            "_the_ceiling_is_the_CONSERVATIVE_one": (
                "1826.57 min, NOT the 1445.54 the measured component speedups "
                "imply. Review kept the previously accepted window for the "
                "first optimized formal search: a hard ceiling derived by "
                "component-level extrapolation can under-authorize a run, and "
                "an optimized implementation that finishes early simply spends "
                "less than its ceiling. The optimized figure is recorded as an "
                "engineering PLANNING ESTIMATE in the pricing record's "
                "optimized_planning_estimate block, and the first optimized "
                "search's own per-expansion telemetry will replace it."),
            "_what_none_of_it_establishes": (
                "that the formal search fits its budget -- the driver "
                "validation ran toy geometry for 20 seconds and the "
                "performance validation measured components over 4 items and "
                "2 removals -- and nothing behavioural whatsoever. The "
                "largest known saving in the search, the reference cache's "
                "36.1% recompute waste, was DIAGNOSED and not fixed, so it is "
                "deliberately not priced in."),
            #: A CONSEQUENCE of the adoption, surfaced HERE because this is
            #: the document the launch review reads. It is not a defect in the
            #: chain and it does not gate this search; it is a second decision
            #: the same review now owes, and burying it in an analysis file
            #: would be the way to get a launch approved without it being seen.
            "a_second_decision_this_review_also_owes": {
                "what_moved": ("adopting the state_eval optimization changed "
                               "src/aadistill/initialization/planning/"
                               "metrics.py, hash a6dd5d56... -> d193cc90..."),
                "what_binds_it": (
                    "logs/stages/stage-1/phase_c2/plans/"
                    "phase_c2_baseline_completion_protocol.json :: "
                    "cross_session_comparability_contract.bound."
                    "evaluator_implementation_sha256, which names four files "
                    "by content. That one moved; the other three did not."),
                "effect": (
                    "bind_identities in the baseline-completion driver now "
                    "REFUSES, which is the gate working. Six tests in "
                    "tests/pod/test_phase_c2_baseline_completion.py are red "
                    "for this single reason and were deliberately left red."),
                "what_is_NOT_affected": (
                    "no completed result: both sides of every finished "
                    "comparison were measured by ONE implementation, so "
                    "c2_baseline_comparison.json stands. THIS search is not "
                    "gated on it -- the full-search protocol does not bind the "
                    "evaluator by hash and rescores all 578 leaves with one "
                    "implementation. What is refused is a FUTURE B "
                    "re-measurement joining the old series, which is already "
                    "barred without a new decision."),
                "measured_disagreement": (
                    "3.03e-05 RELATIVE on pooled per-item KL over four "
                    "calibration items. The decision-level figure -- ABSOLUTE "
                    "drift on the ranked objectives over the complete frozen "
                    "suite, against the 1e-4 Pareto epsilon -- comes from the "
                    "state-eval certification, and is the one a reader should "
                    "use."),
                "options_all_of_which_are_the_maintainers": [
                    "amend the frozen contract to name both hashes, with the "
                    "measured equivalence as the stated justification",
                    "re-measure B with the new evaluator (a GPU session, "
                    "currently barred)",
                    "revert the optimization, forfeiting the measured 76.0x "
                    "and the $7.2748 the search ceiling fell by",
                ],
                "_nothing_was_done_in_any_of_those_directions": (
                    "changing a frozen scientific protocol is an explicit "
                    "stop condition (AGENTS.md P12.1), and loosening the "
                    "guard to make a suite green is the move the guard exists "
                    "to prevent."),
            },
        },

        "authorizes": "nothing",
        "approves": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    doc = proposal()
    body = json.dumps(doc, indent=1) + "\n"
    out = REPO_ROOT / OUT

    money = doc["money"]
    position = doc["budget_position"]
    print(f"provision        : {doc['provisioning']['provision_gb']} GB for "
          f"{doc['provisioning']['required_gib']} GiB required "
          f"(peak {doc['provisioning']['peak_resident_search_states_gib']} GiB "
          f"at level {doc['provisioning']['peak_at_level']})")
    print(f"GPU ceiling      : ${money['gpu']['hard_ceiling_usd']:.4f} at "
          f"${money['gpu']['securePrice_usd_per_hour']}/h over "
          f"{money['hard_ceiling_minutes']:.0f} min")
    print(f"disk ceiling     : ${money['container_disk']['usd']:.4f} "
          f"({money['container_disk']['provisioned_gb']} GB at "
          f"${money['container_disk']['usd_per_hour']:.6f}/h)")
    print(f"TOTAL ceiling    : ${money['TOTAL_hard_ceiling_usd']:.4f} "
          f"(effective ${money['effective_rate_usd_per_hour']:.6f}/h)")
    print(f"chain total      : ${position['chain_total_hard_ceiling_usd']:.4f}")
    print(f"remaining after  : "
          f"${position['remaining_after_the_whole_chain_usd']:.4f} of "
          f"${position['remaining_usd']:.4f}")
    print(f"identities       : {len(doc['bound_identities_the_issuer_will_reproduce'])}")
    print("APPROVES NOTHING. AUTHORIZES NOTHING.")

    if not args.write:
        if out.is_file() and out.read_text() == body:
            print("unchanged (pass --write to rewrite anyway)")
        else:
            print("WOULD CHANGE — pass --write")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
